#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel.h"
#include "Thirdparty/g2o/g2o/core/sparse_optimizer.h"
#include "Thirdparty/g2o/g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "Thirdparty/g2o/g2o/solvers/dense/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/sba/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/types/slam2d/edge_se2.h"
#include "Thirdparty/g2o/g2o/types/slam2d/vertex_se2.h"
#include "Thirdparty/g2o/g2o/types/slam3d/dquat2mat.h"
#include "Thirdparty/g2o/g2o/types/slam3d/types_slam3d.h"
#include "optimizer.h"
#include "test_functions.hpp"
#include "test_optimizer.hpp"

#define USE_LM    // else use GN algorithm

using namespace std;

const std::string outPointCloudFile = "/home/vance/output/test_BA.ply";
size_t nKFVertices = 0;  // num of camera pose


bool inBoard(const Eigen::Vector2d& uv)
{
    static const double& w = Config::ImgSize.width;
    static const double& h = Config::ImgSize.height;

    return uv(0) >= 1. && uv(0) <= w - 1. && uv(1) >= 1. && uv(1) <= h - 1.;
}

void writeToPlyFile(SlamOptimizer* optimizer)
{
    ofstream of(outPointCloudFile.c_str());
    if (!of.is_open()) {
        cerr << "Error on openning the output file: " << outPointCloudFile << endl;
        return;
    }

    auto vertices = optimizer->vertices();
    const size_t nPose = vertices.size();

    of << "ply" << '\n'
       << "format ascii 1.0" << '\n'
       << "element vertex " << nPose * 2 << '\n'
       << "property float x" << '\n'
       << "property float y" << '\n'
       << "property float z" << '\n'
       << "property uchar red" << '\n'
       << "property uchar green" << '\n'
       << "property uchar blue" << '\n'
       << "end_header" << std::endl;

    for (size_t i = 0; i < nPose; ++i) {
        if (i < nKFVertices) {
            auto v = static_cast<g2o::VertexSE2*>(optimizer->vertex(i));
            auto pose = v->estimate();
            of << pose[0] << ' ' << pose[1] << ' ' << 0 << " 0 0 0" << '\n';
        } else {
            auto v = static_cast<g2o::VertexSBAPointXYZ*>(optimizer->vertex(i));
            auto pose = v->estimate();
            of << pose[0] << ' ' << pose[1] << ' ' << pose[2] << " 0 0 0" << '\n';
        }
    }

    of.close();
}

void writeToPlyFileAppend(SlamOptimizer* optimizer)
{
    ofstream of(outPointCloudFile.c_str(), ios::app);
    if (!of.is_open()) {
        cerr << "Error on openning the output file: " << outPointCloudFile << endl;
        return;
    }

    auto vertices = optimizer->vertices();
    const size_t nPose = vertices.size();

    for (size_t i = 0; i < nPose; ++i) {
        if (i < nKFVertices) {
            auto v = static_cast<g2o::VertexSE2*>(optimizer->vertex(i));
            auto pose = v->estimate();
            of << pose[0] << ' ' << pose[1] << ' ' << 0 << " 255 0 0" << '\n';
        } else {
            auto v = static_cast<g2o::VertexSBAPointXYZ*>(optimizer->vertex(i));
            auto pose = v->estimate();
            of << pose[0] << ' ' << pose[1] << ' ' << pose[2] << " 255 0 0" << '\n';
        }
    }

    of.close();
}

//! TODO
int main(int argc, char** argv)
{
    //! check input
    if (argc < 2) {
        cerr << "Usage: test_pointDetection <dataPath> [number_frames_to_process]" << endl;
        exit(-1);
    }
    int num = INT_MAX;
    if (argc == 3) {
        num = atoi(argv[2]);
        cerr << "set number_frames_to_process = " << num << endl << endl;
    }

    //! initialization
    string configPath = "/home/vance/dataset/rk/dibeaDataSet/se2_config/";
    Config::readConfig(configPath);

    //! read data
    string odomRawFile = string(argv[1]) + "odo_raw.txt";  // [mm]
    ifstream rec(odomRawFile);
    if (!rec.is_open()) {
        cerr << "[Main ][Error] Please check if the odo_raw file exists!" << endl;
        rec.close();
        exit(-1);
    }

    const double scale = 0.1;
    vector<g2o::SE2> vOdomData;
    vOdomData.reserve(2000);
    while (rec.peek() != EOF) {
        string line;
        getline(rec, line);
        istringstream iss(line);
        g2o::Vector3D lineData;
        iss >> lineData[0] >> lineData[1] >> lineData[2];
        lineData[0] *= scale;  // mm换成cm
        lineData[1] *= scale;
        vOdomData.push_back(g2o::SE2(lineData));
    }
    const int skip = 100;
    const int delta = 20;
    const int N = min((int)vOdomData.size(), num);
    cout << "Use " << N << " odom data (size of vertices) in the file to test BA. " << endl;
    cout << "skip = " << skip << ", delta = " << delta << endl;

    //! random noise
    cv::RNG rng;  // OpenCV随机数产生器
    const double noiseX = Config::OdoNoiseX * 5;
    const double noiseY = Config::OdoNoiseY * 5;
    const double noiseT = Config::OdoNoiseTheta * 10;
    // const double uncerX = Config::OdoUncertainX ;
    // const double uncerY = Config::OdoUncertainY;
    // const double uncerT = Config::OdoUncertainTheta;

    //! preintergation
    PreSE2 preInfo;
    vector<PreSE2> vPreInfos;
    vPreInfos.reserve(N / delta);

    //! generate observations
    size_t nMPs = 200;
    vector<Eigen::Vector3d> vPoints;
    vPoints.reserve(nMPs);
    for (size_t i = 0; i < nMPs; ++i) {
        double dx = rng.uniform(0., 0.99999);
        double dy = rng.uniform(0., 0.99999);
        const double x = (dx * 5000. - 1000.) * scale;
        const double y = (dy * 5000.) * scale;
        const double z = (5000.0 + rng.gaussian(200.)) * scale;
        vPoints.push_back(Eigen::Vector3d(x, y, z));
    }

    WorkTimer timer;

    //! g2o solver construction
    SlamOptimizer optimizer;
    SlamLinearSolverCholmod* linearSolver = new SlamLinearSolverCholmod();
    SlamBlockSolver* blockSolver = new SlamBlockSolver(linearSolver);
#ifdef USE_LM
    SlamAlgorithmLM* algo = new SlamAlgorithmLM(blockSolver);
#else
    SlamAlgorithmGN* algo = new SlamAlgorithmGN(blockSolver);
#endif
    optimizer.setAlgorithm(algo);
    CamPara* cam = new CamPara(Config::fx, g2o::Vector2D(Config::cx, Config::cy), 0);
    cam->setId(0);
    optimizer.addParameter(cam);
    optimizer.setVerbose(true);

    // vertices of robot
    for (int i = skip; i < N; ++i) {
        // update preInfo
        if (i != skip) {
            Eigen::Map<Eigen::Vector3d> meas(preInfo.meas);
            g2o::SE2 odok = vOdomData[i - 1].inverse() * vOdomData[i];
            Eigen::Vector2d odork = odok.translation();
            Eigen::Matrix2d Phi_ik = Eigen::Rotation2Dd(meas[2]).toRotationMatrix();
            meas.head<2>() += Phi_ik * odork;
            meas[2] += odok.rotation().angle();

            Eigen::Matrix3d Ak = Eigen::Matrix3d::Identity();
            Eigen::Matrix3d Bk = Eigen::Matrix3d::Identity();
            Ak.block<2, 1>(0, 2) = Phi_ik * Eigen::Vector2d(-odork[1], odork[0]);
            Bk.block<2, 2>(0, 0) = Phi_ik;
            Eigen::Map<Eigen::Matrix3d, Eigen::RowMajor> Sigmak(preInfo.cov);
            Eigen::Matrix3d Sigma_vk = Eigen::Matrix3d::Identity();
            Sigma_vk(0, 0) = (Config::OdoNoiseX * Config::OdoNoiseX);
            Sigma_vk(1, 1) = (Config::OdoNoiseY * Config::OdoNoiseY);
            Sigma_vk(2, 2) = (Config::OdoNoiseTheta * Config::OdoNoiseTheta);
            Eigen::Matrix3d Sigma_k_1 = Ak * Sigmak * Ak.transpose() + Bk * Sigma_vk * Bk.transpose();
            Sigmak = Sigma_k_1;
        }

        if ((i - skip) % delta == 0) {
            const int vertexID = (i - skip) / delta;
            g2o::SE2 noise(rng.gaussian(noiseX * noiseX), rng.gaussian(noiseY * noiseY),
                           rng.gaussian(noiseT * noiseT));
            g2o::VertexSE2* v = new g2o::VertexSE2();
            const bool fixed = vertexID < 1 ? true : false;
            v->setId(vertexID);
            v->setFixed(fixed);
            v->setEstimate(vOdomData[i] * noise);
            optimizer.addVertex(v);
            nKFVertices++;

            // reset preInfo
            vPreInfos.push_back(preInfo);  // 有效值从1开始
            for (size_t k = 0; k < 3; k++)
                preInfo.meas[k] = 0;
            for (size_t k = 0; k < 9; k++)
                preInfo.cov[k] = 0;
        }
    }


    double t1 = timer.count();

    optimizer.save("/home/vance/output/test_BA_before.g2o");
    writeToPlyFile(&optimizer);

    timer.start();
    optimizer.initializeOptimization();
    // optimizer.verifyInformationMatrices(true);
#ifdef USE_LM
    optimizer.optimize(15);
#else
    optimizer.optimize(5);
#endif
    cout << "optimizaiton time cost in (building system/optimization): " << t1 << "/"
         << timer.count() << "ms. " << endl;

    // 用于EdgeXYZ的优化效果可视化, 优化结束后添加SE2边
    for (int i = skip + delta; i < N; ++i) {
        if ((i - skip) % delta != 0)
            continue;
        const int vertexIDTo = (i - skip) / delta;
        auto v1 = static_cast<g2o::VertexSE2*>(optimizer.vertex(vertexIDTo - 1));
        auto v2 = static_cast<g2o::VertexSE2*>(optimizer.vertex(vertexIDTo));
        g2o::EdgeSE2* e = new g2o::EdgeSE2();
        e->setMeasurement(g2o::SE2(1,1,0));
        e->setVertex(0, v1);
        e->setVertex(1, v2);
        optimizer.addEdge(e);
    }

    optimizer.save("/home/vance/output/test_BA_after.g2o");
    writeToPlyFileAppend(&optimizer);

    return 0;
}
