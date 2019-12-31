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
//#include "Thirdparty/g2o/g2o/core/hyper_graph.h"
#include "optimizer.h"
#include "test_functions.hpp"
#include "test_optimizer.hpp"

//#define USE_PRESE2  // else use g2o::SE2
//#define USE_XYZCUSTOM  // else use EdgeSE2XYZ
#define USE_LM    // else use GN algorithm
//#define ONLY_EDGEXYZ // else EDGExyz + EdgeSE2

using namespace std;

const std::string g_outPointCloudFile = "/home/vance/output/test_BA_SE2.ply";
size_t g_nKFVertices = 0;  // num of camera pose
const double g_scale = 0.1;  // 数据尺度, 将单位[mm]换成[cm],方便在g2o_viewer中可视化

bool inBoard(const Eigen::Vector2d& uv)
{
    static const double& w = Config::ImgSize.width;
    static const double& h = Config::ImgSize.height;

    return uv(0) >= 1. && uv(0) <= w - 1. && uv(1) >= 1. && uv(1) <= h - 1.;
}

void writeToPlyFile(SlamOptimizer* optimizer)
{
    ofstream of(g_outPointCloudFile.c_str());
    if (!of.is_open()) {
        cerr << "Error on openning the output file: " << g_outPointCloudFile << endl;
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
        if (i < g_nKFVertices) {
            auto v = static_cast<g2o::VertexSE2*>(optimizer->vertex(i));
            auto pose = v->estimate();
            of << pose[0] << ' ' << pose[1] << ' ' << 0 << " 0 255 0" << '\n';
        } else {
            auto v = static_cast<g2o::VertexSBAPointXYZ*>(optimizer->vertex(i));
            auto pose = v->estimate();
            of << pose[0] << ' ' << pose[1] << ' ' << pose[2] << " 0 255 0" << '\n';
        }
    }

    of.close();
}

void writeToPlyFileAppend(SlamOptimizer* optimizer)
{
    ofstream of(g_outPointCloudFile.c_str(), ios::app);
    if (!of.is_open()) {
        cerr << "Error on openning the output file: " << g_outPointCloudFile << endl;
        return;
    }

    auto vertices = optimizer->vertices();
    const size_t nPose = vertices.size();

    for (size_t i = 0; i < nPose; ++i) {
        if (i < g_nKFVertices) {
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


int main(int argc, char** argv)
{
    // check input
    if (argc < 2) {
        cerr << "Usage: test_pointDetection <dataPath> [number_frames_to_process]" << endl;
        exit(-1);
    }
    int num = INT_MAX;
    if (argc == 3) {
        num = atoi(argv[2]);
        cerr << "set number_frames_to_process = " << num << endl << endl;
    }

    // initialization
    string configPath = "/home/vance/dataset/rk/dibeaDataSet/se2_config/";
    Config::readConfig(configPath);

    // read data
    string odomRawFile = string(argv[1]) + "odo_raw.txt";  // [mm]
    ifstream rec(odomRawFile);
    if (!rec.is_open()) {
        cerr << "[Main ][Error] Please check if the odo_raw file exists!" << endl;
        rec.close();
        exit(-1);
    }

    vector<g2o::SE2> vOdomData;
    vOdomData.reserve(2000);
    while (rec.peek() != EOF) {
        string line;
        getline(rec, line);
        istringstream iss(line);
        g2o::Vector3D lineData;
        iss >> lineData[0] >> lineData[1] >> lineData[2];
        lineData[0] *= g_scale;  // mm换成cm
        lineData[1] *= g_scale;
        vOdomData.push_back(g2o::SE2(lineData));
    }
    const int skip = 100;
    const int delta = 20;
    const int N = min((int)vOdomData.size(), num);
    cout << "Use " << N << " odom data (size of vertices) in the file to test BA. " << endl;
    cout << "skip = " << skip << ", delta = " << delta << endl;

    // random noise
    cv::RNG rng;  // OpenCV随机数产生器
    const double noiseX = Config::OdoNoiseX * 5;
    const double noiseY = Config::OdoNoiseY * 5;
    const double noiseT = Config::OdoNoiseTheta * 8;

    // preintergation
    PreSE2 preInfo;
    vector<PreSE2> vPreInfos;
    vPreInfos.reserve(N / delta);

    // generate observations
    size_t nMPs = 200;
    vector<Eigen::Vector3d> vPoints;
    vPoints.reserve(nMPs);
    for (size_t i = 0; i < nMPs; ++i) {
        double dx = rng.uniform(0., 0.99999);
        double dy = rng.uniform(0., 0.99999);
        const double x = (dx * 5000. - 1000.) * g_scale;
        const double y = (dy * 5000.) * g_scale;
        const double z = (5000.0 + rng.gaussian(200.)) * g_scale;
        vPoints.push_back(Eigen::Vector3d(x, y, z));
    }

    WorkTimer timer;

    // g2o solver construction
    SlamOptimizer optimizer;
    SlamLinearSolverCholmod* linearSolver = new SlamLinearSolverCholmod();
    SlamBlockSolver* blockSolver = new SlamBlockSolver(linearSolver);
#ifdef USE_LM
    SlamAlgorithmLM* algo = new SlamAlgorithmLM(blockSolver);
    algo->setMaxTrialsAfterFailure(5);
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
            Eigen::Vector3d& meas = preInfo.meas;
            g2o::SE2 odok = vOdomData[i - 1].inverse() * vOdomData[i];
            Eigen::Vector2d odork = odok.translation();
            Eigen::Matrix2d Phi_ik = Eigen::Rotation2Dd(meas[2]).toRotationMatrix();
            meas.head<2>() += Phi_ik * odork;
            meas[2] += odok.rotation().angle();

            Eigen::Matrix3d Ak = Eigen::Matrix3d::Identity();
            Eigen::Matrix3d Bk = Eigen::Matrix3d::Identity();
            Ak.block<2, 1>(0, 2) = Phi_ik * Eigen::Vector2d(-odork[1], odork[0]);
            Bk.block<2, 2>(0, 0) = Phi_ik;
            Eigen::Matrix3d& Sigmak = preInfo.cov;
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
            g_nKFVertices++;

            // reset preInfo
            vPreInfos.push_back(preInfo);  // 有效值从1开始
            preInfo.meas.setZero();
            preInfo.cov.setZero();
        }
    }

    // edges of odo
    for (int i = skip + delta; i < N; ++i) {
        if ((i - skip) % delta != 0)
            continue;

        const int vertexIDTo = (i - skip) / delta;
        const auto v1 = static_cast<g2o::VertexSE2*>(optimizer.vertex(vertexIDTo - 1));
        const auto v2 = static_cast<g2o::VertexSE2*>(optimizer.vertex(vertexIDTo));

        Eigen::Vector3d& m = vPreInfos[vertexIDTo].meas;  // 第0个不是有效值
        const g2o::SE2 meas(m[0], m[1], m[2]);
        const Eigen::Matrix3d info = vPreInfos[vertexIDTo].cov.inverse();
        // bool isSymmetric = info.transpose() == info;
        // if (!isSymmetric) {
        //     cerr << "非对称信息矩阵: " << endl << info << endl;
        // }

#ifdef USE_PRESE2
        // EdgePreSE2
        g2o::PreEdgeSE2* e = new g2o::PreEdgeSE2();
        e->setMeasurement(m);
#else
        // Edge SE2
        g2o::EdgeSE2* e = new g2o::EdgeSE2();
//        EdgeSE2Custom* e = new EdgeSE2Custom();
        e->setMeasurement(meas);
#endif
        e->setVertex(0, v1);
        e->setVertex(1, v2);
        e->setInformation(info/*Eigen::Matrix3d::Identity()*/);
#ifdef ONLY_EDGEXYZ
#else
        optimizer.addEdge(e);
#endif

        e->computeError();
        cout << "EdgeSE2 from " << v1->id() << " to " << v2->id() << " , chi2 = "
             << e->chi2() << ", error = [" << e->error().transpose() << "]" << endl;
    }
    cout << endl << endl;

    // vertices/edges of MPs
    const g2o::SE3Quat Tbc = toSE3Quat(Config::Tbc);
    size_t MPVertexId = g_nKFVertices;
    for (size_t j = 0; j < nMPs; ++j) {
        // vetex
        const Eigen::Vector3d& lw = vPoints[j];
        g2o::VertexSBAPointXYZ* vj = new g2o::VertexSBAPointXYZ();
        vj->setEstimate(lw);
        vj->setId(MPVertexId++);
        vj->setMarginalized(true);
        // vj->setFixed(true); // pose optimization
        optimizer.addVertex(vj);

        // edge
        for (size_t i = 0; i < g_nKFVertices; ++i) {
            const auto vi = static_cast<g2o::VertexSE2*>(optimizer.vertex(i));
            const int idx = vi->id() * delta + skip;
            const g2o::SE3Quat Tcw = Tbc.inverse() * SE2ToSE3_(vOdomData[idx].inverse());
            const Eigen::Vector3d lc = Tcw.map(lw);
            if (lc(2) < 0)
                continue;

            const Eigen::Vector2d uv = cam->cam_map(lc);
            if (inBoard(uv)) {
                g2o::RobustKernelCauchy* kernel = new g2o::RobustKernelCauchy();
                kernel->setDelta(1);

#ifdef USE_XYZCUSTOM
                EdgeSE2XYZCustom* e = new EdgeSE2XYZCustom();
#else
                g2o::EdgeSE2XYZ* e = new g2o::EdgeSE2XYZ();
#endif
                e->setVertex(0, vi);
                e->setVertex(1, vj);
                e->setMeasurement(uv);
                e->setInformation(Eigen::Matrix2d::Identity());
                e->setCameraParameter(cam);
                e->setExtParameter(Tbc);
                e->setRobustKernel(kernel);
                optimizer.addEdge(e);

//                e->computeError();
//                cout << "EdgeSE2XYZ from " << vi->id() << " to " << vj->id() << " , chi2 = "
//                     << e->chi2() << ", error = [" << e->error().transpose() << "]" << endl;
            }
        }
    }
    double t1 = timer.count();

    optimizer.save("/home/vance/output/test_BA_SE2_before.g2o");
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

#ifdef ONLY_EDGEXYZ
    // 如果只有EdgeXYZ, 则在优化结束后添加SE2边用于可视化轨迹效果
    for (int i = skip + delta; i < N; ++i) {
        if ((i - skip) % delta != 0)
            continue;
        const int vertexIDTo = (i - skip) / delta;
        const auto v1 = static_cast<g2o::VertexSE2*>(optimizer.vertex(vertexIDTo - 1));
        const auto v2 = static_cast<g2o::VertexSE2*>(optimizer.vertex(vertexIDTo));
        g2o::EdgeSE2* e = new g2o::EdgeSE2();
        e->setVertex(0, v1);
        e->setVertex(1, v2);
        optimizer.addEdge(e);
    }
#endif

    optimizer.save("/home/vance/output/test_BA_SE2_after.g2o");
    writeToPlyFileAppend(&optimizer);

    return 0;
}
