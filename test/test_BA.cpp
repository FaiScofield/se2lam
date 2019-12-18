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

#define USE_PRESE2  // else use g2o::SE2
//#define USE_XYZCUSTOM  // else use EdgeSE2XYZ
//#define USE_LM    // else use GN algorithm

using namespace std;

bool inBoard(const Eigen::Vector2d& uv)
{
    static const double& w = Config::ImgSize.width;
    static const double& h = Config::ImgSize.height;

    return uv(0) >= 1. && uv(0) <= w - 1. && uv(1) >= 1. && uv(1) <= h - 1.;
}

size_t nKFVertices = 0;
void writeToPCDFile(const string& filename, SlamOptimizer* optimizer)
{
    ofstream of(filename.c_str());
    if (!of.is_open()) {
        cerr << "Error on openning the output file: " << filename << endl;
        return;
    }

    auto vertices = optimizer->vertices();
    int nPose = vertices.size();

    of << "VERSION .7" << '\n'
       << "FIELDS x y z rgb" << '\n'
       << "SIZE 4 4 4 4" << '\n'
       << "TYPE F F F F" << '\n'
       << "COUNT 1 1 1 1" << '\n'
       << "WIDTH " << nPose << '\n'
       << "HEIGHT 1" << '\n'
       << "VIEWPOINT 0 0 0 1 0 0 0" << '\n'
       << "POINTS " << nPose << '\n'
       << "DATA ascii" << endl;

    uint8_t r1, g1, b1, r2, g2, b2;
    r1 = uint8_t(255);
    g1 = uint8_t(255);
    b1 = uint8_t(255);
    r2 = uint8_t(255);
    g2 = uint8_t(255);
    b2 = uint8_t(0);
    uint32_t rgb1 = ((uint32_t)r1 << 16 | (uint32_t)g1 << 8 | (uint32_t)b1);
    uint32_t rgb2 = ((uint32_t)r2 << 16 | (uint32_t)g2 << 8 | (uint32_t)b2);

    for (int i = 0; i < nPose; ++i) {
        if (i < nKFVertices) {
            auto v = static_cast<g2o::VertexSE2*>(optimizer->vertex(i));
            auto pose = v->estimate();
            of << pose[0] << ' ' << pose[1] << ' ' << 0 << " " << rgb1 << '\n';
        } else {
            auto v = static_cast<g2o::VertexSBAPointXYZ*>(optimizer->vertex(i));
            auto pose = v->estimate();
            of << pose[0] << ' ' << pose[1] << ' ' << pose[2] << ' ' << rgb2 << '\n';
        }
    }

    of.close();
}


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

    bool usePreSE2 = false;
    if (argc == 4)
        usePreSE2 = atoi(argv[3]);
    cerr << "usePreSE2 = " << usePreSE2 << endl << endl;

    //! initialization
    string configPath = "/home/vance/dataset/rk/dibeaDataSet/se2_config/";
    Config::readConfig(configPath);

//    string dataFolder = string(argv[1]) + "slamimg";
//    vector<RK_IMAGE> imgFiles;
//    readImagesRK(dataFolder, imgFiles);

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
    const int skip = 0;
    const int delta = 20;
    const int N = min((int)vOdomData.size(), num);
    cout << "Use " << N << " odom data (size of vertices) in the file to test BA. " << endl;
    cout << "skip = " << skip << ", delta = " << delta << endl;

    //! random noise
    cv::RNG rng;  // OpenCV随机数产生器
    const double noiseX = Config::OdoNoiseX * 5;
    const double noiseY = Config::OdoNoiseY * 5;
    const double noiseT = Config::OdoNoiseTheta * 5;
    // const double uncerX = Config::OdoUncertainX ;
    // const double uncerY = Config::OdoUncertainY;
    // const double uncerT = Config::OdoUncertainTheta;

    //! preintergation
    PreSE2 preInfo;
    vector<PreSE2> vPreInfos;
    vPreInfos.reserve(N / delta);

    //! generate observations
    size_t nMPs = 300;
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

    // edges of odo
    for (int i = skip + delta; i < N; ++i) {
        if ((i - skip) % delta != 0)
            continue;

        const int vertexIDTo = (i - skip) / delta;
        auto v1 = static_cast<g2o::VertexSE2*>(optimizer.vertex(vertexIDTo - 1));
        auto v2 = static_cast<g2o::VertexSE2*>(optimizer.vertex(vertexIDTo));

        double* m = vPreInfos[vertexIDTo].meas;  // 第0个不是有效值
        g2o::SE2 meas(m[0], m[1], m[2]);
        Eigen::Map<Eigen::Matrix3d, Eigen::RowMajor> info(vPreInfos[vertexIDTo].cov);
        // bool isSymmetric = info.transpose() == info;
        // if (!isSymmetric) {
        //     cerr << "非对称信息矩阵: " << endl << info << endl;
        // }

#ifdef USE_PRESE2
        // EdgePreSE2
        g2o::PreEdgeSE2* e = new g2o::PreEdgeSE2();
        e->setMeasurement(g2o::Vector3D(m));
#else
        // Edge SE2
        g2o::EdgeSE2* e = new g2o::EdgeSE2();
        e->setMeasurement(meas);
#endif
        e->setVertex(0, v1);
        e->setVertex(1, v2);
        e->setInformation(info * 1e9);
        optimizer.addEdge(e);
        e->computeError();
        cout << "EdgeSE2 from " << v1->id() << " to " << v2->id() << " , chi2 = "
             << e->chi2() << endl;

//        double thetai = v1->estimate().rotation().angle();

//        g2o::Vector2D dt = v2->estimate().translation() - v1->estimate().translation();
//        double si=sin(thetai), ci=cos(thetai);

//        g2o::Matrix3D J1, J2;
//        J1(0, 0) = -ci; J1(0, 1) = -si; J1(0, 2) = -si*dt.x()+ci*dt.y();
//        J1(1, 0) =  si; J1(1, 1) = -ci; J1(1, 2) = -ci*dt.x()-si*dt.y();
//        J1(2, 0) =  0;  J1(2, 1) = 0;   J1(2, 2) = -1;

//        J2(0, 0) = ci; J2(0, 1)= si; J2(0, 2)= 0;
//        J2(1, 0) =-si; J2(1, 1)= ci; J2(1, 2)= 0;
//        J2(2, 0) = 0;  J2(2, 1)= 0;  J2(2, 2)= 1;

//        g2o::SE2 rmean = g2o::SE2(g2o::Vector3D(m));
//        g2o::Matrix3D z = g2o::Matrix3D::Zero();
//        z.block<2, 2>(0, 0) = rmean.rotation().toRotationMatrix();
//        z(2, 2) = 1.;
//        J1 = z * J1;
//        J2 = z * J2;
//        cout <<  "g2o SE2 Jacobian J1 = " << endl << J1 << endl;
//        cout <<  "g2o SE2 Jacobian J2 = " << endl << J2 << endl;
    }
    cout << endl << endl;

    // vertices/edges of MPs
    g2o::SE3Quat Tbc = toSE3Quat(Config::Tbc);
    size_t MPVertexId = nKFVertices;
    for (size_t j = 0; j < nMPs; ++j) {
        // vetex
        const Eigen::Vector3d& lw = vPoints[j];
        g2o::VertexSBAPointXYZ* vj = new g2o::VertexSBAPointXYZ();
        vj->setEstimate(lw);
        vj->setId(MPVertexId++);
        vj->setMarginalized(true);
        optimizer.addVertex(vj);

        // edge
        for (size_t i = 0; i < nKFVertices; ++i) {
            auto vi = static_cast<g2o::VertexSE2*>(optimizer.vertex(i));
            const int idx = vi->id() * delta * skip;
            const g2o::SE3Quat Tcw = Tbc.inverse() * SE2ToSE3_(vOdomData[idx].inverse());
            const Eigen::Vector3d lc = Tcw.map(lw);
            if (lc(2) < 0)
                continue;

            Eigen::Vector2d uv = cam->cam_map(lc);
            if (inBoard(uv)) {
                g2o::RobustKernelCauchy* kernel = new g2o::RobustKernelCauchy();
                kernel->setDelta(1);

                g2o::Matrix2D info;
                info << uv(0) * uv(0), uv(0) * uv(1), uv(1) * uv(0), uv(1) * uv(1);

#ifdef USE_XYZCUSTOM
                EdgeSE2XYZCustom* e = new EdgeSE2XYZCustom();
#else
                g2o::EdgeSE2XYZ* e = new g2o::EdgeSE2XYZ();
#endif
                e->setVertex(0, vi);
                e->setVertex(1, vj);
                e->setMeasurement(uv);
                e->setInformation(info /*Eigen::Matrix2d::Identity()*/);
                e->setCameraParameter(cam);
                e->setExtParameter(Tbc);
                e->setRobustKernel(kernel);
                optimizer.addEdge(e);

                e->computeError();
                cout << "EdgeSE2XYZ from " << vi->id() << " to " << vj->id() << " , chi2 = "
                     << e->chi2() << ", error = [" << e->error().transpose() << "]" << endl;
            }
        }
    }
    double t1 = timer.count();

    optimizer.save("/home/vance/output/test_BA_before.g2o");
    writeToPCDFile("/home/vance/output/test_BA_before.pcd", &optimizer);

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

    optimizer.save("/home/vance/output/test_BA_after.g2o");
    writeToPCDFile("/home/vance/output/test_BA_after.pcd", &optimizer);

    return 0;
}
