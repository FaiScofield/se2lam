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
//#include "optimizer.h"
#include "test_functions.hpp"

const string g_configPath = "/home/vance/dataset/rk/dibeaDataSet/se2_config/";

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
    Config::readConfig(g_configPath);

//    string dataFolder = string(argv[1]) + "slamimg";
//    vector<RK_IMAGE> imgFiles;
//    readImagesRK(dataFolder, imgFiles);

    string odomRawFile = string(argv[1]) + "odo_raw.txt";  // [mm]
    ifstream rec(odomRawFile);
    if (!rec.is_open()) {
        cerr << "[Main ][Error] Please check if the odo_raw file exists!" << endl;
        rec.close();
        exit(-1);
    }

    //! read data
    const double scale = 0.1;
    vector<g2o::SE2> vOdomData;
    vOdomData.reserve(2000);
    while (rec.peek() != EOF) {
        string line;
        std::getline(rec, line);
        istringstream iss(line);
        g2o::Vector3D lineData;
        iss >> lineData[0] >> lineData[1] >> lineData[2];
        lineData[0] *= scale;  // 换成cm
        lineData[1] *= scale;
        vOdomData.push_back(g2o::SE2(lineData));
    }
    const int skip = 0;
    const int delta = 10;
    const int N = min((int)vOdomData.size(), num);
    cout << "Use " << N << " odom data (size of vertices) in the file to test BA. " << endl;
    cout << "skip = " << skip << ", delta = " << delta << endl;

    //! random noise
    cv::RNG rng;  // OpenCV随机数产生器
    const double noiseX = Config::OdoNoiseX * 15;
    const double noiseY = Config::OdoNoiseY * 15;
    const double noiseT = Config::OdoNoiseTheta * 15;
    // const double uncerX = Config::OdoUncertainX ;
    // const double uncerY = Config::OdoUncertainY;
    // const double uncerT = Config::OdoUncertainTheta;

    //! preintergation
    PreSE2 preInfo;
    vector<PreSE2> vPreInfos;
    vPreInfos.reserve(N / delta);

    WorkTimer timer;

    //! g2o solver construction
    // typedef g2o::LinearSolverCholmod<SlamBlockSolver_3_1::PoseMatrixType> LinearSolverCholmod;

    SlamOptimizer optimizer;
    SlamLinearSolverCholmod* linearSolver = new SlamLinearSolverCholmod();
    SlamBlockSolver* blockSolver = new SlamBlockSolver(linearSolver);
    // LinearSolverCholmod* linearSolver = new LinearSolverCholmod();
    // SlamBlockSolver_3_1* blockSolver = new SlamBlockSolver_3_1(linearSolver);
    // SlamAlgorithmLM* algo = new SlamAlgorithmLM(blockSolver);
    SlamAlgorithmGN* algo = new SlamAlgorithmGN(blockSolver);
    optimizer.setAlgorithm(algo);
    optimizer.setVerbose(true);

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

            // reset preInfo
            vPreInfos.push_back(preInfo);  // 有效值从1开始
            for (size_t k = 0; k < 3; k++)
                preInfo.meas[k] = 0;
            for (size_t k = 0; k < 9; k++)
                preInfo.cov[k] = 0;
        }
    }

    for (int i = skip + delta; i < N; ++i) {
        if ((i - skip) % delta != 0)
            continue;

        const int vertexIDTo = (i - skip) / delta;
        auto v1 = static_cast<g2o::VertexSE2*>(optimizer.vertex(vertexIDTo - 1));
        auto v2 = static_cast<g2o::VertexSE2*>(optimizer.vertex(vertexIDTo));

        double* m = vPreInfos[vertexIDTo].meas;  // 第0个不是有效值
        g2o::SE2 meas(m[0], m[1], m[2]);
        Eigen::Map<Eigen::Matrix3d, Eigen::RowMajor> info(vPreInfos[vertexIDTo].cov);

        // Edge SE2
        g2o::EdgeSE2* e = new g2o::EdgeSE2();
        e->setVertex(0, v1);
        e->setVertex(1, v2);
        e->setMeasurement(meas);
        e->setInformation(info/*Eigen::Matrix3d::Identity()*/);
/*
//        e->computeError();
//        auto error1 = e->error();
        // e->linearizeOplus();
//        double thetai = v1->estimate().rotation().angle();
//        g2o::Vector2D dt = v2->estimate().translation() - v1->estimate().translation();
//        double si=sin(thetai), ci=cos(thetai);
//        g2o::Matrix3D Jxi1, Jxj1;
//        Jxi1(0, 0) = -ci; Jxi1(0, 1) = -si; Jxi1(0, 2) = -si*dt.x()+ci*dt.y();
//        Jxi1(1, 0) =  si; Jxi1(1, 1) = -ci; Jxi1(1, 2) = -ci*dt.x()-si*dt.y();
//        Jxi1(2, 0) =  0;  Jxi1(2, 1) = 0;   Jxi1(2, 2) = -1;
//        Jxj1(0, 0) = ci; Jxj1(0, 1)= si; Jxj1(0, 2)= 0;
//        Jxj1(1, 0) =-si; Jxj1(1, 1)= ci; Jxj1(1, 2)= 0;
//        Jxj1(2, 0) = 0;  Jxj1(2, 1)= 0;  Jxj1(2, 2)= 1;
//        const g2o::SE2& rmean = meas.inverse();
//        g2o::Matrix3D z = g2o::Matrix3D::Zero();
//        z.block<2, 2>(0, 0) = rmean.rotation().toRotationMatrix();
//        z(2, 2) = 1.;
//        Jxi1 = z * Jxi1;
//        Jxj1 = z * Jxj1;
*/
        // EdgePreSE2
        g2o::PreEdgeSE2* pe = new g2o::PreEdgeSE2();
        pe->setVertex(0, v1);
        pe->setVertex(1, v2);
        pe->setMeasurement(g2o::Vector3D(m));
        pe->setInformation(info/*Eigen::Matrix3d::Identity()*/);
/*
//        pe->computeError();
//        auto error2 = pe->error();
//        // pe->linearizeOplus();
//        g2o::Matrix3D Jxi2, Jxj2;
//        g2o::Matrix2D Ri = v1->estimate().rotation().toRotationMatrix();
//        g2o::Vector2D ri = v1->estimate().translation();
//        g2o::Vector2D rj = v2->estimate().translation();
//        g2o::Vector2D rij = rj - ri;
//        g2o::Vector2D rij_x(-rij[1], rij[0]);
//        Jxi2.block<2, 2>(0, 0) = -Ri.transpose();
//        Jxi2.block<2, 1>(0, 2) = -Ri.transpose() * rij_x;
//        Jxi2.block<1, 2>(2, 0).setZero();
//        Jxi2(2, 2) = -1;
//        Jxj2.setIdentity();
//        Jxj2.block<2, 2>(0, 0) = Ri.transpose();
//        g2o::Matrix3D Rm = g2o::Matrix3D::Identity();
//        Rm.block<2, 2>(0, 0) = Eigen::Rotation2Dd(m[2]).toRotationMatrix();
//        Jxi2 = Rm * Jxi2;
//        Jxj2 = Rm * Jxj2;
*/
//        cout << endl;
//        cout << "Edge " << vertexIDTo << ": Error1 = [" << error1.transpose() << "]" << endl;
//        cout << "Edge " << vertexIDTo << ": Error2 = [" << error2.transpose() << "]" << endl;
//        cout << "Edge " << vertexIDTo << ": Jacobian1 is:" << endl << Jxi1 << endl << Jxj1 << endl;
//        cout << "Edge " << vertexIDTo << ": Jacobian2 is:" << endl << Jxi2 << endl << Jxj2 << endl;

        if (usePreSE2)
            optimizer.addEdge(pe);
        else
            optimizer.addEdge(e);
    }
    double t1 = timer.count();

    optimizer.save("/home/vance/output/test_BA_before.g2o");

    timer.start();
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    cout << "optimizaiton time cost in (building system/optimization): " << t1 << "/"
         << timer.count() << "ms. " << endl;

    optimizer.save("/home/vance/output/test_BA_after.g2o");

    return 0;
}
