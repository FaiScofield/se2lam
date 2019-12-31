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

#define USE_LM          1  // else use GN algorithm
#define USE_PRESE3_EDGE 1
#define USE_SE3_EDGE    1
#define SAVE_GT_G2O     0

using namespace std;

const std::string g_outPointCloudFile = "/home/vance/output/test_BA_SE3.ply";
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

    size_t i = 0;
    while (i < g_nKFVertices) {
        auto v = static_cast<g2o::VertexSE3*>(optimizer->vertex(i));
        const auto T = v->estimate();
        const auto pose = T.translation().matrix();
        of << pose[0] << ' ' << pose[1] << ' ' << pose[2] << " 255 255 255" << '\n';
        i++;
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

    size_t i = 0;
    while (i < g_nKFVertices) {
        auto v = static_cast<g2o::VertexSE3*>(optimizer->vertex(i));
        const auto T = v->estimate();
        const auto pose = T.translation().matrix();
        of << pose[0] << ' ' << pose[1] << ' ' << pose[2] << " 255 0 0" << '\n';
        i++;
    }

    of.close();
}

void saveWithSE2(SlamOptimizer* optSE3, const string outfile)
{
    SlamOptimizer optSE2;

    auto mVertices = optSE3->vertices();
    for (const auto& v : mVertices) {
        g2o::VertexSE3* v3 = dynamic_cast<g2o::VertexSE3*>(v.second);
        g2o::Isometry3D est = v3->estimate();
        Se2 Twb;
        Twb.fromCvSE3(toCvMat(est) * Config::Tcb);

        g2o::VertexSE2* v2 = new g2o::VertexSE2();
        v2->setId(v3->id());
        v2->setEstimate(g2o::SE2(Twb.x, Twb.y, Twb.theta));
        optSE2.addVertex(v2);
    }

    auto mEdges = optSE3->edges();
    for (const auto& e3 : mEdges) {
        if (e3->vertices().size() < 2)
            continue;

        g2o::EdgeSE2* e2 = new g2o::EdgeSE2();
        e2->setVertex(0, optSE2.vertex(e3->vertex(0)->id()));
        e2->setVertex(1, optSE2.vertex(e3->vertex(1)->id()));
        optSE2.addEdge(e2);
    }

    optSE2.save(outfile.c_str());
}

//! NOTE g2o::EdgeSE3的残差error平移在前, 旋转在后, jacobian和infomation应该按此顺序计算!
void calcOdoConstraint(const Se2& dOdo, Mat& Tc1c2, g2o::Matrix6d& Info_se3)
{
    const Mat& Tbc = Config::Tbc;
    const Mat& Tcb = Config::Tcb;
    const Mat Tb1b2 = dOdo.toCvSE3(); // [cm]
    Tc1c2 = Tcb * Tb1b2 * Tbc;

    //! Vector order: [trans, rot] 先平移后旋转
    // 不确定度(即协方差), 信息矩阵为其逆.
    double dx = dOdo.x / g_scale * Config::OdoUncertainX + Config::OdoNoiseX;
    double dy = dOdo.y / g_scale * Config::OdoUncertainY + Config::OdoNoiseY;
    double dtheta = dOdo.theta * Config::OdoUncertainTheta + Config::OdoNoiseTheta;

    // 信息矩阵
    g2o::Matrix6d Info_se3_bTb = g2o::Matrix6d::Zero();

    double data[6] = {1.f / (dx * dx), 1.f / (dy * dy), 1e4, 1e4, 1e4, 1 / (dtheta * dtheta)};
    for (int i = 0; i < 6; ++i)
        Info_se3_bTb(i, i) = data[i];

    g2o::Matrix6d J_bTb_cTc = AdjTR(toSE3Quat(Tbc));
    Info_se3 = J_bTb_cTc.transpose() * Info_se3_bTb * J_bTb_cTc;
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

    vector<Se2> vOdomData;
    vOdomData.reserve(2000);
    while (rec.peek() != EOF) {
        string line;
        getline(rec, line);
        istringstream iss(line);
        g2o::Vector3D lineData;
        iss >> lineData[0] >> lineData[1] >> lineData[2];
        lineData[0] *= g_scale;  // mm换成cm
        lineData[1] *= g_scale;
        vOdomData.emplace_back(lineData[0], lineData[1], lineData[2]);
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
    const double noiseT = Config::OdoNoiseTheta * 5;

    //! generate observations
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
#if USE_LM
    SlamAlgorithmLM* algo = new SlamAlgorithmLM(blockSolver);
    algo->setMaxTrialsAfterFailure(10);
#else
    SlamAlgorithmGN* algo = new SlamAlgorithmGN(blockSolver);
#endif
    optimizer.setAlgorithm(algo);
    optimizer.setVerbose(true);

#if SAVE_GT_G2O
    SlamOptimizer opt2;
#endif

    int SE3OffsetParaId = 0;
    addParaSE3Offset(optimizer, g2o::Isometry3D::Identity(), SE3OffsetParaId);

    // SE3 vertices of robot
    vector<Mat> vNoisePose;
    vector<pair<Mat, g2o::Matrix6d>> vOdoConstraint;
    for (int i = skip; i < N; ++i) {
        if ((i - skip) % delta == 0) {
            const int vertexID = (i - skip) / delta;
            const Se2 noiseTwb(vOdomData[i].x + rng.gaussian(noiseX),
                               vOdomData[i].y + rng.gaussian(noiseY),
                               vOdomData[i].theta + rng.gaussian(noiseT));
            Mat noiseTwc = noiseTwb.toCvSE3() * Config::Tbc;
            noiseTwc.at<float>(2, 3) += rng.gaussian(10); // z方向加上10cm噪声.
            vNoisePose.push_back(noiseTwc);

            const bool bIfFix = (vertexID == 0);
            g2o::VertexSE3* vi = new g2o::VertexSE3();
            vi->setEstimate(toIsometry3D(noiseTwc));
            vi->setFixed(bIfFix);
            vi->setId(vertexID);
            optimizer.addVertex(vi);

#if SAVE_GT_G2O
            g2o::VertexSE3* vi0 = new g2o::VertexSE3();
            vi0->setEstimate(toIsometry3D(vOdomData[i].toCvSE3() * Config::Tbc));
            vi0->setFixed(bIfFix);
            vi0->setId(vertexID);
            opt2.addVertex(vi0);
#endif

#if USE_PRESE3_EDGE
            // EdgeSE3Prior
            g2o::SE3Quat Tbc = toSE3Quat(Config::Tbc);
            g2o::SE3Quat Twc = toSE3Quat(toIsometry3D(noiseTwc));
            g2o::SE3Quat Twb = Twc * Tbc.inverse();
            Eigen::AngleAxisd AngleAxis_bw(Twb.rotation());
            Eigen::Vector3d Log_Rbw = AngleAxis_bw.angle() * AngleAxis_bw.axis();
            AngleAxis_bw = Eigen::AngleAxisd(Log_Rbw[2], Eigen::Vector3d::UnitZ());
            Twb.setRotation(Eigen::Quaterniond(AngleAxis_bw));
            Eigen::Vector3d xyz_wb = Twb.translation();
            xyz_wb[2] = 0;
            Twb.setTranslation(xyz_wb);
            Twc = Twb * Tbc;

            //! Vector order: [trans, rot]
            //! Omega_c = J_bc^T * Omega_b * J_bc
            g2o::Matrix6d Info_wb = g2o::Matrix6d::Zero();
            Info_wb(0, 0) = 1e-4;
            Info_wb(1, 1) = 1e-4;
            Info_wb(2, 2) = Config::PlaneMotionInfoZ;       // 1
            Info_wb(3, 3) = Config::PlaneMotionInfoXrot;    // 1e6
            Info_wb(4, 4) = Config::PlaneMotionInfoYrot;    // 1e6
            Info_wb(5, 5) = 1e-4;
            g2o::Matrix6d J_bb_cc = AdjTR(Tbc); // 正确
            g2o::Matrix6d Info_pose = J_bb_cc.transpose() * Info_wb * J_bb_cc; // 正确
            cout << "Info of EdgeSE3Prior = " << endl << Info_pose << endl;

            //! NOTE g2o的残差error平移在前, 旋转在后, jacobian和infomation应该按此顺序计算!
            g2o::EdgeSE3Prior* planeConstraint = new g2o::EdgeSE3Prior();
            planeConstraint->setInformation(Info_pose);
            planeConstraint->setMeasurement(g2o::Isometry3D(Twc));
            planeConstraint->vertices()[0] = vi;
            planeConstraint->setParameterId(0, SE3OffsetParaId);
            optimizer.addEdge(planeConstraint);
#endif

            if (vertexID > 0) {
                const Se2 dodo = vOdomData[i] - vOdomData[i - delta];  // [cm, rad]

                Mat measure;
                g2o::Matrix6d info;
                calcOdoConstraint(dodo, measure, info);
                vOdoConstraint.emplace_back(measure, info);
            }
            g_nKFVertices++;
        }
    }
    assert(vOdoConstraint.size() == g_nKFVertices - 1);

#if USE_SE3_EDGE
    // Add odometry based constraints
//    vector<g2o::EdgeSE3*> vpEdgeOdo;
    for (size_t i = 0; i < g_nKFVertices - 1; ++i) {
        const Mat meas_i = vOdoConstraint[i].first;  // Tc1c2
        g2o::Matrix6d info_i = vOdoConstraint[i].second;
//        info_i.setIdentity();
//        info_i.block(3, 3, 3, 3) *= 1e6;

//        cout << i << " measurement = " << toSE3Quat(meas_i).toMinimalVector().transpose() << endl;
//        g2o::VertexSE3 *from = dynamic_cast<g2o::VertexSE3*>(optimizer.vertex(i));
//        g2o::VertexSE3 *to   = dynamic_cast<g2o::VertexSE3*>(optimizer.vertex(i + 1));
//        g2o::Isometry3D m2   = from->estimate().inverse() * to->estimate();
//        cout << i << " estimate = " << g2o::internal::toVectorMQT(m2).transpose() << endl;
//        g2o::Isometry3D err = toIsometry3D(meas_i).inverse() * m2;
//        cout << i << " error = " << g2o::internal::toVectorMQT(err).transpose() << endl;

//        g2o::Matrix6d J_bb_cc = AdjTR(toSE3Quat(Config::Tbc)); // 正确
//        g2o::Matrix6d Info_pose = J_bb_cc.transpose() * info_i * J_bb_cc; // 正确
        cout << "info_i = " << endl << info_i << endl;
//        cout << "Info_pose = " << endl << Info_pose << endl;
//        cout << "H_bb_cc = " << endl << J_bb_cc.transpose() * J_bb_cc << endl;

        //? EdgeSE3误差集中在平移上! info里必须加大旋转的置信度才能增大旋转的误差占比??
        g2o::RobustKernelCauchy* rkc = new g2o::RobustKernelCauchy();
        rkc->setDelta(1);
        g2o::EdgeSE3* e = new g2o::EdgeSE3();
        e->setMeasurement(toIsometry3D(meas_i));
        e->setVertex(0, optimizer.vertex(i));
        e->setVertex(1, optimizer.vertex(i + 1));
        e->setInformation(info_i);
        e->setRobustKernel(rkc);
        optimizer.addEdge(e);

        e->computeError();
        cout << i << " error = " << e->error().transpose() << ", chi2 = " << e->chi2() << endl;
    }
#endif

    double t1 = timer.count();


    saveWithSE2(&optimizer, "/home/vance/output/test_BA_SE3SE2_before.g2o");
    optimizer.save("/home/vance/output/test_BA_SE3_before.g2o");
    writeToPlyFile(&optimizer);

    timer.start();
    optimizer.initializeOptimization();
    optimizer.optimize(15);
    cout << "optimizaiton time cost in (building system/optimization): " << t1 << "/"
         << timer.count() << "ms. " << endl;


#if USE_SE3_EDGE == 0
    // 如果只有PreSE3Edge, 则在优化结束后添加SE3边用于可视化轨迹效果
    for (int i = skip + delta; i < N; ++i) {
        if ((i - skip) % delta != 0)
            continue;
        const int vertexIDTo = (i - skip) / delta;
        const auto v1 = static_cast<g2o::VertexSE3*>(optimizer.vertex(vertexIDTo - 1));
        const auto v2 = static_cast<g2o::VertexSE3*>(optimizer.vertex(vertexIDTo));
        g2o::EdgeSE3* e = new g2o::EdgeSE3();
        e->setVertex(0, v1);
        e->setVertex(1, v2);
        optimizer.addEdge(e);
    }
#endif

    saveWithSE2(&optimizer, "/home/vance/output/test_BA_SE3SE2_after.g2o");
    optimizer.save("/home/vance/output/test_BA_SE3_after.g2o");
    writeToPlyFileAppend(&optimizer);

#if SAVE_GT_G2O
    for (int i = skip + delta; i < N; ++i) {
        if ((i - skip) % delta != 0)
            continue;
        const int vertexIDTo = (i - skip) / delta;
        const auto v1 = static_cast<g2o::VertexSE3*>(opt2.vertex(vertexIDTo - 1));
        const auto v2 = static_cast<g2o::VertexSE3*>(opt2.vertex(vertexIDTo));
        g2o::EdgeSE3* e = new g2o::EdgeSE3();
        e->setVertex(0, v1);
        e->setVertex(1, v2);
        opt2.addEdge(e);
    }
    opt2.save("/home/vance/output/test_BA_SE3_gt.g2o");
    saveWithSE2(&opt2, "/home/vance/output/test_BA_SE3SE2_gt.g2o");
#endif

    return 0;
}
