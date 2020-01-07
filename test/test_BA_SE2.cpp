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

//#define USE_PRESE2          0 // else use g2o::SE2
//#define USE_XYZCUSTOM       0 // else use EdgeSE2XYZ
#define USE_EDGESE2_ONLY    0
#define USE_EDGEXYZ_ONLY    1
#define EDGEXYZ_FULL_INFO   0
#define SAVE_BA_RESULT      1
#define PRINT_DEBUG_INFO    1
#define NORMALIZE_UNIT      0  // 是否将单位转回[m]进行优化
//! NOTE 在迭代上限足够大的情况下,[m]速度稍快一些, [mm]精度稍高一些

using namespace std;
using namespace Eigen;
using namespace g2o;

const std::string g_outPointCloudFile = "/home/vance/output/test_BA_SE2.ply";
size_t g_nKFVertices = 0;  // num of camera pose
#if NORMALIZE_UNIT
const double g_visualScale = 100;
const double g_unitScale = 1e-3;
#else
const double g_visualScale = 0.1;  // 数据尺度, 将单位[mm]换成[cm],方便在g2o_viewer中可视化
const double g_unitScale = 1.;
#endif

bool inBoard(const Vector2d& uv)
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
            auto v = dynamic_cast<g2o::VertexSE2*>(optimizer->vertex(i));
            auto pose = v->estimate();
            of << pose[0] << ' ' << pose[1] << ' ' << 0 << " 0 255 0" << '\n';
        } else {
            auto v = dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer->vertex(i));
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
            auto v = dynamic_cast<g2o::VertexSE2*>(optimizer->vertex(i));
            auto pose = v->estimate();
            of << pose[0] << ' ' << pose[1] << ' ' << 0 << " 255 0 0" << '\n';
        } else {
            auto v = dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer->vertex(i));
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

    // initialization
    Config::readConfig(string(argv[1]));
    int num = Config::ImgCount;
    if (argc == 3) {
        num = atoi(argv[2]);
        cerr << "set number_frames_to_process = " << num << endl;
    }

    // read data
    string odomRawFile = Config::DataPath + "odo_raw.txt";  // [mm]
    vector<Se2> vOdomData;
    readOdomDatas(odomRawFile, vOdomData);
    if (vOdomData.empty())
        exit(-1);

    const int skip = Config::ImgStartIndex;
    const int delta = 20;
    const int N = min((int)vOdomData.size(), num);
    cout << "Use " << N << " odom data (size of vertices) in the file to test BA. " << endl;
    cout << "skip = " << skip << ", delta = " << delta << endl << endl;

    // random noise
    cv::RNG rng;  // OpenCV随机数产生器
    const double noiseX = Config::OdoNoiseX * 10;
    const double noiseY = Config::OdoNoiseY * 10;
    const double noiseT = Config::OdoNoiseTheta * 8;

    // preintergation
    PreSE2 preInfo;
    vector<PreSE2> vPreInfos;
    vPreInfos.reserve(N / delta);

    // generate observations
    size_t nMPs = 200;
    vector<Vector3d> vPoints;
    vector<double> vSigma2;
    vPoints.reserve(nMPs);
    vSigma2.reserve(nMPs);
    for (size_t i = 0; i < nMPs; ++i) {
        double dx = rng.uniform(0., 0.99999);
        double dy = rng.uniform(0., 0.99999);
        const double x = dx * 5000. - 1000.;
        const double y = dy * 5000.;
        const double z = 3000.0 + rng.gaussian(200.);
        vPoints.push_back(Vector3d(x, y, z));

        const int level = rng.uniform(0, 5);
        vSigma2.push_back(std::pow(1.2, level));
    }

    WorkTimer timer;

    // g2o solver construction
    SlamOptimizer optimizer, optSaver;
    SlamLinearSolverCholmod* linearSolver = new SlamLinearSolverCholmod();
    SlamBlockSolver* blockSolver = new SlamBlockSolver(linearSolver);
    SlamAlgorithmLM* algo = new SlamAlgorithmLM(blockSolver);
    algo->setMaxTrialsAfterFailure(5);
    optimizer.setAlgorithm(algo);

    CamPara* cam = new CamPara(Config::fx, g2o::Vector2D(Config::cx, Config::cy), 0);
    cam->setId(0);
    optimizer.addParameter(cam);
    optimizer.setVerbose(true);

    // vertices of robot
    for (int i = skip; i < N; ++i) {
        // update preInfo
        if (i != skip) {
            Vector3d& meas = preInfo.meas;
            g2o::SE2 odok = toG2OSE2(vOdomData[i - 1]).inverse() * toG2OSE2(vOdomData[i]);
            Vector2d odork = odok.translation() * g_unitScale;
            Matrix2d Phi_ik = Rotation2Dd(meas[2]).toRotationMatrix();
            meas.head<2>() += Phi_ik * odork;
            meas[2] += odok.rotation().angle();

            Matrix3d Ak = Matrix3d::Identity();
            Matrix3d Bk = Matrix3d::Identity();
            Ak.block<2, 1>(0, 2) = Phi_ik * Vector2d(-odork[1], odork[0]) * g_unitScale;
            Bk.block<2, 2>(0, 0) = Phi_ik;
            Matrix3d& Sigmak = preInfo.cov;
            Matrix3d Sigma_vk = Matrix3d::Identity();
            Sigma_vk(0, 0) = (Config::OdoNoiseX * Config::OdoNoiseX) * g_unitScale * g_unitScale;
            Sigma_vk(1, 1) = (Config::OdoNoiseY * Config::OdoNoiseY) * g_unitScale * g_unitScale;
            Sigma_vk(2, 2) = (Config::OdoNoiseTheta * Config::OdoNoiseTheta);
            Matrix3d Sigma_k_1 = Ak * Sigmak * Ak.transpose() + Bk * Sigma_vk * Bk.transpose();
            Sigmak = Sigma_k_1;
        }

        if ((i - skip) % delta == 0) {
            const int vertexID = (i - skip) / delta;
            Se2 noisePose = vOdomData[i];
            noisePose.x = (noisePose.x + rng.gaussian(noiseX * noiseX)) * g_unitScale;
            noisePose.y = (noisePose.y + rng.gaussian(noiseY * noiseY)) * g_unitScale;
            noisePose.theta += rng.gaussian(noiseT * noiseT);

            g2o::VertexSE2* v = new g2o::VertexSE2();
            const bool fixed = (vertexID == 0);
            v->setId(vertexID);
            v->setFixed(fixed);
            v->setEstimate(toG2OSE2(noisePose));
            optimizer.addVertex(v);
            g_nKFVertices++;

            // reset preInfo
            vPreInfos.push_back(preInfo);  // 有效值从1开始
            preInfo.meas.setZero();
            preInfo.cov.setZero();
        }
    }

#if USE_EDGEXYZ_ONLY == 0
    // edges of odo
    for (int i = skip + delta; i < N; ++i) {
        if ((i - skip) % delta != 0)
            continue;

        const int vertexIDTo = (i - skip) / delta;
        const auto v1 = dynamic_cast<g2o::VertexSE2*>(optimizer.vertex(vertexIDTo - 1));
        const auto v2 = dynamic_cast<g2o::VertexSE2*>(optimizer.vertex(vertexIDTo));

        Vector3d& m = vPreInfos[vertexIDTo].meas;  // 第0个不是有效值
        const g2o::SE2 meas(m[0], m[1], m[2]);
        const Matrix3d info = vPreInfos[vertexIDTo].cov.inverse();

    #if USE_PRESE2
        // EdgePreSE2
        g2o::PreEdgeSE2* e = new g2o::PreEdgeSE2();
        e->setMeasurement(meas.toVector());
    #else
        // Edge SE2
        g2o::EdgeSE2* e = new g2o::EdgeSE2();
        e->setMeasurement(meas);
    #endif
        e->setVertex(0, v1);
        e->setVertex(1, v2);
        e->setInformation(info);
        optimizer.addEdge(e);
    #if PRINT_DEBUG_INFO
        e->computeError();
        cout << "EdgeSE2 from " << v1->id() << " to " << v2->id() << " , chi2 = "
             << e->chi2() << ", error = [" << e->error().transpose() << "], info = "
             << /*endl << info << */endl;
    #endif
    }
    cout << endl;
#endif  // USE_EDGEXYZ_ONLY == 0

#if USE_EDGESE2_ONLY == 0
    // vertices/edges of MPs
    const g2o::SE3Quat Tbc = toSE3Quat(Config::Tbc);
    g2o::SE3Quat Tbc_unit = Tbc;
    Tbc_unit.setTranslation(Tbc_unit.translation() * g_unitScale);
    size_t MPVertexId = g_nKFVertices;
    for (size_t j = 0; j < nMPs; ++j) {
        // vetices
        Vector3d noise(rng.gaussian(10), rng.gaussian(10), rng.gaussian(200));
        const Vector3d& lw = (vPoints[j] + noise) * g_unitScale;
        g2o::VertexSBAPointXYZ* vj = new g2o::VertexSBAPointXYZ();
        vj->setEstimate(lw);
        vj->setId(MPVertexId++);
        vj->setMarginalized(true);
        vj->setFixed(false);
        optimizer.addVertex(vj);

        // edge
        for (size_t i = 0; i < g_nKFVertices; ++i) {
            g2o::VertexSE2* vi = dynamic_cast<g2o::VertexSE2*>(optimizer.vertex(i));
            const int idx = vi->id() * delta + skip;
            const g2o::SE3Quat Tciw = Tbc_unit.inverse() *
                    SE2ToSE3_(toG2OSE2(vOdomData[idx] * g_unitScale).inverse());
            const Vector3d lci = Tciw.map(vPoints[j] * g_unitScale);
            if (lci(2) < 0)
                continue;

            const Vector2d uv = cam->cam_map(lci) + Vector2d(rng.gaussian(2), rng.gaussian(2));
            if (inBoard(uv)) {
                g2o::RobustKernelCauchy* kernel = new g2o::RobustKernelCauchy();
                kernel->setDelta(1);

                const Matrix2d Sigma_u = Matrix2d::Identity() * vSigma2[j];
    #if EDGEXYZ_FULL_INFO
                const double zc = lc(2);
                const double zc_inv = 1. / zc;
                const double zc_inv2 = zc_inv * zc_inv;
                const float& fx = Config::fx;
                const float& fy = Config::fy;
                g2o::Matrix23D J_lc;
                J_lc << fx * zc_inv, 0, -fx * lc(0) * zc_inv2, 0, fy * zc_inv, -fy * lc(1) * zc_inv2;
                const Matrix3d Rcw = Tciw.rotation().toRotationMatrix();
                const g2o::Matrix23D J_MPi = J_lc * Rcw;

                const g2o::SE2 Twb = vi->estimate();
                const Vector3d pi(Twb[0], Twb[1], 0);
                const Matrix2d J_KFj_rotxy = (J_MPi * g2o::skew(lw - pi)).block<2, 2>(0, 0);

                const Vector2d J_z = -J_MPi.block<2, 1>(0, 2);

                const float Sigma_rotxy = 1.f / Config::PlaneMotionInfoXrot;
                const float Sigma_z = 1.f / Config::PlaneMotionInfoZ;
                const Matrix2d cov = Sigma_rotxy * J_KFj_rotxy * J_KFj_rotxy.transpose() + Sigma_z * J_z * J_z.transpose() + Sigma_u;
                const Matrix2d info = cov.inverse();
    #else
                const Matrix2d info = Sigma_u.inverse();
    #endif

    #if USE_XYZCUSTOM
                EdgeSE2XYZCustom* e = new EdgeSE2XYZCustom();
    #else
                g2o::EdgeSE2XYZ* e = new g2o::EdgeSE2XYZ();
    #endif
                e->setVertex(0, vi);
                e->setVertex(1, vj);
                e->setMeasurement(uv);

                e->setInformation(info);
                e->setCameraParameter(Config::Kcam);
                e->setExtParameter(Tbc_unit);
                e->setRobustKernel(kernel);
                optimizer.addEdge(e);
    #if PRINT_DEBUG_INFO
                e->computeError();
                cout << "EdgeSE2XYZ from " << vi->id() << " to " << vj->id() << " , chi2 = "
                     << e->chi2() << ", error = [" << e->error().transpose() << "], info = "
                     /*<< endl << info */<< endl;
                //cout << "Sigma_u inv = " << endl << Sigma_u.inverse() << endl;
    #endif
            }
        }
    }
#endif  // USE_EDGESE2_ONLY == 0

    double t1 = timer.count();
    cout << "Load " << optimizer.vertices().size() << " vertices and " << optimizer.edges().size()
         << " edges tatal." << endl;


#if SAVE_BA_RESULT
    for (size_t i = 0; i < g_nKFVertices; ++i) {
        const g2o::VertexSE2* vi = dynamic_cast<const g2o::VertexSE2*>(optimizer.vertex(i));
        const g2o::SE2 est = vi->estimate();
        g2o::SE2 scaledEstimate(est[0] * g_visualScale, est[1] * g_visualScale, est[2]);

        g2o::VertexSE2* v_scale = new g2o::VertexSE2();
        v_scale->setId(i);
        v_scale->setEstimate(scaledEstimate);
        optSaver.addVertex(v_scale);

        if (i > 0) {
            g2o::EdgeSE2* e_tmp = new g2o::EdgeSE2();
            e_tmp->setVertex(0, dynamic_cast<g2o::VertexSE2*>(optSaver.vertex(i - 1)));
            e_tmp->setVertex(1, v_scale);
            optSaver.addEdge(e_tmp);
        }
    }
    optSaver.save("/home/vance/output/test_BA_SE2_before.g2o");
    writeToPlyFile(&optimizer);
#endif

    timer.start();
    optimizer.initializeOptimization(0);
    optimizer.optimize(15);
    cout << "optimizaiton time cost in (building system/optimization): " << t1 << "/"
         << timer.count() << "ms. " << endl;

#if SAVE_BA_RESULT
    // update optimization result
    g2o::Vector3D MAE(0, 0, 0);  // [mm]
    for (size_t i = 0; i < g_nKFVertices; ++i) {
        const g2o::VertexSE2* vi = dynamic_cast<const g2o::VertexSE2*>(optimizer.vertex(i));
        const g2o::SE2 est = vi->estimate();
        g2o::SE2 scaledEstimate(est[0] * g_visualScale, est[1] * g_visualScale, est[2]);

        g2o::VertexSE2* vi_scale = dynamic_cast<g2o::VertexSE2*>(optSaver.vertex(i));
        vi_scale->setEstimate(scaledEstimate);

        // output MAE [mm]
        g2o::Vector3D esti = est.toVector();
        esti[0] /= g_unitScale;
        esti[1] /= g_unitScale;
        g2o::Vector3D abs_ei = esti - toG2OSE2(vOdomData[i]).toVector();
        for (size_t i = 0; i < 3; ++i)
            abs_ei[i] = abs(abs_ei[i]);
        MAE += abs_ei;
    }
    MAE = MAE / g_nKFVertices;
    cout << "MAE[mm] = " << MAE.transpose() << endl;
    optSaver.save("/home/vance/output/test_BA_SE2_after.g2o");
    writeToPlyFileAppend(&optimizer);
#endif

    return 0;
}
