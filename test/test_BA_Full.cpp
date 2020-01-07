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

#define NORMALIZE_UNIT      0
#define SAVE_BA_RESULT      1
#define PRINT_DEBUG_INFO    1
// local BA
#define DO_LOCAL_BA         1
#define USE_EDGE_SE2        1
#define USE_EDGE_SE2XYZ     1
// globa BA
#define DO_GLOBAL_BA        1
#define USE_EDGE_PRESE3     1
#define USE_EDGE_SE3        1

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace g2o;

const string g_outPointCloudFile = "/home/vance/output/testBA_Full.ply";
const string g_outGTfile = "/home/vance/output/testBA_Full_GT.ply";
#if NORMALIZE_UNIT
const double g_visualScale = 100;
const double g_unitScale = 1e-3;
const bool g_normalizeUnit = true;
#else
const double g_visualScale = 0.1;  // 数据尺度, 将单位[mm]换成[cm],方便在g2o_viewer中可视化
const double g_unitScale = 1.;
const bool g_normalizeUnit = false;
#endif

size_t g_nKFs = 0, g_nMPs = 400;    // num of camera pose and MPs.
size_t g_skip = 0, g_delta = 0;
Mat Tcb_unit, Tbc_unit;

struct TestFrame;
struct TestFeature;

Eigen::Matrix<double, 4, 4> toMatrix4d(const cv::Mat& cvMat4)
{
    Eigen::Matrix<double, 4, 4> M;

    M << cvMat4.at<float>(0, 0), cvMat4.at<float>(0, 1), cvMat4.at<float>(0, 2), cvMat4.at<float>(0, 3),
        cvMat4.at<float>(1, 0), cvMat4.at<float>(1, 1), cvMat4.at<float>(1, 2), cvMat4.at<float>(1, 3),
        cvMat4.at<float>(2, 0), cvMat4.at<float>(2, 1), cvMat4.at<float>(2, 2), cvMat4.at<float>(2, 3),
        cvMat4.at<float>(3, 0), cvMat4.at<float>(3, 1), cvMat4.at<float>(3, 2), cvMat4.at<float>(3, 3);

    return M;
}

struct TestFrame
{
    TestFrame(size_t id_, const Se2& odo_) : id(id_), Twb_gt(odo_) {
        const Mat Twc_mat = Twb_gt.toCvSE3() * Config::Tbc;
        Twc_gt = toMatrix4d(Twc_mat);
        Tcw_gt = toMatrix4d(cvu::inv(Twc_mat));
    }
    void setNoisePose(const Se2& noise) {
        Twb_noise = noise;
        const Mat NoiseTwc_mat = Twb_noise.toCvSE3() * Config::Tbc;
        Twc_noise = toMatrix4d(NoiseTwc_mat);
        Tcw_noise = toMatrix4d(cvu::inv(NoiseTwc_mat));
    }

    size_t id = 0;
    Se2 Twb_gt, Twb_noise, Twb_opt;
    Matrix4d Twc_gt, Tcw_gt, Twc_noise, Tcw_noise, Twc_opt, Tcw_opt;
    PreSE2 PreInfo;
    unordered_map<size_t, Vector2d> IdxObs;
    unordered_map<size_t, Vector2d> IdxObs_NoiseMeasure;
};

struct TestFeature
{
    TestFeature(size_t id_, const Vector3d& pos_, TestFrame* kf_ = nullptr)
        : id(id_), Pose_gt(pos_), MainKF(kf_) {}
    void setNoisePose(const Vector3d& noise) {
        Pose_noise = noise;
    }
    void setLevel(int l) {
        Level = l;
        Sigma2 = pow(1.2, Level);
        invSigma2 = 1. / Sigma2;
    }

    size_t id = 0;
    Vector3d Pose_gt, Pose_noise, Pose_opt;
    unordered_map<size_t, Vector2d> IdxObs;
    TestFrame* MainKF = nullptr;
    int Level = 0;
    double Sigma2 = 1.;
    double invSigma2 = 1.;
};

bool inBoard(const Vector2d& uv)
{
    static const double& w = Config::ImgSize.width;
    static const double& h = Config::ImgSize.height;

    return uv(0) >= 1. && uv(0) <= w - 1. && uv(1) >= 1. && uv(1) <= h - 1.;
}

void saveGTsToPly(const vector<TestFrame>& vGTFrames, const vector<TestFeature>& vGTFeatures)
{
    ofstream of(g_outGTfile.c_str());
    if (!of.is_open()) {
        cerr << "Error on openning the output file: " << g_outGTfile << endl;
        return;
    }

    const size_t nFrames = vGTFrames.size();
    const size_t nFeatures = vGTFeatures.size();

    of << "ply" << '\n'
       << "format ascii 1.0" << '\n'
       << "element vertex " << (nFrames + nFeatures) * 2 << '\n'
       << "property float x" << '\n'
       << "property float y" << '\n'
       << "property float z" << '\n'
       << "property uchar red" << '\n'
       << "property uchar green" << '\n'
       << "property uchar blue" << '\n'
       << "end_header" << endl;

    for (size_t i = 0; i < nFrames; ++i) {
        const Se2& pose1 = vGTFrames[i].Twb_gt;
        const Se2& pose2 = vGTFrames[i].Twb_noise;
        of << pose1.x << ' ' << pose1.y << ' ' << 0 << " 0 255 0" << '\n';
        of << pose2.x << ' ' << pose2.y << ' ' << 0 << " 255 255 255" << '\n';
    }
    for (size_t j = 0; j < nFeatures; ++j) {
        const Vector3d& pose1 = vGTFeatures[j].Pose_gt;
        const Vector3d& pose2 = vGTFeatures[j].Pose_noise;
        of << pose1[0] << ' ' << pose1[1] << ' ' << pose1[2] << " 0 255 0" << '\n';
        of << pose2[0] << ' ' << pose2[1] << ' ' << pose2[2] << " 255 255 255" << '\n';
    }

    of.close();
}

void writeToPly(SlamOptimizer* optimizer)
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
       << "end_header" << endl;

    for (size_t i = 0; i < nPose; ++i) {
        if (i < g_nKFs) {
            auto v = dynamic_cast<VertexSE2*>(optimizer->vertex(i));
            auto pose = v->estimate();
            of << pose[0] << ' ' << pose[1] << ' ' << 0 << " 0 255 0" << '\n';
        } else {
            auto v = dynamic_cast<VertexSBAPointXYZ*>(optimizer->vertex(i));
            auto pose = v->estimate();
            of << pose[0] << ' ' << pose[1] << ' ' << pose[2] << " 0 255 0" << '\n';
        }
    }

    of.close();
}

void writeToPlyAppend(SlamOptimizer* optimizer)
{
    ofstream of(g_outPointCloudFile.c_str(), ios::app);
    if (!of.is_open()) {
        cerr << "Error on openning the output file: " << g_outPointCloudFile << endl;
        return;
    }

    auto vertices = optimizer->vertices();
    const size_t nPose = vertices.size();

    for (size_t i = 0; i < nPose; ++i) {
        if (i < g_nKFs) {
            auto v = dynamic_cast<VertexSE2*>(optimizer->vertex(i));
            auto pose = v->estimate();
            of << pose[0] << ' ' << pose[1] << ' ' << 0 << " 255 0 0" << '\n';
        } else {
            auto v = dynamic_cast<VertexSBAPointXYZ*>(optimizer->vertex(i));
            auto pose = v->estimate();
            of << pose[0] << ' ' << pose[1] << ' ' << pose[2] << " 255 0 0" << '\n';
        }
    }

    of.close();
}

void generateSimData(const vector<Se2>& vOdomData, vector<TestFrame>& vGTFrames,
                     vector<TestFeature>& vGTFeatures, size_t N)
{
    assert(g_skip >= 0);
    assert(g_delta > 0);
    assert(N > g_skip);

    // random noise
    RNG rng;  // OpenCV随机数产生器
    const double noiseX = Config::OdoNoiseX * 5;
    const double noiseY = Config::OdoNoiseY * 5;
    const double noiseT = Config::OdoNoiseTheta * 5;

    // generate KFs
    double maxX = 0, maxY = 0, minX = 99999, minY = 99999;
    PreSE2 preInfo;
    vGTFrames.clear();
    vGTFrames.reserve(N / g_delta);
    for (size_t i = g_skip; i < N; ++i) {
        // preintergration
        if (i != g_skip) {
            Se2 odoLast = vOdomData[i - 1];
            odoLast.x += noiseX;
            odoLast.y += noiseY;
            odoLast.theta += noiseT;
            Se2 odoCurr = vOdomData[i];
            odoCurr.x += noiseX;
            odoCurr.y += noiseY;
            odoCurr.theta += noiseT;

            Vector3d& meas = preInfo.meas;
            SE2 odok = toG2OSE2(odoLast).inverse() * toG2OSE2(odoCurr);
            Vector2d odork = odok.translation();
            Matrix2d Phi_ik = Rotation2Dd(meas[2]).toRotationMatrix();
            meas.head<2>() += Phi_ik * odork;
            meas[2] += odok.rotation().angle();

            Matrix3d Ak = Matrix3d::Identity();
            Matrix3d Bk = Matrix3d::Identity();
            Ak.block<2, 1>(0, 2) = Phi_ik * Vector2d(-odork[1], odork[0]);
            Bk.block<2, 2>(0, 0) = Phi_ik;
            Matrix3d& Sigmak = preInfo.cov;
            Matrix3d Sigma_vk = Matrix3d::Identity();
            Sigma_vk(0, 0) = (Config::OdoNoiseX * Config::OdoNoiseX);
            Sigma_vk(1, 1) = (Config::OdoNoiseY * Config::OdoNoiseY);
            Sigma_vk(2, 2) = (Config::OdoNoiseTheta * Config::OdoNoiseTheta);
            Matrix3d Sigma_k_1 = Ak * Sigmak * Ak.transpose() + Bk * Sigma_vk * Bk.transpose();
            Sigmak = Sigma_k_1;
        }

        // KF
        if ((i - g_skip) % g_delta == 0) {
            const size_t vertexID = (i - g_skip) / g_delta;
            const Se2& Twb = vOdomData[i];
            Se2 noiseTwb = Twb;
            noiseTwb.x += rng.gaussian(noiseX * 5);
            noiseTwb.y += rng.gaussian(noiseY * 5);
            noiseTwb.theta += rng.gaussian(noiseT * 5);

            TestFrame tf(vertexID, Twb);
            tf.setNoisePose(noiseTwb);
            tf.PreInfo = preInfo;
            vGTFrames.push_back(tf);

            preInfo.meas.setZero();
            preInfo.cov.setZero();
            g_nKFs++;

            if (maxX < Twb.x)
                maxX = Twb.x;
            if (maxY < Twb.y)
                maxY = Twb.y;
            if (minX > Twb.x)
                minX = Twb.x;
            if (minY > Twb.y)
                minY = Twb.y;
        }
    }
    cout << g_nKFs << " KeyFrames Generated. Limitation is: X ~ [" << minX << ", "
         << maxX << "], Y ~ [" << minY << ", " << maxY << "]" << endl;

    // generate MPs
    vGTFeatures.clear();
    vGTFeatures.reserve(g_nMPs);
    const size_t m = sqrt(g_nMPs);
    const double stepX = (maxX - minX + 1000.) / m;
    const double stepY = (maxY - minY + 1000.) / m;
    size_t MPid = 0;
    for (size_t i = 0; i < m; ++i) {
        Vector3d pose;
        pose[0] = minX - 500 + i * stepX;
        for (size_t j = 0; j < m; ++j) {
            pose[1] = minY - 500 + j * stepY;
            pose[2] = 2800.;

            Vector3d noise(rng.gaussian(10), rng.gaussian(10), rng.gaussian(200));
            TestFeature mp(MPid++, pose);
            mp.setNoisePose(pose + noise);
            mp.setLevel(rng.uniform(0, 5));
            vGTFeatures.push_back(mp);
        }
    }
    g_nMPs = MPid;
    cout << g_nMPs << " MapPoints Generated. Limitation is: X ~ [" << minX - 500 << ", "
         << maxX + 500 << "], Y ~ [" << minY - 500 << ", " << maxY + 500
         << "], Z ~ [2800, 3000]" << endl;

    // generate relationship (observation)
    size_t nEdges1 = 0, nEdges2 = 0;
    for (size_t i = 0; i < g_nMPs; ++i) {
        const Vector3d& Pw = vGTFeatures[i].Pose_gt;
        for (size_t j = 0; j < g_nKFs; ++j) {
            const Matrix4d& Tcwj = vGTFrames[j].Tcw_gt;
            const Vector3d Pcj = Tcwj.block(0, 0, 3, 3) * Pw + Tcwj.block(0, 3, 3, 1);
            assert(Pcj[2] > 0);
            const Vector2d uvj = camera_map(Pcj);
            const Vector2d uvj_n = uvj + Vector2d(rng.gaussian(2), rng.gaussian(2));
            if (inBoard(uvj)) {
                vGTFrames[j].IdxObs.emplace(vGTFeatures[i].id, uvj);
                vGTFeatures[i].IdxObs.emplace(vGTFrames[j].id, uvj);
                nEdges1++;
                if (inBoard(uvj_n)) {
                    vGTFrames[j].IdxObs_NoiseMeasure.emplace(vGTFeatures[i].id, uvj_n);
                    nEdges2++;
                }
            }
        }
    }
    cout << nEdges2 << "/" << nEdges1 << " Edges will be added to the Graph." << endl;
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
    if (argc > 2) {
        num = atoi(argv[2]);
        cout << " - set number_frames_to_process to " << num << endl;
    }
    g_skip = Config::ImgStartIndex;
    g_delta = 20;
    cout << " - set g_skip and g_delta to " << g_skip << ", " << g_delta << endl;
    cout << " - set unitScale and visualScale to " << g_unitScale << ", " << g_visualScale << endl;

    Tcb_unit = Config::Tcb;
    Tcb_unit.rowRange(0, 3).col(3) *= g_unitScale;
    Tbc_unit = cvu::inv(Tcb_unit);
    cout << " - set Tcb_unit to " << endl << Tcb_unit << endl;
    cout << " - set Tbc_unit to " << endl << Tbc_unit << endl;

    // read data
    string odomRawFile = Config::DataPath + "odo_raw.txt";  // [mm]
    vector<Se2> vOdomData;
    readOdomDatas(odomRawFile, vOdomData);
    if (vOdomData.empty())
        exit(-1);
    const int N = min((int)vOdomData.size(), num);
    cout << " - use " << N << " odom data (size of vertices) in the file to test BA. " << endl;
    cout << endl;

    // generate ground truth and noise data
    vector<TestFrame> vGTFrames;
    vector<TestFeature> vGTFeatures;
    generateSimData(vOdomData, vGTFrames, vGTFeatures, N);
    saveGTsToPly(vGTFrames, vGTFeatures);

    WorkTimer timer;

    // g2o solver construction
    SlamOptimizer optimizer, optSaver;
    SlamLinearSolverCholmod* linearSolver = new SlamLinearSolverCholmod();
    linearSolver->setBlockOrdering(true);
    SlamBlockSolver* blockSolver = new SlamBlockSolver(linearSolver);
    SlamAlgorithmLM* algo = new SlamAlgorithmLM(blockSolver);
    algo->setMaxTrialsAfterFailure(5);
    optimizer.setAlgorithm(algo);
    optimizer.setVerbose(true);

    // vertices of robot poses
#if DO_LOCAL_BA
    cout << "Loading Local BA,  this will take a while..." << endl;
    for (size_t i = 0; i < g_nKFs; ++i) {
        const TestFrame& tfi = vGTFrames[i];

        VertexSE2* vi = new VertexSE2();
        vi->setId(tfi.id);
        vi->setFixed(tfi.id == 0);
        vi->setEstimate(toG2OSE2(tfi.Twb_noise, g_normalizeUnit));
        optimizer.addVertex(vi);
        assert(tfi.id == i);

#if USE_EDGE_SE2
        if (i > 0) {
            VertexSE2* vf = dynamic_cast<VertexSE2*>(optimizer.vertex(i - 1));
            EdgeSE2* e = new EdgeSE2();
            e->setVertex(0, vf);
            e->setVertex(1, vi);
            Matrix3d info = tfi.PreInfo.cov.inverse();
            e->setMeasurement(SE2(tfi.PreInfo.meas));
            e->setInformation(info);
            e->setRobustKernel(new RobustKernelCauchy());
            optimizer.addEdge(e);

    #if PRINT_DEBUG_INFO
            e->computeError();
            cout << "EdgeSE2 from " << vf->id() << " to " << vi->id() << " , chi2 = "
                 << e->chi2() << ", error = [" << e->error().transpose() << "], info = "
                 << endl << info << endl;
    #endif
        }
#endif
    }

#if USE_EDGE_SE2XYZ
    // vertices of Landmarks
    for (size_t j = 0; j < g_nMPs; ++j) {
        const TestFeature& tlj = vGTFeatures[j];
        VertexSBAPointXYZ* vj = new VertexSBAPointXYZ();
        vj->setId(tlj.id + g_nKFs);
        vj->setFixed(false);
        vj->setMarginalized(true);
        vj->setEstimate(tlj.Pose_noise);
        optimizer.addVertex(vj);
        assert(tlj.id == j);

        for (const auto& obs : tlj.IdxObs) {
            const size_t idxKF = obs.first;
            if (vGTFrames[idxKF].IdxObs_NoiseMeasure.count(tlj.id)) {
                VertexSE2* vi = dynamic_cast<VertexSE2*>(optimizer.vertex(idxKF));
                EdgeSE2XYZ* e = new EdgeSE2XYZ();
                e->setVertex(0, vi);
                e->setVertex(1, vj);
                e->setMeasurement(vGTFrames[idxKF].IdxObs_NoiseMeasure[tlj.id]);
                e->setInformation(Matrix2d::Identity() * tlj.invSigma2);
                e->setCameraParameter(Config::Kcam);
                e->setExtParameter(toSE3Quat(Tbc_unit));
                e->setRobustKernel(new RobustKernelCauchy());
                optimizer.addEdge(e);
    #if PRINT_DEBUG_INFO
                e->computeError();
                cout << "EdgeSE2XYZ from " << vi->id() << " to " << vj->id() << " , chi2 = "
                     << e->chi2() << ", error = [" << e->error().transpose() << "], info = "
                     << tlj.invSigma2 << " * I2 " << endl;
                cout << "EdgeSE2XYZ from " << vi->id() << " to " << vj->id() << ", measurement = "
                     << vGTFrames[idxKF].IdxObs_NoiseMeasure[tlj.id].transpose() << ", gt = "
                     << vGTFrames[idxKF].IdxObs[tlj.id].transpose() << endl;
    #endif
            }
        }
    }
#endif

#if SAVE_BA_RESULT
    for (size_t i = 0; i < g_nKFs; ++i) {
        const VertexSE2* vi = dynamic_cast<const VertexSE2*>(optimizer.vertex(i));
        const SE2 est = vi->estimate();
        SE2 scaledEstimate(est[0] * g_visualScale, est[1] * g_visualScale, est[2]);

        VertexSE2* v_scale = new VertexSE2();
        v_scale->setId(i);
        v_scale->setEstimate(scaledEstimate);
        optSaver.addVertex(v_scale);

        if (i > 0) {
            EdgeSE2* e_tmp = new EdgeSE2();
            e_tmp->setVertex(0, dynamic_cast<VertexSE2*>(optSaver.vertex(i - 1)));
            e_tmp->setVertex(1, v_scale);
            optSaver.addEdge(e_tmp);
        }
    }
    optSaver.save("/home/vance/output/testBA_Full_SE2_before.g2o");
    writeToPly(&optimizer);
#endif
    double t1 = timer.count();

    // Do Local BA
    timer.start();
    optimizer.initializeOptimization(0);
    optimizer.optimize(15);
    double t2 = timer.count();
    cout << "Local BA load vertives and edges = " << optimizer.vertices().size() << " / "
         << optimizer.edges().size() << ", time cost in (building system/optimization): "
         << t1 << "/" << t2 << "ms. " << endl;

#if SAVE_BA_RESULT
    Vector3d rmse(0, 0, 0);  // [mm]
    for (size_t i = 0; i < g_nKFs; ++i) {
        const VertexSE2* vi = dynamic_cast<const VertexSE2*>(optimizer.vertex(i));
        const SE2 est = vi->estimate();
        SE2 scaledEstimate(est[0] * g_visualScale, est[1] * g_visualScale, est[2]);

        VertexSE2* v_scale = dynamic_cast<VertexSE2*>(optSaver.vertex(i));
        v_scale->setEstimate(scaledEstimate);

        // output rmse [mm]
        Vector3D esti = est.toVector();
        esti[0] /= g_unitScale;
        esti[1] /= g_unitScale;
        const Vector3D delta = esti - toG2OSE2(vGTFrames[i].Twb_gt).toVector();
        rmse += Vector3d(delta[0] * delta[0], delta[1] * delta[1], delta[2] * delta[2]);
    }
    rmse /= g_nKFs;
    rmse[0] = sqrt(rmse[0]);
    rmse[1] = sqrt(rmse[1]);
    rmse[2] = sqrt(rmse[2]);
    cout << "Local BA RMSE = " << rmse.transpose() << endl;

    optSaver.save("/home/vance/output/testBA_Full_SE2_after.g2o");
    writeToPlyAppend(&optimizer);
#endif

#endif  // DO_LOCAL_BA

    return 0;
}
