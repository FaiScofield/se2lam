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

#define SAVE_BA_RESULT      1
#define PRINT_DEBUG_INFO    1
// local BA
#define DO_LOCAL_BA         1
#define USE_EDGEXYZ_ONLY    0
// globa BA
#define DO_GLOBAL_BA        1
#define USE_EDGEXYZ_GLOBAL  1

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace g2o;

const string g_outPointCloudFile = "/home/vance/output/test_BA_SE2.ply";
const string g_outGTfile = "/home/vance/output/test_BA_Full_GT.ply";
const double g_scale = 0.1;  // 数据尺度, 将单位[mm]换成[cm],方便在g2o_viewer中可视化
size_t g_nKFs = 0, g_nMPs = 900;    // num of camera pose and MPs.
size_t g_skip = 0, g_delta = 0;

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
    TestFrame(size_t id_, const Se2& odo_) : id(id_), Twb(odo_) {
        const Mat Twc_mat = Twb.toCvSE3() * Config::Tbc;
        Twc = toMatrix4d(Twc_mat);
        Tcw = toMatrix4d(cvu::inv(Twc_mat));
    }
    void setNoisePose(const Se2& noise) {
        NoiseTwb = noise;
        const Mat NoiseTwc_mat = NoiseTwb.toCvSE3() * Config::Tbc;
        NoiseTwc = toMatrix4d(NoiseTwc_mat);
        NoiseTcw = toMatrix4d(cvu::inv(NoiseTwc_mat));
    }

    size_t id = 0;
    Se2 Twb, NoiseTwb;
    Matrix4d Twc, Tcw, NoiseTwc, NoiseTcw;
    PreSE2 PreInfo;
    unordered_map<size_t, Vector2d> IdxObs;
    unordered_map<size_t, Vector2d> IdxObs_NoiseMeasure;
};

struct TestFeature
{
    TestFeature(size_t id_, const Vector3d& pos_, TestFrame* kf_ = nullptr)
        : id(id_), Pose(pos_), MainKF(kf_) {}
    void setNoisePose(const Vector3d& noise) {
        NoisePose = noise;
    }
    void setLevel(int l) {
        Level = l;
        Sigma2 = pow(1.2, Level);
    }

    size_t id = 0;
    Vector3d Pose;
    Vector3d NoisePose;
    set<size_t> Obs;
    TestFrame* MainKF = nullptr;
    int Level = 0;
    double Sigma2 = 1;
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
       << "element vertex " << nFrames + nFeatures << '\n'
       << "property float x" << '\n'
       << "property float y" << '\n'
       << "property float z" << '\n'
       << "property uchar red" << '\n'
       << "property uchar green" << '\n'
       << "property uchar blue" << '\n'
       << "end_header" << endl;

    for (size_t i = 0; i < nFrames; ++i) {
        const Se2& pose = vGTFrames[i].Twb;
        of << pose.x << ' ' << pose.y << ' ' << 0 << " 255 255 255" << '\n';
    }
    for (size_t j = 0; j < nFeatures; ++j) {
        const Vector3d& pose = vGTFeatures[j].Pose;
        of << pose[0] << ' ' << pose[1] << ' ' << pose[2] << " 0 255 0" << '\n';
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
                     vector<TestFeature>& vGTFeatures)
{
    const size_t N = vOdomData.size();
    assert(g_skip >= 0);
    assert(g_delta > 0);
    assert(N > g_skip);

    // random noise
    RNG rng;  // OpenCV随机数产生器
    const double noiseX = Config::OdoNoiseX * 10;
    const double noiseY = Config::OdoNoiseY * 10;
    const double noiseT = Config::OdoNoiseTheta * 8;

    // generate KFs
    double maxX = 0, maxY = 0, minX = 99999, minY = 99999;
    PreSE2 preInfo;
    vGTFrames.clear();
    vGTFrames.reserve(N / g_delta);
    for (size_t i = g_skip; i < N; ++i) {
        // preintergration
        if (i != g_skip) {
            Vector3d& meas = preInfo.meas;
            SE2 odok = toG2OSE2(vOdomData[i - 1]).inverse() * toG2OSE2(vOdomData[i]);
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
            noiseTwb.x += rng.gaussian(noiseX * noiseX);
            noiseTwb.y += rng.gaussian(noiseY * noiseY);
            noiseTwb.theta += rng.gaussian(noiseT * noiseT);

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
            pose[2] = rng.uniform(2800., 3000.);

            Vector3d noise(rng.gaussian(10), rng.gaussian(10), rng.gaussian(200));
            TestFeature mp(MPid++, pose);
            mp.setNoisePose(noise);
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
        const Vector3d& Pw = vGTFeatures[i].Pose;
        const Vector3d& Pw_n = vGTFeatures[i].NoisePose;
        for (size_t j = 0; j < g_nKFs; ++j) {
            const Matrix4d& Tcwj = vGTFrames[j].Tcw;
            const Matrix4d& Tcwj_n = vGTFrames[j].NoiseTcw;
            const Vector3d Pcj = Tcwj.block(0, 0, 3, 3) * Pw + Tcwj.block(0, 3, 3, 1);
            const Vector3d Pcj_n = Tcwj_n.block(0, 0, 3, 3) * Pw_n + Tcwj_n.block(0, 3, 3, 1);
            Vector2d uvj = camera_map(Pcj);
            Vector2d uvj_n = camera_map(Pcj_n);
            if (inBoard(uvj)) {
                vGTFrames[j].IdxObs.emplace(vGTFeatures[i].id, uvj);
                vGTFeatures[i].Obs.insert(vGTFrames[j].id);
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

    g_skip = Config::ImgStartIndex;
    g_delta = 20;
    const int N = min((int)vOdomData.size(), num);
    cout << "Use " << N << " odom data (size of vertices) in the file to test BA. " << endl;
    cout << "g_skip = " << g_skip << ", g_delta = " << g_delta << endl << endl;

    // generate ground truth and noise data
    vector<TestFrame> vGTFrames;
    vector<TestFeature> vGTFeatures;
    generateSimData(vOdomData, vGTFrames, vGTFeatures);
    saveGTsToPly(vGTFrames, vGTFeatures);

    WorkTimer timer;

    // g2o solver construction
    SlamOptimizer optimizer, optSaver;
    SlamLinearSolverCholmod* linearSolver = new SlamLinearSolverCholmod();
    SlamBlockSolver* blockSolver = new SlamBlockSolver(linearSolver);
    SlamAlgorithmLM* algo = new SlamAlgorithmLM(blockSolver);
    algo->setMaxTrialsAfterFailure(5);
    optimizer.setAlgorithm(algo);
    optimizer.setVerbose(true);

//    // vertices of robot
//    for (int i = g_skip; i < N; ++i) {
//        // update preInfo
//        if (i != g_skip) {
//            Vector3d& meas = preInfo.meas;
//            SE2 odok = toG2OSE2(vOdomData[i - 1]).inverse() * toG2OSE2(vOdomData[i]);
//            Vector2d odork = odok.translation();
//            Matrix2d Phi_ik = Rotation2Dd(meas[2]).toRotationMatrix();
//            meas.head<2>() += Phi_ik * odork;
//            meas[2] += odok.rotation().angle();

//            Matrix3d Ak = Matrix3d::Identity();
//            Matrix3d Bk = Matrix3d::Identity();
//            Ak.block<2, 1>(0, 2) = Phi_ik * Vector2d(-odork[1], odork[0]);
//            Bk.block<2, 2>(0, 0) = Phi_ik;
//            Matrix3d& Sigmak = preInfo.cov;
//            Matrix3d Sigma_vk = Matrix3d::Identity();
//            Sigma_vk(0, 0) = (Config::OdoNoiseX * Config::OdoNoiseX);
//            Sigma_vk(1, 1) = (Config::OdoNoiseY * Config::OdoNoiseY);
//            Sigma_vk(2, 2) = (Config::OdoNoiseTheta * Config::OdoNoiseTheta);
//            Matrix3d Sigma_k_1 = Ak * Sigmak * Ak.transpose() + Bk * Sigma_vk * Bk.transpose();
//            Sigmak = Sigma_k_1;
//        }

//        if ((i - g_skip) % g_delta == 0) {
//            const int vertexID = (i - g_skip) / g_delta;


//            VertexSE2* v = new VertexSE2();
//            const bool fixed = (vertexID == 0);
//            v->setId(vertexID);
//            v->setFixed(fixed);
//            v->setEstimate(toG2OSE2(noisePose));
//            optimizer.addVertex(v);
//            g_nKFs++;

//            // reset preInfo
//            vPreInfos.push_back(preInfo);  // 有效值从1开始
//            preInfo.meas.setZero();
//            preInfo.cov.setZero();
//        }
//    }

//#if USE_EDGEXYZ_ONLY == 0
//    // edges of odo
//    for (int i = g_skip + g_delta; i < N; ++i) {
//        if ((i - g_skip) % g_delta != 0)
//            continue;

//        const int vertexIDTo = (i - g_skip) / g_delta;
//        const auto v1 = dynamic_cast<VertexSE2*>(optimizer.vertex(vertexIDTo - 1));
//        const auto v2 = dynamic_cast<VertexSE2*>(optimizer.vertex(vertexIDTo));

//        Vector3d& m = vPreInfos[vertexIDTo].meas;  // 第0个不是有效值
//        const SE2 meas(m[0], m[1], m[2]);
//        const Matrix3d info = vPreInfos[vertexIDTo].cov.inverse();

//#if USE_PRESE2
//        // EdgePreSE2
//        PreEdgeSE2* e = new PreEdgeSE2();
//        e->setMeasurement(m);
//#else
//        // Edge SE2
//        EdgeSE2* e = new EdgeSE2();
//        // EdgeSE2Custom* e = new EdgeSE2Custom();
//        e->setMeasurement(meas);
//#endif
//        e->setVertex(0, v1);
//        e->setVertex(1, v2);
//        e->setInformation(info/*Matrix3d::Identity()*/);
//        optimizer.addEdge(e);
//#if PRINT_DEBUG_INFO
//        e->computeError();
//        cout << "EdgeSE2 from " << v1->id() << " to " << v2->id() << " , chi2 = "
//             << e->chi2() << ", error = [" << e->error().transpose() << "], info = "
//             << endl << info << endl;
//#endif
//    }
//    cout << endl;
//#endif

//    // vertices/edges of MPs
//    const SE3Quat Tbc = toSE3Quat(Config::Tbc);
//    size_t MPVertexId = g_nKFs;
//    for (size_t j = 0; j < nMPs; ++j) {
//        // vetices
//        Vector3d noise(rng.gaussian(10), rng.gaussian(10), rng.gaussian(200));
//        const Vector3d& lw = vGTFeatures[j] + noise;
//        VertexSBAPointXYZ* vj = new VertexSBAPointXYZ();
//        vj->setEstimate(lw);
//        vj->setId(MPVertexId++);
//        vj->setMarginalized(true);
//        vj->setFixed(false);
//        optimizer.addVertex(vj);

//        // edge
//        for (size_t i = 0; i < g_nKFs; ++i) {
//            VertexSE2* vi = dynamic_cast<VertexSE2*>(optimizer.vertex(i));
//            const int idx = vi->id() * g_delta + g_skip;
//            const SE3Quat Tcw = Tbc.inverse() * SE2ToSE3_(toG2OSE2(vOdomData[idx]).inverse());
//            const Vector3d lc = Tcw.map(vGTFeatures[j]);
//            if (lc(2) < 0)
//                continue;

//            const Vector2d uv = cam->cam_map(lc);
//            if (inBoard(uv)) {
//                RobustKernelCauchy* kernel = new RobustKernelCauchy();
//                kernel->setDelta(1);

//                const Matrix2d Sigma_u = Matrix2d::Identity() * vSigma2[j];
//#if EDGEXYZ_FULL_INFO
//                const double zc = lc(2);
//                const double zc_inv = 1. / zc;
//                const double zc_inv2 = zc_inv * zc_inv;
//                const float& fx = Config::fx;
//                const float& fy = Config::fy;
//                Matrix23D J_lc;
//                J_lc << fx * zc_inv, 0, -fx * lc(0) * zc_inv2, 0, fy * zc_inv, -fy * lc(1) * zc_inv2;
//                const Matrix3d Rcw = Tcw.rotation().toRotationMatrix();
//                const Matrix23D J_MPi = J_lc * Rcw;

//                const SE2 Twb = vi->estimate();
//                const Vector3d pi(Twb[0], Twb[1], 0);
//                const Matrix2d J_KFj_rotxy = (J_MPi * skew(lw - pi)).block<2, 2>(0, 0);

//                const Vector2d J_z = -J_MPi.block<2, 1>(0, 2);

//                const float Sigma_rotxy = 1.f / Config::PlaneMotionInfoXrot;
//                const float Sigma_z = 1.f / Config::PlaneMotionInfoZ;
//                const Matrix2d cov = Sigma_rotxy * J_KFj_rotxy * J_KFj_rotxy.transpose() + Sigma_z * J_z * J_z.transpose() + Sigma_u;
//                const Matrix2d info = cov.inverse();
//#else
//                const Matrix2d info = Sigma_u.inverse();
//#endif

//#if USE_XYZCUSTOM
//                EdgeSE2XYZCustom* e = new EdgeSE2XYZCustom();
//#else
//                EdgeSE2XYZ* e = new EdgeSE2XYZ();
//#endif
//                e->setVertex(0, vi);
//                e->setVertex(1, vj);
//                e->setMeasurement(uv);

//                e->setInformation(info);
//                e->setCameraParameter(Config::Kcam);
//                e->setExtParameter(Tbc);
//                e->setRobustKernel(kernel);
//                optimizer.addEdge(e);
//#if PRINT_DEBUG_INFO
//                e->computeError();
//                cout << "EdgeSE2XYZ from " << vi->id() << " to " << vj->id() << " , chi2 = "
//                     << e->chi2() << ", error = [" << e->error().transpose() << "], info = "
//                     << endl << info << endl;
//                cout << "Sigma_u inv = " << endl << Sigma_u.inverse() << endl;
//#endif
//            }
//        }
//    }
//    double t1 = timer.count();
//    cout << "Load " << optimizer.vertices().size() << " vertices and " << optimizer.edges().size()
//         << " edges tatal." << endl;


//#if SAVE_BA_RESULT
//    for (size_t i = 0; i < g_nKFs; ++i) {
//        const VertexSE2* vi = dynamic_cast<const VertexSE2*>(optimizer.vertex(i));
//        const SE2 est = vi->estimate();
//        SE2 scaledEstimate(est[0] * g_scale, est[1] * g_scale, est[2]);

//        VertexSE2* v_scale = new VertexSE2();
//        v_scale->setId(i);
//        v_scale->setEstimate(scaledEstimate);
//        optSaver.addVertex(v_scale);

//        if (i > 0) {
//            EdgeSE2* e_tmp = new EdgeSE2();
//            e_tmp->setVertex(0, dynamic_cast<VertexSE2*>(optSaver.vertex(i - 1)));
//            e_tmp->setVertex(1, v_scale);
//            optSaver.addEdge(e_tmp);
//        }
//    }
//    optSaver.save("/home/vance/output/test_BA_SE2_before.g2o");
//    writeToPlyFile(&optimizer);
//#endif

//    timer.start();
//    optimizer.initializeOptimization(0);
//    optimizer.optimize(15);
//    cout << "optimizaiton time cost in (building system/optimization): " << t1 << "/"
//         << timer.count() << "ms. " << endl;

//#if SAVE_BA_RESULT
//    // update optimization result
//    for (size_t i = 0; i < g_nKFs; ++i) {
//        const VertexSE2* vi = dynamic_cast<const VertexSE2*>(optimizer.vertex(i));
//        const SE2 est = vi->estimate();
//        SE2 scaledEstimate(est[0] * g_scale, est[1] * g_scale, est[2]);

//        VertexSE2* vi_scale = dynamic_cast<VertexSE2*>(optSaver.vertex(i));
//        vi_scale->setEstimate(scaledEstimate);
//    }
//    optSaver.save("/home/vance/output/test_BA_SE2_after.g2o");
//    writeToPlyFileAppend(&optimizer);
//#endif

    return 0;
}
