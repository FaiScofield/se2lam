/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG
* (github.com/hbtang)
*/

#include "Track.h"
#include "LocalMapper.h"
#include "Map.h"
#include "ORBmatcher.h"
#include "converter.h"
#include "cvutil.h"
#include "optimizer.h"
#include <ros/ros.h>


namespace se2lam
{
using namespace std;
using namespace cv;
using namespace Eigen;

typedef lock_guard<mutex> locker;

bool Track::mbUseOdometry = true;
const string trajectoryFile = "/home/vance/output/rk_se2lam/trajectory.txt";
ofstream ofs;

Track::Track()
{
    mState = cvu::FIRST_FRAME;   // NO_READY_YET
    mLocalMPs = vector<Point3f>(Config::MaxFtrNumber, Point3f(-1, -1, -1));
    nMinFrames = 8;
    nMaxFrames = Config::FPS;
    mnGoodPrl = 0;
    // mbTriangulated = false;

    mpORBextractor = new ORBextractor(Config::MaxFtrNumber, Config::ScaleFactor, Config::MaxLevel);
    mMatchIdx.clear();
    mvbGoodPrl.clear();

    mbFinished = false;
    mbFinishRequested = false;

    nLostFrames = 0;
//    try {
//        ofs.open(trajectoryFile, ios_base::out);
//    } catch (exception e) {
//        cout << e.what() << endl;
//    }
//    ofs.close();
}

Track::~Track() {}

void Track::setMap(Map *pMap)
{
    mpMap = pMap;
}

void Track::setLocalMapper(LocalMapper *pLocalMapper)
{
    mpLocalMapper = pLocalMapper;
}

void Track::setSensors(Sensors *pSensors)
{
    mpSensors = pSensors;
}

// TODO 首帧位姿要设为0，Odom也要归零
void Track::run()
{
    if (Config::LOCALIZATION_ONLY)
        return;

    ros::Rate rate(Config::FPS * 5);

    while (ros::ok()) {
        cv::Mat img;
        Se2 odo;

        WorkTimer timer;
        timer.start();

        bool sensorUpdated = mpSensors->update();
        Point3f odo_3f;

        if (sensorUpdated) {
            mpSensors->readData(odo_3f, img);
            odo = Se2(odo_3f.x, odo_3f.y, odo_3f.z);

            {
                locker lock(mMutexForPub);
                bool noFrame = !(Frame::nextId);
                if (noFrame) {
//                    if (mState == FIRST_FRAME)
//                    mFirstFrameOdom = odo;
                    mCreateFrame(img, odo);  // 为初始帧创建帧信息
                    mLastState = mState;
                    mState = cvu::OK;
                } else {
//                    odo.x -= mFirstFrameOdom.x /** cos(mFirstFrameOdom.theta)*/;
//                    odo.y -= mFirstFrameOdom.y /** sin(mFirstFrameOdom.theta)*/;
//                    odo.theta -= mFirstFrameOdom.theta;
//                    cvu::normalizeYawAngle(odo);
                    mTrack(img, odo);  // 非初始帧则进行tracking,计算位姿
                }
            }
            mpMap->setCurrentFramePose(mFrame.Tcw);
            mLastOdom = odo;

            timer.stop();
            printf("[Track] #%d Tracking consuming time: %fms, Pose:[%f, %f]\n",
                   mFrame.id, timer.time, mFrame.Twb.x/1000, mFrame.Twb.y/1000);
            printf("[Track] #%d Odom_input:[%f, %f, %f]\n", mFrame.id,
                   mFrame.odom.x/1000, mFrame.odom.y/1000, mFrame.odom.theta);

            //            writePose();
        }

        if (checkFinish())
            break;

        rate.sleep();
    }

    cerr << "[Track] Exiting tracking .." << endl;
    printf("[Track] Save trajectory to file: %s\n", trajectoryFile.c_str());

    setFinish();
}

void Track::mCreateFrame(const Mat &img, const Se2 &odo)
{

    mFrame = Frame(img, odo, mpORBextractor, Config::Kcam, Config::Dcam);

    mFrame.Twb = Se2(0, 0, 0);  //!@Vance: 当前帧World->Body，即Body的世界坐标，首帧为原点
    mFrame.Tcw = Config::cTb.clone();  //!@Vance: 当前帧Camera->World

    if (mFrame.keyPoints.size() > 80) { // 100
        cout << "========================================================"
             << endl;
        cout << "[Track] #" << mFrame.id << " Create first frame with "
             << mFrame.N << " features." << endl;
        cout << "========================================================"
             << endl;
        mpKF = make_shared<KeyFrame>(mFrame);  // 首帧为关键帧
        mpMap->insertKF(mpKF);
        resetLocalTrack();  // 数据转至参考帧
    } else {
        cout << "[Track] Failed to create first frame for too less keyPoints: "
             << mFrame.keyPoints.size() << endl;
        Frame::nextId = 0;
    }
}

void Track::mTrack(const Mat &img, const Se2 &odo)
{

    WorkTimer timer;
    timer.start();

    mFrame = Frame(img, odo, mpORBextractor, Config::Kcam, Config::Dcam);

    bool bTrackOK;

//    if (mState == TEMPORARY_LOST) {
//        bTrackOK = relocalization();
//        if (bTrackOK) {
//            nLostFrames = 0;
//            mState = OK;
//        } else {
//            nLostFrames++;
//        }
////        if (nLostFrames > 50) {
////            mState = LOST;
////        }
//    }

    assert(mState == cvu::OK);

    ORBmatcher matcher(0.9);
    int nMatchedTmp = matcher.MatchByWindow(mRefFrame, mFrame, mPrevMatched, 20, mMatchIdx);
    // 利用基础矩阵F计算匹配内点，内点数大于10才能继续
    int nMatched = removeOutliers(mRefFrame.keyPointsUn, mFrame.keyPointsUn, mMatchIdx);
    cout << "[Track] #" << mFrame.id << " ORBmatcher get valid/tatal matches "
         << nMatched << "/" << nMatchedTmp << endl;

    updateFramePose();

    // Check parallax and do triangulation
    int nTrackedOld = doTriangulate();

    // Need new KeyFrame decision
    if (needNewKF(nTrackedOld, nMatched)) {
        // Insert KeyFrame
        PtrKeyFrame pKF = make_shared<KeyFrame>(mFrame);

        assert(mpMap->getCurrentKF()->mIdKF == mpKF->mIdKF);
        mpKF->preOdomFromSelf = make_pair(pKF, preSE2);
        pKF->preOdomToSelf = make_pair(mpKF, preSE2);
        mpLocalMapper->addNewKF(pKF, mLocalMPs, mMatchIdx, mvbGoodPrl);

        resetLocalTrack();

        mpKF = pKF;

        printf("[Track] Add new KF at #%d(KF#%d)\n", mFrame.id, mpKF->mIdKF);
    }

    timer.stop();
}

///@Vance: 根据Last KF和当前帧的里程计更新先验位姿和变换关系
//!@Vance: odom是绝对值而非增量
void Track::updateFramePose()
{
    //! 参考帧Body->当前帧Body，向量，这里就是delta_odom
    //! mpKF是Last KF，如果当前帧不是KF，则当前帧位姿由Last
    //! KF叠加上里程计数据计算得到
    //! 这里默认场景是工业级里程计，精度比较准，KF之间里程误差可以忽略
    mFrame.Trb = mFrame.odom - mpKF->odom;

    //! 当前帧Body->参考帧Body？ Trb和Tbr不能直接取负，而是差了一个旋转
    Se2 dOdo = mpKF->odom - mFrame.odom;  //!@Vance: delta_odom,即t_cr?
    //@Vance: 当前帧Camera->参考帧Camera，矩阵
    mFrame.Tcr =
        Config::cTb * dOdo.toCvSE3() * Config::bTc;  //!@Vance: 为何是右乘？
    //! NOTE 当前帧Camera->World，矩阵，为何右乘？
    mFrame.Tcw = mFrame.Tcr * mpKF->Tcw;
    //@Vance: 当前帧World->Body，向量，故相加
    mFrame.Twb = mpKF->Twb + mFrame.Trb;  //!@Vance: 用上变量mFrame.Trb
    //        mFrame.Twb = mpKF->Twb + (mFrame.odom - mpKF->odom);  //!@Vance:
    //        即 mpKF.Twb + mFrame.Trb

    //!@Vance: 换个版本？
    //    mFrame.Trb = mFrame.odom - mpKF->odom;
    //    mFrame.Tcr = Config::bTc * dOdo.toCvSE3() * Config::cTb;
    //    mFrame.Tcw = mpKF->Tcw * mFrame.Tcr;
    //    mFrame.Twb = mpKF->Twb + mFrame.Trb;

    // preintegration 预积分
    //! NOTE 这里并没有使用上预积分？都是局部变量，且实际一帧图像仅对应一帧Odom数据
    Eigen::Map<Vector3d> meas(preSE2.meas);
    Se2 odok = mFrame.odom - mLastOdom;
    Vector2d odork(odok.x, odok.y);
    Matrix2d Phi_ik = Rotation2Dd(meas[2]).toRotationMatrix();
    meas.head<2>() += Phi_ik * odork;
    meas[2] += odok.theta;

    Matrix3d Ak = Matrix3d::Identity();
    Matrix3d Bk = Matrix3d::Identity();
    Ak.block<2, 1>(0, 2) = Phi_ik * Vector2d(-odork[1], odork[0]);
    Bk.block<2, 2>(0, 0) = Phi_ik;
    Eigen::Map<Matrix3d, RowMajor> Sigmak(preSE2.cov);
    Matrix3d Sigma_vk = Matrix3d::Identity();
    Sigma_vk(0, 0) = (Config::ODO_X_NOISE * Config::ODO_X_NOISE);
    Sigma_vk(1, 1) = (Config::ODO_Y_NOISE * Config::ODO_Y_NOISE);
    Sigma_vk(2, 2) = (Config::ODO_T_NOISE * Config::ODO_T_NOISE);
    Matrix3d Sigma_k_1 = Ak * Sigmak * Ak.transpose() + Bk * Sigma_vk * Bk.transpose();
    Sigmak = Sigma_k_1;
}

//!@Vance: 当前帧变参考帧
void Track::resetLocalTrack()
{
    mFrame.Tcr = cv::Mat::eye(4, 4, CV_32FC1);  //!@Vance: 这里值归不归回初始值都没关系吧？
    mFrame.Trb = Se2(0, 0, 0);
    // cv::KeyPoint 转 cv::Point2f
    KeyPoint::convert(mFrame.keyPoints, mPrevMatched);
    mRefFrame = mFrame;
    mLocalMPs = mpKF->mViewMPs;
    mnGoodPrl = 0;
    mMatchIdx.clear();

    for (int i = 0; i < 3; i++)
        preSE2.meas[i] = 0;
    for (int i = 0; i < 9; i++)
        preSE2.cov[i] = 0;
}

int Track::copyForPub(vector<KeyPoint> &kp1, vector<KeyPoint> &kp2, Mat &img1,
                      Mat &img2, vector<int> &vMatches12)
{

    locker lock(mMutexForPub);
    mRefFrame.copyImgTo(img1);
    mFrame.copyImgTo(img2);

    kp1 = mRefFrame.keyPoints;
    kp2 = mFrame.keyPoints;
    vMatches12 = mMatchIdx;

    return !mMatchIdx.empty();
}

void Track::calcOdoConstraintCam(const Se2 &dOdo, Mat &cTc,
                                 g2o::Matrix6d &Info_se3)
{

    const Mat bTc = Config::bTc;
    const Mat cTb = Config::cTb;

    const Mat bTb = Se2(dOdo.x, dOdo.y, dOdo.theta).toCvSE3();

    cTc = cTb * bTb * bTc;

    float dx = dOdo.x * Config::ODO_X_UNCERTAIN + Config::ODO_X_NOISE;
    float dy = dOdo.y * Config::ODO_Y_UNCERTAIN + Config::ODO_Y_NOISE;
    float dtheta = dOdo.theta * Config::ODO_T_UNCERTAIN + Config::ODO_T_NOISE;

    g2o::Matrix6d Info_se3_bTb = g2o::Matrix6d::Zero();
    //    float data[6] = { 1.f/(dx*dx), 1.f/(dy*dy), 1, 1e4, 1e4,
    //    1.f/(dtheta*dtheta) };
    float data[6] = {1.f / (dx * dx), 1.f / (dy * dy), 1e-4,
                     1e-4, 1e-4, 1.f / (dtheta * dtheta)};
    for (int i = 0; i < 6; i++)
        Info_se3_bTb(i, i) = data[i];
    Info_se3 = Info_se3_bTb;


    //    g2o::Matrix6d J_bTb_cTc = toSE3Quat(bTc).adj();
    //    J_bTb_cTc.block(0,3,3,3) = J_bTb_cTc.block(3,0,3,3);
    //    J_bTb_cTc.block(3,0,3,3) = g2o::Matrix3D::Zero();

    //    Info_se3 = J_bTb_cTc.transpose() * Info_se3_bTb * J_bTb_cTc;

    //    for (int i = 0; i < 6; i++)
    //        for (int j = 0; j < i; j++)
    //            Info_se3(i,j) = Info_se3(j,i);

    // assert(verifyInfo(Info_se3));
}

void Track::calcSE3toXYZInfo(Point3f xyz1, const Mat &Tcw1, const Mat &Tcw2,
                             Eigen::Matrix3d &info1, Eigen::Matrix3d &info2)
{

    Point3f O1 = Point3f(cvu::inv(Tcw1).rowRange(0, 3).col(3));
    Point3f O2 = Point3f(cvu::inv(Tcw2).rowRange(0, 3).col(3));
    Point3f xyz = cvu::se3map(cvu::inv(Tcw1), xyz1);
    Point3f vO1 = xyz - O1;
    Point3f vO2 = xyz - O2;
    float sinParallax = cv::norm(vO1.cross(vO2)) / (cv::norm(vO1) * cv::norm(vO2));

    Point3f xyz2 = cvu::se3map(Tcw2, xyz);
    float length1 = cv::norm(xyz1);
    float length2 = cv::norm(xyz2);
    float dxy1 = 2.f * length1 / Config::fxCam;
    float dxy2 = 2.f * length2 / Config::fxCam;
    float dz1 = dxy2 / sinParallax;
    float dz2 = dxy1 / sinParallax;

    Mat info_xyz1 = (Mat_<float>(3, 3) << 1.f / (dxy1 * dxy1), 0, 0, 0,
                     1.f / (dxy1 * dxy1), 0, 0, 0, 1.f / (dz1 * dz1));

    Mat info_xyz2 = (Mat_<float>(3, 3) << 1.f / (dxy2 * dxy2), 0, 0, 0,
                     1.f / (dxy2 * dxy2), 0, 0, 0, 1.f / (dz2 * dz2));

    Point3f z1 = Point3f(0, 0, length1);
    Point3f z2 = Point3f(0, 0, length2);
    Point3f k1 = xyz1.cross(z1);
    float normk1 = cv::norm(k1);
    float sin1 = normk1 / (cv::norm(z1) * cv::norm(xyz1));
    k1 = k1 * (std::asin(sin1) / normk1);
    Point3f k2 = xyz2.cross(z2);
    float normk2 = cv::norm(k2);
    float sin2 = normk2 / (cv::norm(z2) * cv::norm(xyz2));
    k2 = k2 * (std::asin(sin2) / normk2);

    Mat R1, R2;
    Mat k1mat = (Mat_<float>(3, 1) << k1.x, k1.y, k1.z);
    Mat k2mat = (Mat_<float>(3, 1) << k2.x, k2.y, k2.z);
    cv::Rodrigues(k1mat, R1);
    cv::Rodrigues(k2mat, R2);

    info1 = toMatrix3d(R1.t() * info_xyz1 * R1);
    info2 = toMatrix3d(R2.t() * info_xyz2 * R2);
}

int Track::removeOutliers(const vector<KeyPoint> &kp1,
                          const vector<KeyPoint> &kp2, vector<int> &matches)
{
    vector<Point2f> pt1, pt2;
    vector<int> idx;
    pt1.reserve(kp1.size());
    pt2.reserve(kp2.size());
    idx.reserve(kp1.size());

    for (int i = 0, iend = kp1.size(); i < iend; i++) {
        if (matches[i] < 0)
            continue;
        idx.push_back(i);
        pt1.push_back(kp1[i].pt);
        pt2.push_back(kp2[matches[i]].pt);
    }

    vector<unsigned char> mask;

    if (pt1.size() != 0)
        findFundamentalMat(pt1, pt2, mask);

    int nInlier = 0;
    for (int i = 0, iend = mask.size(); i < iend; i++) {
        if (!mask[i])
            matches[idx[i]] = -1;
        else
            nInlier++;
    }

    // If too few match inlier, discard all matches. The enviroment might not be
    // suitable for image tracking.
    if (nInlier < 10) {
        nInlier = 0;
        std::fill(mMatchIdx.begin(), mMatchIdx.end(), -1);
    }

    return nInlier;
}

bool Track::needNewKF(int nTrackedOldMP, int nMatched)
{
    int nOldKP = mpKF->getSizeObsMP();
    bool c0 = mFrame.id - mpKF->id > nMinFrames;  // 间隔不能太小
    bool c1 = (float)nTrackedOldMP <= (float)nOldKP * 0.5f;
    bool c2 = mnGoodPrl > 40;  // 重叠视野小于之前的一半，且当前视野内点数大于40
    bool c3 = mFrame.id - mpKF->id > nMaxFrames;  // 或间隔太大
    bool c4 = nMatched < 0.1f * Config::MaxFtrNumber || nMatched < 20;  // 或匹配对小于1/10最大特征点数
    bool bNeedNewKF = c0 && ((c1 && c2) || c3 || c4);  // 则可能需要加入关键帧，实际还得看odom那边的条件

    bool bNeedKFByOdo = true;
    if (mbUseOdometry) {
        Se2 dOdo = mFrame.odom - mpKF->odom;
        bool c5 = dOdo.theta >= 0.0349f;  // Larger than 2 degree
        // cv::Mat cTc = Config::cTb * toT4x4(dOdo.x, dOdo.y, dOdo.theta) *
        // Config::bTc;
        cv::Mat cTc = Config::cTb * Se2(dOdo.x, dOdo.y, dOdo.theta).toCvSE3() * Config::bTc;
        cv::Mat xy = cTc.rowRange(0, 2).col(3);
        bool c6 = cv::norm(xy) >= (0.0523f * Config::UPPER_DEPTH * 0.1f);  // 3 degree = 0.0523 rad

        bNeedKFByOdo = c5 || c6;  // 里程计旋转超过2°，或者移动距离超过一定距离(约2-5cm,
                       // for 4-10m UPPER_DEPTH)
    }
    bNeedNewKF = bNeedNewKF && bNeedKFByOdo;  // 加上odom的移动条件

    if (mpLocalMapper->acceptNewKF()) {
        return bNeedNewKF;
    } else if (c0 && (c4 || c3) && bNeedKFByOdo) {
        mpLocalMapper->setAbortBA();
    }

    return false;
}

int Track::doTriangulate()
{
    if (mFrame.id - mpKF->id < nMinFrames) {
        return 0;
    }

    Mat TfromRefKF = cvu::inv(mFrame.Tcr);
    Point3f Ocam = Point3f(TfromRefKF.rowRange(0, 3).col(3));
    int nTrackedOld = 0;
    mvbGoodPrl = vector<bool>(mRefFrame.N, false);
    mnGoodPrl = 0;

    for (int i = 0; i < mRefFrame.N; i++) {
        if (mMatchIdx[i] < 0)
            continue;

        //!@Vance: 如果上一关键帧看得到当前特征点，则局部地图点更新为此点？
        if (mpKF->hasObservation(i)) {
            mLocalMPs[i] = mpKF->mViewMPs[i];
            nTrackedOld++;
            continue;
        }

        Point2f pt_KF = mpKF->keyPointsUn[i].pt;
        Point2f pt = mFrame.keyPointsUn[mMatchIdx[i]].pt;
        cv::Mat P_KF = Config::PrjMtrxEye;
        cv::Mat P = Config::Kcam * mFrame.Tcr.rowRange(0, 3);
        Point3f pos = cvu::triangulate(pt_KF, pt, P_KF, P);

        if (Config::acceptDepth(pos.z)) {
            mLocalMPs[i] = pos;
            // 检查视差
            if (cvu::checkParallax(Point3f(0, 0, 0), Ocam, pos, 2)) {
                mnGoodPrl++;
                mvbGoodPrl[i] = true;
            }
        } else {
            mMatchIdx[i] = -1;
        }
    }

    return nTrackedOld;
}

void Track::requestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool Track::checkFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

bool Track::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void Track::setFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

void Track::writePose()
{
    Mat wTb = mpMap->getCurrentFramePose();
    Mat wRb = wTb.rowRange(0, 3).colRange(0, 3);
    g2o::Vector3D euler = g2o::internal::toEuler(se2lam::toMatrix3d(wRb));
    float x = wTb.at<float>(0, 3);
    float y = wTb.at<float>(1, 3);
    float theta = euler(2);
    printf("[Track] #%d Tcw:[%f, %f, %f]\n", mFrame.id, x, y, theta);

    ofs.open(trajectoryFile, ios_base::app);
    if (!ofs.is_open()) {
        fprintf(stderr, "[Track] Failed to open trajectory file!\n");
        return;
    }
    ofs << mFrame.id << " " << x << " " << y << " " << theta << endl;
    ofs.close();
}

//! TODO 完成重定位功能
/*
bool Track::relocalization()
{
    // 步骤1：计算当前帧特征点的Bow映射
    mFrame.ComputeBoW();

    // 步骤2：找到与当前帧相似的候选关键帧
    vector<KeyFrame *> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mFrame);

    if (vpCandidateKFs.empty())
        return false;

    const size_t nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75, true);

    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint *>> vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates = 0;

    for (size_t i = 0; i < nKFs; i++) {
        KeyFrame *pKF = vpCandidateKFs[i];
        if (pKF->isBad())
            vbDiscarded[i] = true;
        else {
            // 步骤3：通过BoW进行匹配
            int nmatches =
                matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
            if (nmatches < 15) {
                vbDiscarded[i] = true;
                continue;
            } else {
                // 初始化PnPsolver
                PnPsolver *pSolver =
                    new PnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99, 10, 300, 4, 0.5, 5.991f);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9f, true);

    while (nCandidates > 0 && !bMatch) {
        for (size_t i = 0; i < nKFs; i++) {
            if (vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            // 步骤4：通过EPnP算法估计姿态
            PnPsolver *pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if (bNoMore) {
                vbDiscarded[i] = true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if (!Tcw.empty()) {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint *> sFound;

                const size_t np = vbInliers.size();

                for (size_t j = 0; j < np; j++) {
                    if (vbInliers[j]) {
                        mCurrentFrame.mvpMapPoints[j] =
                            vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    } else
                        mCurrentFrame.mvpMapPoints[j] = NULL;
                }

                // 步骤5：通过PoseOptimization对姿态进行优化求解
                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if (nGood < 10)
                    continue;

                for (int io = 0; io < mCurrentFrame.N; io++)
                    if (mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io] =
                            static_cast<MapPoint *>(NULL);

                // If few inliers, search by projection in a coarse window and
                // optimize again
                // 步骤6：如果内点较少，则通过投影的方式对之前未匹配的点进行匹配，再进行优化求解
                if (nGood < 50) {
                    int nadditional = matcher2.SearchByProjection(
                        mCurrentFrame, vpCandidateKFs[i], sFound, 10, 100);

                    if (nadditional + nGood >= 50) {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by
                        // projection again in a narrower window
                        // the camera has been already optimized with many
                        // points
                        if (nGood > 30 && nGood < 50) {
                            sFound.clear();
                            for (int ip = 0; ip < mCurrentFrame.N; ip++)
                                if (mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(
                                        mCurrentFrame.mvpMapPoints[ip]);
                            nadditional = matcher2.SearchByProjection(
                                mCurrentFrame, vpCandidateKFs[i], sFound, 3,
                                64);

                            // Final optimization
                            if (nGood + nadditional >= 50) {
                                nGood =
                                    Optimizer::PoseOptimization(&mCurrentFrame);

                                for (int io = 0; io < mCurrentFrame.N; io++)
                                    if (mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io] = NULL;
                            }
                        }
                    }
                }

                // If the pose is supported by enough inliers stop ransacs and
                // continue
                if (nGood >= 50) {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if (!bMatch) {
        return false;
    } else {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }
}
*/
}  // namespace se2lam
