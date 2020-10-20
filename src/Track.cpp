/**
 * This file is part of se2lam
 *
 * Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
 */

#include "Track.h"
#include "LocalMapper.h"
#include "Map.h"
#include "MapPublish.h"
#include "ORBmatcher.h"
#include "converter.h"
#include "cvutil.h"
#include "optimizer.h"
#include <opencv2/calib3d/calib3d.hpp>
#include <ros/ros.h>


namespace se2lam
{
using namespace std;
using namespace cv;
using namespace Eigen;

typedef lock_guard<mutex> locker;

bool Track::mbUseOdometry = true;

Track::Track()
{
    mLocalMPs = vector<Point3f>(Config::MaxFtrNumber, Point3f(-1, -1, -1));
    nMinFrames = min(8, cvCeil(0.25 * Config::FPS));  // 上溢
    nMaxFrames = cvFloor(1 * Config::FPS);            // 下溢
    mnGoodPrl = 0;
    mnKPMatches = 0;
    mnKPsInline = 0;
    mnKPMatchesGood = 0;
    mnMPsTracked = 0;

    mpORBextractor = new ORBextractor(Config::MaxFtrNumber, Config::ScaleFactor, Config::MaxLevel);
    mpORBmatcher = new ORBmatcher(0.9, true);

    mAffineMatrix = Mat::eye(2, 3, CV_64FC1);

    mbNeedVisualization = Config::NeedVisualization;
    mbFinished = false;
    mbFinishRequested = false;
}

Track::~Track()
{
    delete mpORBextractor;
    delete mpORBmatcher;
}

void Track::Run()
{
    if (Config::LocalizationOnly)
        return;

    LOGT("--st-- Track::Run()");

    CheckReady();

    WorkTimer timer;
    Mat img;
    Se2 odo;
    double trackTimeTatal = 0.;
    double t1 = 0., t2 = 0.;

    ros::Rate rate(Config::FPS * 5);
    while (ros::ok()) {
        if (CheckFinish())
            break;

        if (!mpSensors->update()) {
            rate.sleep();
            continue;
        }

        // read data
        timer.start();
        mpSensors->readData(odo, img);
        t1 = timer.count();

        // tracking
        timer.start();
        {
            locker lock(mMutexForPub);
            bool noFrame = !(Frame::nextId);
            if (noFrame)
                ProcessFirstFrame(img, odo);
            else
                TrackReferenceKF(img, odo);
        }
        mpMap->setCurrentFramePose(mCurrentFrame.Tcw);
        lastOdom = odo;
        t2 = timer.count();
        trackTimeTatal += t1 + t2;
        LOGT(setiosflags(ios::fixed) << setprecision(2));  // 设置浮点数保留2位小数
        LOGT("#" << mCurrentFrame.id << "-#" << mpReferenceKF->id << " TIME COST: \t" << t1 + t2
                 << "ms, AVE: " << trackTimeTatal / mCurrentFrame.id << "ms");

        CopyForPub();

        ResetLocalTrack();

        rate.sleep();
    }

    LOGT("--en-- Track::Run()");

    SetFinish();
}


void Track::ProcessFirstFrame(const Mat& img, const Se2& odo)
{
    LOGT("--st-- Track::ProcessFirstFrame()");
    mCurrentFrame = Frame(img, odo, mpORBextractor, Config::Kcam, Config::Dcam);

    const size_t th_nMaxFtrNum = (Config::MaxFtrNumber >> 1);
    if (mCurrentFrame.N > th_nMaxFtrNum) {  // 首帧特征点需要超过最大点数的一半
        cout << "========================================================" << endl;
        cout << "[Track][Info ] Create first frame with " << mCurrentFrame.N << " features. "
             << "And the start odom is: " << mCurrentFrame.odom << endl;
        cout << "========================================================" << endl;

        mCurrentFrame.Twb = Se2(0, 0, 0);
        mCurrentFrame.Tcw = Config::Tcb.clone();
        mpReferenceKF = make_shared<KeyFrame>(mCurrentFrame);
        mpMap->insertKF(mpReferenceKF);

        // reset
        KeyPoint::convert(mCurrentFrame.mvKeyPoints, mPrevMatched);
        mLastFrame = mCurrentFrame;
        mLocalMPs = mpReferenceKF->mViewMPs;
        mvKPMatchIdx.clear();
        mAffineMatrix = Mat::eye(2, 3, CV_64FC1);
        mnGoodPrl = mnKPMatches = mnKPsInline = mnKPMatchesGood = mnMPsTracked = 0;
        for (int i = 0; i < 3; i++)
            preSE2.meas[i] = 0;
        for (int i = 0; i < 9; i++)
            preSE2.cov[i] = 0;

    } else {
        cerr << "[Track][Warni] Failed to create first frame for too less keyPoints: "
             << mCurrentFrame.N << endl;

        Frame::nextId = 0;
    }
    LOGT("--en-- Track::ProcessFirstFrame()");
}

void Track::TrackReferenceKF(const Mat& img, const Se2& odo)
{
    mCurrentFrame = Frame(img, odo, mpORBextractor, Config::Kcam, Config::Dcam);
    UpdateFramePose();

    // int nMatched = mpORBmatcher->MatchByWindow(mLastFrame, mCurrentFrame, mPrevMatched, 20, mvKPMatchIdx);
    mnKPMatches = mpORBmatcher->MatchByWindowWarp(mLastFrame, mCurrentFrame, mAffineMatrix, mvKPMatchIdx, 20);
    mnKPsInline = RemoveOutliers(mLastFrame.keyPointsUn, mCurrentFrame.keyPointsUn, mvKPMatchIdx);

    // Check parallax and do triangulation
    mnMPsTracked = DoTriangulate();

    cout << "[Track][Info ] #" << mCurrentFrame.id << "-#" << mpReferenceKF->id
         << " 追踪匹配情况: 关联/内点数/总匹配数/潜在好视差数 = " << mnMPsTracked << "/"
         << mnKPsInline << "/" << mnKPMatches << "/" << mnGoodPrl << endl;
}

void Track::UpdateFramePose()
{
    mCurrentFrame.Trb = mCurrentFrame.odom - mpReferenceKF->odom;
    Se2 dOdo = mpReferenceKF->odom - mCurrentFrame.odom;
    mCurrentFrame.Tcr = Config::Tcb * dOdo.toCvSE3() * Config::Tbc;
    mCurrentFrame.Tcw = mCurrentFrame.Tcr * mpReferenceKF->Tcw;
    mCurrentFrame.Twb = mpReferenceKF->Twb + mCurrentFrame.Trb;

    // preintegration
    Eigen::Map<Vector3d> meas(preSE2.meas);
    Se2 odok = mCurrentFrame.odom - lastOdom;
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
    Sigma_vk(0, 0) = (Config::OdoNoiseX * Config::OdoNoiseX);
    Sigma_vk(1, 1) = (Config::OdoNoiseY * Config::OdoNoiseY);
    Sigma_vk(2, 2) = (Config::OdoNoiseTheta * Config::OdoNoiseTheta);
    Matrix3d Sigma_k_1 = Ak * Sigmak * Ak.transpose() + Bk * Sigma_vk * Bk.transpose();
    Sigmak = Sigma_k_1;
}

void Track::ResetLocalTrack()
{
    // Need new KeyFrame decision
    if (NeedNewKF()) {
        assert(mpMap->getCurrentKF()->mIdKF == mpReferenceKF->mIdKF);

        // Insert KeyFrame
        PtrKeyFrame pKF = make_shared<KeyFrame>(mCurrentFrame);
        mpReferenceKF->preOdomFromSelf = make_pair(pKF, preSE2);
        pKF->preOdomToSelf = make_pair(mpReferenceKF, preSE2);
        mpLocalMapper->addNewKF(pKF, mLocalMPs, mvKPMatchIdx, mvbGoodPrl);
        mpReferenceKF = pKF;

        cout << "[Track][Info ] #" << mCurrentFrame.id << "-#" << mpReferenceKF->id
             << " 成为了新的KF#" << pKF->mIdKF << endl;

        KeyPoint::convert(mCurrentFrame.mvKeyPoints, mPrevMatched);
        mLastFrame = mCurrentFrame;
        mLocalMPs = mpReferenceKF->mViewMPs;
        mvKPMatchIdx.clear();
        mAffineMatrix = Mat::eye(2, 3, CV_64FC1);
        mnGoodPrl = mnKPMatches = mnKPsInline = mnKPMatchesGood = mnMPsTracked = 0;
        for (int i = 0; i < 3; i++)
            preSE2.meas[i] = 0;
        for (int i = 0; i < 9; i++)
            preSE2.cov[i] = 0;
    }
}

void Track::CopyForPub()
{
    locker lock1(mMutexForPub);
    locker lock2(mpMapPublisher->mMutexUpdate);

    mvKPMatchIdxGood = mvKPMatchIdx;
    mpMapPublisher->mnCurrentFrameID = mCurrentFrame.id;
    mpMapPublisher->mCurrentFramePose = mCurrentFrame.Tcw;
    mpMapPublisher->mCurrentImage = mCurrentFrame.mImage.clone();
    mpMapPublisher->mvCurrentKPs = mCurrentFrame.mvKeyPoints;
    mpMapPublisher->mvMatchIdx = mvKPMatchIdx;
    mpMapPublisher->mvMatchIdxGood = mvKPMatchIdxGood;
    mpMapPublisher->mAffineMatrix = mAffineMatrix.clone();
    assert(!mAffineMatrix.empty());

    char strMatches[64];
    if (1) {  // 正常情况和刚丢失情况
        mpMapPublisher->mReferenceImage = mpReferenceKF->mImage.clone();
        mpMapPublisher->mvReferenceKPs = mpReferenceKF->mvKeyPoints;
        const int d = mCurrentFrame.id - mpReferenceKF->id;
        std::snprintf(strMatches, 64, "F: %d, KF: %d(%d), D: %d, M: %ld", mCurrentFrame.id,
                      mpReferenceKF->id, mpReferenceKF->mIdKF, d, mvKPMatchIdx.size());
    } else {  // 丢失情况和刚完成重定位
        if (mpLoopKF != nullptr) {
            mpMapPublisher->mReferenceImage = mpLoopKF->mImage.clone();
            mpMapPublisher->mvReferenceKPs = mpLoopKF->mvKeyPoints;
        } else {
            mpMapPublisher->mReferenceImage = Mat::zeros(mCurrentFrame.mImage.size(), CV_8UC1);
            mpMapPublisher->mvReferenceKPs.clear();
        }
    }

    mpMapPublisher->mFrontText = string(strMatches);
    mpMapPublisher->mbFrontUpdated = true;
}

void Track::calcOdoConstraintCam(const Se2& dOdo, Mat& cTc, g2o::Matrix6d& Info_se3)
{
    const Mat bTc = Config::Tbc;
    const Mat cTb = Config::Tcb;

    const Mat bTb = Se2(dOdo.x, dOdo.y, dOdo.theta).toCvSE3();

    cTc = cTb * bTb * bTc;

    float dx = dOdo.x * Config::OdoUncertainX + Config::OdoNoiseX;
    float dy = dOdo.y * Config::OdoUncertainY + Config::OdoNoiseY;
    float dtheta = dOdo.theta * Config::OdoUncertainTheta + Config::OdoNoiseTheta;

    g2o::Matrix6d Info_se3_bTb = g2o::Matrix6d::Zero();
    // float data[6] = { 1.f/(dx*dx), 1.f/(dy*dy), 1, 1e4, 1e4, 1.f/(dtheta*dtheta) };
    float data[6] = {1.f / (dx * dx), 1.f / (dy * dy), 1e-4, 1e-4, 1e-4, 1.f / (dtheta * dtheta)};
    for (int i = 0; i < 6; i++)
        Info_se3_bTb(i, i) = data[i];
    Info_se3 = Info_se3_bTb;

    // g2o::Matrix6d J_bTb_cTc = toSE3Quat(bTc).adj();
    // J_bTb_cTc.block(0,3,3,3) = J_bTb_cTc.block(3,0,3,3);
    // J_bTb_cTc.block(3,0,3,3) = g2o::Matrix3D::Zero();
    // Info_se3 = J_bTb_cTc.transpose() * Info_se3_bTb * J_bTb_cTc;
    // for(int i = 0; i < 6; i++)
    //     for(int j = 0; j < i; j++)
    //         Info_se3(i,j) = Info_se3(j,i);
    // assert(verifyInfo(Info_se3));
}

void Track::calcSE3toXYZInfo(Point3f xyz1, const Mat& Tcw1, const Mat& Tcw2, Eigen::Matrix3d& info1,
                             Eigen::Matrix3d& info2)
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
    float dxy1 = 2.f * length1 / Config::fx;
    float dxy2 = 2.f * length2 / Config::fx;
    float dz1 = dxy2 / sinParallax;
    float dz2 = dxy1 / sinParallax;

    Mat info_xyz1 = (Mat_<float>(3, 3) << 1.f / (dxy1 * dxy1), 0, 0, 0, 1.f / (dxy1 * dxy1), 0, 0,
                     0, 1.f / (dz1 * dz1));

    Mat info_xyz2 = (Mat_<float>(3, 3) << 1.f / (dxy2 * dxy2), 0, 0, 0, 1.f / (dxy2 * dxy2), 0, 0,
                     0, 1.f / (dz2 * dz2));

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

int Track::RemoveOutliers(const vector<KeyPoint>& kp1, const vector<KeyPoint>& kp2, vector<int>& matches)
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
        // findFundamentalMat(pt1, pt2, mask);
        mAffineMatrix = estimateAffinePartial2D(pt1, pt2, mask, RANSAC, 2.0);

    int nInlier = 0;
    for (int i = 0, iend = mask.size(); i < iend; i++) {
        if (!mask[i])
            matches[idx[i]] = -1;
        else
            nInlier++;
    }

    // If too few match inlier, discard all matches. The enviroment might not be suitable for image
    // tracking.
    if (nInlier < 10) {
        nInlier = 0;
        std::fill(mvKPMatchIdx.begin(), mvKPMatchIdx.end(), -1);
    }

    return nInlier;
}

bool Track::NeedNewKF()
{
    size_t nOldKP = mpReferenceKF->countObservations();
    bool c0 = mCurrentFrame.id - mpReferenceKF->id > nMinFrames;
    bool c1 = mnMPsTracked <= (float)nOldKP * 0.5f;
    bool c2 = mnGoodPrl > 40;
    bool c3 = mCurrentFrame.id - mpReferenceKF->id > nMaxFrames;
    bool c4 = mnKPMatches < 0.1f * Config::MaxFtrNumber || mnKPMatches < 20;
    bool bNeedNewKF = c0 && ((c1 && c2) || c3 || c4);

    bool bNeedKFByOdo = true;
    bool c5 = false, c6 = false;
    if (mbUseOdometry) {
        Se2 dOdo = mCurrentFrame.odom - mpReferenceKF->odom;
        c5 = abs(dOdo.theta) >= 0.8727;  // 0.8727(50deg). orig: 0.0349(2deg)
        cv::Mat cTc = Config::Tcb * Se2(dOdo.x, dOdo.y, dOdo.theta).toCvSE3() * Config::Tbc;
        cv::Mat xy = cTc.rowRange(0, 2).col(3);
        // c6 = cv::norm(xy) >= (0.0523f * Config::UpperDepth * 0.1f);  // orig: 3deg*0.1*UpperDepth
        c6 = cv::norm(xy) >= (0.05 * Config::UpperDepth);  // 10m高走半米

        bNeedKFByOdo = c5 || c6;
    }
    bNeedNewKF = bNeedNewKF && bNeedKFByOdo;

    if (mpLocalMapper->acceptNewKF()) {
        cout << "[Track][Timer] #" << mCurrentFrame.id << "-#" << mpReferenceKF->id
             << " 当前帧即将成为新的KF, 其KF条件满足情况: "
             << "(关联少+视差好(" << c1 << "+" << mnGoodPrl << ")/达上限(" << c3 << ")/匹配少("
             << c4 << "))&&(旋转大(" << c5 << ")/平移大(" << c6 << "))" << endl;
        return bNeedNewKF;
    } else if (c0 && (c4 || c3) && bNeedKFByOdo) {
        cout << "[Track][Timer] #" << mCurrentFrame.id << "-#" << mpReferenceKF->id
             << " 强制中断LM的BA过程, 当前帧KF条件满足情况: "
             << "(关联少+视差好(" << c1 << "+" << mnGoodPrl << ")/达上限(" << c3 << ")/匹配少("
             << c4 << "))&&(旋转大(" << c5 << ")/平移大(" << c6 << "))" << endl;
        // mpLocalMapper->setAbortBA();
    }

    return false;
}

int Track::DoTriangulate()
{
    if (mCurrentFrame.id - mpReferenceKF->id < nMinFrames) {
        return 0;
    }

    Mat TfromRefKF = cvu::inv(mCurrentFrame.Tcr);
    Point3f Ocam = Point3f(TfromRefKF.rowRange(0, 3).col(3));
    int nTrackedOld = 0;
    mvbGoodPrl = vector<bool>(mLastFrame.N, false);
    mnGoodPrl = 0;

    for (int i = 0; i < mLastFrame.N; i++) {
        if (mvKPMatchIdx[i] < 0)
            continue;

        if (mpReferenceKF->hasObservation(i)) {
            mLocalMPs[i] = mpReferenceKF->mViewMPs[i];
            nTrackedOld++;
            continue;
        }

        Point2f pt_KF = mpReferenceKF->keyPointsUn[i].pt;
        Point2f pt = mCurrentFrame.keyPointsUn[mvKPMatchIdx[i]].pt;
        cv::Mat P_KF = Config::PrjMtrxEye;
        cv::Mat P = Config::Kcam * mCurrentFrame.Tcr.rowRange(0, 3);
        Point3f pos = cvu::triangulate(pt_KF, pt, P_KF, P);

        if (Config::acceptDepth(pos.z)) {
            mLocalMPs[i] = pos;
            if (cvu::checkParallax(Point3f(0, 0, 0), Ocam, pos, 2)) {
                mnGoodPrl++;
                mvbGoodPrl[i] = true;
            }
        } else {
            mvKPMatchIdx[i] = -1;
        }
    }

    return nTrackedOld;
}

void Track::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool Track::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

bool Track::IsFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void Track::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

void Track::CheckReady()
{
    assert(mpMap != nullptr);
    assert(mpLocalMapper != nullptr);
    assert(mpGlobalMapper != nullptr);
    assert(mpSensors != nullptr);
    assert(mpMapPublisher != nullptr);
    assert(mpORBextractor != nullptr);
    assert(mpORBmatcher != nullptr);
}

}  // namespace se2lam
