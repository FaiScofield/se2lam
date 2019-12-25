/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG
* (github.com/hbtang)
*/

#include "Track.h"
#include "Config.h"
#include "GlobalMapper.h"
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

typedef unique_lock<mutex> locker;

bool Track::mbUseOdometry = true;

Track::Track()
    : mState(cvu::NO_READY_YET), mLastState(cvu::NO_READY_YET), mbPrint(true),
      mbNeedVisualization(false), mbRelocalized(false), mpReferenceKF(nullptr), mpLoopKF(nullptr),
      mnMPsNewAdded(0), mnKPsInline(0), mnMPsInline(0), mnMPsTracked(0), mnLostFrames(0),
      mbFinishRequested(false), mbFinished(false)
{
    mpORBextractor = new ORBextractor(Config::MaxFtrNumber, Config::ScaleFactor, Config::MaxLevel);
    mpORBmatcher = new ORBmatcher(0.9, true);

    nMinFrames = min(2, cvCeil(0.25 * Config::FPS));  // 上溢
    nMaxFrames = cvFloor(5 * Config::FPS);  // 下溢
    nMinMatches = std::min(cvFloor(0.1 * Config::MaxFtrNumber), 40);
    mMaxAngle = static_cast<float>(g2o::deg2rad(80.));
    mMaxDistance = 0.3f * Config::UpperDepth;

    mAffineMatrix = Mat::eye(2, 3, CV_64FC1);  // double
    preSE2.meas.setZero();
    preSE2.cov.setZero();

    mbPrint = Config::GlobalPrint;
    mbNeedVisualization = Config::NeedVisualization;

    fprintf(stderr, "[Track][Info ] 相关参数如下: \n - 最小/最大KF帧数: %d/%d\n"
                    " - 最大移动距离/角度: %.0fmm/%.0fdeg\n - 最少匹配数量: %d\n",
            nMinFrames, nMaxFrames, mMaxDistance, g2o::rad2deg(mMaxAngle), nMinMatches);
}

Track::~Track()
{
    delete mpORBextractor;
    delete mpORBmatcher;
}

void Track::run()
{
    checkReady();

    if (Config::LocalizationOnly)
        return;

    if (mState == cvu::NO_READY_YET)
        mState = cvu::FIRST_FRAME;

    unsigned long lastRefKFid = 0; // 这里仅用于输出logs
    ros::Rate rate(Config::FPS * 5);
    while (ros::ok()) {
        if (checkFinish())
            break;

        if (!mpSensors->update()) {
            rate.sleep();
            continue;
        }

        WorkTimer timer;
        mLastState = mState;

        Mat img;
        Se2 odo;
        double time = 0;
        mpSensors->readData(odo, img);
        double t1 = timer.count(), t2 = 0, t3 = 0;

        timer.start();
        bool bOK = false;
        {
            locker lock(mMutexForPub);

            mCurrentFrame = Frame(img, odo, mpORBextractor, time);
            t2 = timer.count();

            timer.start();
            if (mState == cvu::FIRST_FRAME) {
                processFirstFrame();
                rate.sleep();
                continue;
            } else if (mState == cvu::OK) {
                bOK = trackReferenceKF();
                if (!bOK)
                    bOK = trackLocalMap();  // 刚丢的还可以再抢救一下
            } else if (mState == cvu::LOST) {
                bOK = doRelocalization();  // 没追上的直接检测回环重定位
            }
            t3 = timer.count();
        }

        if (bOK) {
            // TODO 更新一下MPCandidates里面Tc2w
            // FIXME 回环通过产生新的KF的MP观测数太少!
            mnLostFrames = 0;
            mState = cvu::OK;
        } else {
            mnLostFrames++;
            mState = cvu::LOST;
            if (mnLostFrames >= 50) {
                startNewTrack();
                rate.sleep();
                continue;
            }
        }
        mpMap->setCurrentFramePose(mCurrentFrame.getPose());

        {  // Reset
            locker lock(mMutexForPub);
            lastRefKFid = mpReferenceKF->id;
            resetLocalTrack();  // KF判断在这里
        }

        trackTimeTatal += t2 + t3;
        printf("[Track][Timer] #%ld-#%ld 前端总耗时: 读取数据/构建Frame/追踪/总耗时为: "
               "%.2f/%.2f/%.2f/%.2fms, 平均追踪耗时: %.2fms\n",
               mCurrentFrame.id, lastRefKFid, t1, t2, t3, t1 + t2 + t3,
               trackTimeTatal * 1.f / mCurrentFrame.id);

        rate.sleep();
    }

    cerr << "[Track][Info ] Exiting tracking .." << endl;
    setFinish();
}

void Track::processFirstFrame()
{
    size_t th = Config::MaxFtrNumber;
    if (mCurrentFrame.N > (th >> 1)) {  // 首帧特征点需要超过最大点数的一半
        cout << "========================================================" << endl;
        cout << "[Track][Info ] Create first frame with " << mCurrentFrame.N << " features. "
             << "And the start odom is: " << mCurrentFrame.odom << endl;
        cout << "========================================================" << endl;

        mCurrentFrame.setPose(Se2(0, 0, 0));
        mpReferenceKF = make_shared<KeyFrame>(mCurrentFrame);  // 首帧为关键帧
        mpMap->insertKF(mpReferenceKF);  // 首帧的KF直接给Map, 没有给LocalMapper

        mLastFrame = mCurrentFrame;
        mState = cvu::OK;
    } else {
        cerr << "[Track][Warni] Failed to create first frame for too less keyPoints: "
             << mCurrentFrame.N << endl;

        Frame::nextId = 0;
        mState = cvu::FIRST_FRAME;
    }
}

bool Track::trackReferenceKF()
{
    if (mCurrentFrame.isNull())  // 图像挂掉的时候Frame会变成null.
        return false;
    if (mCurrentFrame.mTimeStamp < mLastFrame.mTimeStamp) {
        fprintf(stderr, "[Track][Warni] #%ld-#%ld 图像序列在时间上不连续: Last = %.3f, Curr = %.3f\n",
                mCurrentFrame.id, mLastFrame.id, mLastFrame.mTimeStamp, mCurrentFrame.mTimeStamp);
        return false;
    }

    assert(mnKPsInline == 0);

    // 1.根据里程计设置初始位姿
    updateFramePoseFromRef();

    // 2.基于仿射变换先验估计投影Cell的位置
    //! TODO  仿射变换要改成欧氏变换!! 只有旋转和平移, 没有缩放!
    mnKPMatches =
        mpORBmatcher->MatchByWindowWarp(*mpReferenceKF, mCurrentFrame, mAffineMatrix, mvKPMatchIdx, 20);
    if (mnKPMatches < 15) {
        printf("[Track][Warni] #%ld-#%ld 与参考帧匹配[总]点数少于15(%d), 即将转为重定位!\n",
               mCurrentFrame.id, mpReferenceKF->id, mnKPMatches);
        return false;
    }

    //! 3.利用仿射矩阵A计算KP匹配的内点，内点数大于10才能继续
    mnKPsInline = removeOutliers(mpReferenceKF, &mCurrentFrame, mvKPMatchIdx, mAffineMatrix);
    if (mnKPsInline < 10) {
        printf("[Track][Warni] #%ld-#%ld 与参考帧匹配[内]点数少于10(%d/%d), 即将转为重定位!\n",
               mCurrentFrame.id, mpReferenceKF->id, mnKPsInline, mnKPMatches);
        return false;
    }

    //! 4.三角化生成潜在MP, 由LocalMap线程创造MP
    doTriangulate(mpReferenceKF);  // 更新 mnTrackedOld, mnGoodInliers, mvGoodMatchIdx

    //! TODO FIXME 5.最小化重投影误差优化当前帧位姿
    const int nObs = mCurrentFrame.countObservations();
    if (0 && mnMPsTracked > 0 && nObs > 0) {  // 保证有MP观测才做优化 // TODO
        Se2 Twb = mCurrentFrame.getTwb();
        printf(
            "[Track][Warni] #%ld-#%ld 位姿优化更新前的值为: [%.2f, %.2f, %.2f], 可视MP数为:%d\n",
            mCurrentFrame.id, mpReferenceKF->id, Twb.x, Twb.y, Twb.theta, nObs);

        int nCros = 0;
        double projError = 10000.;
        poseOptimization(&mCurrentFrame, nCros, projError);  // 更新 mPose, mvbOutlier
        // mCurrentFrame.updateObservationsAfterOpt();  // 根据 mvbOutlier 更新Frame的观测情况

        Twb = mCurrentFrame.getTwb();
        printf("[Track][Warni] #%ld-#%ld 位姿优化更新后的值为: [%.2f, %.2f, %.2f], 可视MP数为:%ld\n",
               mCurrentFrame.id, mpReferenceKF->id, Twb.x, Twb.y, Twb.theta, mCurrentFrame.countObservations());
        printf("[Track][Warni] #%ld-#%ld 位姿优化情况: MP内点数/重投影误差为: %d/%.3f,\n",
               mCurrentFrame.id, mpReferenceKF->id, nCros, projError);

//        bool lost = detectIfLost(nCros, projError);
//        if (lost) {
//            printf("[Track][Warni] #%ld-#%ld 由MP优化位姿失败! MP内点数/重投影误差为: %d/%.3f, "
//                   "即将转为重定位!\n",
//                   mCurrentFrame.id, mpReferenceKF->id, nCros, projError);
//            return false;
//        }
    } /*else if (mCurrentFrame.id - mpReferenceKF->id >= 0.8 * nMaxFrames) {  //
    没有MP观测说明太仅或者太远
        return false;
    }*/

    return true;
}

// 把局部地图的点往当前帧上投
bool Track::trackLocalMap()
{
    const vector<PtrMapPoint> vpLocalMPs = mpMap->getLocalMPs();
    vector<int> vMatchedIdxMPs;  // 一定是新投上的点
    int nProj = mpORBmatcher->SearchByProjection(mCurrentFrame, vpLocalMPs, vMatchedIdxMPs, 20, 1);
    if (nProj > 0) {
        assert(vMatchedIdxMPs.size() == mCurrentFrame.N);
        for (int i = 0, iend = mCurrentFrame.N; i < iend; ++i) {
            if (vMatchedIdxMPs[i] < 0)
                continue;
            const PtrMapPoint& pMP = vpLocalMPs[vMatchedIdxMPs[i]];  // 新匹配上的MP
            mCurrentFrame.setObservation(pMP, i);  // TODO 要不要管 mvbOutlier?
        }
        doLocalBA(mCurrentFrame);
        return !detectIfLost();
    } else {
        return false;
    }
}

//! 根据RefKF和当前帧的里程计更新先验位姿和变换关系, odom是se2绝对位姿而非增量
//! 如果当前帧不是KF，则当前帧位姿由RefK叠加上里程计数据计算得到
//! 这里默认场景是工业级里程计，精度比较准
void Track::updateFramePoseFromRef()
{
    const Se2 Tb1b2 = mCurrentFrame.odom - mpReferenceKF->odom;
    const Mat Tc2c1 = Config::Tcb * Tb1b2.inv().toCvSE3() * Config::Tbc;

    // Tb1b2, 当前帧Body->参考帧Body, se2形式
    mCurrentFrame.setTrb(Tb1b2);
    // Tc2c1, 参考帧Camera->当前帧Camera, Tc2c1 = Tcb * Tb2b1 * Tbc
    mCurrentFrame.setTcr(Tc2c1);
    // Tc2w, World->当前帧Camera, Tc2w = Tc2c1 * Tc1w
    mCurrentFrame.setPose(Tc2c1 * mpReferenceKF->getPose());
    // Twb2, Body->当前帧World，se2形式，故相加， Twb2 = Twb1 * Tb1b2
    // mCurrentFrame.setPose(mpReferenceKF->getTwb() + Tb1b2);

    assert(mCurrentFrame.id - mLastFrame.id == 1);

    // preintegration 预积分
    Vector3d& meas = preSE2.meas;
    Se2 odok = mCurrentFrame.odom - mLastFrame.odom;
    Vector2d odork(odok.x, odok.y);
    Matrix2d Phi_ik = Rotation2Dd(meas[2]).toRotationMatrix();
    meas.head<2>() += Phi_ik * odork;
    meas[2] += odok.theta;
    // 协方差传递参考卡尔曼滤波
    Matrix3d Ak = Matrix3d::Identity();  // dodok_dxk-1
    Matrix3d Bk = Matrix3d::Identity();  // dodok_dxk
    Ak.block<2, 1>(0, 2) = -Phi_ik * Vector2d(-odork[1], odork[0]);
    Bk.block<2, 2>(0, 0) = -Phi_ik;
    Matrix3d& Sigmak = preSE2.cov;
    Matrix3d Sigma_vk = Matrix3d::Identity();
    Sigma_vk(0, 0) = Config::OdoNoiseX * Config::OdoNoiseX;
    Sigma_vk(1, 1) = Config::OdoNoiseY * Config::OdoNoiseY;
    Sigma_vk(2, 2) = Config::OdoNoiseTheta * Config::OdoNoiseTheta;
    Matrix3d Sigma_k_1 = Ak * Sigmak * Ak.transpose() + Bk * Sigma_vk * Bk.transpose();
    Sigmak = Sigma_k_1;
}

//! 根据仿射矩阵A剔除外点，利用了RANSAC算法
int Track::removeOutliers(const PtrKeyFrame& pKFRef, const Frame* pFCur, vector<int>& vKPMatchIdx12, Mat& A12)
{
    assert(pKFRef->N == vKPMatchIdx12.size());
    if (vKPMatchIdx12.empty())
        return 0;

    int nKPInliers = 0;

    vector<Point2f> vPtRef, vPtCur;
    vector<size_t> vIdxRef;
    vIdxRef.reserve(pKFRef->N);
    vPtRef.reserve(pKFRef->N);
    vPtCur.reserve(pFCur->N);
    for (size_t i = 0, iend = pKFRef->N; i < iend; ++i) {
        if (vKPMatchIdx12[i] < 0)
            continue;
        vIdxRef.push_back(i);
        vPtRef.push_back(pKFRef->mvKeyPoints[i].pt);
        vPtCur.push_back(pFCur->mvKeyPoints[vKPMatchIdx12[i]].pt);
    }
    if (vPtRef.size() == 0)
        return 0;

    vector<uchar> vInlier;
    A12 = estimateAffinePartial2D(vPtRef, vPtCur, vInlier, RANSAC, 2.0);

    assert(vIdxRef.size() == vInlier.size());
    for (size_t i = 0, iend = vInlier.size(); i < iend; ++i) {
        if (!vInlier[i]) {
            assert(vKPMatchIdx12[vIdxRef[i]] >= 0);
            vKPMatchIdx12[vIdxRef[i]] = -1;
        } else
            nKPInliers++;
    }

    return nKPInliers;
}

int Track::removeOutliers(const PtrKeyFrame& pKFRef, const Frame* pFCur, map<int, int>& mapMatch12, Mat& A12)
{
    if (mapMatch12.empty())
        return 0;

    map<int, int> mapMatchGood;
    vector<int> vIdxRef, vIdxCur;
    vector<Point2f> vPtRef, vPtCur;

    for (auto iter = mapMatch12.begin(); iter != mapMatch12.end(); iter++) {
        const int idxRef = iter->first;
        const int idxCur = iter->second;
        vIdxRef.push_back(idxRef);
        vIdxCur.push_back(idxCur);
        vPtRef.push_back(pKFRef->mvKeyPoints[idxRef].pt);
        vPtCur.push_back(pFCur->mvKeyPoints[idxCur].pt);
    }

    vector<uchar> vInlier;
    if (!A12.empty())  // 一般情况下更新A
        A12 = estimateAffinePartial2D(vPtRef, vPtCur, vInlier, RANSAC, 3.0);
    else  //NOTE 重定位情况中不能用A去外点, 因为KP匹配本来就是通过A来匹配的. 要用H来剔除外点
        A12 = findHomography(vPtRef, vPtCur, RANSAC, 3, vInlier);

    for (size_t i = 0, iend = vInlier.size(); i < iend; ++i) {
        if (vInlier[i] == true)
            mapMatchGood.emplace(vIdxRef[i], vIdxCur[i]);
    }

    mapMatch12 = mapMatchGood;

    return mapMatchGood.size();
}

Mat Track::computeA12(const PtrKeyFrame &pKFRef, const Frame *pFCur, std::vector<int> &vKPMatchIdx12)
{
    if (vKPMatchIdx12.empty())
        return Mat::eye(2, 3L, CV_64F);

    vector<Point2f> vPtRef, vPtCur;
    vector<size_t> vIdxRef;
    vIdxRef.reserve(pKFRef->N);
    vPtRef.reserve(pKFRef->N);
    vPtCur.reserve(pFCur->N);

    assert(pKFRef->N == vKPMatchIdx12.size());
    int m = 0;
    for (size_t i = 0, iend = pKFRef->N; i < iend; ++i) {
        if (vKPMatchIdx12[i] < 0)
            continue;
        vIdxRef.push_back(i);
        vPtRef.push_back(pKFRef->mvKeyPoints[i].pt);
        vPtCur.push_back(pFCur->mvKeyPoints[vKPMatchIdx12[i]].pt);
        m++;
    }

    if (vPtRef.size() == 0)
        return Mat::eye(2, 3L, CV_64F);

    Mat A(2 * m, 4, CV_64F);
    for (int i = 0; i < m; i++) {
        //const float u1 = vPtRef[i].x;
        //const float v1 = vPtRef[i].y;
        const float u2 = vPtCur[i].x;
        const float v2 = vPtCur[i].y;

        A.at<double>(2 * i, 0) = -v2;
        A.at<double>(2 * i, 1) = -u2;
        A.at<double>(2 * i, 2) = 0;
        A.at<double>(2 * i, 3) = -1;
        A.at<double>(2 * i + 1, 0) = u2;
        A.at<double>(2 * i + 1, 1) = -v2;
        A.at<double>(2 * i + 1, 2) = 1;
        A.at<double>(2 * i + 1, 3) = 0;
    }

    cv::Mat u, w, vt;

    cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    return vt.row(3).reshape(0, 2);  // v的最后一列
}

/**
 * @brief 关键函数, 三角化获取特征点的深度, 视差好的可直接生成MP, 视差不好的先被标记为候选MP
 * 更新needNewKF()函数需要考虑到的相关变量, 更新Frame和KF的MP观测情况.
 * @note    这里的变量对应关系不能出错, 很重要!
 * @date    2019.11.22
 */
void Track::doTriangulate(PtrKeyFrame& pKF)
{
    if (mvKPMatchIdx.empty() || mCurrentFrame.id - mpReferenceKF->id < nMinFrames)
        return;

    // 以下成员变量在这个函数中会被修改
    mnMPsCandidate = mMPCandidates.size();
    mvKPMatchIdxGood = mvKPMatchIdx;
    mnMPsInline = 0;
    mnMPsTracked = 0;
    mnMPsNewAdded = 0;
    mnKPMatchesBad = 0;
    int nKPMatchesGood = 0;
    int n11 = 0, n121 = 0, n21 = 0, n22 = 0, n31 = 0, n32 = 0, n33 = 0;
    int n2 = mnMPsCandidate;
    int nObsCur = mCurrentFrame.countObservations();
    int nObsRef = pKF->countObservations();
    assert(nObsCur == 0);  // 当前帧应该还没有观测!

    // 相机1和2的投影矩阵
    const Mat Tc1w = pKF->getPose();
    const Mat Tc2w = mCurrentFrame.getPose();
    const Mat Tcr = mCurrentFrame.getTcr();
    const Mat Tc1c2 = cvu::inv(Tcr);
    const Point3f Ocam1 = Point3f(0.f, 0.f, 0.f);
    const Point3f Ocam2 = Point3f(Tc1c2.rowRange(0, 3).col(3));
    const cv::Mat Proj1 = Config::PrjMtrxEye;  // P1 = K * cv::Mat::eye(3, 4, CV_32FC1)
    const cv::Mat Proj2 = Config::Kcam * Tcr.rowRange(0, 3);  // P2 = K * Tc2c1(3*4)

    /* 遍历参考帧的KP, 根据'mvKPMatchIdx'对有匹配点对的KP进行处理, 如果:
     * 1. 参考帧KP已经有对应的MP观测:
     *  - 1.1 对于视差好的MP, 直接给当前帧关联一下MP观测;
     *  - 1.2 对于视差不好的MP, 再一次三角化更新其坐标值. 如果更新后:
     *      - 1.2.1 深度符合(理应保持)且视差好, 更新MP的属性;
     *      - 1.2.2 深度符合但视差没有被更新为好, 更新MP坐标
     *      - 1.2.3 深度被更新为不符合(出现概率不大), 则不处理.
     * 2. 参考帧KP已经有对应的MP候选: (说明深度符合但视差还不太好)
     *  - 三角化更新一下候选MP的坐标, 如果:
     *      - 2.1 深度符合(理应保持)且视差好, 生成新的MP, 为KF和F添加MP观测, 为MP添加KF观测,
     * 删除候选;
     *      - 2.2 深度符合(理应保持)但视差没有被更新为好, 更新候选MP的坐标;
     *      - 2.3 深度被更新为不符合(出现概率不大), 则不处理.
     * 3. 参考帧KP一无所有:
     *  - 三角化, 如果:
     *      - 3.1 深度符合且视差好, 生成新的MP, 为KF和F添加MP观测, 为MP添加KF观测;
     *      - 3.2 深度符合但视差不好, 将其添加候选MP;
     *      - 3.3 深度不符合, 则丢弃此匹配点对.
     *
     * 几个变量的关系:
     *  - mnMPsTracked  = 1.1 + 1.2.1
     *  - mnMPsNewAdded = 2.1 + 3.1
     *  - mnKPMatchesBad  = 3.3
     *  - mnMPsInline = mnMPsTracked + mnMPsNewAdded
     *  - mnCandidateMPs = mnCandidateMPs - 2.1 + 3.2
     *
     * NOTE 对视差不好的MP或者MP候选, 说明它是参考帧和其他帧生成的, 并不是和自己生成的,
     * 如果把视差不好的MP关联给当前帧, 会影响当前帧的投影精度.
     * 能成为MP或MP候选的点说明其深度都是符合预期的.
     * 最后MP候选将在LocalMap线程中生成视差不好的MP, 他们在添加KF观测后有机会更新视差.
     */
    for (size_t i = 0, iend = pKF->N; i < iend; ++i) {
        if (mvKPMatchIdx[i] < 0)
            continue;

        nKPMatchesGood++;
        PtrMapPoint pObservedMP = nullptr;  // 对于的MP观测
        const bool bObserved = pKF->hasObservationByIndex(i);  // 是否有对应MP观测
        if (bObserved) {
            pObservedMP = pKF->getObservation(i);
            if (pObservedMP->isGoodPrl()) {  // 情况1.1
                mCurrentFrame.setObservation(pObservedMP, mvKPMatchIdx[i]);
                mnMPsTracked++;
                mnMPsInline++;
                n11++;
                continue;
            }
        }

        // 如果参考帧KP没有对应的MP, 或者对应MP视差不好, 则为此匹配对KP三角化计算深度(相对参考帧的坐标)
        // 由于两个投影矩阵是两KF之间的相对投影, 故三角化得到的坐标是相对参考帧的坐标, 即Pc1
        const Point2f pt1 = pKF->mvKeyPoints[i].pt;
        const Point2f pt2 = mCurrentFrame.mvKeyPoints[mvKPMatchIdx[i]].pt;
        const Point3f Pc1 = cvu::triangulate(pt1, pt2, Proj1, Proj2);
        const Point3f Pw = cvu::se3map(cvu::inv(Tc1w), Pc1);

        // Pc2用Tcr计算出来的是预测值, 故这里用Pc1的深度判断即可
        const bool bAccepDepth = Config::acceptDepth(Pc1.z);  // 深度是否符合
        const bool bCandidated = mMPCandidates.count(i);  // 是否有对应的MP候选
        assert(!(bObserved && bCandidated));  // 不能即有MP观测又是MP候选

        if (bAccepDepth) {  // 如果深度计算符合预期
            const bool bGoodPrl = cvu::checkParallax(Ocam1, Ocam2, Pc1, 2);  // 视差是否良好
            if (bGoodPrl) {  // 如果视差好
                Eigen::Matrix3d xyzinfo1, xyzinfo2;
                calcSE3toXYZInfo(Pc1, Tc1w, Tc2w, xyzinfo1, xyzinfo2);
                if (bObserved) {  // 情况1.2.1
                    pObservedMP->setPos(Pw);
                    pObservedMP->setGoodPrl(true);
                    pKF->setObsAndInfo(pObservedMP, i, xyzinfo1);
                    mCurrentFrame.setObservation(pObservedMP, mvKPMatchIdx[i]);
                    mnMPsTracked++;
                    mnMPsInline++;
                    n121++;
                } else {  // 情况2.1和3.1
                    PtrMapPoint pNewMP = make_shared<MapPoint>(Pw, true);
                    pKF->setObsAndInfo(pNewMP, i, xyzinfo1);
                    pNewMP->addObservation(pKF, i);  // MP只添加KF的观测
                    mCurrentFrame.setObservation(pNewMP, mvKPMatchIdx[i]);
                    mpMap->insertMP(pNewMP);
                    mnMPsNewAdded++;
                    mnMPsInline++;
                    if (bCandidated) {  // 对于情况2.1, 候选转正后还要删除候选名单
                        mMPCandidates.erase(i);
                        mnMPsCandidate--;
                        n21++;
                    } else {
                        n31++;
                    }
                }
            } else {  // 如果视差不好
                if (bObserved) {  // 情况1.2.2, 更新Pw
                    pObservedMP->setPos(Pw);
                } else if (bCandidated) {  // 情况2.2, 更新候选MP
                    assert(!pKF->hasObservationByIndex(i));
                    mMPCandidates[i].Pc1 = Pc1;
                    mMPCandidates[i].id2 = mCurrentFrame.id;
                    mMPCandidates[i].kpIdx2 = mvKPMatchIdx[i];
                    mMPCandidates[i].Tc2w = mCurrentFrame.getPose().clone();
                    n22++;
                } else {  // 情况3.2
                    assert(!pKF->hasObservationByIndex(i));
                    MPCandidate MPcan(pKF, Pc1, mCurrentFrame.id, mvKPMatchIdx[i], Tc2w);
                    mMPCandidates.emplace(i, MPcan);
                    mnMPsCandidate++;
                    n32++;
                }
            }
        } else {  // 如果深度计算不符合预期
            if (!bObserved && !bCandidated) {  // 情况3.3
                n33++;
                mnKPMatchesBad++;
                nKPMatchesGood--;
                mvKPMatchIdxGood[i] = -1;
            }
            // 情况1.2.3和2.3不处理
        }
    }
    printf("[Track][Info ] #%ld-#%ld KP匹配结果: KP内点数/KP匹配数: %d/%d, "
           "MP总观测数(Ref)/关联数/视差好的/更新到好的: %d/%d/%d/%d\n",
           mCurrentFrame.id, pKF->id, mnKPsInline, mnKPMatches, nObsRef, mnMPsTracked, n11, n121);
    printf("[Track][Info ] #%ld-#%ld 三角化结果: 候选MP原总数/转正数/更新数/增加数/现总数: "
           "%d/%d/%d/%d/%d, "
           "三角化新增MP数/新增候选数/剔除匹配数: %d/%d/%d, 新生成MP数: %d\n",
           mCurrentFrame.id, pKF->id, n2, n21, n22, n32, mnMPsCandidate, n31, n32, n33, mnMPsNewAdded);

    // KF判定依据变量更新 FIXME
    mCurrRatioGoodDepth = mnKPsInline > 0 ? mnMPsCandidate / mnKPsInline : 0;
    mCurrRatioGoodParl = mnMPsCandidate > 0 ? mnMPsNewAdded / mnMPsCandidate : 0;

    assert(n11 + n121 == mnMPsTracked);
    assert(n21 + n31 == mnMPsNewAdded);
    assert(n33 == mnKPMatchesBad);
    assert(nKPMatchesGood + mnKPMatchesBad == mnKPsInline);
    assert(mnMPsTracked + mnMPsNewAdded == mnMPsInline);
    assert((n2 - n21 + n32 == mnMPsCandidate) && (mnMPsCandidate == mMPCandidates.size()));
    assert(nObsCur + mnMPsTracked + mnMPsNewAdded == mCurrentFrame.countObservations());
    assert(nObsRef + mnMPsNewAdded == pKF->countObservations());
}

void Track::resetLocalTrack()
{
    // 正常状态进行关键帧判断
    if (mState == cvu::OK) {
        if (needNewKF()) {
            PtrKeyFrame pNewKF = make_shared<KeyFrame>(mCurrentFrame);

            // 重定位成功将当前帧变成KF
            if (mState == cvu::OK && mLastState == cvu::LOST) {
                assert(mbRelocalized == true);
                pNewKF->addCovisibleKF(mpLoopKF);
                mpLoopKF->addCovisibleKF(pNewKF);
                printf("[Track][Info ] #%ld 成为了新的KF(#%ld), 因为刚刚重定位成功!\n", pNewKF->id,
                       pNewKF->mIdKF);
                mbRelocalized = false;

                //! FIXME 重定位成功(回环验证通过)构建特征图和约束
                //! NOTE mKPMatchesLoop 必须是有MP的匹配!
                //                for (auto& m : mKPMatchesLoop) {
                //                    if (!mCurrentFrame.hasObservationByIndex(m.first) ||
                //                        !mpReferenceKF->hasObservationByIndex(m.second))
                //                        mKPMatchesLoop.erase(m.first);
                //                }
                //                SE3Constraint Se3_Curr_Loop;
                //                bool bFtrCnstrErr = GlobalMapper::createFeatEdge(pNewKF, mpLoopKF,
                //                mKPMatchesLoop, Se3_Curr_Loop);
                //                if (!bFtrCnstrErr) {
                //                    pNewKF->addFtrMeasureFrom(mpLoopKF, Se3_Curr_Loop.measure,
                //                    Se3_Curr_Loop.info);
                //                    mpLoopKF->addFtrMeasureTo(pNewKF, Se3_Curr_Loop.measure,
                //                    Se3_Curr_Loop.info);
                //                }
            }

            mpLocalMapper->addNewKF(pNewKF, mMPCandidates);  // 新的KF观测和共视关系在LocalMap里更新

            if (mbNeedVisualization)
                copyForPub();  // Visualization

            mpReferenceKF = pNewKF;
            mAffineMatrix = Mat::eye(2, 3, CV_64FC1);
            mMPCandidates.clear();  // 加了新的KF后才清空. 和普通帧的匹配也会生成MP
            mnMPsCandidate = 0;
            mLastRatioGoodDepth = 0.;
            mLastRatioGoodParl = 0.;
        } else {
            mLastRatioGoodDepth = mCurrRatioGoodDepth;
            mLastRatioGoodParl = mCurrRatioGoodParl;
        }
        mLastFrame = mCurrentFrame;
        mnKPMatches = 0;
        mnKPsInline = 0;
        mnKPMatchesBad = 0;
        mnMPsTracked = 0;
        mnMPsNewAdded = 0;
        mnMPsInline = 0;
        preSE2.meas.setZero();
        preSE2.cov.setZero();
        return;
    }

    // 当前帧刚刚丢失要确保上一帧是最新的KF
    assert(mState == cvu::LOST);
    if (mState == cvu::LOST && mLastState == cvu::OK && !mLastFrame.isNull() &&
        mLastFrame.id != mpReferenceKF->id) {
        printf("[Track][Info ] #%ld-#%ld 上一帧(#%ld)成为了新的KF(#%ld)! 因为当前帧刚刚丢失!\n",
               mCurrentFrame.id, mpReferenceKF->id, mLastFrame.id, KeyFrame::mNextIdKF);

        PtrKeyFrame pNewKF = make_shared<KeyFrame>(mLastFrame);
        mpLocalMapper->addNewKF(pNewKF, mMPCandidates);
        mpReferenceKF = pNewKF;
    }

    // 处于丢失的状态则直接交换前后帧数据
    //    const Point2f rotationCenter(160.5827 - 0.01525, 117.7329 - 3.6984);
    //    const double angle = normalizeAngle(mCurrentFrame.odom.theta - mpReferenceKF->odom.theta);
    //    mAffineMatrix = getRotationMatrix2D(rotationCenter, angle * 180 / CV_PI, 1.);
    // mAffineMatrix = Mat::eye(2, 3, CV_64FC1);
    mMPCandidates.clear();
    mLastFrame = mCurrentFrame;
    mnKPMatches = 0;
    mnKPsInline = 0;
    mnKPMatchesBad = 0;
    mnMPsTracked = 0;
    mnMPsNewAdded = 0;
    mnMPsInline = 0;
    mLastRatioGoodDepth = 0.f;
    mLastRatioGoodParl = 0.f;
    preSE2.meas.setZero();
    preSE2.cov.setZero();
}

/**
 * @brief Track::needNewKF
 * c2 - KP匹配内点数(mnInliers): 多(间隔近, false), 少(间隔远, true)
 * c3 -
 * 其他参数基本都依赖于KP匹配内点数(mnInliers), 且在间隔较近或较远时表现相似, 不适合做判定标准
 */
bool Track::needNewKF()
{
    // 刚重定位成功需要建立新的KF
    if (mState == cvu::OK && mLastState == cvu::LOST) {
        assert(mbRelocalized == true);
        printf("[Track][Info ] #%ld 刚重定位成功! 需要加入新的KF! \n", mCurrentFrame.id);
        return true;
    }

    // int nOldObs = mpReferenceKF->countObservations();
    int deltaFrames = static_cast<int>(mCurrentFrame.id - mpReferenceKF->id);

    //! TODO  目前先按固定间隔增加KF, 以统计跟跟踪情况, 方便设置阈值!(先不开启重定位) 待改证.
    //    const int fixDelta = 10;
    //    if (deltaFrames < fixDelta)
    //        return false;
    //    else
    //        return true;

    // 必要条件
    bool c0 = deltaFrames >= nMinFrames;  // 下限

    // 充分条件
    bool c1 = deltaFrames >= nMaxFrames;  // 上限
    bool c2 = mnKPMatches < 20 || mnKPsInline < 15;  // 和参考帧匹配的内点数太少.
    bool c3 = mnMPsCandidate > 100;  // 候选MP多且新增MP少(远)已经新增的MP数够多
    bool c4 = (mCurrRatioGoodDepth < mLastRatioGoodDepth) && (mCurrRatioGoodParl < mLastRatioGoodParl);
    bool bNeedKFByVo = c0 && (c1 || c2 || c3 || c4);

    bool bNeedKFByOdo = false;
    bool c5 = false, c6 = false;
    if (mbUseOdometry) {  //! 旋转平移的条件可以和跟踪条件结合考虑
        Se2 dOdo = mCurrentFrame.odom - mpReferenceKF->odom;
        c5 = static_cast<double>(abs(dOdo.theta)) >= mMaxAngle;  // 旋转量超过50°
        cv::Mat cTc = Config::Tcb * dOdo.toCvSE3() * Config::Tbc;
        cv::Mat xy = cTc.rowRange(0, 2).col(3);
        c6 = cv::norm(xy) >= mMaxDistance;  // 相机的平移量足够大
        bNeedKFByOdo = c5 || c6;  // 相机移动取决于深度上限,考虑了不同深度下视野的不同
    }

    bool bNeedNewKF = bNeedKFByVo || bNeedKFByOdo;
    if (!bNeedNewKF) {
        return false;
    } else if (mpLocalMapper->checkIfAcceptNewKF()) {
        printf("[Track][Info ] #%ld-#%ld 成为了新的KF(#%ld), 其KF条件满足情况: "
               "下限(%d)/上限(%d)/内点(%d)/潜在或已增(%d)/追踪(%d)/旋转(%d)/平移(%d)\n",
               mCurrentFrame.id, mpReferenceKF->id, KeyFrame::mNextIdKF, c0, c1, c2, c3, c4, c5, c6);
        return true;
    }

    // 强制更新KF的条件
    if (bNeedKFByOdo && (c2 || c3 || c4)) {
        printf("[Track][Warni] #%ld-#%ld 强制添加KF(#%ld), 其KF条件满足情况: "
               "下限(%d)/上限(%d)/内点(%d)/潜在或已增(%d)/追踪(%d)/旋转(%d)/平移(%d)\n",
               mCurrentFrame.id, mpReferenceKF->id, KeyFrame::mNextIdKF, c0, c1, c2, c3, c4, c5, c6);
        mpLocalMapper->setAbortBA();  // 如果必须要加入关键帧,则终止LocalMap优化
        mpLocalMapper->setAcceptNewKF(true);
        return true;
    } else {
        printf("[Track][Warni] #%ld-#%ld 应该成为了新的KF, 但局部地图繁忙! 其KF条件满足情况:"
               "下限(%d)/上限(%d)/内点(%d)/潜在或已增(%d)/追踪(%d)/旋转(%d)/平移(%d)\n",
               mCurrentFrame.id, mpReferenceKF->id, c0, c1, c2, c3, c4, c5, c6);
        return false;
    }
}

void Track::addNewKF()
{
    PtrKeyFrame pNewKF = make_shared<KeyFrame>(mCurrentFrame);
    // 新的KF观测和共视关系在LocalMap里更新
    mpLocalMapper->addNewKF(pNewKF, mMPCandidates);
    printf("[Track][Info ] #%ld-#%ld 成为了新的KF!\n", pNewKF->id, mpReferenceKF->id);

    mpReferenceKF = pNewKF;
    mAffineMatrix = Mat::eye(2, 3, CV_64FC1);
    mLastRatioGoodDepth = 0.;
    mLastRatioGoodParl = 0.;
    mMPCandidates.clear();  // 加了新的KF后才清空. 和普通帧的匹配也会生成MP
}

void Track::copyForPub()
{
    locker lock1(mMutexForPub);
    locker lock2(mpMapPublisher->mMutexPub);

    mpMapPublisher->mnCurrentFrameID = mCurrentFrame.id;
    mpMapPublisher->mCurrentFramePose = mCurrentFrame.getPose().clone();
    mpMapPublisher->mCurrentImage = mCurrentFrame.mImage.clone();
    mpMapPublisher->mvCurrentKPs = mCurrentFrame.mvKeyPoints;
    mpMapPublisher->mvMatchIdx = mvKPMatchIdx;
    mpMapPublisher->mvMatchIdxGood = mvKPMatchIdxGood;
    mpMapPublisher->mAffineMatrix = mAffineMatrix.clone();

    char strMatches[64];
    if (mLastState == cvu::OK) {  // 正常情况和刚丢失情况
        mpMapPublisher->mReferenceImage = mpReferenceKF->mImage.clone();

        mpMapPublisher->mvReferenceKPs = mpReferenceKF->mvKeyPoints;
        std::snprintf(strMatches, 64, "F: %ld-%ld, KF: %ld(%ld)-%ld, D: %ld, M: %d/%d/%d",
                      mCurrentFrame.id, mCurrentFrame.countObservations(), mpReferenceKF->id,
                      mpReferenceKF->mIdKF, mpReferenceKF->countObservations(),
                      mCurrentFrame.id - mpReferenceKF->id, mnMPsInline, mnKPsInline - mnKPMatchesBad,
                      mnKPMatches);
    } else {  // 丢失情况和刚完成重定位
        if (mpLoopKF) {
            mpMapPublisher->mReferenceImage = mpLoopKF->mImage.clone();
            mpMapPublisher->mvReferenceKPs = mpLoopKF->mvKeyPoints;
            std::snprintf(strMatches, 64, "F: %ld-%ld, LoopKF: %ld(%ld)-%ld, Score: %.3f, M: %d/%d/%d",
                          mCurrentFrame.id, mCurrentFrame.countObservations(), mpLoopKF->id,
                          mpLoopKF->mIdKF, mpLoopKF->countObservations(), mLoopScore, mnMPsInline,
                          mnKPsInline - mnKPMatchesBad, mnKPMatches);
        } else {
            mpMapPublisher->mReferenceImage = Mat::zeros(mCurrentFrame.mImage.size(), CV_8UC1);
            mpMapPublisher->mvReferenceKPs.clear();
            std::snprintf(strMatches, 64, "F: %ld-%ld, Have no LoopKF!", mCurrentFrame.id,
                          mCurrentFrame.countObservations());
        }
    }

    mpMapPublisher->mImageText = string(strMatches);
    mpMapPublisher->mbUpdated = true;
}


void Track::requestFinish()
{
    locker lock(mMutexFinish);
    mbFinishRequested = true;
}

bool Track::checkFinish()
{
    locker lock(mMutexFinish);
    return mbFinishRequested;
}

bool Track::isFinished()
{
    locker lock(mMutexFinish);
    return mbFinished;
}

void Track::setFinish()
{
    locker lock(mMutexFinish);
    mbFinished = true;
}

void Track::checkReady()
{
    assert(mpMap != nullptr);
    assert(mpLocalMapper != nullptr);
    assert(mpGlobalMapper != nullptr);
    assert(mpSensors != nullptr);
    assert(mpMapPublisher != nullptr);
}

// 根据优化内点数和重投影误差来判断是否丢失定位
bool Track::detectIfLost(int nCros, double projError)
{
    const int th = mCurrentFrame.countObservations() * 0.6;
    if (nCros < th)
        return true;
    if (projError > 1000.)  // TODO 阈值待调整
        return true;

    return false;
}

// 暂时没用
bool Track::detectIfLost()
{
    if (mCurrentFrame.id == mpReferenceKF->id)
        return false;

    //    if (mCurrentFrame.id > 190 && mCurrentFrame.id < 200)
    //        return true;

    const int df = mCurrentFrame.id - mpReferenceKF->id;
    const Se2 dOdo1 = mCurrentFrame.odom - mLastFrame.odom;
    const Se2 dOdo2 = mCurrentFrame.odom - mpReferenceKF->odom;
    const Se2 dVo = mCurrentFrame.getTwb() - mpReferenceKF->getTwb();

    const float th_angle = Config::MaxAngularSpeed * 2.f / Config::FPS;
    const float th_dist = Config::MaxLinearSpeed * 2.f / Config::FPS;

    // 分析输入数据是否符合标准
    //    if (abs(normalizeAngle(dOdo1.theta)) > th_angle) {
    //        fprintf(stderr, "[Track][Warni] #%ld-#%ld 因Odo突变角度过大而丢失!\n",
    //        mCurrentFrame.id,
    //                mpReferenceKF->id);
    //        cerr << "[Track][Warni] 因Odo突变角度过大而丢失! Last odom: " << mLastFrame.odom
    //             << ", Current odom: " << mCurrentFrame.odom << endl;
    //        return true;
    //    }
    //    if (cv::norm(Point2f(dOdo1.x, dOdo1.y)) > th_dist) {
    //        fprintf(stderr, "[Track][Warni] #%ld-#%ld 因Odo突变距离过大而丢失!\n",
    //        mCurrentFrame.id,
    //                mpReferenceKF->id);
    //        cerr << "[Track][Warni] 因Odo突变距离过大而丢失! Last odom: " << mLastFrame.odom
    //             << ", Current odom: " << mCurrentFrame.odom << endl;
    //        return true;
    //    }

    // 分析计算结果的合理性
    if (abs(normalizeAngle(dVo.theta)) > th_angle * df) {
        fprintf(stderr, "[Track][Warni] #%ld-#%ld 因VO角度过大而丢失!\n", mCurrentFrame.id,
                mpReferenceKF->id);
        return true;
    }
    if (cv::norm(Point2f(dVo.x, dVo.y)) > th_dist * df) {
        fprintf(stderr, "[Track][Warni] #%ld-#%ld 因VO距离过大而丢失!\n", mCurrentFrame.id,
                mpReferenceKF->id);
        return true;
    }
    if (abs(normalizeAngle(dVo.theta - dOdo2.theta)) > mMaxAngle * 0.5) {
        fprintf(stderr, "[Track][Warni] #%ld-#%ld 因VO角度和Odo相差过大而丢失!\n", mCurrentFrame.id,
                mpReferenceKF->id);
        return true;
    }
    if (cv::norm(Point2f(dVo.x, dVo.y)) - cv::norm(Point2f(dOdo2.x, dOdo2.y)) > mMaxDistance * 0.5) {
        fprintf(stderr, "[Track][Warni] #%ld-#%ld 因VO距离和Odo相差过大而丢失!\n", mCurrentFrame.id,
                mpReferenceKF->id);
        return true;
    }
    return false;
}


/* @breif   通过相对变换关系计算等距变换
 * dOdo = RefFrame.odom - CurFrame.odom
 */
Mat Track::getAffineMatrix(const Se2& dOdo)
{
    Point2f rotationCenter;
    rotationCenter.x = Config::cx - Config::Tbc.at<float>(1, 3) * Config::fx / 3000.f;
    rotationCenter.y = Config::cy - Config::Tbc.at<float>(0, 3) * Config::fy / 3000.f;
    Mat R0 = getRotationMatrix2D(rotationCenter, dOdo.theta * 180.f / CV_PI, 1);

    Mat Tc1c2 = Config::Tcb * dOdo.inv().toCvSE3() * Config::Tbc;  // 4x4
    Mat Rc1c2 = Tc1c2.rowRange(0, 3).colRange(0, 3).clone();  // 3x3
    Mat tc1c2 = Tc1c2.rowRange(0, 3).col(3).clone();  // 3x1
    Mat R = Config::Kcam * Rc1c2 * (Config::Kcam).inv();  // 3x3 相当于A, 但少了部分平移信息
    Mat t = Config::Kcam * tc1c2 / 3000.f;  // 3x1 假设MP平均深度3m

    Mat A;
    R.rowRange(0, 2).convertTo(A, CV_64FC1);
    R0.colRange(0, 2).copyTo(A.colRange(0, 2));  // 去掉尺度变换
    A.at<double>(0, 2) += (double)t.at<float>(0, 0);  // 加上平移对图像造成的影响
    A.at<double>(1, 2) += (double)t.at<float>(1, 0);

    return A.clone();
}

//! 重定位. BoW搜索匹配，通过重投影误差看是否成功
bool Track::doRelocalization()
{
    if (mCurrentFrame.isNull())  // 图像挂掉的时候Frame会变成null.
        return false;
    if (mCurrentFrame.mTimeStamp < mLastFrame.mTimeStamp)
        return false;

    mnMPsNewAdded = mnMPsCandidate = mnKPMatchesBad = 0;
    mnKPsInline = mnMPsInline = mnMPsTracked = 0;

    updateFramePoseFromRef();

    bool bDetected = detectLoopClose();
    bool bVerified = false;
    if (bDetected) {
        bVerified = verifyLoopClose();
        if (bVerified) {
            // Get Local Map and do local BA
            // trackLocalMap();  // FIEME

            // setPose
            // mCurrentFrame.setPose(mpLoopKF->getTwb());

            mbRelocalized = true;
            return true;
        } else {
            return false;
        }
    } else {
        return false;
    }
}

bool Track::detectLoopClose()
{
    bool bDetected = false;

    mCurrentFrame.computeBoW(mpGlobalMapper->mpORBVoc);
    const DBoW2::BowVector& BowVecCurr = mCurrentFrame.mBowVec;
    const double minScoreBest = 0.015 /*Config::MinScoreBest*/;  // 0.02

    vector<PtrKeyFrame> vpKFsAll = mpMap->getAllKFs();
    for_each(vpKFsAll.begin(), vpKFsAll.end(), [&](PtrKeyFrame& pKF) {
        if (!pKF->mbBowVecExist)
            pKF->computeBoW(mpGlobalMapper->mpORBVoc);
    });

    double scoreBest = 0;
    PtrKeyFrame pKFBest = nullptr;
    for (int i = 0, iend = vpKFsAll.size(); i < iend; ++i) {
        const PtrKeyFrame& pKFi = vpKFsAll[i];
        const DBoW2::BowVector& BowVec = pKFi->mBowVec;

        const double score = mpGlobalMapper->mpORBVoc->score(BowVecCurr, BowVec);
        if (score > scoreBest) {
            scoreBest = score;
            pKFBest = pKFi;
        }
    }

    if (pKFBest != nullptr) {
        mLoopScore = scoreBest;
        if (mLoopScore >= minScoreBest) {
            mpLoopKF = pKFBest;
            bDetected = true;
            fprintf(stderr, "[Track][Info ] #%ld-KF#%ld(Loop) 重定位-检测到回环#%ld! score = %.3f >= "
                            "%.3f, 等待验证!\n",
                    mCurrentFrame.id, pKFBest->mIdKF, pKFBest->id, mLoopScore, minScoreBest);
        } else {
            mpLoopKF.reset();
            fprintf(stderr,
                    "[Track][Warni] #%ld 重定位-回环检测#%ld(KF#%ld)失败! score = %.3f < %.3f\n",
                    mCurrentFrame.id, pKFBest->id, pKFBest->mIdKF, mLoopScore, minScoreBest);
        }
    }

    return bDetected;
}

bool Track::verifyLoopClose()
{
    assert(mpLoopKF != nullptr && !mCurrentFrame.isNull());

    const int nMinKPMatch = Config::MinKPMatchNum * 0.6;  // TODO. *0.6要去掉
    // const float minRatioMPMatch = Config::MinMPMatchRatio;  // TODO. 0.05.

    mKPMatchesLoop.clear();
    assert(mnKPMatches == 0);
    assert(mnKPsInline == 0);
    assert(mnMPsInline == 0);
    assert(mnMPsTracked == 0);
    assert(mnMPsNewAdded == 0);

    // BoW匹配
    bool bIfMatchMPOnly = false;  // 要看总体匹配点数多不多
    mnKPMatches = mpORBmatcher->SearchByBoW(static_cast<Frame*>(mpLoopKF.get()), &mCurrentFrame, mKPMatchesLoop, bIfMatchMPOnly);
    if (mnKPMatches < nMinKPMatch) {
        fprintf(stderr, "[Track][Warni] #%ld-KF#%ld(Loop) 重定位-回环验证失败! 与回环帧的KP匹配数过少: %d < %d\n",
                mCurrentFrame.id, mpLoopKF->mIdKF, mnKPMatches, nMinKPMatch);
        return false;
    }

    // 匹配点数足够则剔除外点
    Se2 dOdo = mCurrentFrame.odom - mpLoopKF->odom;
    mAffineMatrix = getAffineMatrix(dOdo);  // 计算先验A
    mnKPMatches = mpORBmatcher->MatchByWindowWarp(*mpLoopKF, mCurrentFrame, mAffineMatrix, mvKPMatchIdx, 20);
    if (mnKPMatches < nMinKPMatch) {
        printf("[Track][Warni] #%ld-KF#%ld(Loop) 重定位-回环验证失败! 与回环帧的Warp KP匹配数过少: %d < %d\n",
               mCurrentFrame.id, mpLoopKF->mIdKF, mnKPMatches, nMinKPMatch);
        return false;
    }

//    Mat H; // 重定位这里要用H去除外点, 不能用先验A
    mnKPsInline = removeOutliers(mpLoopKF, &mCurrentFrame, mvKPMatchIdx, mAffineMatrix);
    if (mnKPsInline < 10) {
        printf("[Track][Warni] #%ld-KF#%ld(Loop) 重定位-回环验证失败! 与回环帧的Warp KP匹配内点数少于10(%d/%d)!\n",
               mCurrentFrame.id, mpLoopKF->mIdKF, mnKPsInline, mnKPMatches);
        return false;
    }
    // H.rowRange(0, 2).copyTo(mAffineMatrix);

    assert(mMPCandidates.empty());
    doTriangulate(mpLoopKF);  //  对MP视差好的会关联
    const int nObs = mCurrentFrame.countObservations();
    const float ratio = nObs == 0 ? 0 : mnMPsInline * 1.f / nObs;
    fprintf(stderr, "[Track][Info ] #%ld-KF#%ld(Loop) 重定位-回环验证, 和回环帧的MP关联/KP内点/KP匹配数为: %d/%d/%d, "
                    "当前帧观测数/MP匹配率为: %d/%.2f (TODO)\n",
            mCurrentFrame.id, mpLoopKF->mIdKF, mnMPsInline, mnKPsInline, mnKPMatches, nObs, ratio);

    //! FIXME 优化位姿, 并计算重投影误差
    int nCorres = 0;
    double projError = 1000.;
    Se2 Twb = mCurrentFrame.getTwb();
    printf("[Track][Warni] #%ld 位姿优化更新前的值为: [%.2f, %.2f, %.2f], 可视MP数为:%ld\n",
           mCurrentFrame.id, Twb.x, Twb.y, Twb.theta, mCurrentFrame.countObservations());
    poseOptimization(&mCurrentFrame, nCorres, projError);  // 确保已经setViewMP()
    Twb = mCurrentFrame.getTwb();
    printf("[Track][Warni] #%ld 位姿优化更新后的值为: [%.2f, %.2f, %.2f], 重投影误差为:%.2f\n",
           mCurrentFrame.id, Twb.x, Twb.y, Twb.theta, projError);

    const bool bProjLost = detectIfLost(nCorres, projError);
    if (bProjLost) {
        fprintf(stderr,
                "[Track][Warni] #%ld-KF#%ld(Loop) 重定位-回环验证失败! 优化时MP优化内点数/总数/重投影误差为: %d/%d/%.2f\n",
                mCurrentFrame.id, mpLoopKF->mIdKF, nCorres, nObs, projError);
        return false;
    } else {
        mnKPsInline = mnMPsInline;
        mvKPMatchIdx = mvKPMatchIdxGood;
        fprintf(stderr, "[Track][Info ] #%ld-KF#%ld(Loop) 重定位-回环验证成功! MP内点数/总数/重投影误差为: %d/%d/%.2f\n",
                mCurrentFrame.id, mpLoopKF->mIdKF, nCorres, nObs, projError);
        return true;
    }
}


/**
 * @brief 根据MP观测优化当前帧位姿. 二元边
 */
void Track::doLocalBA(Frame& frame)
{
    WorkTimer timer;

    SlamOptimizer optimizer;
    SlamLinearSolverCholmod* linearSolver = new SlamLinearSolverCholmod();
    SlamBlockSolver* blockSolver = new SlamBlockSolver(linearSolver);
    SlamAlgorithmLM* solver = new SlamAlgorithmLM(blockSolver);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(Config::LocalVerbose);

    int camParaId = 0;
    addCamPara(optimizer, Config::Kcam, camParaId);

    // 当前帧位姿节点(待优化变量)
    addVertexSE3Expmap(optimizer, toSE3Quat(frame.getPose()), 0, false);
    // 设置平面运动约束(边)
    addEdgeSE3ExpmapPlaneConstraint(optimizer, toSE3Quat(frame.getPose()), 0, Config::Tbc);
    int vertexId = 1;

    // Add MPs in local map as fixed
    const float delta = Config::ThHuber;
    vector<PtrMapPoint> allViewMPs = frame.getObservations(false, false);
    for (size_t ftrIdx = 0, N = frame.N; ftrIdx < N; ++ftrIdx) {
        if (allViewMPs[ftrIdx] == nullptr)
            continue;

        const PtrMapPoint& pMP = allViewMPs[ftrIdx];

        // 添加MP节点
        bool marginal = false;
        bool fixed = pMP->isGoodPrl();  // 视差好的点才固定
        addVertexSBAXYZ(optimizer, toVector3d(pMP->getPos()), vertexId, marginal, fixed);

        // 添加边
        int octave = frame.mvKeyPoints[ftrIdx].octave;
        const float invSigma2 = frame.mvInvLevelSigma2[octave];
        Eigen::Vector2d uv = toVector2d(frame.mvKeyPoints[ftrIdx].pt);
        Eigen::Matrix2d info = Eigen::Matrix2d::Identity() * invSigma2;

        g2o::EdgeProjectXYZ2UV* ei = new g2o::EdgeProjectXYZ2UV();
        ei->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(vertexId)));
        ei->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        ei->setMeasurement(uv);
        ei->setParameterId(0, camParaId);
        ei->setInformation(info);
        ei->setLevel(0);
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        rk->setDelta(delta);
        ei->setRobustKernel(rk);
        optimizer.addEdge(ei);

        vertexId++;
    }

    optimizer.initializeOptimization(0);
    optimizer.verifyInformationMatrices(true);
    optimizer.optimize(Config::LocalIterNum);

    Mat Tcw = toCvMat(estimateVertexSE3Expmap(optimizer, 0));
    frame.setPose(Tcw);  // 更新Tcw

    Se2 Twb = frame.getTwb();
    fprintf(stderr, "[Track][Info ] #%ld-#%ld 局部地图投影优化成功, 参与优化的MP观测数 = %d, 耗时 "
                    "= %.2fms, 位姿更新为:[%.2f, %.2f, %.2f]\n",
            frame.id, mpReferenceKF->id, vertexId, timer.count(), Twb.x, Twb.y, Twb.theta);
}

void Track::startNewTrack()
{
    fprintf(stderr, "\n***** 连续丢失超过50帧! 清空地图从当前帧重新开始运行!! *****\n");
    mpMap->clear();
    mState = cvu::FIRST_FRAME;
    mpReferenceKF = nullptr;
    mpLoopKF = nullptr;
    mMPCandidates.clear();
    mvKPMatchIdx.clear();
    mvKPMatchIdxGood.clear();
    N1 = N2 = N3 = 0;
    mnLostFrames = 0;
    mAffineMatrix = Mat::eye(2, 3, CV_64FC1);
}

}  // namespace se2lam
