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
    : mState(cvu::NO_READY_YET), mLastState(cvu::NO_READY_YET), mbPrint(true), mbNeedVisualization(false),
      mpReferenceKF(nullptr), mpLoopKF(nullptr), mnNewAddedMPs(0), mnInliers(0), mnGoodInliers(0),
      mnTrackedOld(0), mnLostFrames(0), mbFinishRequested(false), mbFinished(false)
{
    mpORBextractor = new ORBextractor(Config::MaxFtrNumber, Config::ScaleFactor, Config::MaxLevel);
    mpORBmatcher = new ORBmatcher(0.9, true);

    std::fill(mMPCandidates.begin(), mMPCandidates.end(), Point3f(-1.f, -1.f, -1.f));
    nMinFrames = min(1, cvCeil(0.25 * Config::FPS));  // 上溢
    nMaxFrames = cvFloor(3 * Config::FPS);            // 下溢
    nMinMatches = std::min(cvFloor(0.1 * Config::MaxFtrNumber), 50);
    mMaxAngle = static_cast<float>(g2o::deg2rad(30.));
    mMaxDistance = 0.2f * Config::UpperDepth;

    mK = Config::Kcam;
    mD = Config::Dcam;
    mAffineMatrix = Mat::eye(2, 3, CV_64FC1);  // double

    mbPrint = Config::GlobalPrint;
    mbNeedVisualization = Config::NeedVisualization;

    for (int i = 0; i < 3; ++i)
        preSE2.meas[i] = 0;
    for (int i = 0; i < 9; ++i)
        preSE2.cov[i] = 0;

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
    if (Config::LocalizationOnly)
        return;

    if (mState == cvu::NO_READY_YET)
        mState = cvu::FIRST_FRAME;

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
        mpSensors->readData(odo, img);
        double t1 = timer.count(), t2 = 0, t3 = 0;

        // Tracking
        timer.start();
        bool bOK = false;
        {
            locker lock(mMutexForPub);

            double imgTime = 0.;
            mCurrentFrame = Frame(img, imgTime, odo, mpORBextractor, mK, mD);
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
                bOK = relocalization();  // 没追上的直接检测回环重定位
            }

            if (bOK) {
                // TODO 更新一下MPCandidates里面Tc2w
                mnLostFrames = 0;
                mState = cvu::OK;
            } else {
                mnLostFrames++;
                mState = cvu::LOST;
                if (mnLostFrames >= 50) {
                    resartTracking();
                    rate.sleep();
                    continue;
                }
            }

            // Visualization
            if (mbNeedVisualization) {
                char strMatches[64];
                std::snprintf(strMatches, 64, "F: %ld, KF: %ld-%ld, M: %d/%d", mCurrentFrame.id,
                              mpReferenceKF->mIdKF, mCurrentFrame.id - mpReferenceKF->id,
                              mnGoodInliers, mnInliers);
                mpMapPublisher->update(this, strMatches);
            }

            // Reset
            mpMap->setCurrentFramePose(mCurrentFrame.getPose());
            resetLocalTrack();  // KF判断在这里
        }
        t3 = timer.count();
        trackTimeTatal += t2 + t3;
        fprintf(stdout, "[Track][Timer] #%ld-#%ld T3.当前帧前端读取数据/构建Frame/追踪/总耗时为: "
                        "%.2f/%.2f/%.2f/%.2fms, 平均追踪耗时: %.2fms\n",
                mCurrentFrame.id, mpReferenceKF->id, t1, t2, t3, t1 + t2 + t3,
                trackTimeTatal / (mCurrentFrame.id + 1.f));

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

bool Track::trackLastFrame()
{
    if (mCurrentFrame.isNull())  // 图像挂掉的时候Frame会变成null.
        return false;
    if (mCurrentFrame.mTimeStamp < mLastFrame.mTimeStamp) {
        fprintf(stderr,
                "[Track][Warni] #%ld-#%ld 图像序列在时间上不连续: Last = %.3f, Curr = %.3f\n",
                mCurrentFrame.id, mLastFrame.id, mLastFrame.mTimeStamp, mCurrentFrame.mTimeStamp);
        return false;
    }

    updateFramePoseFromLast();

    int nMatches = mpORBmatcher->SearchByProjection(mLastFrame, mCurrentFrame, 7);
    if (nMatches < 20) {
        printf("[Track][Warni] #%ld-#%ld 与上一帧的MP匹配点数过少, 扩大搜索半径重新搜索.\n",
               mCurrentFrame.id, mLastFrame.id);
        std::fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), nullptr);
        nMatches = mpORBmatcher->SearchByProjection(mLastFrame, mCurrentFrame, 15);
    }
    if (nMatches < 20) {
        printf("[Track][Warni] #%ld-#%ld 与上一帧的MP匹配点数仍然过少, 即将转为与参考帧匹配!\n",
               mCurrentFrame.id, mLastFrame.id);
        return false;
    }
    int nCorres = 0;
    double projError = 10000.;
    Se2 Twb = mCurrentFrame.getTwb();
    printf("[Track][Warni] #%ld-#%ld 位姿优化更新前的值为: [%.2f, %.2f, %.2f]\n", mCurrentFrame.id,
           mLastFrame.id, Twb.x, Twb.y, Twb.theta);
    poseOptimization(&mCurrentFrame, nCorres, projError);  // 更新 mPose, mvbOutlier
    mCurrentFrame.updateObservationsAfterOpt();
    Twb = mCurrentFrame.getTwb();
    printf("[Track][Warni] #%ld-#%ld 位姿优化更新后的值为: [%.2f, %.2f, %.2f]\n", mCurrentFrame.id,
           mLastFrame.id, Twb.x, Twb.y, Twb.theta);
    bool lost = detectIfLost(nCorres, projError);
    if (lost) {
        printf("[Track][Warni] #%ld-#%ld 由MP优化位姿失败! MP内点数/重投影误差为: %d/%.3f, "
               "即将转为重定位!\n",
               mCurrentFrame.id, mLastFrame.id, nCorres, projError);
        return false;
    }
    return true;
}

bool Track::trackReferenceKF()
{
    if (mCurrentFrame.isNull())  // 图像挂掉的时候Frame会变成null.
        return false;
    if (mCurrentFrame.mTimeStamp < mLastFrame.mTimeStamp) {
        fprintf(stderr,
                "[Track][Warni] #%ld-#%ld 图像序列在时间上不连续: Last = %.3f, Curr = %.3f\n",
                mCurrentFrame.id, mLastFrame.id, mLastFrame.mTimeStamp, mCurrentFrame.mTimeStamp);
        return false;
    }

    //! 1.根据里程计设置初始位姿
    updateFramePoseFromRef();

    //! 2.基于仿射变换先验估计投影Cell的位置
    int nMatches =
        mpORBmatcher->MatchByWindowWarp(*mpReferenceKF, mCurrentFrame, mAffineMatrix, mvMatchIdx, 20);
    if (nMatches < 15) {
        printf("[Track][Warni] #%ld-#%ld 与参考帧匹配[总]点数过少(%d), 即将转为重定位!\n",
               mCurrentFrame.id, mpReferenceKF->id, nMatches);
        return false;
    }

    //! 3.利用仿射矩阵A计算KP匹配的内点，内点数大于10才能继续
    removeOutliers();  // 更新A, mnInliers
    if (mnInliers < 10) {
        printf("[Track][Warni] #%ld-#%ld 与参考帧匹配[内]点数过少(%d), 即将转为重定位!\n",
               mCurrentFrame.id, mpReferenceKF->id, mnInliers);
        return false;
    }

    //! 4.三角化生成潜在MP, 由LocalMap线程创造MP
    doTriangulate();  // 更新 mnTrackedOld, mnGoodInliers, mvGoodMatchIdx

    Se2 Twb = mCurrentFrame.getTwb();
    printf("[Track][Warni] #%ld-#%ld 位姿优化更新前的值为: [%.2f, %.2f, %.2f]\n", mCurrentFrame.id,
           mLastFrame.id, Twb.x, Twb.y, Twb.theta);

    //! 5.最小化重投影误差优化当前帧位姿
    int nCros = 0;
    double projError = 10000.;
    poseOptimization(&mCurrentFrame, nCros, projError);  // 更新 mPose, mvbOutlier
    mCurrentFrame.updateObservationsAfterOpt();  // 根据 mvbOutlier 更新Frame的观测情况

    Twb = mCurrentFrame.getTwb();
    printf("[Track][Warni] #%ld-#%ld 位姿优化更新后的值为: [%.2f, %.2f, %.2f]\n", mCurrentFrame.id,
           mLastFrame.id, Twb.x, Twb.y, Twb.theta);

    bool lost = detectIfLost(nCros, projError);
    if (lost) {
        printf("[Track][Warni] #%ld-#%ld 由MP优化位姿失败! MP内点数/重投影误差为: %d/%.3f, "
               "即将转为重定位!\n",
               mCurrentFrame.id, mpReferenceKF->id, nCorres, projError);
        return false;
    }

    return true;
}

// 把局部地图的点往当前帧上投
bool Track::trackLocalMap()
{
    mpMap->updateLocalGraph();  // 并不一定需要在这里更新, 因为在LocalMap线程里会更新
    vector<PtrMapPoint> vpLocalMPs = mpMap->getLocalMPs();
    vector<size_t> vMatchedIdxMPs;  // 一定是新投上的点
    mpORBmatcher->SearchByProjection(&mCurrentFrame, vpLocalMPs, 20, 1, vMatchedIdxMPs);
    for (int i = 0, iend = mCurrentFrame.N; i < iend; ++i) {
        if (vMatchedIdxMPs[i] < 0)
            continue;
        PtrMapPoint& pMP = vpLocalMPs[vMatchedIdxMPs[i]];  // 新匹配上的MP
        mCurrentFrame.setObservation(pMP, i);              // TODO 要不要管 mvbOutlier?
    }

    doLocalBA(mCurrentFrame);

    return !detectIfLost();
}

void Track::updateFramePoseFromLast()
{
    Se2 Tb1b2 = mCurrentFrame.odom - mLastFrame.odom;
    Se2 Twb = mLastFrame.getTwb() + Tb1b2;
    mCurrentFrame.setPose(Twb);
}

//! 根据RefKF和当前帧的里程计更新先验位姿和变换关系, odom是se2绝对位姿而非增量
//! 如果当前帧不是KF，则当前帧位姿由RefK叠加上里程计数据计算得到
//! 这里默认场景是工业级里程计，精度比较准
void Track::updateFramePoseFromRef()
{
    // Tb1b2, 参考帧Body->当前帧Body, se2形式
    // mCurrentFrame.Trb = mCurrentFrame.odom - mpReferenceKF->odom;
    // Tb2b1, 当前帧Body->参考帧Body, se2形式
    Se2 Tb1b2 = mCurrentFrame.odom - mpReferenceKF->odom;
    Se2 Tb2b1 = mpReferenceKF->odom - mCurrentFrame.odom;
    // Tc2c1, 当前帧Camera->参考帧Camera, Tc2c1 = Tcb * Tb2b1 * Tbc
    // mCurrentFrame.Tcr = Config::Tcb * dOdo.toCvSE3() * Config::Tbc;
    // Tc2w, 当前帧Camera->World, Tc2w = Tc2c1 * Tc1w
    // mCurrentFrame.Tcw = mCurrentFrame.Tcr * mpReferenceKF->Tcw;
    // Twb2, 当前帧World->Body，se2形式，故相加， Twb2 = Twb1 * Tb1b2
    // mCurrentFrame.Twb = mpReferenceKF->Twb + mCurrentFrame.Trb;


    Mat Tc2c1 = Config::Tcb * Tb2b1.toCvSE3() * Config::Tbc;
    Mat Tc1w = mpReferenceKF->getPose();
    mCurrentFrame.setTrb(Tb1b2);
    mCurrentFrame.setTcr(Tc2c1);
    mCurrentFrame.setPose(Tc2c1 * Tc1w);

    // preintegration 预积分
    //! TODO 这里并没有使用上预积分？都是局部变量，且实际一帧图像仅对应一帧Odom数据
    /*
        Eigen::Map<Vector3d> meas(preSE2.meas);
        Se2 odok = mCurrentFrame.odom - mLastFrame.odom;
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
    */
}

/**
 * @brief   根据仿射矩阵A剔除外点，利用了RANSAC算法
 * 更新匹配关系 mvMatchIdx 和内点数 mnInliers
 */
void Track::removeOutliers()
{
    mnInliers = 0;

    vector<Point2f> ptRef, ptCur;
    vector<size_t> idxRef;
    idxRef.reserve(mpReferenceKF->N);
    ptRef.reserve(mpReferenceKF->N);
    ptCur.reserve(mCurrentFrame.N);

    for (size_t i = 0, iend = mpReferenceKF->N; i < iend; ++i) {
        if (mvMatchIdx[i] < 0)
            continue;
        idxRef.push_back(i);
        ptRef.push_back(mpReferenceKF->mvKeyPoints[i].pt);
        ptCur.push_back(mCurrentFrame.mvKeyPoints[mvMatchIdx[i]].pt);
    }

    if (ptRef.size() == 0)
        return;

    vector<unsigned char> mask;
    mAffineMatrix = estimateAffine2D(ptRef, ptCur, mask, RANSAC, 3.0);
    // Mat H = findHomography(pt1, pt2, RANSAC, 3, mask);  // 朝天花板摄像头应该用H矩阵, F会退化

    assert(idxRef.size() == mask.size());
    for (size_t i = 0, iend = mask.size(); i < iend; ++i) {
        if (!mask[i]) {
            assert(mvMatchIdx[idxRef[i]] >= 0);
            mvMatchIdx[idxRef[i]] = -1;
        } else
            mnInliers++;
    }
}

/**
 * @brief 关键函数, 三角化获取特征点的深度, 视差好的可直接生成MP, 视差不好的先被标记为候选MP
 * 更新needNewKF()函数需要考虑到的相关变量, 更新Frame和KF的MP观测情况.
 * @note    这里的变量对应关系不能出错, 很重要!
 * @date    2019.11.22
 */
void Track::doTriangulate()
{
    // 以下成员变量在这个函数中会被修改
    mvGoodMatchIdx = mvMatchIdx;
    mnTrackedOld = 0;
    mnNewAddedMPs = 0;
    mnBadMatches = 0;
    mnGoodInliers = mnInliers;
    mnCandidateMPs = mMPCandidates.size();
    int n11 = 0, n121 = 0, n21 = 0, n22 = 0, n31 = 0, n32 = 0, n33 = 0;
    int nObsRef = mpReferenceKF->countObservations();

    // 相机1和2的投影矩阵
    const Mat Tc1w = mpReferenceKF->getPose();
    const Mat Tc2w = mCurrentFrame.getPose();
    const Mat Tcr = mCurrentFrame.getTcr();
    const Mat Tc1c2 = cvu::inv(Tcr);
    const Point3f Ocam1 = Point3f(0.f, 0.f, 0.f);
    const Point3f Ocam2 = Point3f(Tc1c2.rowRange(0, 3).col(3));
    const cv::Mat Proj1 = Config::PrjMtrxEye;  // P1 = K * cv::Mat::eye(3, 4, CV_32FC1)
    const cv::Mat Proj2 = Config::Kcam * Tcr.rowRange(0, 3);  // P2 = K * Tc2c1(3*4)

    /* 遍历参考帧的KP, 对于有匹配点对的KP进行处理, 如果:
     * 1. 参考帧KP已经有对应的MP观测:
     *  - 1.1 对于视差好的MP, 直接给当前帧关联一下MP观测;
     *  - 1.2 对于视差不好的MP, 再一次三角化更新其坐标值. 如果更新后:
     *      - 1.2.1 深度符合(理应保持)且视差好, 更新MP的属性;
     *      - 1.2.2 深度符合但视差没有被更新为好, 或深度被更新为不符合(出现概率不大), 则不处理.
     * 2. 参考帧KP已经有对应的MP候选: (说明深度符合但视差还不太好)
     *  - 三角化更新一下候选MP的坐标, 如果:
     *      - 2.1 深度符合(理应保持)且视差好, 生成新的MP, 为KF和F添加MP观测, 为MP添加KF观测, 删除候选;
     *      - 2.2 深度符合(理应保持)但视差没有被更新为好, 更新候选MP的坐标;
     *      - 2.3 深度被更新为不符合(出现概率不大), 则不处理.
     * 3. 参考帧KP一无所有:
     *  - 三角化, 如果:
     *      - 3.1 深度符合且视差好, 生成新的MP, 为KF和F添加MP观测, 为MP添加KF观测;
     *      - 3.2 深度符合但视差不好, 将其添加候选MP;
     *      - 3.3 深度不符合, 则丢弃此匹配点对.
     *
     * 几个变量的关系:
     *  - mnTrackedOld  = 1.1 + 1.2.1
     *  - mnNewAddedMPs = 2.1 + 3.1
     *  - mnBadMatches  = 3.3
     *  - mnGoodInliers = mnInliers - 3.3
     *  - mnCandidateMPs = mnCandidateMPs - 2.1 + 3.2
     *
     * NOTE 对视差不好的MP或者MP候选, 说明它是参考帧和其他帧生成的, 并不是和自己生成的,
     * 如果把视差不好的MP关联给当前帧, 会影响当前帧的投影精度.
     * 能成为MP或MP候选的点说明其深度都是符合预期的.
     * 最后MP候选将在LocalMap线程中生成视差不好的MP, 他们在添加KF观测后有机会更新视差.
     */
    for (size_t i = 0, iend = mpReferenceKF->N; i < iend; ++i) {
        if (mvMatchIdx[i] < 0)
            continue;

        PtrMapPoint pObservedMP = nullptr;                               // 对于的MP观测
        const bool bObserved = mpReferenceKF->hasObservationByIndex(i);  // 是否有对应MP观测
        if (bObserved) {
            assert(mpReferenceKF->mvbViewMPsInfoExist[i] == true);
            pObservedMP = mpReferenceKF->getObservation(i);
            if (pObservedMP->isGoodPrl()) {  // 情况1.1
                mCurrentFrame.setObservation(pObservedMP, mvMatchIdx[i]);
                mnTrackedOld++;
                n11++;
                continue;
            }
        }

        // 如果参考帧KP没有对应的MP,或者对应MP视差不好，则为此匹配对KP三角化计算深度(相对参考帧的坐标)
        // 由于两个投影矩阵是两KF之间的相对投影, 故三角化得到的坐标是相对参考帧的坐标, 即Pc1
        Point2f pt1 = mpReferenceKF->mvKeyPoints[i].pt;
        Point2f pt2 = mCurrentFrame.mvKeyPoints[mvMatchIdx[i]].pt;
        Point3f Pc1 = cvu::triangulate(pt1, pt2, Proj1, Proj2);
        Point3f Pw = cvu::se3map(cvu::inv(Tc1w), Pc1);

        // Pc2用Tcr计算出来的是预测值, 故这里用Pc1的深度判断即可
        const bool bAccepDepth = Config::acceptDepth(Pc1.z);  // 深度是否符合
        const bool bCandidated = mMPCandidates.count(i);      // 是否有对应的MP候选
        assert(bObserved != bCandidated);

        if (bAccepDepth) {  // 如果深度计算符合预期
            const bool bGoodPrl = cvu::checkParallax(Ocam1, Ocam2, Pc1, 2);  // 视差是否良好
            if (bGoodPrl) {                                                  // 如果视差好
                Eigen::Matrix3d xyzinfo1, xyzinfo2;
                calcSE3toXYZInfo(Pc1, Tc1w, Tc2w, xyzinfo1, xyzinfo2);
                if (bObserved) {  // 情况1.2.1
                    pObservedMP->setGoodPrl(true);
                    pObservedMP->setPos(Pw);
                    mpReferenceKF->setObsAndInfo(pObservedMP, i, xyzinfo1);
                    mCurrentFrame.setObservation(pObservedMP, mvMatchIdx[i]);
                    mnTrackedOld++;
                    n121++;
                } else {  // 情况2.1和3.1
                    PtrMapPoint pNewMP = make_shared<MapPoint>(Pw, true);
                    mpReferenceKF->setObsAndInfo(pNewMP, i, xyzinfo1);
                    mCurrentFrame.setObservation(pNewMP, mvMatchIdx[i]);
                    pNewMP->addObservation(mpReferenceKF, i);  // MP只添加KF的观测
                    mpMap->insertMP(pNewMP);
                    mnNewAddedMPs++;
                    n21++;
                    n31++;
                    if (bCandidated) { // 对于情况2.1, 候选转正后还要删除候选名单
                        mMPCandidates.erase(i);
                        mnCandidateMPs--;
                    }
                }
            } else {  // 如果视差不好
                if (bCandidated) {  // 情况2.2
                    mMPCandidates[i].Pc1 = Pc1;
                    mMPCandidates[i].id2 = mCurrentFrame.id;
                    mMPCandidates[i].kpIdx2 = mvMatchIdx[i];
                    n22++;
                } else {  // 情况3.2
                    assert(!mpReferenceKF->hasObservationByIndex(i));
                    MPCandidate MPcan(Pc1, mCurrentFrame.id, mvMatchIdx[i], Tc2w);
                    mMPCandidates.emplace(i, MPcan);
                    mnCandidateMPs++;
                    n32++;
                }
                // 情况1.2.2不处理
            }
        } else {  // 如果深度计算不符合预期
            if (!bObserved && !bCandidated) {  // 情况3.3
                n33++;
                mnBadMatches++;
                mnGoodInliers--;
                mvGoodMatchIdx[i] = -1;
            }
            // 情况1.2.3和2.3不处理
        }
    }
    printf("[Track][Info ] #%ld-#%ld 三角化结果: MP关联视差好点数(%d)/MP关联视差更新到好点数(%d)/MP总关联数(%d), "
           "候选更新新增MP数(%d)/KP三角化新增MP数(%d)/MP新增总数(%d), "
           "候选更新数(%d)/候选增加数(%d)/剔除匹配数(%d)/目前候选总数(%d)\n",
           mCurrentFrame.id, mpReferenceKF->id, n11, n121, mnTrackedOld,
           n21, n31, mnNewAddedMPs, n22, n32, mnBadMatches, mnCandidateMPs);

    // KF判定依据变量更新
    mCurrRatioGoodDepth = mnCandidateMPs / mnInliers;
    mCurrRatioGoodParl = mnNewAddedMPs / mnCandidateMPs;

    assert(n11 + n121 == mnTrackedOld);
    assert(n21 + n31 == mnNewAddedMPs);
    assert(n33 == mnBadMatches);
    assert(mnGoodInliers + mnBadMatches == mnInliers);
    assert(mnCandidateMPs == mMPCandidates.size());
    assert(mnTrackedOld + mnNewAddedMPs == mCurrentFrame.countObservations());
    assert(nObsRef + mnNewAddedMPs == mpReferenceKF->countObservations());
}

void Track::resetLocalTrack()
{
    // 正常状态进行关键帧判断
    if (mState == cvu::OK) {
        if (needNewKF()) {
            PtrKeyFrame pNewKF = make_shared<KeyFrame>(mCurrentFrame);
            // 新的KF观测和共视关系在LocalMap里更新
            mpLocalMapper->processNewKF(pNewKF, mMPCandidates);

            mpReferenceKF = pNewKF;
            mAffineMatrix = Mat::eye(2, 3, CV_64FC1);
            mLastRatioGoodDepth = 0.;
            mLastRatioGoodParl = 0.;
            mMPCandidates.clear();  // 加了新的KF后才清空. 和普通帧的匹配也会生成MP
        } else {
            mLastRatioGoodDepth = mCurrRatioGoodDepth;
            mLastRatioGoodParl = mCurrRatioGoodParl;
        }
        mLastFrame = mCurrentFrame;
        return;
    }

    // 当前帧刚丢失要确保上一帧是最新的KF
    if (mState == cvu::LOST && mLastState == cvu::OK && !mLastFrame.isNull()) {
        PtrKeyFrame pNewKF = make_shared<KeyFrame>(mLastFrame);
        mpLocalMapper->processNewKF(pNewKF, mMPCandidates);  // TODO
        mpReferenceKF = pNewKF;
    }

    // 处于丢失的状态则直接交换前后帧数据
    assert(mState == cvu::LOST);
    mAffineMatrix = Mat::eye(2, 3, CV_64FC1);
    mLastFrame = mCurrentFrame;
    mMPCandidates.clear();
}

bool Track::needNewKF()
{
    // 刚重定位成功需要建立新的KF
    if (mState == cvu::OK && mLastState == cvu::LOST)
        return true;

    // int nOldObs = mpReferenceKF->countObservations();
    int deltaFrames = static_cast<int>(mCurrentFrame.id - mpReferenceKF->id);

    // 必要条件
    bool c0 = deltaFrames > nMinFrames;  // 下限

    // 充分条件
    bool c1 = deltaFrames > nMaxFrames;  // 上限
    bool c2 = mnCandidateMPs > 10;       // 此帧可生成的潜在MP数
    bool c3 = mnNewAddedMPs > static_cast<int>(mnCandidateMPs * 0.8);  // 视差好的点多, 比较理想
    bool c4 = (mCurrRatioGoodDepth < mLastRatioGoodDepth) && (mCurrRatioGoodParl < mLastRatioGoodParl);
    bool bNeedKFByVo = c0 && (c1 || (c2 && c3) || c4);

    bool bNeedKFByOdo = false;
    bool c5 = false, c6 = false;
    if (mbUseOdometry) {
        Se2 dOdo = mCurrentFrame.odom - mpReferenceKF->odom;
        c5 = static_cast<double>(abs(dOdo.theta)) >= mMaxAngle;  // 旋转量超过40°
        cv::Mat cTc = Config::Tcb * dOdo.toCvSE3() * Config::Tbc;
        cv::Mat xy = cTc.rowRange(0, 2).col(3);
        c6 = cv::norm(xy) >= mMaxDistance;  // 相机的平移量足够大
        bNeedKFByOdo = c5 || c6;  // 相机移动取决于深度上限,考虑了不同深度下视野的不同
    }

    bool bNeedNewKF = bNeedKFByVo || bNeedKFByOdo;
    if (!bNeedNewKF) {
        return false;
    } else if (mpLocalMapper->checkIfAcceptNewKF()) {
        printf("[Track][Info ] #%ld-#%ld 成为了新的KF, 其KF条件满足情况: "
               "下限(%d)/上限(%d)/潜在(%d)/视差(%d)/追踪(%d)/旋转(%d)/平移(%d)\n",
               mCurrentFrame.id, mpReferenceKF->id, c0, c1, c2, c3, c4, c5, c6);
        return true;
    }

    // 强制更新KF的条件
    if ((c2 && c3) && c4 && bNeedKFByOdo) {
        printf("[Track][Warni] #%ld-#%ld 强制添加KF, 其KF条件满足情况: "
               "下限(%d)/上限(%d)/视差(%d)/追踪(%d)/旋转(%d)/平移(%d)\n",
               mCurrentFrame.id, mpReferenceKF->id, c0, c1, c2, c3, c4, c5, c6);
        mpLocalMapper->setAbortBA();  // 如果必须要加入关键帧,则终止LocalMap优化
        mpLocalMapper->setAcceptNewKF(true);
        return true;
    } else {
        printf("[Track][Warni] #%ld-#%ld 应该成为了新的KF, 但局部地图繁忙!\n", mCurrentFrame.id,
               mpReferenceKF->id);
        return false;
    }
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

// 根据优化内点数和重投影误差来判断是否丢失定位
bool Track::detectIfLost(int nCros, double projError)
{
    const int th = (mnTrackedOld + mnNewAddedMPs) * 0.6;
    if (nCros < th)
        return true;
    if (projError > 100.)  // TODO 阈值待调整
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

    const float th_angle = Config::MaxAngularSpeed / Config::FPS;
    const float th_dist = Config::MaxLinearSpeed / Config::FPS;

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


//! 重定位. BoW搜索匹配，通过重投影误差看是否成功
bool Track::relocalization()
{
    if (mCurrentFrame.isNull())  // 图像挂掉的时候Frame会变成null.
        return false;
    if (mCurrentFrame.mTimeStamp < mLastFrame.mTimeStamp)
        return false;

    mnNewAddedMPs = mnCandidateMPs = mnBadMatches = 0;
    mnInliers = mnGoodInliers = mnTrackedOld = 0;

    updateFramePoseFromRef();

    bool bDetected = detectLoopClose();
    bool bVerified = false;
    if (bDetected) {
        bVerified = verifyLoopClose();
        if (bVerified) {
            // Get Local Map and do local BA
            trackLocalMap();

            PtrKeyFrame mpNewKF = make_shared<KeyFrame>(mCurrentFrame);
            mpNewKF->addCovisibleKF(mpLoopKF);
            mpLoopKF->addCovisibleKF(mpNewKF);

            mpMap->insertKF(mpNewKF);


            fprintf(stderr, "[Track][Warni] #%ld-#%ld 重定位成功!\n", mCurrentFrame.id,
                    mpReferenceKF->id);

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
    const double minScoreBest = Config::MinScoreBest;  // 0.003

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

        double score = mpGlobalMapper->mpORBVoc->score(BowVecCurr, BowVec);
        if (score > scoreBest) {
            scoreBest = score;
            pKFBest = pKFi;
        }
    }

    if (pKFBest != nullptr && scoreBest > minScoreBest) {
        mpLoopKF = pKFBest;
        bDetected = true;
        fprintf(stderr, "[Track][Info ] #%ld-#%ld 重定位-回环检测成功! score = %.3f >= %.3f\n",
                mCurrentFrame.id, mpReferenceKF->id, scoreBest, minScoreBest);
    } else {
        mpLoopKF.reset();
        fprintf(stderr, "[Track][Warni] #%ld-#%ld 重定位-回环检测失败! score = %.3f < %.3f\n",
                mCurrentFrame.id, mpReferenceKF->id, scoreBest, minScoreBest);
    }

    return bDetected;
}

bool Track::verifyLoopClose()
{
    assert(mpLoopKF != nullptr && !mCurrentFrame.isNull());

    const int nMinKPMatch = Config::MinKPMatchNum * 0.6;  // 30, KP最少匹配数
    map<int, int> mapMatches;

    // BoW匹配
    ORBmatcher matcher;
    bool bIfMatchMPOnly = false;  // 要看总体匹配点数多不多
    matcher.SearchByBoW(mCurrentFrame, mpLoopKF, mapMatches, bIfMatchMPOnly);
    const int nGoodKPMatch = mapMatches.size();
    const bool bMatchGood = nGoodKPMatch >= nMinKPMatch;
    if (!bMatchGood) {
        fprintf(stderr, "[Track][Warni] #%ld-#%ld 重定位-回环验证失败! nGoodMatch = %d < %d\n",
                mCurrentFrame.id, mpReferenceKF->id, nGoodKPMatch, nMinKPMatch);
        return false;
    }

    // 匹配点数足够则剔除外点
    removeMatchOutlierRansac(&mCurrentFrame, mpLoopKF, mapMatches);
    mnGoodInliers = mapMatches.size();  // 总的KP匹配内点数

    // 根据和LoopKF的MP匹配情况更新自己的MP可视情况
    mnNewAddedMPs = 0;
    mnTrackedOld = 0;
    mvGoodMatchIdx.clear();
    mvGoodMatchIdx.resize(mpReferenceKF->N, -1);
    for (auto iter = mapMatches.begin(); iter != mapMatches.end(); iter++) {
        int idxCurr = iter->first;
        int idxLoop = iter->second;
        mvGoodMatchIdx[idxLoop] = idxCurr;

        bool isMPLoop = mpLoopKF->hasObservationByIndex(idxLoop);
        if (isMPLoop) {
            PtrMapPoint pMP = mpLoopKF->getObservation(idxLoop);
            mCurrentFrame.setObservation(pMP, idxCurr);
            mnTrackedOld++;
        }
    }

    // 优化位姿, 并计算重投影误差
    int nCorres = 0;
    double projError = 1000.;
    poseOptimization(&mCurrentFrame, nCorres, projError);  // 确保已经setViewMP()
    const bool bProjLost = detectIfLost(nCorres, projError);

    if (bProjLost) {
        fprintf(stderr,
                "[Track][Warni] #%ld-#%ld 重定位-回环验证失败! MP内点数/重投影误差为: %d/%.3f\n",
                mCurrentFrame.id, mpReferenceKF->id, nCorres, projError);
        return false;
    } else {
        mnInliers = mnGoodInliers;
        mvMatchIdx = mvGoodMatchIdx;
        fprintf(stderr,
                "[Track][Info ] #%ld-#%ld 重定位-回环验证成功! MP内点数/重投影误差为: %d/%.3f\n",
                mCurrentFrame.id, mpReferenceKF->id, nCorres, projError);
        return true;
    }
}

void Track::removeMatchOutlierRansac(const Frame* _pFCurr, const PtrKeyFrame& _pKFLoop, map<int, int>& mapMatch)
{
    const int numMinMatch = 10;

    int numMatch = mapMatch.size();
    if (numMatch < numMinMatch) {
        mapMatch.clear();
        return;
    }

    map<int, int> mapMatchGood;
    vector<int> vIdxCurr, vIdxLoop;
    vector<Point2f> vPtCurr, vPtLoop;

    for (auto iter = mapMatch.begin(); iter != mapMatch.end(); iter++) {
        int idxCurr = iter->first;
        int idxLoop = iter->second;

        vIdxCurr.push_back(idxCurr);
        vIdxLoop.push_back(idxLoop);

        vPtCurr.push_back(_pFCurr->mvKeyPoints[idxCurr].pt);
        vPtLoop.push_back(_pKFLoop->mvKeyPoints[idxLoop].pt);
    }

    // RANSAC with fundemantal matrix
    vector<uchar> vInlier;  // 1 when inliers, 0 when outliers
    findHomography(vPtCurr, vPtLoop, FM_RANSAC, 3.0, vInlier);
    for (size_t i = 0, iend = vInlier.size(); i < iend; ++i) {
        int idxCurr = vIdxCurr[i];
        int idxLoop = vIdxLoop[i];
        if (vInlier[i] == true) {
            mapMatchGood[idxCurr] = idxLoop;
        }
    }

    mapMatch = mapMatchGood;
}

/**
 * @brief 根据MP观测优化当前帧位姿. 二元边
 */
void Track::doLocalBA(Frame& frame)
{
    WorkTimer timer;

    SlamOptimizer optimizer;
    SlamLinearSolver* linearSolver = new SlamLinearSolver();
    SlamBlockSolver* blockSolver = new SlamBlockSolver(linearSolver);
    SlamAlgorithm* solver = new SlamAlgorithm(blockSolver);
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
    vector<PtrMapPoint> allViewMPs = frame.getAllLandMarks();
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
    frame.setPose(Tcw);  // 更新Tcw和Twb

    Se2 Twb = frame.getTwb();
    fprintf(stderr, "[Track][Info ] #%ld-#%ld 局部地图投影优化成功, 参与优化的MP观测数 = %d, 耗时 "
                    "= %.2fms, 位姿更新为:[%.2f, %.2f, %.2f]\n",
            frame.id, mpReferenceKF->id, vertexId, timer.count(), Twb.x, Twb.y, Twb.theta);
}

void Track::resartTracking()
{
    fprintf(stderr, "\n***** 连续丢失超过50帧! 清空地图从当前帧重新开始运行!! *****\n");
    mpMap->clear();
    mState = cvu::FIRST_FRAME;
    mpReferenceKF = nullptr;
    mpLoopKF = nullptr;
    mMPCandidates.clear();
    mvMatchIdx.clear();
    mvGoodMatchIdx.clear();
    N1 = N2 = N3 = 0;
    mnLostFrames = 0;
    mAffineMatrix = Mat::eye(2, 3, CV_64FC1);
}

}  // namespace se2lam
