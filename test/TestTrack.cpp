#include "TestTrack.h"
#include "Map.h"
#include "MapPublish.h"
#include "ORBmatcher.h"
#include "converter.h"
#include "cvutil.h"
#include "optimizer.h"
#include "GlobalMapper.h"
#include <opencv2/calib3d/calib3d.hpp>
#include <ros/ros.h>

namespace se2lam
{

#define DO_LOCAL_BA  0
#define DO_GLOBAL_BA 0

using namespace std;
using namespace cv;
using namespace Eigen;

typedef unique_lock<mutex> locker;


TestTrack::TestTrack()
    : mState(cvu::NO_READY_YET), mLastState(cvu::NO_READY_YET), mpReferenceKF(nullptr),
      mpNewKF(nullptr), mpLoopKF(nullptr), mnMPsCandidate(0), mnKPMatches(0), mnKPsInline(0),
      mnKPMatchesGood(0), mnMPsTracked(0), mnMPsNewAdded(0), mnMPsInline(0), mnLostFrames(0),
      mnLessMatchFrames(0), mLoopScore(0)
{
    mpORBextractor = new ORBextractor(Config::MaxFtrNumber, Config::ScaleFactor, Config::MaxLevel);
    mpORBmatcher = new ORBmatcher(0.9, true);
    mpORBvoc = new ORBVocabulary();
    string strVocFile = Config::DataPath + "../se2_config/ORBvoc.bin";
    bool bVocLoad = mpORBvoc->loadFromBinaryFile(strVocFile);
    if (!bVocLoad)
        cerr << "[Track][Error] Wrong path to vocabulary, Falied to open it." << endl;

    nMinFrames = min(3, cvCeil(0.3 * Config::FPS));  // 上溢
    nMaxFrames = cvFloor(5 * Config::FPS);  // 下溢
    nMinMatches = std::min(cvFloor(0.1 * Config::MaxFtrNumber), 40);
    mMaxAngle = static_cast<float>(g2o::deg2rad(80.));
    mMinDistance = 330.f;

    mAffineMatrix = Mat::eye(2, 3, CV_64FC1);  // double
    preSE2.meas.setZero();
    preSE2.cov.setZero();

    fprintf(stderr, "[Track][Info ] KF选择策略相关参数如下: \n - 最小/最大KF帧数: %d/%d\n"
                    " - 最小移动距离/最大角度: %.0fmm/%.0fdeg\n - 最少匹配数量: %d\n",
            nMinFrames, nMaxFrames, mMinDistance, g2o::rad2deg(mMaxAngle), nMinMatches);
}

TestTrack::~TestTrack()
{
    delete mpORBextractor;
    delete mpORBmatcher;
    delete mpORBvoc;
}

bool TestTrack::checkReady()
{
    if (!mpMap || !mpMapPublisher)
        return false;
    return true;
}

void TestTrack::run(const cv::Mat& img, const Se2& odo, const double time)
{
    if (mState == cvu::NO_READY_YET)
        mState = cvu::FIRST_FRAME;

    WorkTimer timer;
    mLastState = mState;
    bool bOK = false;
    double t1 = 0, t2 = 0, t3 = 0;

    {
        locker lock(mMutexForPub);

        mCurrentFrame = Frame(img, odo, mpORBextractor, time);
        t1 = timer.count();

        timer.start();
        if (mState == cvu::FIRST_FRAME) {
            processFirstFrame();
            return;
        } else if (mState > 0) {
            mLastRefKFid = mpReferenceKF->id;  // 这里仅用于输出log

            // 每次Tracikng前先判断是否处于纯旋转状态
            const Se2 dOdom = mCurrentFrame.odom - mLastFrame.odom;
            const Point2f dxy(dOdom.x, dOdom.y);
            const bool bPureRotation = cv::norm(dxy) < 10.f && abs(dOdom.theta) > 0.1;  // [mm/rad]
            if (bPureRotation) {
                mState = cvu::ROTATE;
                printf("[Track][Info ] #%ld-#%ld 当前处于纯旋转状态! dx = %.2f, dtheta = %.2f\n",
                       mCurrentFrame.id, mLastRefKFid, cv::norm(dxy), dOdom.theta);
            } else {
                mState = cvu::OK;
            }

            bOK = trackReferenceKF();
            if (!bOK)
                bOK = trackLocalMap();  // 刚丢的还可以再抢救一下
        } else if (mState == cvu::LOST) {
            bOK = doRelocalization();  // 没追上的直接检测回环重定位
            if (bOK) {
                mnLostFrames = 0;
                mState = cvu::OK;
            }
        }
        t2 = timer.count();
    }

    if (!bOK) {
        mnLostFrames++;
        mState = cvu::LOST;
        //if (mnLostFrames > 50)
        //    startNewTrack();
    }

    copyForPub();

    newKFDecision();  // KF判断在这里

    t3 = t1 + t2;
    trackTimeTatal += t3;
//    printf("[Track][Timer] #%ld-#%ld 前端构建Frame/追踪/总耗时为: %.2f/%.2f/%.2fms, 平均耗时: %.2fms\n",
//           mCurrentFrame.id, mLastRefKFid, t1, t2, t3,  trackTimeTatal * 1.f / mCurrentFrame.id);
}

void TestTrack::processFirstFrame()
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

bool TestTrack::trackReferenceKF()
{
    if (mCurrentFrame.isNull())
        return false;
    if (mCurrentFrame.mTimeStamp < mLastFrame.mTimeStamp) {
        fprintf(stderr, "[Track][Warni] #%ld-#%ld 图像序列在时间上不连续: Last = %.3f, Curr = %.3f\n",
                mCurrentFrame.id, mLastFrame.id, mLastFrame.mTimeStamp, mCurrentFrame.mTimeStamp);
        return false;
    }

    assert(mnKPsInline == 0);

    // 1.根据里程计设置初始位姿
    updateFramePoseFromRef();

    // 2.基于等距变换先验估计投影Cell的位置
    mnKPMatches = mpORBmatcher->MatchByWindowWarp(*mpReferenceKF, mCurrentFrame, mAffineMatrix,
                                                  mvKPMatchIdx, 25);

    bool bLessKPmatch = false;
    if (mState == cvu::OK && mnKPMatches < 20) {// 纯旋转时不要考虑匹配点数
        bLessKPmatch = true;
//        fprintf(stderr, "[Track][Warni] #%ld-#%ld 与参考帧匹配总点数过少(%ld), Less即将+1\n",
//               mCurrentFrame.id, mLastRefKFid, mnKPMatches);
    }

    // 3.利用仿射矩阵A计算KP匹配的内点，内点数大于10才能继续
    mnKPsInline = removeOutliers(mpReferenceKF, &mCurrentFrame, mvKPMatchIdx, mAffineMatrix);
    if (mState == cvu::OK && mnKPsInline < 15) { // 纯旋转时不要考虑匹配点数
        bLessKPmatch = true;
//        fprintf(stderr, "[Track][Warni] #%ld-#%ld 与参考帧匹配内点数过少(%ld), Less即将+1\n",
//               mCurrentFrame.id, mLastRefKFid, mnKPsInline);
    }

    assert(mnLessMatchFrames < 5);
    if (bLessKPmatch) {
        if (++mnLessMatchFrames >= 5) {
            fprintf(stderr, "[Track][Warni] #%ld-#%ld 连续5帧与参考帧匹配总/内点数过少, 即将转为trackLocalMap()\n",
                   mCurrentFrame.id, mLastRefKFid);
            mnLessMatchFrames = 0;
            return false;
        }
    }

    // 4.三角化生成潜在MP, 由LocalMap线程创造MP
    int nObs = doTriangulate(mpReferenceKF, &mCurrentFrame);  // 更新 mnTrackedOld, mnGoodInliers, mvGoodMatchIdx

    assert(nObs == mCurrentFrame.countObservations());
    if (0 && nObs > 10) { //! 这里不用执行
        WorkTimer timer;

        const Se2 Twb1 = mCurrentFrame.getTwb();
        const Mat Tcw_opt = poseOptimize(&mCurrentFrame);
        Se2 Twb2;
        Twb2.fromCvSE3(cvu::inv(Tcw_opt) * Config::Tcb);
        cout << "[Track][Info ] #" << mCurrentFrame.id << "-#" << mLastRefKFid << " 位姿优化情况: 观测数: "
             << nObs << ", 耗时: " << timer.count() << ", 位姿优化前后的值为: " << Twb1 << " ==> " << Twb2 << endl;

        bool lost = detectIfLost(mCurrentFrame, Tcw_opt);
        if (lost)
            return false;
        else
            mCurrentFrame.setPose(Tcw_opt);
    }

    return true;
}

bool TestTrack::trackLocalMap()
{
    cerr << "[Track][Warni] #" << mCurrentFrame.id << "-#" << mLastRefKFid
         << " TrackRefKF()丢失, 正在从局部地图中计算当前帧位姿..." << endl;

    const vector<PtrMapPoint> vpLocalMPs = mpMap->getLocalMPs();
    if (vpLocalMPs.empty())
        return false;

    vector<int> vMatchedIdxMPs;  // 一定是新投上的点
    int nProj = mpORBmatcher->SearchByProjection(mCurrentFrame, vpLocalMPs, vMatchedIdxMPs, 20, 1);
    if (nProj > 0 && nProj + mCurrentFrame.countObservations() >= 10) {
        assert(vMatchedIdxMPs.size() == mCurrentFrame.N);
        for (size_t i = 0, iend = mCurrentFrame.N; i < iend; ++i) {
            if (vMatchedIdxMPs[i] < 0)
                continue;
            const PtrMapPoint& pMP = vpLocalMPs[vMatchedIdxMPs[i]];  // 新匹配上的MP
            mCurrentFrame.setObservation(pMP, i);
        }

        const Mat Tcw_opt = poseOptimize(&mCurrentFrame);
        if (detectIfLost(mCurrentFrame, Tcw_opt))
            return false;
        else
            mCurrentFrame.setPose(Tcw_opt);
    } else {
        return false;
    }
    cerr << "[Track][Info ] #" << mCurrentFrame.id << "-#" << mLastRefKFid
         << " 从局部地图中计算当前帧位姿成功!" << endl;
    return true;
}

void TestTrack::updateFramePoseFromRef()
{
    const Se2 Tb1b2 = mCurrentFrame.odom - mpReferenceKF->odom;
    const Mat Tc2c1 = Config::Tcb * Tb1b2.inv().toCvSE3() * Config::Tbc;
    mCurrentFrame.setTrb(Tb1b2);
    mCurrentFrame.setTcr(Tc2c1);
    mCurrentFrame.setPose(Tc2c1 * mpReferenceKF->getPose()); // 位姿的预测初值用odo进行更新
    //mCurrentFrame.setPose(mpReferenceKF->getTwb() + Tb1b2);

    assert(mCurrentFrame.id - mLastFrame.id == 1);

    // Eigen::Map 是一个引用, 这里更新了到上一帧的积分
    Vector3d& meas = preSE2.meas;
    Se2 odok = mCurrentFrame.odom - mLastFrame.odom;
    Vector2d odork(odok.x, odok.y);
    Matrix2d Phi_ik = Rotation2Dd(meas[2]).toRotationMatrix();
    meas.head<2>() += Phi_ik * odork;
    meas[2] += odok.theta;

    Matrix3d Ak = Matrix3d::Identity();
    Matrix3d Bk = Matrix3d::Identity();
    Ak.block<2, 1>(0, 2) = -Phi_ik * Vector2d(-odork[1], odork[0]);
    Bk.block<2, 2>(0, 0) = -Phi_ik;
    Matrix3d& Sigmak = preSE2.cov;
    Matrix3d Sigma_vk = Matrix3d::Identity();
    Sigma_vk(0, 0) = (Config::OdoNoiseX * Config::OdoNoiseX);
    Sigma_vk(1, 1) = (Config::OdoNoiseY * Config::OdoNoiseY);
    Sigma_vk(2, 2) = (Config::OdoNoiseTheta * Config::OdoNoiseTheta);
    Matrix3d Sigma_k_1 = Ak * Sigmak * Ak.transpose() + Bk * Sigma_vk * Bk.transpose();
    Sigmak = Sigma_k_1;

    mCurrentFrame.mPreSE2 = preSE2;

//    cout << "[Debug] #" << mCurrentFrame.id << "-#" << mLastRefKFid << " prein_odo ref to cur info  = "
//         << endl << Sigmak << endl;
}

int TestTrack::doTriangulate(PtrKeyFrame& pKF, Frame* frame)
{
    // 以下成员变量在这个函数中会被修改
    mnMPsCandidate = mMPCandidates.size();
    mvKPMatchIdxGood = mvKPMatchIdx;
    mnMPsInline = 0;
    mnMPsTracked = 0;
    mnMPsNewAdded = 0;
    mnKPMatchesGood = 0;

    if (mvKPMatchIdx.empty()) {
        mnKPMatchesGood = mnKPsInline;
        return 0;
    }

    // 纯旋转/与参考帧相距太近, 不做三角化但要做MP关联
    if (mState == cvu::ROTATE || mCurrentFrame.id - mpReferenceKF->id < nMinFrames) {
        mnKPMatchesGood = 0;
        printf("[Track][Info ] #%ld-#%ld 纯旋转/与参考帧相距太近, 不进行三角化, 只进行MP关联. \n",
               mCurrentFrame.id, mLastRefKFid);

        for (size_t i = 0, iend = pKF->N; i < iend; ++i) {
            if (mvKPMatchIdx[i] < 0)
                continue;

            mnKPMatchesGood++;
            const bool bObserved = pKF->hasObservationByIndex(i);  // 是否有对应MP观测
            if (bObserved) {
                PtrMapPoint pObservedMP = pKF->getObservation(i);
                if (pObservedMP->isGoodPrl()) {
                    frame->setObservation(pObservedMP, mvKPMatchIdx[i]);
                    mnMPsTracked++;
                    mnMPsInline++;
                }
            }
        }
        return mnMPsInline;;
    }


    int n11 = 0, n121 = 0, n21 = 0, n22 = 0, n31 = 0, n32 = 0, n33 = 0;
    int n2 = mnMPsCandidate;
    int nObsCur = frame->countObservations();
    int nObsRef = pKF->countObservations();
    assert(nObsCur == 0);  // 当前帧应该还没有观测!

    // 相机1和2的投影矩阵
    const Mat Tc1w = pKF->getPose();
    const Mat Tc2w = frame->getPose();
    const Mat Tcr = frame->getTcr();
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

        mnKPMatchesGood++;
        PtrMapPoint pObservedMP = nullptr;  // 对于的MP观测
        const bool bObserved = pKF->hasObservationByIndex(i);  // 是否有对应MP观测
        if (bObserved) {
            pObservedMP = pKF->getObservation(i);
            if (pObservedMP->isGoodPrl()) {  // 情况1.1
                frame->setObservation(pObservedMP, mvKPMatchIdx[i]);
                mnMPsTracked++;
                mnMPsInline++;
                n11++;
                continue;
            }
        }

        // 如果参考帧KP没有对应的MP, 或者对应MP视差不好, 则为此匹配对KP三角化计算深度(相对参考帧的坐标)
        // 由于两个投影矩阵是两KF之间的相对投影, 故三角化得到的坐标是相对参考帧的坐标, 即Pc1
        const Point2f pt1 = pKF->mvKeyPoints[i].pt;
        const Point2f pt2 = frame->mvKeyPoints[mvKPMatchIdx[i]].pt;
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
                    frame->setObservation(pObservedMP, mvKPMatchIdx[i]);
                    mnMPsTracked++;
                    mnMPsInline++;
                    n121++;
                } else {  // 情况2.1和3.1
                    PtrMapPoint pNewMP = make_shared<MapPoint>(Pw, true);
                    pKF->setObsAndInfo(pNewMP, i, xyzinfo1);
                    pNewMP->addObservation(pKF, i);  // MP只添加KF的观测
                    frame->setObservation(pNewMP, mvKPMatchIdx[i]);
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
                    mMPCandidates[i].id2 = frame->id;
                    mMPCandidates[i].kpIdx2 = mvKPMatchIdx[i];
                    mMPCandidates[i].Tc2w = frame->getPose().clone();
                    n22++;
                } else {  // 情况3.2
                    assert(!pKF->hasObservationByIndex(i));
                    MPCandidate MPcan(pKF, Pc1, i, frame->id, mvKPMatchIdx[i], Tc2w);
                    mMPCandidates.emplace(i, MPcan);
                    mnMPsCandidate++;
                    n32++;
                }
            }
        } else {  // 如果深度计算不符合预期
            if (!bObserved && !bCandidated) {  // 情况3.3
                n33++;
                mnKPMatchesGood--;
                mvKPMatchIdxGood[i] = -1;
            }
            // 情况1.2.3和2.3不处理
        }
    }

    printf("[Track][Info ] #%ld-#%ld KP匹配结果: KP好匹配数/总内点数/总匹配数: %d/%d/%d, "
           "MP总观测数(Ref)/关联数/视差好的/更新到好的: %d/%d/%d/%d\n", frame->id, pKF->id,
           mnKPMatchesGood, mnKPsInline, mnKPMatches, nObsRef, mnMPsTracked, n11, n121);
    printf("[Track][Info ] #%ld-#%ld 三角化结果: 候选MP原总数/转正数/更新数/增加数/现总数: %d/%d/%d/%d/%d, "
           "三角化新增MP数/新增候选数/剔除匹配数: %d/%d/%d, 新生成MP数: %d\n",
           frame->id, pKF->id, n2, n21, n22, n32, mnMPsCandidate, n31, n32, n33, mnMPsNewAdded);

    assert(n11 + n121 == mnMPsTracked);
    assert(n21 + n31 == mnMPsNewAdded);
    assert(n33 + mnKPMatchesGood == mnKPsInline);
    assert(mnMPsTracked + mnMPsNewAdded == mnMPsInline);
    assert((n2 - n21 + n32 == mnMPsCandidate) && (mnMPsCandidate == mMPCandidates.size()));
    assert(nObsCur + mnMPsTracked + mnMPsNewAdded == frame->countObservations());
    assert(nObsRef + mnMPsNewAdded == pKF->countObservations());

    return mnMPsInline;
}

int  TestTrack::removeOutliers(const PtrKeyFrame& pKFRef, const Frame* pFCur,
                               vector<int>& vKPMatchIdx12, Mat& A12)
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
    A12 = estimateAffinePartial2D(vPtRef, vPtCur, vInlier, RANSAC, 1.5);
//    if (pFCur->id > 100) {
//        const double scale = A12.at<double>(0, 0) / cos(asin(A12.at<double>(1, 0)));
//        A12.at<double>(0, 0) /= scale;
//        A12.at<double>(1, 1) /= scale;
//        cout << "[Track][Info ] #" << pFCur->id << " 当前帧的 scale = " << scale
//             << ", A12 = " << endl << A12 << endl;
//    }

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

void TestTrack::newKFDecision()
{
    if (needNewKFForLastFrame()) {
        PtrKeyFrame pNewKF = make_shared<KeyFrame>(mLastFrame);

        mpReferenceKF->preOdomFromSelf = make_pair(pNewKF, pNewKF->mPreSE2);
        pNewKF->preOdomToSelf = make_pair(mpReferenceKF, pNewKF->mPreSE2);

        addNewKF(pNewKF, mMPCandidates);

        mpReferenceKF = pNewKF;

        resetTrackingData(true);

        return;
    }

    // 当前帧处于正常tracking状态
    if (mState > 0) {
        if (needNewKFForCurrentFrame()) {
            PtrKeyFrame pNewKF = make_shared<KeyFrame>(mCurrentFrame);

            // 重定位成功将当前帧变成KF, 需添加特征图约束. 没有odo约束?(addNewKF()函数里会添加)
            if (mState > 0 && mLastState == cvu::LOST) {
                pNewKF->addCovisibleKF(mpLoopKF);
                mpLoopKF->addCovisibleKF(pNewKF);
                printf("[Track][Info ] #%ld 成为了新的KF(#%ld), 因为刚刚重定位成功!\n", pNewKF->id, pNewKF->mIdKF);

                //! TODO 重定位成功(回环验证通过)构建特征图和约束
                //! NOTE mKPMatchesLoop 必须是有MP的匹配!
            } else {
                mpReferenceKF->preOdomFromSelf = make_pair(pNewKF, pNewKF->mPreSE2);
                pNewKF->preOdomToSelf = make_pair(mpReferenceKF, pNewKF->mPreSE2);
            }

            addNewKF(pNewKF, mMPCandidates);  // 新的KF观测和共视关系在LocalMap里更新

            mpReferenceKF = pNewKF;

            resetTrackingData(true);
        } else {
            resetTrackingData(false);
        }
        return;
    }

    // 处于丢失的状态则直接交换前后帧数据
    assert(mState == cvu::LOST);
    assert(mLastState == cvu::LOST);
    if (mState == cvu::LOST && mLastState == cvu::LOST) {
        resetTrackingData(true);
    }
}

bool TestTrack::needNewKFForLastFrame()
{
    // 上一帧已经是KF.
    if (mLastFrame.isNull() || mLastFrame.id == mpReferenceKF->id)
        return false;

    // 当前帧刚丢失要确保上一帧是最新的KF
    if (mState == cvu::LOST && mLastState > 0) {
        printf("[Track][Info ] #%ld-#%ld 当前帧刚丢失, 需要将上一帧(#%ld)加入新KF(%ld)!\n",
               mCurrentFrame.id, mLastRefKFid, mLastFrame.id, KeyFrame::mNextIdKF);
        return true;
    }

    // 纯旋转前要建立新的KF
    if (mState == cvu::ROTATE && mLastState == cvu::OK) {
        printf("[Track][Info ] #%ld-#%ld 刚进入纯旋转状态,需要将上一帧(#%ld)加入新KF(%ld)!\n",
               mCurrentFrame.id, mLastRefKFid, mLastFrame.id, KeyFrame::mNextIdKF);
        return true;
    }

    return false;
}

bool TestTrack::needNewKFForCurrentFrame()
{
    // 刚重定位成功需要建立新的KF
    if (mState > 0 && mLastState == cvu::LOST) {
        printf("[Track][Info ] #%ld-KF#%ld(Loop) 成为了新的KF(#%ld), 因为其刚重定位成功.\n",
               mCurrentFrame.id, mpLoopKF->id, KeyFrame::mNextIdKF);
        return true;
    }

    // 纯旋转后要建立新的KF
    if (mState == cvu::OK && mLastState == cvu::ROTATE) {
        printf("[Track][Info ] #%ld-#%ld 成为了新的KF(#%ld), 因为其刚从纯旋转状态中脱离.\n",
               mCurrentFrame.id, mLastRefKFid, KeyFrame::mNextIdKF);
        return true;
    }

    const Se2 dOdo = mCurrentFrame.odom - mpReferenceKF->odom;

    // 1. 如果在旋转, 则只看旋转角度是否达到上限
    if (mState == cvu::ROTATE) {
        if (abs(dOdo.theta) >= mMaxAngle) {
            printf("[Track][Info ] #%ld-#%ld 成为了新的KF(#%ld), 因为其旋转角度(%.2f)达到上限.\n",
                   mCurrentFrame.id, mLastRefKFid, KeyFrame::mNextIdKF, abs(dOdo.theta));
            mCurrentFrame.mPreSE2.cov(2, 2) *= 0.1;  // 增大旋转时的置信度.
            return true;
        } else {
            return false;
        }
    }

    // 2. 如果在平移, 则尽量选择和参考帧视差最好的那一帧
    const int deltaFrames = static_cast<int>(mCurrentFrame.id - mpReferenceKF->id);
    if (deltaFrames < nMinFrames)  // 必要条件, 不能太近
        return false;

    // 充分条件
    const cv::Mat cTc = Config::Tcb * dOdo.toCvSE3() * Config::Tbc;
    const cv::Mat xy = cTc.rowRange(0, 2).col(3);
    const bool c1 = cv::norm(xy) >= mMinDistance;  // 相机的平移量足够大, 视差才好
    const bool c2 = mnMPsNewAdded > 20 || mnMPsInline > 45;  // 关联/新增MP数量够多, 说明这时候比较理想 60
    if (c1 && c2) {
        printf("[Track][Info ] #%ld-#%ld 成为了新的KF(#%ld), 因为其平移适中(%.2f)或新增MP多(%d/%d)\n",
               mCurrentFrame.id, mLastRefKFid, KeyFrame::mNextIdKF, cv::norm(xy), mnMPsNewAdded, mnMPsInline);
        return true;
    }

    // 快要丢失时一定要插入.  这里顺序很重要! 30/20/15f
    const bool c3 = mnKPMatches < 20 || mnKPsInline < 15 || mnKPMatchesGood < 10;
    if (c1 && c3) {
        printf("[Track][Info ] #%ld-#%ld 成为了新的KF(#%ld), 因为其匹配数过少(%d/%d/%d)\n",
           mCurrentFrame.id, mLastRefKFid, KeyFrame::mNextIdKF, mnKPMatches, mnKPsInline, mnKPMatchesGood);
        return true;
    }

    return false;
}

void TestTrack::resetTrackingData(bool newKFInserted)
{
    mLastFrame = mCurrentFrame;
    mpLoopKF.reset();
    mnKPMatches = 0;
    mnKPsInline = 0;
    mnKPMatchesGood = 0;
    mnMPsTracked = 0;
    mnMPsNewAdded = 0;
    mnMPsInline = 0;

    if (newKFInserted) {
        preSE2.meas.setZero();
        preSE2.cov.setZero();

        mAffineMatrix = Mat::eye(2, 3, CV_64FC1);

        mMPCandidates.clear();
        mnMPsCandidate = 0;
        mnLessMatchFrames = 0;
    }
}

void TestTrack::copyForPub()
{
    locker lock1(mMutexForPub);
    locker lock2(mpMapPublisher->mMutexPub);

    assert(!mvKPMatchIdx.empty());
    assert(!mvKPMatchIdxGood.empty());

    mpMapPublisher->mnCurrentFrameID = mCurrentFrame.id;
    mpMapPublisher->mCurrentFramePose = mCurrentFrame.getPose()/*.clone()*/;
    mpMapPublisher->mCurrentImage = mCurrentFrame.mImage.clone();
    mpMapPublisher->mvCurrentKPs = mCurrentFrame.mvKeyPoints;
    mpMapPublisher->mvMatchIdx = mvKPMatchIdx;
    mpMapPublisher->mvMatchIdxGood = mvKPMatchIdxGood;
    mpMapPublisher->mAffineMatrix = mAffineMatrix.clone();

    char strMatches[64];
    if (mLastState > 0) {  // 正常情况和刚丢失情况
        mpMapPublisher->mReferenceImage = mpReferenceKF->mImage.clone();

        mpMapPublisher->mvReferenceKPs = mpReferenceKF->mvKeyPoints;
        std::snprintf(strMatches, 64, "F: %ld->%ld, KF: %ld(%ld)->%ld, D: %ld, M: %d/%d/%d/%d",
                      mCurrentFrame.id, mCurrentFrame.countObservations(), mpReferenceKF->id,
                      mpReferenceKF->mIdKF, mpReferenceKF->countObservations(),
                      mCurrentFrame.id - mpReferenceKF->id, mnMPsInline, mnKPMatchesGood,
                      mnKPsInline, mnKPMatches);
    } else {  // 丢失情况和刚完成重定位
        if (mpLoopKF) {
            mpMapPublisher->mReferenceImage = mpLoopKF->mImage.clone();
            mpMapPublisher->mvReferenceKPs = mpLoopKF->mvKeyPoints;
            std::snprintf(strMatches, 64, "F: %ld->%ld, LoopKF: %ld(%ld)->%ld, Score: %.3f, M: %d/%d/%d/%d",
                          mCurrentFrame.id, mCurrentFrame.countObservations(), mpLoopKF->id,
                          mpLoopKF->mIdKF, mpLoopKF->countObservations(), mLoopScore, mnMPsInline,
                          mnKPMatchesGood, mnKPsInline, mnKPMatches);
        } else {
            mpMapPublisher->mReferenceImage = Mat::zeros(mCurrentFrame.mImage.size(), CV_8UC1);
            mpMapPublisher->mvReferenceKPs.clear();
            std::snprintf(strMatches, 64, "F: %ld->%ld, Have no LoopKF!", mCurrentFrame.id,
                          mCurrentFrame.countObservations());
        }
    }

    mpMapPublisher->mImageText = string(strMatches);
    mpMapPublisher->mbUpdated = true;
}

bool TestTrack::detectIfLost(Frame& f, const Mat& Tcw_opt)
{
    Se2 Twb_opt;
    Twb_opt.fromCvSE3(Tcw_opt);
    Se2 Twb_ori = f.getTwb();
    Se2 dSe2 = Twb_opt - Twb_ori;
    const double dt = sqrt(dSe2.x * dSe2.x + dSe2.y * dSe2.y);
    const double da = abs(dSe2.theta);
    const double thl = Config::MaxLinearSpeed / Config::FPS;
    const double tha = Config::MaxAngularSpeed / Config::FPS;
    if (dt > thl) {
        fprintf(stderr, "[Track][Warni] #%ld\t 位姿优化结果相差太大: dt(%.2f) > MaxLinearSpeed(%.2f)\n",
                f.id, dt, thl);
        return true;
    }
    if (da > tha) {
        fprintf(stderr, "[Track][Warni] #%ld\t 位姿优化结果相差太大: da(%.2f) < MaxAngularSpeed(%.2f)\n",
                f.id, da, tha);
        return true;
    }
    return false;
}

Mat TestTrack::getAffineMatrix(const Se2& dOdo)
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

bool TestTrack::doRelocalization()
{
    if (mCurrentFrame.isNull())  // 图像挂掉的时候Frame会变成null.
        return false;
    if (mCurrentFrame.mTimeStamp < mLastFrame.mTimeStamp)
        return false;

    mnMPsNewAdded = mnMPsCandidate = mnKPMatchesGood = 0;
    mnKPsInline = mnMPsInline = mnMPsTracked = 0;

    updateFramePoseFromRef();

    fprintf(stderr, "[Track][Info ] #%ld-#%ld 正在进行重定位...\n", mCurrentFrame.id, mLastRefKFid);
    vector<PtrKeyFrame> vpKFsCandidate;
    const bool bDetected = detectLoopClose(&mCurrentFrame, vpKFsCandidate);
    if (bDetected) {
        const bool bVerified = verifyLoopClose(&mCurrentFrame, vpKFsCandidate);
        if (bVerified) {// TODO 要么全局优化, 要么先EPNP求解位姿初值再做局部优化.
            Mat optPose;
            const bool bOptimizePassed = optimizeLoopClose(&mCurrentFrame, optPose);
            if (bOptimizePassed) {
                mCurrentFrame.setPose(optPose);
                return true;
            }
        }
    }
    return false;
}

// 重定位后应该是纯视觉的优化. 利用pnp算法
bool TestTrack::optimizeLoopClose(Frame* frame, cv::Mat& optPose)
{
    SlamOptimizer optimizer;
    SlamLinearSolverDense* linearSolver = new SlamLinearSolverDense();
    SlamBlockSolver* blockSolver = new SlamBlockSolver(linearSolver);
    SlamAlgorithmLM* solver = new SlamAlgorithmLM(blockSolver);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);
    optimizer.verifyInformationMatrices(true);

    CamPara* camPara = addCamPara(optimizer, Config::Kcam, 0);

    mpMap->updateLocalGraph_new(optPose, 0, Config::MaxLocalFrameNum, Config::LocalFrameSearchRadius);
    const vector<PtrKeyFrame> vpLocalKFs = mpMap->getLocalKFs();
    const vector<PtrKeyFrame> vpLocalRefKFs = mpMap->getRefKFs();
    const vector<PtrMapPoint> vpLocalMPs = mpMap->getLocalMPs();
    for (size_t i = 0, iend = vpLocalKFs.size(); i < iend; ++i) {
        const PtrKeyFrame& pKFi = vpLocalKFs[i];
        assert(!pKFi->isNull());

        const int vertexIdKF = i;
        const bool fixed = true;

        const Se2 Twb = pKFi->getTwb();
        const g2o::SE2 pose(Twb.x, Twb.y, Twb.theta);
        addVertexSE2(optimizer, pose, vertexIdKF, fixed);
    }

    optPose = poseOptimize(frame);
    return true;
}

bool TestTrack::detectLoopClose(Frame* frame, vector<PtrKeyFrame>& vpKFsCandidate)
{
    bool bDetected = false;

    frame->computeBoW(mpORBvoc);
    const DBoW2::BowVector& BowVecCurr = frame->mBowVec;
    const double minScoreBest = 0.015 /*Config::MinScoreBest*/;  // 0.02

    vector<PtrKeyFrame> vpKFsAll = mpMap->getAllKFs();
    for_each(vpKFsAll.begin(), vpKFsAll.end(), [&](PtrKeyFrame& pKF) {
        if (!pKF->mbBowVecExist)
            pKF->computeBoW(mpORBvoc);
    });

    vector<PtrKeyFrame> vCans;
    vCans.reserve(vpKFsAll.size());

    double scoreBest = 0;
    PtrKeyFrame pKFBest = nullptr;
    for (int i = 0, iend = vpKFsAll.size(); i < iend; ++i) {
        const PtrKeyFrame& pKFi = vpKFsAll[i];
        const DBoW2::BowVector& BowVec = pKFi->mBowVec;

        const double score = mpORBvoc->score(BowVecCurr, BowVec);
        if (score >= minScoreBest) {
            vCans.emplace_back(pKFi);
            if (score > scoreBest) {
                scoreBest = score;
                pKFBest = pKFi;
            }
        }
    }

    if (pKFBest != nullptr) {
        mpLoopKF = pKFBest;
        mLoopScore = scoreBest;
        assert(mLoopScore >= minScoreBest);

        bDetected = true;
        fprintf(stderr, "[Track][Info ] #%ld 重定位ing, 检测到回环候选个数: %ld, "
                        "其中 bestKFid = #%ld, bestScore = %.3f >= %.3f, 等待验证!\n",
                frame->id, vCans.size(), mpLoopKF->id, mLoopScore, minScoreBest);
    } else {
        fprintf(stderr, "[Track][Warni] #%ld 重定位ing, 回环检测失败! 所有场景相似度太低! 最高得分仅为: %.3f\n",
                frame->id, scoreBest);
    }

    vpKFsCandidate.swap(vCans);
    return bDetected;
}

/**
 * 流程:
 *  1. 对每个候选回环帧做词袋匹配, 剔除外点;
 *  2. 关联每个候选得到的MP, 然后计算初始位姿(PNP);
 *  3. 利用初始位姿做KP匹配(只对得分最高的那一帧做), 再次关联MP, 进一步优化位姿.
 *
 * 回环验证通过的条件:
 *  1. 2D-2D: fundamental matrix test with RANSAC
 *  2. 3D-3D: PnP test with RANSAC
 *  3. At least 30 inliers
 *
 * TODO 回环验证中, 当前帧的初始位姿要用视觉信息来计算. 3D-2D + 2D-2D
 */
bool TestTrack::verifyLoopClose(Frame* frame, const vector<PtrKeyFrame>& vpKFsCandidate)
{
    assert(mpLoopKF != nullptr && !frame->isNull());
    assert(!vpKFsCandidate.empty());

    assert(mnKPMatches == 0);
    assert(mnKPsInline == 0);
    assert(mnMPsInline == 0);
    assert(mnMPsTracked == 0);
    assert(mnMPsNewAdded == 0);
    assert(mMPCandidates.empty());

    mKPMatchesLoop.clear();

    const int nMinKPMatch = Config::MinKPMatchNum;  // 30
    const int nMinMPMatch = Config::MinMPMatchNum;  // 15
    const float ratioMinMPMatch = Config::MinMPMatchRatio;  // 0.05
    const size_t nCandidate = vpKFsCandidate.size();

    // BoW匹配
    const bool bIfMatchMPOnly = true;
    for (size_t i = 0; i < nCandidate; ++i) {
        std::map<int, int> MPMatchesLoop;
        int nMPs = mpORBmatcher->SearchByBoW(static_cast<Frame*>(vpKFsCandidate[i].get()), frame,
                                             MPMatchesLoop, bIfMatchMPOnly);
        //? TODO remove outlier?
        if (nMPs > 0) {
            mnMPsTracked += nMPs;
            for (const auto& m : MPMatchesLoop) {
                const PtrMapPoint pMP = vpKFsCandidate[i]->getObservation(m.first);
                assert(!pMP || pMP->isNull());
                frame->setObservation(pMP, m.second);
            }
        }
    }
    assert(frame->countObservations() == mnMPsTracked);

    if (mnMPsTracked < nMinMPMatch) {
        fprintf(stderr, "[Track][Warni] #%ld 重定位ing, 回环验证失败! 与回环候选帧的总关联MP数过少: %d < %d\n",
                frame->id, mnMPsTracked, nMinMPMatch);
        return false;
    }

    // 再用warp匹配提高一下匹配点对
    const Se2 dOdo = frame->odom - mpLoopKF->odom;
    mAffineMatrix = getAffineMatrix(dOdo);  // 计算先验A
    mnKPMatches = mpORBmatcher->MatchByWindowWarp(*mpLoopKF, *frame, mAffineMatrix, mvKPMatchIdx, 20);
    //? TODO A不准怎么办?
    mnKPsInline = removeOutliers(mpLoopKF, frame, mvKPMatchIdx, mAffineMatrix);
    if (mnKPsInline < nMinKPMatch * 0.5) {
        fprintf(stderr, "[Track][Warni] #%ld 重定位ing, 回环验证失败! 与回环帧的KP匹配内点数(%d/%d)少于匹配阈值的一半!\n",
               frame->id, mnKPsInline, nMinKPMatch * 0.5);
        return false;
    }

    mnMPsInline = doTriangulate(mpLoopKF, frame);  //  对MP视差好的会关联
    const int nObs = frame->countObservations();
    const float ratio = nObs == 0 ? 0 : mnMPsInline * 1.f / nObs;
    fprintf(stderr, "[Track][Info ] #%ld-KF#%ld(Loop) 重定位-回环验证通过! 和回环帧的MP关联/KP内点/KP匹配数为: %d/%d/%d, "
                    "当前帧观测数/MP匹配率为: %d/%.2f (TODO)\n",
            frame->id, mpLoopKF->mIdKF, mnMPsInline, mnKPsInline, mnKPMatches, nObs, ratio);

    return true;
}

void TestTrack::startNewTrack()
{
    fprintf(stderr, "\n***** 连续丢失超过50帧! 清空地图从当前帧重新开始运行!! *****\n");
    mpMap->clear();
    mState = cvu::FIRST_FRAME;
    mpReferenceKF = nullptr;
    mpLoopKF = nullptr;
    mMPCandidates.clear();
    mvKPMatchIdx.clear();
    mvKPMatchIdxGood.clear();
    mnLostFrames = 0;
    mAffineMatrix = Mat::eye(2, 3, CV_64FC1);
}

//! LocalMap 的工作
void TestTrack::addNewKF(PtrKeyFrame& pKF, const map<size_t, MPCandidate>& MPCandidates)
{
    WorkTimer timer;

    mpNewKF = pKF;

    // 1.根据自己可视MP更新信息矩阵, 局部地图投影关联MP，并由MP候选生成新的MP, 关键函数!!!
    findCorresponds(MPCandidates);
    double t1 = timer.count();
    printf("[Track][Timer] #%ld(KF#%ld) L1.关联地图点总耗时: %.2fms\n", pKF->id, pKF->mIdKF, t1);

    // 2.更新局部地图里的共视关系
    timer.start();
    // mpNewKF->updateCovisibleGraph();
    mpMap->updateCovisibility(mpNewKF);

    PtrKeyFrame pKFLast = mpMap->getCurrentKF();
    assert(mpNewKF->mIdKF - pKFLast->mIdKF == 1);
    assert(pKFLast->mIdKF == mpReferenceKF->mIdKF);

    // 添加KF之间的位姿约束, 用于全局优化
    Mat measure;
    g2o::Matrix6d info;
    calcOdoConstraintCam(mpNewKF->odom - pKFLast->odom, measure, info);
    pKFLast->setOdoMeasureFrom(mpNewKF, measure, toCvMat6f(info));
    mpNewKF->setOdoMeasureTo(pKFLast, measure, toCvMat6f(info));

    double t2 = timer.count();
    printf("[Track][Timer] #%ld(KF#%ld) L2.更新共视关系耗时: %.2fms, 共获得%ld个共视KF, MP观测数: %ld\n",
           pKF->id, pKF->mIdKF, t2, pKF->countCovisibleKFs(), pKF->countObservations());

    // 3.将KF插入地图
    timer.start();
    mpMap->insertKF(mpNewKF);
    double t3 = timer.count();
    printf("[Track][Timer] #%ld(KF#%ld) L3.将KF插入地图, LocalMap的预处理总耗时: %.2fms\n",
           pKF->id, pKF->mIdKF, t1 + t2 + t3);

    // 4.更新局部地图
    timer.start();
    updateLocalGraph();
    double t4 = timer.count();
    printf("[Track][Timer] #%ld(KF#%ld) L4.第一次更新局部地图耗时: %.2fms\n", pKF->id, pKF->mIdKF, t4);

    if (mpNewKF->mIdKF < 2)
        return;

    // do local BA
#if DO_LOCAL_BA
    // 5.修剪冗余KF
//    timer.start();
//    pruneRedundantKFinMap();  // 会自动更新LocalMap
//    double t5 = timer.count();
//    printf("[Track][Timer] #%ld(KF#%ld) L5.修剪冗余KF耗时: %.2fms\n", pKF->id, pKF->mIdKF, t5);

    // 6.删除MP外点
//    timer.start();
//    removeOutlierChi2();
//    double t6 = timer.count();
//    printf("[Track][Timer] #%ld(KF#%ld) L6.删除MP外点耗时: %.2fms\n", pKF->id, pKF->mIdKF, t6);

    // 7.再一次更新局部地图
//    timer.start();
//    updateLocalGraph();
//    double t7 = timer.count();
//    printf("[Track][Timer] #%ld(KF#%ld) L7.第二次更新局部地图耗时: %.2fms\n", pKF->id, pKF->mIdKF, t7);

    // 8.局部图优化
    timer.start();
    localBA();
    double t8 = timer.count();
    printf("[Track][Timer] #%ld(KF#%ld) L8.局部BA耗时: %.2fms\n", pKF->id, pKF->mIdKF, t8);
#endif

    // do global BA
#if DO_GLOBAL_BA
    // 刚重定位成功不需要全局优化
    if (mLastState == cvu::LOST && mState > 0)
        return;

    bool bFeatGraphRenewed = false, bLoopCloseDetected = false, bLoopCloseVerified = false;

    timer.start();
    bFeatGraphRenewed = mpMap->updateFeatGraph(mpNewKF);
    double tg1 = timer.count();
    printf("[Track][Timer] #%ld(KF#%ld) G1.更新特征图耗时: %.2fms\n", pKF->id, pKF->mIdKF, tg1);

    timer.start();
    bLoopCloseDetected = detectLoopClose_Global(mpNewKF);
    double tg2 = timer.count();
    printf("[Track][Timer] #%ld(KF#%ld) G2.回环检测耗时: %.2fms\n", pKF->id, pKF->mIdKF, tg2);

    double tg3 = 0;
    if (bLoopCloseDetected) {
        timer.start();
        bLoopCloseVerified = verifyLoopClose_Global(mpNewKF);
        tg3 = timer.count();
        printf("[Track][Timer] #%ld(KF#%ld) G3.回环验证耗时: %.2fms\n", pKF->id, pKF->mIdKF, tg3);
    }

    double tg4 = 0;
    if (bLoopCloseVerified || bFeatGraphRenewed) {
        timer.start();
        globalBA();
        tg4 = timer.count();
        printf("[Track][Timer] #%ld(KF#%ld) G4.全局BA耗时: %.2fms\n", pKF->id, pKF->mIdKF, tg4);
    }
#endif
}

void TestTrack::findCorresponds(const map<size_t, MPCandidate>& MPCandidates)
{
    WorkTimer timer;

    PtrKeyFrame pRefKF = nullptr;
    if (MPCandidates.empty())
        pRefKF = mpMap->getCurrentKF();
    else
        pRefKF = (*MPCandidates.begin()).second.pKF;  // 回环成功refKF为LoopKF
    assert(pRefKF != nullptr);
    assert(mpNewKF->id > pRefKF->id);

    const Mat Tc1w = pRefKF->getPose();
    const Mat Tc2w = mpNewKF->getPose();

    // 1.为newKF在Track线程中添加的可视MP添加info. 可以保证视差都是好的
    int nAddInfo= 0;
    const int nObs = mpNewKF->countObservations();
    for (size_t i = 0, iend = mpNewKF->N; i < iend; ++i) {
        PtrMapPoint pMP = mpNewKF->getObservation(i);
        if (pMP) {
            assert(pMP->isGoodPrl());
            Point3f Pc1 = cvu::se3map(Tc1w, pMP->getPos());
            Eigen::Matrix3d xyzinfo1, xyzinfo2;
            calcSE3toXYZInfo(Pc1, Tc1w, Tc2w, xyzinfo1, xyzinfo2);
            mpNewKF->setObsAndInfo(pMP, i, xyzinfo2);
            pMP->addObservation(mpNewKF, i);
            nAddInfo++;
        }
    }
    double t1 = timer.count();
    printf("[Track][Info ] #%ld(KF#%ld) L1.1.关联地图点1/3, 可视MP添加信息矩阵数/KF可视MP数: %d/%d, 耗时: %.2fms\n",
           mpNewKF->id, mpNewKF->mIdKF, nAddInfo, nObs, t1);
    assert(nObs == mpNewKF->countObservations());

    // 2.局部地图中非newKF的MPs投影到newKF, 新投影的MP可能会把MP候选的坑占了
    //! NOTE 视差不好的不要投
    timer.start();
    const vector<PtrMapPoint> vLocalMPs = mpMap->getLocalMPs();
    const int nLocalMPs = vLocalMPs.size();
    int nProjLocalMPs = 0;
    if (nLocalMPs > 0) {
        vector<int> vMatchedIdxMPs;
        ORBmatcher matcher;
        int m = matcher.SearchByProjection(mpNewKF, vLocalMPs, vMatchedIdxMPs, 20, 1);
        if (m > 0) {
            for (int i = 0, iend = mpNewKF->N; i < iend; ++i) {
                if (vMatchedIdxMPs[i] < 0)  // vMatchedIdxMPs.size() = mpNewKF->N
                    continue;

                const PtrMapPoint& pMP = vLocalMPs[vMatchedIdxMPs[i]];
                if (!pMP || pMP->isNull())
                    continue;
                assert(pMP->isGoodPrl());  // 视差好的才投

                // 通过三角化验证一下投影匹配对不对
                Point3f Pw = cvu::triangulate(pMP->getMainMeasureProjection(), mpNewKF->mvKeyPoints[i].pt,
                                     Config::Kcam * pMP->getMainKF()->getPose().rowRange(0, 3),
                                     Config::Kcam * Tc2w.rowRange(0, 3));
                Point3f Pc2 = cvu::se3map(Tc2w, Pw);
                if (!Config::acceptDepth(Pc2.z))
                    continue;
                if (!pMP->acceptNewObserve(Pc2, mpNewKF->mvKeyPoints[i]))
                    continue;

                // 验证通过给newKF关联此MP.
                Eigen::Matrix3d infoOld, infoNew;
                calcSE3toXYZInfo(Pc2, Tc2w, pMP->getMainKF()->getPose(), infoNew, infoOld);
                mpNewKF->setObsAndInfo(pMP, i, infoNew);
                pMP->addObservation(mpNewKF, i);
                nProjLocalMPs++;
            }
            double t2 = timer.count();
            printf("[Track][Info ] #%ld(KF#%ld) L1.1.关联地图点2/3, "
                   "KF关联MP数/投影MP数/局部MP总数: %d/%d/%d, 耗时: %.2fms\n",
                   mpNewKF->id, mpNewKF->mIdKF, nProjLocalMPs, m, nLocalMPs, t2);
        } else {
            printf("[Track][Info ] #%ld(KF#%ld) L1.1.关联地图点2/3, 投影MP数为0, 局部MP总数为: %d\n",
                   mpNewKF->id, mpNewKF->mIdKF, nLocalMPs);
        }
    } else {
        printf("[Track][Info ] #%ld(KF#%ld) L1.1.关联地图点2/3, 当前局部地图对当前帧没有有效投影.\n",
               mpNewKF->id, mpNewKF->mIdKF);
    }

    // 3.处理所有的候选MP.(候选观测完全是新的MP)
    timer.start();
    int nAddNewMP = 0, nReplacedCands = 0;
    for (const auto& cand : MPCandidates) {
        const size_t idx1 = cand.second.kpIdx1;
        const size_t idx2 = cand.second.kpIdx2;
        const unsigned long frameId = cand.second.id2;

        assert(pRefKF == cand.second.pKF);
        assert(!pRefKF->hasObservationByIndex(idx1));

        // 局部地图投影到newKF中的MP, 如果把候选的坑占了, 则取消此候选.
        if (frameId == mpNewKF->id && mpNewKF->hasObservationByIndex(idx2)) {  // 只有可能是局部地图投影下来的
            PtrMapPoint pMP = mpNewKF->getObservation(idx2);
            Point3f Pc1 = cvu::se3map(Tc1w, pMP->getPos());
            Eigen::Matrix3d xyzinfo1, xyzinfo2;
            calcSE3toXYZInfo(Pc1, Tc1w, Tc2w, xyzinfo1, xyzinfo2);
            pRefKF->setObsAndInfo(pMP, idx1, xyzinfo1);
            pMP->addObservation(pRefKF, idx1);
            nReplacedCands++;
            continue;
        }

        Eigen::Matrix3d xyzinfo1, xyzinfo2;
        calcSE3toXYZInfo(cand.second.Pc1, Tc1w, cand.second.Tc2w, xyzinfo1, xyzinfo2);
        Point3f Pw = cvu::se3map(Tc1w, cand.second.Pc1);
        PtrMapPoint pNewMP = make_shared<MapPoint>(Pw, false);  // 候选的视差都是不好的

        pRefKF->setObsAndInfo(pNewMP, idx1, xyzinfo1);
        pNewMP->addObservation(pRefKF, idx1);
        if (frameId == mpNewKF->id) {
            assert(!mpNewKF->hasObservationByIndex(idx2));
            mpNewKF->setObsAndInfo(pNewMP, idx2, xyzinfo2);
            pNewMP->addObservation(mpNewKF, idx2);
        }
        mpMap->insertMP(pNewMP);
        nAddNewMP++;
    }
    double t3 = timer.count();
    printf("[Track][Info ] #%ld(KF#%ld) L1.1.关联地图点3/3, MP候选数/新增数/替换数/当前MP总数: "
           "%ld/%d/%d/%ld, 当前KF观测数: %ld, 耗时: %.2fms\n",
           mpNewKF->id, mpNewKF->mIdKF, MPCandidates.size(), nAddNewMP, nReplacedCands,
           mpMap->countMPs(), mpNewKF->countObservations(), t3);

    assert(nReplacedCands <= nProjLocalMPs);
    assert(nAddNewMP + nReplacedCands == MPCandidates.size());
}

void TestTrack::updateLocalGraph()
{
    mpMap->updateLocalGraph_new(mpReferenceKF->getPose(), Config::LocalFrameSearchLevel,
                                Config::MaxLocalFrameNum, Config::LocalFrameSearchRadius);
//    mpMap->updateLocalGraph(Config::LocalFrameSearchLevel, Config::MaxLocalFrameNum,
//                            Config::LocalFrameSearchRadius);
}

void TestTrack::pruneRedundantKFinMap()
{
    WorkTimer timer;

    int nPruned = mpMap->pruneRedundantKF();

    printf("[Track][Info ] #%ld(KF#%ld) L5.修剪冗余KF, 共修剪了%d帧KF, 耗时:%.2fms.\n",
           mpNewKF->id, mpNewKF->mIdKF, nPruned, timer.count());
}

void TestTrack::removeOutlierChi2()
{
    WorkTimer timer;

    if (mpMap->countMPs() == 0)
        return;

    SlamOptimizer optimizer;
    initOptimizer(optimizer);

    int nBadMP = mpMap->removeLocalOutlierMP(optimizer);

    printf("[Track][Info ] #%ld(KF#%ld) L6.移除离群点%d成功! 耗时: %.2fms\n", mpNewKF->id, mpNewKF->mIdKF,
           nBadMP, timer.count());
}

void TestTrack::localBA()
{
    WorkTimer timer;

    printf("[Track][Info ] #%ld(KF#%ld) L8.正在执行localBA()...\n", mpNewKF->id, mpNewKF->mIdKF);
    const Se2 Twb1 = mpNewKF->getTwb();

    SlamOptimizer optimizer;
    SlamLinearSolverCholmod* linearSolver = new SlamLinearSolverCholmod();
//    SlamLinearSolverCSparse* linearSolver = new SlamLinearSolverCSparse();
    SlamBlockSolver* blockSolver = new SlamBlockSolver(linearSolver);
    SlamAlgorithmLM* solver = new SlamAlgorithmLM(blockSolver);
    solver->setMaxTrialsAfterFailure(5);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    loadLocalGraph(optimizer);
    if (optimizer.edges().empty()) {
        fprintf(stderr, "[Track][Error] #%ld(KF#%ld) No MPs in graph, leaving localBA().\n",
                mpNewKF->id, mpNewKF->mIdKF);
        return;
    }
    double t1 = timer.count();

    const string g2oFile1 = Config::G2oResultsStorePath + to_string(mpNewKF->mIdKF) + "-local-1.g2o";
    optimizer.save(g2oFile1.c_str());

    timer.start();
    optimizer.initializeOptimization(0);
    optimizer.optimize(Config::LocalIterNum);
    double t2 = timer.count();

    if (solver->currentLambda() > 100.0) {
        cerr << "[Track][Error] current lambda too large " << solver->currentLambda()
             << " , reject optimized result!" << endl;
        //return;
    }

    timer.start();
    mpMap->optimizeLocalGraph(optimizer);  // 用优化后的结果更新KFs和MPs
    double t3 = timer.count();
    double t4 = t1 + t2 + t3;
    printf("[Track][Timer] #%ld(KF#%ld) L8.localBA()加载/优化/更新/总耗时: %.2f/%.2f/%.2f/%.2fms\n",
           mpNewKF->id, mpNewKF->mIdKF, t1, t2, t3, t4);

    const Se2 Twb2 = mpNewKF->getTwb();
    printf("[Track][Info ] #%ld(KF#%ld) L8.localBA()优化成功, 优化前后位姿为: [%.2f, %.2f, %.2f] ==> [%.2f, %.2f, %.2f]\n",
           mpNewKF->id, mpNewKF->mIdKF, Twb1.x, Twb1.y, Twb1.theta, Twb2.x, Twb2.y, Twb2.theta);

    const string g2oFile2 = Config::G2oResultsStorePath + to_string(mpNewKF->mIdKF) + "-local-2.g2o";
    optimizer.save(g2oFile2.c_str());
}

void TestTrack::loadLocalGraph(SlamOptimizer& optimizer) {
    const vector<PtrKeyFrame> vpLocalKFs = mpMap->getLocalKFs();
    const vector<PtrKeyFrame> vpLocalRefKFs = mpMap->getRefKFs();
    const vector<PtrMapPoint> vpLocalMPs = mpMap->getLocalMPs();

    CamPara* campr = addCamPara(optimizer, Config::Kcam, 0);

    const float delta = Config::ThHuber;
    const size_t nLocalKFs = vpLocalKFs.size();
    const size_t nRefKFs = vpLocalRefKFs.size();
    const size_t nLocalMPs = vpLocalMPs.size();
    const size_t maxVertexIdKF = nLocalKFs + nRefKFs;
    unsigned long minKFid = nRefKFs > 0 ? 0 : vpLocalKFs[0]->mIdKF;
    if (nRefKFs == 0)
        fprintf(stderr, "[Track][Debug] #%ld(KF#%ld) L8.localBA() RefKF.size = 0, minKFid = %ld\n",
               mpNewKF->id, mpNewKF->mIdKF, minKFid);

    // Vertices for LocalKFs
    for (size_t i = 0; i < nLocalKFs; ++i) {
        const PtrKeyFrame& pKFi = vpLocalKFs[i];
        assert(!pKFi->isNull());

        const int vertexIdKF = i;
        const bool fixed = (pKFi->mIdKF == minKFid) || (pKFi->mIdKF == 0);

        const Se2 Twb = pKFi->getTwb();
        const g2o::SE2 pose(Twb.x, Twb.y, Twb.theta);
        addVertexSE2(optimizer, pose, vertexIdKF, fixed);
    }

    // odo Edges for LocalKFs
    for (size_t i = 0; i < nLocalKFs; ++i) {
        const PtrKeyFrame& pKFi = vpLocalKFs[i];
        const PtrKeyFrame pKFj = pKFi->preOdomFromSelf.first;
        if (!pKFj || pKFj->isNull())
            continue;
        assert(pKFj->mIdKF > pKFi->mIdKF); // from是指自己的下一KF

        PreSE2 meas = pKFi->preOdomFromSelf.second;
        auto it = std::find(vpLocalKFs.begin(), vpLocalKFs.end(), pKFj);
        if (it == vpLocalKFs.end())
            continue;

        const int id1 = distance(vpLocalKFs.begin(), it);
        auto e = addEdgeSE2(optimizer, meas.meas, i, id1, meas.cov.inverse()*1e2);

//        e->computeError();
//        cout << "EdgeSE2 from " << pKFi->mIdKF << " to " << pKFj->mIdKF << " , chi2 = "
//             << e->chi2() << ", error = [" << e->error().transpose() << "]" << endl;
//        Se2 dodo = pKFj->odom - pKFi->odom;
//        cout << "OdomSe2 from " << pKFi->mIdKF << " to " << pKFj->mIdKF << ", dodo = "
//             << dodo << ", Edge measurement = [" << meas.meas.transpose() << "]" << endl;
    }

    // Vertices for LocalRefKFs
    for (size_t j = 0; j < nRefKFs; ++j) {
        const PtrKeyFrame& pKFj = vpLocalRefKFs[j];
        assert(!pKFj->isNull());

        const int vertexIdKF = j + nLocalKFs;

        const Se2 Twb = pKFj->getTwb();
        const g2o::SE2 pose(Twb.x, Twb.y, Twb.theta);
        addVertexSE2(optimizer, pose, vertexIdKF, true);
    }

    // Vertices and edges for LocalMPs
    for (size_t i = 0; i < nLocalMPs; ++i) {
        const PtrMapPoint& pMP = vpLocalMPs[i];
        assert(!pMP->isNull());

        const int vertexIdMP = i + maxVertexIdKF;
        const Eigen::Vector3d lw = toVector3d(pMP->getPos());
        addVertexSBAXYZ(optimizer, lw, vertexIdMP);

        const vector<PtrKeyFrame> pObsKFs = pMP->getObservations();
        for (auto j = pObsKFs.begin(), jend = pObsKFs.end(); j != jend; ++j) {
            const PtrKeyFrame pKFj = (*j);
            if (mpMap->checkAssociationErr(pKFj, pMP)) {
                fprintf(stderr, "[Track][Warni] localBA() 索引错误! For KF#%ld-%d and MP#%ld-%d\n",
                        pKFj->mIdKF, pKFj->getFeatureIndex(pMP), pMP->mId, pMP->getKPIndexInKF(pKFj));
                continue;
            }

            const int octave = pMP->getOctave(pKFj);
            const int ftrIdx = pMP->getKPIndexInKF(pKFj);
            const float Sigma2 = pKFj->mvLevelSigma2[octave];  // 单层时都是1.0
            const Eigen::Vector2d uv = toVector2d(pKFj->mvKeyPoints[ftrIdx].pt);

            // 针对当前MPi的某一个观测KFj, 如果KFj在图里(是一个顶点)则给它加上边
            int vertexIdKF = -1;
            auto it1 = std::find(vpLocalKFs.begin(), vpLocalKFs.end(), pKFj);
            if (it1 != vpLocalKFs.end()) {
                vertexIdKF = distance(vpLocalKFs.begin(), it1);
            } else {
                auto it2 = std::find(vpLocalRefKFs.begin(), vpLocalRefKFs.end(), pKFj);
                if (it2 != vpLocalRefKFs.end())
                    vertexIdKF = nLocalKFs + distance(vpLocalRefKFs.begin(), it2);
            }
            if (vertexIdKF == -1)
                continue;

            // compute covariance/information
            const Matrix2d Sigma_u = Eigen::Matrix2d::Identity() * Sigma2;
            const Vector3d lc = toVector3d(pKFj->getMPPoseInCamareFrame(ftrIdx));

            const double zc = lc(2);
            const double zc_inv = 1. / zc;
            const double zc_inv2 = zc_inv * zc_inv;
            const float& fx = Config::fx;
            g2o::Matrix23D J_lc;
            J_lc << fx * zc_inv, 0, -fx * lc(0) * zc_inv2, 0, fx * zc_inv, -fx * lc(1) * zc_inv2;
            const Matrix3d Rcw = toMatrix3d(pKFj->getPose().rowRange(0, 3).colRange(0, 3));
            const Se2 Twb = pKFj->getTwb();
            const Vector3d pi(Twb.x, Twb.y, 0);

            const g2o::Matrix23D J_MPi = J_lc * Rcw;

            const Matrix2d J_KFj_rotxy = (J_MPi * g2o::skew(lw - pi)).block<2, 2>(0, 0);
            const Vector2d J_z = -J_MPi.block<2, 1>(0, 2);
            const float Sigma_rotxy = 1.f / Config::PlaneMotionInfoXrot;
            const float Sigma_z = 1.f / Config::PlaneMotionInfoZ;
            const Matrix2d Sigma_all =
                Sigma_rotxy * J_KFj_rotxy * J_KFj_rotxy.transpose() + Sigma_z * J_z * J_z.transpose() + Sigma_u;

            addEdgeSE2XYZ(optimizer, uv, vertexIdKF, vertexIdMP, campr, toSE3Quat(Config::Tbc),
                          Sigma_all.inverse(), delta);
        }
    }

    printf("[Track][Timer] #%ld(KF#%ld) L8.localBA()规模: KFs/RefKFs/MPs = %ld/%ld/%ld, Edges = %ld\n",
           mpNewKF->id, mpNewKF->mIdKF, nLocalKFs, nRefKFs, nLocalMPs, optimizer.edges().size());
}

Mat TestTrack::poseOptimize(Frame* pFrame)
{
#define USE_SE2 0 // USE_SE2效果不好
#define REMOVE_OUTLIERS 0

    SlamOptimizer optimizer;
    SlamLinearSolverDense* linearSolver = new SlamLinearSolverDense();
    SlamBlockSolver* blockSolver = new SlamBlockSolver(linearSolver);
    SlamAlgorithmLM* solver = new SlamAlgorithmLM(blockSolver);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);
    optimizer.verifyInformationMatrices(true);

    CamPara* camPara = addCamPara(optimizer, Config::Kcam, 0);

    // 当前帧位姿节点(待优化变量)
    const Se2 Twb1 = pFrame->getTwb();

#if USE_SE2
    g2o::VertexSE2* v0 = addVertexSE2(optimizer, g2o::SE2(Twb1.x, Twb1.y, Twb1.theta), 0, false);
#else
    g2o::VertexSE3Expmap* v0 = new g2o::VertexSE3Expmap();
    v0->setEstimate(toSE3Quat(pFrame->getPose()));
    v0->setId(0);
    v0->setFixed(false);
    optimizer.addVertex(v0);

    // 设置平面运动约束(边)
    addEdgeSE3ExpmapPlaneConstraint(optimizer, toSE3Quat(pFrame->getPose()), 0, Config::Tbc);
#endif

#if USE_SE2
    vector<g2o::EdgeSE2XYZ*> vpEdges;
#else
    vector<EdgeSE3ProjectXYZOnlyPose*> vpEdges;
#endif
    const size_t N = pFrame->N;
    vpEdges.reserve(N);
    vector<size_t> vnIndexEdge;
    vnIndexEdge.reserve(N);

    // 设置MP观测约束(边)
    const float delta = Config::ThHuber;
    const vector<PtrMapPoint> vAllMPs = pFrame->getObservations(false);
    assert(vAllMPs.size() == N);
    int vertexIdMP = 1;
    for (size_t i = 0; i < N; ++i) {
        const PtrMapPoint& pMP = vAllMPs[i];
        if (!pMP || pMP->isNull())
            continue;

        pFrame->mvbMPOutlier[i] = false;  //! TODO

        const Eigen::Vector3d lw = toVector3d(pMP->getPos());
        const cv::KeyPoint& kpUn = pFrame->mvKeyPoints[i];
        const Eigen::Vector2d uv = toVector2d(kpUn.pt);
        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];

        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        rk->setDelta(delta);

#if USE_SE2
        // 固定MP
        g2o::VertexSBAPointXYZ* vi = addVertexSBAXYZ(optimizer, lw, vertexIdMP++, false, true);
        g2o::EdgeSE2XYZ* e = new g2o::EdgeSE2XYZ();
        e->setVertex(0, v0);
        e->setVertex(1, vi);
        e->setCameraParameter(camPara);
        e->setExtParameter(toSE3Quat(Config::Tbc));
#else
        EdgeSE3ProjectXYZOnlyPose* e = new EdgeSE3ProjectXYZOnlyPose();
        e->setVertex(0, v0);
        e->Xw = lw;
#endif
        e->setMeasurement(uv);
        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);
        e->setRobustKernel(rk);
        optimizer.addEdge(e);
        vpEdges.push_back(e);
        vnIndexEdge.push_back(i);
    }

    if (vpEdges.size() < 5) {
        return pFrame->getPose();
    }

#if REMOVE_OUTLIERS
    for (size_t it = 0; it < 4; it++) {
        optimizer.initializeOptimization(0);
        optimizer.optimize(10);

        bool bEdgeCured = false;
        for (size_t i = 0, iend = vpEdges.size(); i < iend; i++) {
            auto e = vpEdges[i];
            const size_t idx = vnIndexEdge[i];  // MP对应的KP索引

            if (!pFrame->mvbMPOutlier[idx]) {
                const float chi2 = e->chi2();

                if (chi2 > 5.991) {
                    pFrame->mvbMPOutlier[idx] = true;
                    e->setLevel(1);
                    bEdgeCured = true;
                } else {
                    pFrame->mvbMPOutlier[idx] = false;
                    e->setLevel(0);
                }
            } else {  //! TODO 如果在优化过程中被剔除外点, 新的优化结果可能又会使其变回内点?
                e->computeError();
                const float chi2 = e->chi2();
                if (chi2 > 5.991) {
                    pFrame->mvbMPOutlier[idx] = true;
                    e->setLevel(1);
                    bEdgeCured = true;
                } else {
                    pFrame->mvbMPOutlier[idx] = false;
                    e->setLevel(0);
                }
            }

            if (it == 2)
                e->setRobustKernel(0);
        }

        if (optimizer.edges().size() < 10 || !bEdgeCured)
            break;
    }
#else
    optimizer.initializeOptimization(0);
    optimizer.optimize(10);
#endif

    // Recover optimized pose and return number of inliers
#if USE_SE2
    const g2o::SE3Quat Tcw = g2o::SE2ToSE3(v0->estimate());
#else
    const g2o::SE3Quat Tcw = v0->estimate();
#endif

    Se2 Twb2;
    Twb2.fromCvSE3(toCvMat(Tcw));
    cout << "[Track][Info ] #" << pFrame->id << " 位姿优化更新前后为: " << Twb1 << " ==> " << Twb2
         << ", 可视MP数为: " << N << endl;

    return toCvMat(Tcw);
}

void TestTrack::globalBA()
{
    WorkTimer timer;

    printf("[Track][Info ] #%ld(KF#%ld) G4.正在执行globalBA()...\n", mpNewKF->id, mpNewKF->mIdKF);

    const vector<PtrKeyFrame> vAllKFs = mpMap->getAllKFs();

    SlamOptimizer optimizer;
    SlamLinearSolverCholmod* linearSolver = new SlamLinearSolverCholmod();
    SlamBlockSolver* blockSolver = new SlamBlockSolver(linearSolver);
    SlamAlgorithmLM* solver = new SlamAlgorithmLM(blockSolver);
    optimizer.setAlgorithm(solver);

    int SE3OffsetParaId = 0;
    addParaSE3Offset(optimizer, g2o::Isometry3D::Identity(), SE3OffsetParaId);

    unsigned long maxKFid = 0;

    // Add all KFs
    map<unsigned long, PtrKeyFrame> mapId2pKF;
    vector<g2o::EdgeSE3Prior*> vpEdgePlane;

    for (auto it = vAllKFs.begin(); it != vAllKFs.end(); ++it) {
        const PtrKeyFrame pKF = (*it);
        if (!pKF || pKF->isNull())
            continue;

        const Mat Twc = cvu::inv(pKF->getPose());
        const bool bIfFix = (pKF->mIdKF == 0);
        g2o::EdgeSE3Prior* pEdge = addVertexSE3AndEdgePlaneMotion(optimizer, toIsometry3D(Twc), pKF->mIdKF,
                                                           Config::Tbc, SE3OffsetParaId, bIfFix);
        vpEdgePlane.push_back(pEdge);

        mapId2pKF[pKF->mIdKF] = pKF;
        if (pKF->mIdKF > maxKFid)
            maxKFid = pKF->mIdKF;
    }

    // Add odometry based constraints
    int numOdoCnstr = 0;
    vector<g2o::EdgeSE3*> vpEdgeOdo;
    for (auto it = vAllKFs.begin(); it != vAllKFs.end(); ++it) {
        const PtrKeyFrame pKF = (*it);
        if (!pKF || pKF->isNull())
            continue;
        if (pKF->mOdoMeasureFrom.first == nullptr)
            continue;

        const g2o::Matrix6d info = toMatrix6d(pKF->mOdoMeasureFrom.second.info);
        g2o::EdgeSE3* pEdgeOdoTmp = addEdgeSE3(optimizer, toIsometry3D(pKF->mOdoMeasureFrom.second.measure),
                                               pKF->mIdKF, pKF->mOdoMeasureFrom.first->mIdKF, info);
        vpEdgeOdo.push_back(pEdgeOdoTmp);

        numOdoCnstr++;
    }

    // Add feature based constraints
    int numFtrCnstr = 0;
    vector<g2o::EdgeSE3*> vpEdgeFeat;
    for (auto it = vAllKFs.begin(); it != vAllKFs.end(); ++it) {
        const PtrKeyFrame ptrKFFrom = (*it);
        if (!ptrKFFrom || ptrKFFrom->isNull())
            continue;

        for (auto it2 = ptrKFFrom->mFtrMeasureFrom.begin(); it2 != ptrKFFrom->mFtrMeasureFrom.end();
             it2++) {
            const PtrKeyFrame ptrKFTo = (*it2).first;
            if (std::find(vAllKFs.begin(), vAllKFs.end(), ptrKFTo) == vAllKFs.end())
                continue;

            const Mat meas = (*it2).second.measure;
            const g2o::Matrix6d info = toMatrix6d((*it2).second.info);

            g2o::EdgeSE3* pEdgeFeatTmp =
                addEdgeSE3(optimizer, toIsometry3D(meas), ptrKFFrom->mIdKF, ptrKFTo->mIdKF, info);
            vpEdgeFeat.push_back(pEdgeFeatTmp);

            numFtrCnstr++;
        }
    }
    double t1 = timer.count();
    timer.start();

    optimizer.setVerbose(Config::GlobalVerbose);
    optimizer.initializeOptimization(0);
    optimizer.optimize(Config::GlobalIterNum);

    double t2 = timer.count();
    timer.start();

    // Update local graph KeyFrame poses
    for (auto it = vAllKFs.begin(), iend = vAllKFs.end(); it != iend; ++it) {
        PtrKeyFrame pKF = (*it);
        if (!pKF || pKF->isNull())
            continue;

        const Mat Twc = toCvMat(estimateVertexSE3(optimizer, pKF->mIdKF));
        pKF->setPose(cvu::inv(Twc));
    }

    // Update local graph MapPoint positions
    const vector<PtrMapPoint> vMPsAll = mpMap->getAllMPs();
    for (auto it = vMPsAll.begin(); it != vMPsAll.end(); ++it) {
        PtrMapPoint pMP = (*it);
        if (!pMP || pMP->isNull())
            continue;

        const PtrKeyFrame pKF = pMP->getMainKF();
        if (!pKF->hasObservationByPointer(pMP))
            continue;

        const Mat Twc = pKF->getPose().inv();
        const Mat Rwc = Twc.rowRange(0, 3).colRange(0, 3);
        const Mat twc = Twc.rowRange(0, 3).colRange(3, 4);
        const int idx = pKF->getFeatureIndex(pMP);
        const Point3f Pt3_MP_KF = pKF->getMPPoseInCamareFrame(idx);
        const Mat t3_MP_KF = (Mat_<float>(3, 1) << Pt3_MP_KF.x, Pt3_MP_KF.y, Pt3_MP_KF.z);
        const Mat t3_MP_w = Rwc * t3_MP_KF + twc;
        const Point3f Pt3_MP_w(t3_MP_w);
        pMP->setPos(Pt3_MP_w);
    }
    double t3 = timer.count();
    double t4 = t1 + t2 + t3;
    printf("[Track][Timer] #%ld(KF#%ld) G4.globalBA()加载/优化/更新/总耗时: %.2f/%.2f/%.2f/%.2fms\n",
           mpNewKF->id, mpNewKF->mIdKF, t1, t2, t3, t4);
}

bool TestTrack::detectLoopClose_Global(PtrKeyFrame& pKF)
{
    bool bDetected = false;

    pKF->computeBoW(mpORBvoc);
    const DBoW2::BowVector& BowVecCurr = pKF->mBowVec;
    const size_t minKFIdOffset = static_cast<size_t>(Config::MinKFidOffset);   // 25
    const double minScoreBest = /*0.015*/ Config::MinScoreBest;  // 0.02

    vector<PtrKeyFrame> vpKFsAll = mpMap->getAllKFs();
    for_each(vpKFsAll.begin(), vpKFsAll.end(), [&](PtrKeyFrame& pKFj) {
        if (!pKFj->mbBowVecExist)
            pKFj->computeBoW(mpORBvoc);
    });

    double scoreBest = 0;
    PtrKeyFrame pKFBest = nullptr;
    for (size_t i = 0, iend = vpKFsAll.size(); i < iend; ++i) {
        const PtrKeyFrame& pKFi = vpKFsAll[i];
        if (abs(pKFi->mIdKF - pKF->mIdKF) < minKFIdOffset)
            continue;

        const DBoW2::BowVector& BowVec = pKFi->mBowVec;
        const double score = mpORBvoc->score(BowVecCurr, BowVec);
        if (score > scoreBest) {
            scoreBest = score;
            pKFBest = pKFi;
        }
    }

    if (pKFBest != nullptr && scoreBest >= minScoreBest) {
        mpLoopKF = pKFBest;
        bDetected = true;
    } else {
        mpLoopKF.reset();
    }

    return bDetected;
}

bool TestTrack::verifyLoopClose_Global(PtrKeyFrame& pKF)
{
    assert(mpLoopKF != nullptr && !pKF->isNull());

    const int nMinMPMatch = Config::MinMPMatchNum;  // 15
    const int nMinKPMatch = Config::MinKPMatchNum;  // 30
    const float ratioMinMPMatch = Config::MinMPMatchRatio;  // 0.05

    // BoW匹配
    const bool bIfMatchMPOnly = false;
    std::map<int, int> KPMatchesLoop, MPMatchesLoop;
    int nKPMatches1 = mpORBmatcher->SearchByBoW(static_cast<Frame*>(mpLoopKF.get()), static_cast<Frame*>(pKF.get()),
                                                KPMatchesLoop, bIfMatchMPOnly);
    if (nKPMatches1 < nMinKPMatch) {
        fprintf(stderr, "[Track][Warni] #%ld-KF#%ld(Loop) 回环验证失败! 与回环帧的KP匹配数(%d)<阈值(%d)\n",
                pKF->id, mpLoopKF->mIdKF, nKPMatches1, nMinKPMatch);
        return false;
    }

    // 保留MP匹配
    GlobalMapper::removeKPMatch(pKF, mpLoopKF, KPMatchesLoop);
    MPMatchesLoop = KPMatchesLoop;

    // 匹配点数足够则剔除外点
    const Se2 dOdo = pKF->odom - mpLoopKF->odom;
    std::vector<int> vKPMatchIdx;
    Mat affine = getAffineMatrix(dOdo);  // 计算先验A
    int nKPMatches = mpORBmatcher->MatchByWindowWarp(*mpLoopKF, *pKF, affine, vKPMatchIdx, 20);
    if (nKPMatches < nKPMatches1) {  // 如果仿射变换的先验不准确, 则丢弃此匹配
        std::fill(vKPMatchIdx.begin(), vKPMatchIdx.end(), -1);
        for (const auto& m : KPMatchesLoop)
            vKPMatchIdx[m.first] = m.second;
        nKPMatches = nKPMatches1;
        affine = Mat::eye(2, 3, CV_64FC1);
    }

    int nKPsInline = removeOutliers(mpLoopKF, static_cast<Frame*>(pKF.get()), vKPMatchIdx, affine);
    if (nKPsInline < nMinMPMatch) {
        printf("[Track][Warni] #%ld-KF#%ld(Loop) 回环验证失败! 与回环帧的KP匹配内点数(%d)<阈值(%d)\n",
               pKF->id, mpLoopKF->mIdKF, nKPsInline, nMinMPMatch);
        return false;
    }

    const int nObs = pKF->countObservations();
    const float ratio = nObs * 1.f / nKPsInline;
    if (ratio < ratioMinMPMatch) {
        printf("[Track][Warni] #%ld-KF#%ld(Loop) 回环验证失败! 与回环帧的MP匹配率(%.2f)<阈值(%.2f)\n",
               pKF->id, mpLoopKF->mIdKF, ratio, ratioMinMPMatch);
        return false;
    }

    fprintf(stderr, "[Track][Info ] #%ld-KF#%ld(Loop) 回环验证通过! 和回环帧的KP匹配内点/KP匹配数为: %d/%d, "
                    "当前帧MP观测数与MP匹配率为: %d/%.2f\n",
            pKF->id, mpLoopKF->mIdKF, nKPsInline, nKPMatches, nObs, ratio);

    // 构建特征图
    SE3Constraint Se3_Curr_Loop;
    bool bFtrCnstrErr = GlobalMapper::createFeatEdge(pKF, mpLoopKF, MPMatchesLoop, Se3_Curr_Loop);
    if (!bFtrCnstrErr) {
        pKF->addFtrMeasureFrom(mpLoopKF, Se3_Curr_Loop.measure, Se3_Curr_Loop.info);
        mpLoopKF->addFtrMeasureTo(pKF, Se3_Curr_Loop.measure, Se3_Curr_Loop.info);
        fprintf(stderr, "[Track][Info ] #%ld-KF#%ld(Loop) 为回环帧添加了特征图约束!\n", pKF->id, mpLoopKF->mIdKF);
    }

    mpMap->mergeLoopClose(MPMatchesLoop, pKF, mpLoopKF);
    return true;
}

/*
int TestTrack::doTriangulate_Global(PtrKeyFrame& pKF, PtrKeyFrame& frame, vector<int>& vKPMatchIdx)
{
    if (vKPMatchIdx.empty())
        return 0;

    int n11 = 0, n121 = 0, n122 = 0, n21 = 0, n22 = 0;

    // 相机1和2的投影矩阵
    const Mat Tc1w = pKF->getPose();
    const Mat Tc2w = frame->getPose();
    const Mat Tcr = frame->getTcr();
    const Mat Tc1c2 = cvu::inv(Tcr);
    const Point3f Ocam1 = Point3f(0.f, 0.f, 0.f);
    const Point3f Ocam2 = Point3f(Tc1c2.rowRange(0, 3).col(3));
    const cv::Mat Proj1 = Config::PrjMtrxEye;  // P1 = K * cv::Mat::eye(3, 4, CV_32FC1)
    const cv::Mat Proj2 = Config::Kcam * Tcr.rowRange(0, 3);  // P2 = K * Tc2c1(3*4)

//    * 遍历回环帧的KP, 根据'vKPMatchIdx'对有匹配点对的KP进行处理, 如果:
//     * 1. 回环帧KP已经有对应的MP观测:
//     *  - 1.1 对于视差好的MP, 直接给当前帧关联一下MP观测;
//     *  - 1.2 对于视差不好的MP, 再一次三角化更新其坐标值. 如果更新后:
//     *      - 1.2.1 深度符合(理应保持)且视差好, 更新MP的属性;
//     *      - 1.2.2 深度符合但视差没有被更新为好, 更新MP坐标
//     *      - 1.2.3 深度被更新为不符合(出现概率不大), 则不处理.
//     * 2. 回环帧KP没有对应的MP观测:
//     *  - 三角化, 如果:
//     *      - 2.1 深度符合且视差好, 生成新的MP, 为KF和F添加MP观测, 为MP添加KF观测;
//     *      - 2.2 深度不符合, 则丢弃此匹配点对.
    for (size_t i = 0, iend = pKF->N; i < iend; ++i) {
        if (vKPMatchIdx[i] < 0)
            continue;

        PtrMapPoint pObservedMP = nullptr;  // 对于的MP观测
        const bool bObserved = pKF->hasObservationByIndex(i);  // 是否有对应MP观测
        if (bObserved) {
            pObservedMP = pKF->getObservation(i);
            if (pObservedMP->isGoodPrl()) {  // 情况1.1
                frame->setObservation(pObservedMP, vKPMatchIdx[i]);
                n11++;
                continue;
            }
        }

        // 如果参考帧KP没有对应的MP, 或者对应MP视差不好, 则为此匹配对KP三角化计算深度(相对参考帧的坐标)
        // 由于两个投影矩阵是两KF之间的相对投影, 故三角化得到的坐标是相对参考帧的坐标, 即Pc1
        const Point2f pt1 = pKF->mvKeyPoints[i].pt;
        const Point2f pt2 = frame->mvKeyPoints[vKPMatchIdx[i]].pt;
        const Point3f Pc1 = cvu::triangulate(pt1, pt2, Proj1, Proj2);
        const Point3f Pw = cvu::se3map(cvu::inv(Tc1w), Pc1);

        // Pc2用Tcr计算出来的是预测值, 故这里用Pc1的深度判断即可
        const bool bAccepDepth = Config::acceptDepth(Pc1.z);  // 深度是否符合
        if (bAccepDepth) {  // 如果深度计算符合预期
            const bool bGoodPrl = cvu::checkParallax(Ocam1, Ocam2, Pc1, 2);  // 视差是否良好
            if (bGoodPrl) {  // 如果视差好
                Eigen::Matrix3d xyzinfo1, xyzinfo2;
                calcSE3toXYZInfo(Pc1, Tc1w, Tc2w, xyzinfo1, xyzinfo2);
                if (bObserved) {  // 情况1.2.1
                    pObservedMP->setPos(Pw);
                    pObservedMP->setGoodPrl(true);
                    pKF->setObsAndInfo(pObservedMP, i, xyzinfo1);
                    frame->setObservation(pObservedMP, vKPMatchIdx[i]);
                    n121++;
                } else {  // 情况2.1
                    PtrMapPoint pNewMP = make_shared<MapPoint>(Pw, true);
                    pKF->setObsAndInfo(pNewMP, i, xyzinfo1);
                    pNewMP->addObservation(pKF, i);  // MP只添加KF的观测
                    frame->setObservation(pNewMP, vKPMatchIdx[i]);
                    mpMap->insertMP(pNewMP);
                    n21++;
                }
            } else {  // 如果视差不好
                if (bObserved) {// 情况1.2.2, 更新Pw
                    pObservedMP->setPos(Pw);
                    n122++;
                }
            }
        } else {  // 如果深度计算不符合预期
            if (!bObserved) {  // 情况2.2
                n22++;
                vKPMatchIdx[i] = -1;
            }
            // 情况1.2.3不处理
        }
    }

    return n11 + n121 + n21;  // nMPsInline
}
*/

}  // namespace se2lam
