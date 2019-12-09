#include "TestTrack.h"
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


TestTrack::TestTrack()
    : mState(cvu::NO_READY_YET), mLastState(cvu::NO_READY_YET), mpReferenceKF(nullptr),
      mpNewKF(nullptr), mpLoopKF(nullptr), mnMPsCandidate(0), mnKPMatches(0), mnKPsInline(0),
      mnKPMatchesGood(0), mnMPsTracked(0), mnMPsNewAdded(0), mnMPsInline(0), mnLostFrames(0), mLoopScore(0)
{
    mpORBextractor = new ORBextractor(Config::MaxFtrNumber, Config::ScaleFactor, Config::MaxLevel);
    mpORBmatcher = new ORBmatcher(0.9, true);
    mpORBvoc = new ORBVocabulary();
    string strVocFile = Config::DataPath + "../se2_config/ORBvoc.bin";
    bool bVocLoad = mpORBvoc->loadFromBinaryFile(strVocFile);
    if (!bVocLoad)
        cerr << "[ERRO] Wrong path to vocabulary, Falied to open it." << endl;

    nMinFrames = min(2, cvCeil(0.25 * Config::FPS));  // 上溢
    nMaxFrames = cvFloor(5 * Config::FPS);  // 下溢
    nMinMatches = std::min(cvFloor(0.1 * Config::MaxFtrNumber), 40);
    mMaxAngle = static_cast<float>(g2o::deg2rad(50.));
    mMaxDistance = 0.3f * Config::UpperDepth;

    mCurrRatioGoodDepth = mCurrRatioGoodParl = 0;
    mLastRatioGoodDepth = mLastRatioGoodParl = 0;

    mAffineMatrix = Mat::eye(2, 3, CV_64FC1);  // double

    fprintf(stderr, "[INFO] 相关参数如下: \n - 最小/最大KF帧数: %d/%d\n"
                    " - 最大移动距离/角度: %.0fmm/%.0fdeg\n - 最少匹配数量: %d\n",
            nMinFrames, nMaxFrames, mMaxDistance, g2o::rad2deg(mMaxAngle), nMinMatches);
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
        } else if (mState == cvu::OK) {
            bOK = trackReferenceKF();
            //if (!bOK)
            //    bOK = trackLocalMap();  // 刚丢的还可以再抢救一下
        } else if (mState == cvu::LOST) {
            // FIXME 回环通过产生新的KF的MP观测数太少!
            bOK = doRelocalization();  // 没追上的直接检测回环重定位
        }
        t2 = timer.count();
    }

    if (bOK) {
        // TODO 更新一下MPCandidates里面Tc2w
        mnLostFrames = 0;
        mState = cvu::OK;
    } else {
        mnLostFrames++;
        mState = cvu::LOST;
    }

    //copyForPub();
    mLastRefKFid = mpReferenceKF->id;  // cur更新成ref后ref的id就和cur相等了, 这里仅用于输出log
    resetLocalTrack();  // KF判断在这里

    t3 = t1 + t2;
    trackTimeTatal += t3;
    printf("[TIME] #%ld-#%ld 前端构建Frame/追踪/总耗时为: %.2f/%.2f/%.2fms, 平均耗时: %.2fms\n",
           mCurrentFrame.id, mLastRefKFid, t1, t2, t3,  trackTimeTatal * 1.f / mCurrentFrame.id);
}

void TestTrack::processFirstFrame()
{
    size_t th = Config::MaxFtrNumber;
    if (mCurrentFrame.N > (th >> 1)) {  // 首帧特征点需要超过最大点数的一半
        cout << "========================================================" << endl;
        cout << "[INFO] Create first frame with " << mCurrentFrame.N << " features. "
             << "And the start odom is: " << mCurrentFrame.odom << endl;
        cout << "========================================================" << endl;

        mCurrentFrame.setPose(Config::Tcb);
        mCurrentFrame.setTwb(Se2(0, 0, 0));
        mpReferenceKF = make_shared<KeyFrame>(mCurrentFrame);  // 首帧为关键帧
        mpMap->insertKF(mpReferenceKF);  // 首帧的KF直接给Map, 没有给LocalMapper

        mLastFrame = mCurrentFrame;
        mState = cvu::OK;
    } else {
        cerr << "[WARN] Failed to create first frame for too less keyPoints: "
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
        fprintf(stderr, "[WARN] #%ld-#%ld 图像序列在时间上不连续: Last = %.3f, Curr = %.3f\n",
                mCurrentFrame.id, mLastFrame.id, mLastFrame.mTimeStamp, mCurrentFrame.mTimeStamp);
        return false;
    }

    assert(mnKPsInline == 0);

    //! 1.根据里程计设置初始位姿
    updateFramePoseFromRef();

    //! 2.基于等距变换先验估计投影Cell的位置
    mnKPMatches = mpORBmatcher->MatchByWindowWarp(*mpReferenceKF, mCurrentFrame, mAffineMatrix,
                                                  mvKPMatchIdx, 25);
    if (mnKPMatches < 15) {
        printf("[WARN] #%ld-#%ld 与参考帧匹配[总]点数少于15(%d), 即将转为重定位!\n",
               mCurrentFrame.id, mpReferenceKF->id, mnKPMatches);
        return false;
    }

    //! 3.利用仿射矩阵A计算KP匹配的内点，内点数大于10才能继续
    mnKPsInline = removeOutliers(mpReferenceKF, &mCurrentFrame, mvKPMatchIdx, mAffineMatrix);
    if (mnKPsInline < 10) {
        printf("[WARN] #%ld-#%ld 与参考帧匹配[内]点数少于10(%d/%d), 即将转为重定位!\n",
               mCurrentFrame.id, mpReferenceKF->id, mnKPsInline, mnKPMatches);
        return false;
    }

    //! 4.三角化生成潜在MP, 由LocalMap线程创造MP
    doTriangulate(mpReferenceKF);  // 更新 mnTrackedOld, mnGoodInliers, mvGoodMatchIdx

    //! TODO FIXME 5.最小化重投影误差优化当前帧位姿
    const int nObs = mCurrentFrame.countObservations();
    if (0 && mnMPsTracked > 0 && nObs > 0) {  // 保证有MP观测才做优化 // TODO
        Se2 Twb1 = mCurrentFrame.getTwb();

        int nCros = 0;
        double projError = 10000.;
        poseOptimization(&mCurrentFrame, nCros, projError);  // 更新 mPose, mvbOutlier
        // mCurrentFrame.updateObservationsAfterOpt();  // 根据 mvbOutlier 更新Frame的观测情况

        Se2 Twb2 = mCurrentFrame.getTwb();
        printf("[INFO] #%ld-#%ld VO优化位姿更新前后的值为: [%.2f, %.2f, %.2f] ==> [%.2f, %.2f, %.2f]\n",
               mCurrentFrame.id, mpReferenceKF->id, Twb1.x, Twb1.y, Twb1.theta, Twb2.x, Twb2.y, Twb2.theta);
        printf("[WARN] #%ld-#%ld 位姿优化情况: 可视MP数/MP内点数/重投影误差为: %d/%d/%.3f,\n",
               mCurrentFrame.id, mpReferenceKF->id, nObs, nCros, projError);

//        bool lost = detectIfLost(nCros, projError);
//        if (lost) {
//            printf("[WARN] #%ld-#%ld 由MP优化位姿失败! MP内点数/重投影误差为: %d/%.3f, 即将转为重定位!\n",
//                   mCurrentFrame.id, mpReferenceKF->id, nCros, projError);
//            return false;
//        }
    }

    return true;
}

bool TestTrack::trackLocalMap()
{
    printf("[TIME] #%ld-#%ld 正在从局部地图中计算当前帧位姿..\n", mCurrentFrame.id, mLastRefKFid);
    const vector<PtrMapPoint> vpLocalMPs = mpMap->getLocalMPs();
    vector<int> vMatchedIdxMPs;  // 一定是新投上的点
    int nProj = mpORBmatcher->SearchByProjection(mCurrentFrame, vpLocalMPs, vMatchedIdxMPs, 20, 1);
    if (nProj > 0) {
        assert(vMatchedIdxMPs.size() == mCurrentFrame.N);
        for (size_t i = 0, iend = mCurrentFrame.N; i < iend; ++i) {
            if (vMatchedIdxMPs[i] < 0)
                continue;
            const PtrMapPoint& pMP = vpLocalMPs[vMatchedIdxMPs[i]];  // 新匹配上的MP
            mCurrentFrame.setObservation(pMP, i);  // TODO 要不要管 mvbOutlier?
        }
        int nCros = 0;
        double projError = 10000.;
        poseOptimization(&mCurrentFrame, nCros, projError);
        // doLocalBA(mCurrentFrame);
        return !detectIfLost(nCros, projError);
    } else {
        return false;
    }
}

void TestTrack::updateFramePoseFromRef()
{
    const Se2 Tb1b2 = mCurrentFrame.odom - mpReferenceKF->odom;
    const Mat Tc2c1 = Config::Tcb * Tb1b2.inv().toCvSE3() * Config::Tbc;
    mCurrentFrame.setTrb(Tb1b2);
    mCurrentFrame.setTcr(Tc2c1);
    mCurrentFrame.setPose(Tc2c1 * mpReferenceKF->getPose()); // 初始位姿用odo进行更新, 这里是位姿的预测值
    mCurrentFrame.setTwb(mpReferenceKF->getTwb() + Tb1b2);   // Twb是位姿的测量值
    //mCurrentFrame.setPose(Config::Tcb * (mpReferenceKF->getTwb() + Tb1b2).inv().toCvSE3());

    // Eigen::Map 是一个引用, 这里更新了到当前帧的积分
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
}

void TestTrack::doTriangulate(PtrKeyFrame& pKF)
{
    if (mvKPMatchIdx.empty())
        return;

    // 以下成员变量在这个函数中会被修改
    mnMPsCandidate = mMPCandidates.size();
    mvKPMatchIdxGood = mvKPMatchIdx;
    mnMPsInline = 0;
    mnMPsTracked = 0;
    mnMPsNewAdded = 0;
    mnKPMatchesGood = 0;
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

        mnKPMatchesGood++;
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
                mnKPMatchesGood--;
                mvKPMatchIdxGood[i] = -1;
            }
            // 情况1.2.3和2.3不处理
        }
    }
    printf("[INFO] #%ld-#%ld KP匹配结果: KP好匹配数/总内点数/总匹配数: %d/%d/%d, "
           "MP总观测数(Ref)/关联数/视差好的/更新到好的: %d/%d/%d/%d\n", mCurrentFrame.id, pKF->id,
           mnKPMatchesGood, mnKPsInline, mnKPMatches, nObsRef, mnMPsTracked, n11, n121);
    printf("[INFO] #%ld-#%ld 三角化结果: 候选MP原总数/转正数/更新数/增加数/现总数: %d/%d/%d/%d/%d, "
           "三角化新增MP数/新增候选数/剔除匹配数: %d/%d/%d, 新生成MP数: %d\n",
           mCurrentFrame.id, pKF->id, n2, n21, n22, n32, mnMPsCandidate, n31, n32, n33, mnMPsNewAdded);

    // KF判定依据变量更新 FIXME
    mCurrRatioGoodDepth = mnKPsInline > 0 ? mnMPsCandidate / mnKPsInline : 0;
    mCurrRatioGoodParl = mnMPsCandidate > 0 ? mnMPsNewAdded / mnMPsCandidate : 0;

    assert(n11 + n121 == mnMPsTracked);
    assert(n21 + n31 == mnMPsNewAdded);
    assert(n33 + mnKPMatchesGood == mnKPsInline);
    assert(mnMPsTracked + mnMPsNewAdded == mnMPsInline);
    assert((n2 - n21 + n32 == mnMPsCandidate) && (mnMPsCandidate == mMPCandidates.size()));
    assert(nObsCur + mnMPsTracked + mnMPsNewAdded == mCurrentFrame.countObservations());
    assert(nObsRef + mnMPsNewAdded == pKF->countObservations());
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

void TestTrack::resetLocalTrack()
{
    // 正常状态进行关键帧判断
    if (mState == cvu::OK) {
        if (needNewKF()) {
            PtrKeyFrame pNewKF = make_shared<KeyFrame>(mCurrentFrame);
            mpReferenceKF->preOdomFromSelf = make_pair(pNewKF, preSE2);
            pNewKF->preOdomToSelf = make_pair(mpReferenceKF, preSE2);

            // 重定位成功将当前帧变成KF
            // TODO 重定位的预积分信息怎么解决??
            if (mState == cvu::OK && mLastState == cvu::LOST) {
                pNewKF->addCovisibleKF(mpLoopKF);
                mpLoopKF->addCovisibleKF(pNewKF);
                printf("[INFO] #%ld 成为了新的KF(#%ld), 因为刚刚重定位成功!\n", pNewKF->id,
                       pNewKF->mIdKF);

                //! FIXME 重定位成功(回环验证通过)构建特征图和约束
                //! NOTE mKPMatchesLoop 必须是有MP的匹配!
            }

            addNewKF(pNewKF, mMPCandidates);  // 新的KF观测和共视关系在LocalMap里更新

            copyForPub();
            mpReferenceKF = pNewKF;
            mAffineMatrix = Mat::eye(2, 3, CV_64FC1);
            mMPCandidates.clear();  // 加了新的KF后才清空. 和普通帧的匹配也会生成MP
            mnMPsCandidate = 0;
            mLastRatioGoodDepth = 0.;
            mLastRatioGoodParl = 0.;
        } else {
            copyForPub();
            mLastRatioGoodDepth = mCurrRatioGoodDepth;
            mLastRatioGoodParl = mCurrRatioGoodParl;
        }
        mLastFrame = mCurrentFrame;
        mnKPMatches = 0;
        mnKPsInline = 0;
        mnKPMatchesGood = 0;
        mnMPsTracked = 0;
        mnMPsNewAdded = 0;
        mnMPsInline = 0;
        for(int i = 0; i < 3; i++)
            preSE2.meas[i] = 0;
        for(int i = 0; i < 9; i++)
            preSE2.cov[i] = 0;
        return;
    }

    // 当前帧刚刚丢失要确保上一帧是最新的KF
    assert(mState == cvu::LOST);
    if (mState == cvu::LOST && mLastState == cvu::OK && !mLastFrame.isNull() &&
        mLastFrame.id != mpReferenceKF->id) {
        printf("[INFO] #%ld-#%ld 上一帧(#%ld)成为了新的KF(#%ld)! 因为当前帧刚刚丢失!\n",
               mCurrentFrame.id, mpReferenceKF->id, mLastFrame.id, KeyFrame::mNextIdKF);

        PtrKeyFrame pNewKF = make_shared<KeyFrame>(mLastFrame);
        //! TODO 这里要扣掉当前帧的预积分信息
        mpReferenceKF->preOdomFromSelf = make_pair(pNewKF, preSE2);
        pNewKF->preOdomToSelf = make_pair(mpReferenceKF, preSE2);
        addNewKF(pNewKF, mMPCandidates);

        copyForPub();
        mpReferenceKF = pNewKF;
        mMPCandidates.clear();
        mLastFrame = mCurrentFrame;
        mnKPMatches = 0;
        mnKPsInline = 0;
        mnKPMatchesGood = 0;
        mnMPsTracked = 0;
        mnMPsNewAdded = 0;
        mnMPsInline = 0;
        mLastRatioGoodDepth = 0.f;
        mLastRatioGoodParl = 0.f;
        for(int i = 0; i < 3; i++)
            preSE2.meas[i] = 0;
        for(int i = 0; i < 9; i++)
            preSE2.cov[i] = 0;
        return;
    }

    // 处于丢失的状态则直接交换前后帧数据
    copyForPub();
    mMPCandidates.clear();
    mLastFrame = mCurrentFrame;
    mnKPMatches = 0;
    mnKPsInline = 0;
    mnKPMatchesGood = 0;
    mnMPsTracked = 0;
    mnMPsNewAdded = 0;
    mnMPsInline = 0;
    mLastRatioGoodDepth = 0.f;
    mLastRatioGoodParl = 0.f;
    for(int i = 0; i < 3; i++)
        preSE2.meas[i] = 0;
    for(int i = 0; i < 9; i++)
        preSE2.cov[i] = 0;
}

bool TestTrack::needNewKF()
{
    // 刚重定位成功需要建立新的KF
    if (mState == cvu::OK && mLastState == cvu::LOST) {
        printf("[INFO] #%ld 刚重定位成功! 需要加入新的KF! \n", mCurrentFrame.id);
        return true;
    }

    int deltaFrames = static_cast<int>(mCurrentFrame.id - mpReferenceKF->id);

    // 必要条件
    bool c0 = deltaFrames >= nMinFrames;  // 下限

    // 充分条件
    bool c1 = mnKPMatches < 50 && (mnKPsInline < 30 || mnKPMatchesGood < 20); // 和参考帧匹配的内点数太少. 这里顺序很重要! 30/20/15
    bool c2 = mnMPsCandidate > 50;  // 候选MP多且新增MP少(远)已经新增的MP数够多
    bool c3 = (mCurrRatioGoodDepth < mLastRatioGoodDepth) && (mCurrRatioGoodParl < mLastRatioGoodParl);
    bool bNeedKFByVO = c0 && (c1 || c2 || c3);

    //! 1.跟踪要跪了必须要加入新的KF
    if (bNeedKFByVO) {
        printf("[INFO] #%ld-#%ld 成为了新的KF(#%ld), 其KF条件满足情况: 内点少(%d)/潜在或已增多(%d)/追踪要丢(%d)\n",
               mCurrentFrame.id, mpReferenceKF->id, KeyFrame::mNextIdKF, c1, c2, c3);
        return true;
    }

    // 上限/旋转/平移
    bool c4 = deltaFrames > nMaxFrames;  // 上限
    Se2 dOdo = mCurrentFrame.odom - mpReferenceKF->odom;
    bool c5 = static_cast<double>(abs(dOdo.theta)) >= mMaxAngle;  // 旋转量超过50°
    cv::Mat cTc = Config::Tcb * dOdo.toCvSE3() * Config::Tbc;
    cv::Mat xy = cTc.rowRange(0, 2).col(3);
    bool c6 = cv::norm(xy) >= mMaxDistance;  // 相机的平移量足够大
    bool bNeedKFByOdo = c4 || (c5 || c6);  // 相机移动取决于深度上限,考虑了不同深度下视野的不同

    //! 2.如果跟踪效果还可以, 就看旋转平移条件
    if (bNeedKFByOdo) {
        printf("[INFO] #%ld-#%ld 成为了新的KF(#%ld), 其KF条件满足情况: 达上限(%d)/大旋转(%d)/大平移(%d)\n",
               mCurrentFrame.id, mpReferenceKF->id, KeyFrame::mNextIdKF, c4, c5, c6);
        return true;
    }

    return false;
}

void TestTrack::copyForPub()
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

bool TestTrack::detectIfLost(int nCros, double projError)
{
    const int th = mCurrentFrame.countObservations() * 0.6;
    if (nCros < th)
        return true;
    if (projError > 1000.)  // TODO 阈值待调整
        return true;

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

    fprintf(stderr, "[INFO] #%ld 正在进行重定位...\n", mCurrentFrame.id);
    bool bDetected = detectLoopClose();
    bool bVerified = false;
    if (bDetected) {
        bVerified = verifyLoopClose();
        if (bVerified) {
            // Get Local Map and do local BA
            // trackLocalMap();  // FIEME

            // setPose
            // mCurrentFrame.setPose(mpLoopKF->getTwb());
            return true;
        } else {
            return false;
        }
    } else {
        return false;
    }
}

bool TestTrack::detectLoopClose()
{
    bool bDetected = false;

    mCurrentFrame.computeBoW(mpORBvoc);
    const DBoW2::BowVector& BowVecCurr = mCurrentFrame.mBowVec;
    const double minScoreBest = 0.015 /*Config::MinScoreBest*/;  // 0.02

    vector<PtrKeyFrame> vpKFsAll = mpMap->getAllKFs();
    for_each(vpKFsAll.begin(), vpKFsAll.end(), [&](PtrKeyFrame& pKF) {
        if (!pKF->mbBowVecExist)
            pKF->computeBoW(mpORBvoc);
    });

    double scoreBest = 0;
    PtrKeyFrame pKFBest = nullptr;
    for (int i = 0, iend = vpKFsAll.size(); i < iend; ++i) {
        const PtrKeyFrame& pKFi = vpKFsAll[i];
        const DBoW2::BowVector& BowVec = pKFi->mBowVec;

        const double score = mpORBvoc->score(BowVecCurr, BowVec);
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
            fprintf(stderr, "[INFO] #%ld-KF#%ld(Loop) 重定位-检测到回环#%ld! score = %.3f >= "
                            "%.3f, 等待验证!\n",
                    mCurrentFrame.id, pKFBest->mIdKF, pKFBest->id, mLoopScore, minScoreBest);
        }
    } else {
        fprintf(stderr, "[WARN] #%ld 重定位-回环检测失败! 所有的KF场景相识度都太低! 最高得分仅为: %.3f\n",
                mCurrentFrame.id, scoreBest);
    }

    return bDetected;
}

bool TestTrack::verifyLoopClose()
{
    assert(mpLoopKF != nullptr && !mCurrentFrame.isNull());

    const int nMinKPMatch = Config::MinKPMatchNum;

    mKPMatchesLoop.clear();
    assert(mnKPMatches == 0);
    assert(mnKPsInline == 0);
    assert(mnMPsInline == 0);
    assert(mnMPsTracked == 0);
    assert(mnMPsNewAdded == 0);

    // BoW匹配
    bool bIfMatchMPOnly = false;  // 要看总体匹配点数多不多
    mnKPMatches = mpORBmatcher->SearchByBoW(&(*mpLoopKF), &mCurrentFrame, mKPMatchesLoop, bIfMatchMPOnly);
    if (mnKPMatches < nMinKPMatch * 0.6) {
        fprintf(stderr, "[WARN] #%ld-KF#%ld(Loop) 重定位-回环验证失败! 与回环帧的KP匹配数过少: %d < %d\n",
                mCurrentFrame.id, mpLoopKF->mIdKF, mnKPMatches, int(nMinKPMatch * 0.6));
        return false;
    }

    // 匹配点数足够则剔除外点
    Se2 dOdo = mCurrentFrame.odom - mpLoopKF->odom;
    mAffineMatrix = getAffineMatrix(dOdo);  // 计算先验A
    mnKPMatches = mpORBmatcher->MatchByWindowWarp(*mpLoopKF, mCurrentFrame, mAffineMatrix, mvKPMatchIdx, 20);
    if (mnKPMatches < nMinKPMatch) {
        printf("[WARN] #%ld-KF#%ld(Loop) 重定位-回环验证失败! 与回环帧的Warp KP匹配数过少: %d < %d\n",
               mCurrentFrame.id, mpLoopKF->mIdKF, mnKPMatches, nMinKPMatch);
        return false;
    }

    mnKPsInline = removeOutliers(mpLoopKF, &mCurrentFrame, mvKPMatchIdx, mAffineMatrix);
    if (mnKPsInline < 10) {
        printf("[WARN] #%ld-KF#%ld(Loop) 重定位-回环验证失败! 与回环帧的Warp KP匹配内点数少于10(%d/%d)!\n",
               mCurrentFrame.id, mpLoopKF->mIdKF, mnKPsInline, mnKPMatches);
        return false;
    }

    assert(mMPCandidates.empty());
    doTriangulate(mpLoopKF);  //  对MP视差好的会关联
    const int nObs = mCurrentFrame.countObservations();
    const float ratio = nObs == 0 ? 0 : mnMPsInline * 1.f / nObs;
    fprintf(stderr, "[INFO] #%ld-KF#%ld(Loop) 重定位-回环验证, 和回环帧的MP关联/KP内点/KP匹配数为: %d/%d/%d, "
                    "当前帧观测数/MP匹配率为: %d/%.2f (TODO)\n",
            mCurrentFrame.id, mpLoopKF->mIdKF, mnMPsInline, mnKPsInline, mnKPMatches, nObs, ratio);

    //! FIXME 优化位姿, 并计算重投影误差
    int nCorres = 0;
    double projError = 1000.;
    Se2 Twb = mCurrentFrame.getTwb();
    printf("[WARN] #%ld 位姿优化更新前的值为: [%.2f, %.2f, %.2f], 可视MP数为:%ld\n",
           mCurrentFrame.id, Twb.x, Twb.y, Twb.theta, mCurrentFrame.countObservations());
    poseOptimization(&mCurrentFrame, nCorres, projError);  // 确保已经setViewMP()
    Twb = mCurrentFrame.getTwb();
    printf("[WARN] #%ld 位姿优化更新后的值为: [%.2f, %.2f, %.2f], 重投影误差为:%.2f\n",
           mCurrentFrame.id, Twb.x, Twb.y, Twb.theta, projError);

    const bool bProjLost = detectIfLost(nCorres, projError);
    if (bProjLost) {
        fprintf(stderr,
                "[WARN] #%ld-KF#%ld(Loop) 重定位-回环验证失败! 优化时MP优化内点数/总数/重投影误差为: %d/%d/%.2f\n",
                mCurrentFrame.id, mpLoopKF->mIdKF, nCorres, nObs, projError);
        return false;
    } else {
        mnKPsInline = mnMPsInline;
        mvKPMatchIdx = mvKPMatchIdxGood;
        fprintf(stderr, "[INFO] #%ld-KF#%ld(Loop) 重定位-回环验证成功! MP内点数/总数/重投影误差为: %d/%d/%.2f\n",
                mCurrentFrame.id, mpLoopKF->mIdKF, nCorres, nObs, projError);
        return true;
    }
}

void TestTrack::doLocalBA(Frame& frame)
{
    WorkTimer timer;

    Se2 Twb1 = frame.getTwb();

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
    frame.setPose(Tcw);  // 更新Tcw和Twb

    Se2 Twb2 = frame.getTwb();
    printf("[INFO] #%ld-#%ld 局部地图投影优化成功, 参与优化的MP观测数为%d, 耗时%.2fms, "
           "优化前后位姿为: [%.2f, %.2f, %.2f] ==> [%.2f, %.2f, %.2f]\n",
           frame.id, mpReferenceKF->id, vertexId, timer.count(),
           Twb1.x, Twb1.y, Twb1.theta, Twb2.x, Twb2.y, Twb2.theta);

    const string g2oFile = Config::G2oResultsStorePath + to_string(mpNewKF->mIdKF) + "-ba.g2o";
    optimizer.save(g2oFile.c_str());
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

    //! 1.根据自己可视MP更新信息矩阵, 局部地图投影关联MP，并由MP候选生成新的MP, 关键函数!!!
    findCorresponds(MPCandidates);
    double t1 = timer.count();
    printf("[TIME] #%ld(KF#%ld) L1.关联地图点总耗时: %.2fms\n", pKF->id, pKF->mIdKF, t1);

    //! 2.更新局部地图里的共视关系，MP共同观测超过自身的30%则添加共视关系, 更新 mspCovisibleKFs
    timer.start();
    mpNewKF->updateCovisibleGraph();

    PtrKeyFrame pKFLast = mpMap->getCurrentKF();
    assert(mpNewKF->mIdKF - pKFLast->mIdKF == 1);
    Mat measure;
    g2o::Matrix6d info;
    calcOdoConstraintCam(mpNewKF->odom - pKFLast->odom, measure, info);
    pKFLast->setOdoMeasureFrom(mpNewKF, measure, toCvMat6f(info));
    mpNewKF->setOdoMeasureTo(pKFLast, measure, toCvMat6f(info));

    double t2 = timer.count();
    printf("[TIME] #%ld(KF#%ld) L2.更新共视关系耗时: %.2fms, 共获得%ld个共视KF, MP观测数: %ld\n",
           pKF->id, pKF->mIdKF, t2, pKF->countCovisibleKFs(), pKF->countObservations());

    //! 3.将KF插入地图
    timer.start();
    mpMap->insertKF(mpNewKF);
    double t3 = timer.count();
    printf("[TIME] #%ld(KF#%ld) L3.将KF插入地图, LocalMap的预处理总耗时: %.2fms\n",
           pKF->id, pKF->mIdKF, t1 + t2 + t3);

    //! 4.更新局部地图
    timer.start();
    updateLocalGraph();
    double t4 = timer.count();
    printf("[TIME] #%ld(KF#%ld) L4.第一次更新局部地图耗时: %.2fms\n", pKF->id, pKF->mIdKF, t4);

    if (mpNewKF->mIdKF < 2)
        return;

    //! 5.修剪冗余KF
    timer.start();
    pruneRedundantKFinMap();  // 会自动更新LocalMap
    double t5 = timer.count();
    printf("[TIME] #%ld(KF#%ld) L5.修剪冗余KF耗时: %.2fms\n", pKF->id, pKF->mIdKF, t5);

    //! 6.删除MP外点
//    timer.start();
//    removeOutlierChi2();
//    double t6 = timer.count();
//    printf("[TIME] #%ld(KF#%ld) L6.删除MP外点耗时: %.2fms\n", pKF->id, pKF->mIdKF, t6);

    //! 7.再一次更新局部地图
//    timer.start();
//    updateLocalGraph();
//    double t7 = timer.count();
//    printf("[TIME] #%ld(KF#%ld) L7.第二次更新局部地图耗时: %.2fms\n", pKF->id, pKF->mIdKF, t7);

    //! 8.局部图优化
//    timer.start();
//    localBA();
//    doLocalBA(*mpNewKF);
//    double t8 = timer.count();
//    printf("[TIME] #%ld(KF#%ld) L8.局部BA耗时: %.2fms\n", pKF->id, pKF->mIdKF, t8);


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
    printf("[INFO] #%ld(KF#%ld) L1.1.关联地图点1/3, 可视MP添加信息矩阵数/KF可视MP数: %d/%d, 耗时: %.2fms\n",
           mpNewKF->id, mpNewKF->mIdKF, nAddInfo, nObs, t1);
    assert(nObs == mpNewKF->countObservations());

    // 2.局部地图中非newKF的MPs投影到newKF, 新投影的MP可能会把MP候选的坑占了
    //! NOTE 视差不好的要不要投?? 投上了说不定可以更新?
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
                // assert(pMP->isGoodPrl());  // 视差好的才投

                // 通过三角化验证一下投影匹配对不对
                Point3f Pw =
                    cvu::triangulate(pMP->getMainMeasureProjection(), mpNewKF->mvKeyPoints[i].pt,
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
            printf("[INFO] #%ld(KF#%ld) L1.1.关联地图点2/3, "
                   "KF关联MP数/投影MP数/局部MP总数: %d/%d/%d, 耗时: %.2fms\n",
                   mpNewKF->id, mpNewKF->mIdKF, nProjLocalMPs, m, nLocalMPs, t2);
        } else {
            printf("[INFO] #%ld(KF#%ld) L1.1.关联地图点2/3, "
                   "投影MP数为0, 局部MP总数为: %d\n",
                   mpNewKF->id, mpNewKF->mIdKF, nLocalMPs);
        }
    } else {
        printf("[INFO] #%ld(KF#%ld) L1.1.关联地图点2/3, 当前局部地图对当前帧没有有效投影.\n",
               mpNewKF->id, mpNewKF->mIdKF);
    }

    // 3.处理所有的候选MP.(候选观测完全是新的MP)
    timer.start();
    int nAddNewMP = 0, nReplacedCands = 0;
    for (const auto& cand : MPCandidates) {
        const size_t idx1 = cand.first;
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
    printf("[INFO] #%ld(KF#%ld) L1.1.关联地图点3/3, MP候选数/新增数/替换数/当前MP总数: "
           "%ld/%d/%d/%ld, 耗时: %.2fms\n",
           mpNewKF->id, mpNewKF->mIdKF, MPCandidates.size(), nAddNewMP, nReplacedCands, mpMap->countMPs(), t3);
    assert(nReplacedCands <= nProjLocalMPs);
    assert(nAddNewMP + nReplacedCands == MPCandidates.size());
}

void TestTrack::updateLocalGraph()
{
    mpMap->updateLocalGraph_new(Config::LocalFrameSearchLevel, Config::MaxLocalFrameNum,
                                Config::LocalFrameSearchRadius);
//    mpMap->updateLocalGraph(Config::LocalFrameSearchLevel, Config::MaxLocalFrameNum,
//                            Config::LocalFrameSearchRadius);
}

void TestTrack::pruneRedundantKFinMap()
{
    WorkTimer timer;

    int nLocalKFs = mpMap->countLocalKFs();
    int nLocalMPs = mpMap->countLocalMPs();
    int nLocalRefs = mpMap->countRefKFs();

    int nPruned = mpMap->pruneRedundantKF();

    printf("[INFO] #%ld(KF#%ld) L5.修剪冗余KF前后, 局部KF/MP/RefKF的数量为: %d/%d/%d, %ld/%ld/%ld\n",
           mpNewKF->id, mpNewKF->mIdKF, nLocalKFs, nLocalMPs, nLocalRefs,
           mpMap->countLocalKFs(), mpMap->countLocalMPs(), mpMap->countRefKFs());
    printf("[INFO] #%ld(KF#%ld) L5.修剪冗余KF, 共修剪了%d帧KF, 耗时:%.2fms.\n",
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

    printf("[INFO] #%ld(KF#%ld) L6.移除离群点%d成功! 耗时: %.2fms\n", mpNewKF->id, mpNewKF->mIdKF,
           nBadMP, timer.count());
}

void TestTrack::localBA()
{
    WorkTimer timer;
    timer.start();

    printf("[INFO] #%ld(KF#%ld) L8.正在执行localBA()...\n", mpNewKF->id, mpNewKF->mIdKF);
    Se2 Twb1 = mpNewKF->getTwb();

    SlamOptimizer optimizer;
    SlamLinearSolverCholmod* linearSolver = new SlamLinearSolverCholmod();
//    SlamLinearSolverCSparse* linearSolver = new SlamLinearSolverCSparse();
    SlamBlockSolver* blockSolver = new SlamBlockSolver(linearSolver);
    SlamAlgorithmLM* solver = new SlamAlgorithmLM(blockSolver);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    mpMap->loadLocalGraph(optimizer);
    if (optimizer.edges().empty()) {
        fprintf(stderr, "[ERRO] #%ld(KF#%ld) No MPs in graph, leaving localBA().\n",
                mpNewKF->id, mpNewKF->mIdKF);
        return;
    }
    double t1 = timer.count();
    timer.start();

    //optimizer.verifyInformationMatrices(true);
    //assert(optimizer.verifyInformationMatrices(true)); // TODO

    optimizer.initializeOptimization(0);
    optimizer.optimize(Config::LocalIterNum);
    double t2 = timer.count();

    if (solver->currentLambda() > 100.0) {
        cerr << "[ERRO] current lambda too large " << solver->currentLambda()
             << " , reject optimized result!" << endl;
        return;
    }

    timer.start();
    mpMap->optimizeLocalGraph(optimizer);  // 用优化后的结果更新KFs和MPs
    double t3 = timer.count();
    double t4 = t1 + t2 + t3;
    printf("[TIME] #%ld(KF#%ld) L8.localBA()加载/优化/更新/总耗时: %.2f/%.2f/%.2f/%.2fms\n",
           mpNewKF->id, mpNewKF->mIdKF, t1, t2, t3, t4);

    Se2 Twb2 = mpNewKF->getTwb();
    printf("[INFO] #%ld-#%ld L8.localBA()优化成功, 优化前后位姿为: [%.2f, %.2f, %.2f] ==> [%.2f, %.2f, %.2f]\n",
           mpNewKF->id, mpReferenceKF->id, Twb1.x, Twb1.y, Twb1.theta, Twb2.x, Twb2.y, Twb2.theta);

    const string g2oFile = Config::G2oResultsStorePath + to_string(mpNewKF->mIdKF) + "-local.g2o";
    optimizer.save(g2oFile.c_str());
}

}  // namespace se2lam
