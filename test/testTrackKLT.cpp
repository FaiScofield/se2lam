#include "testTrackKLT.h"
#include "LineDetector.h"
#include "test_functions.hpp"
#include "Map.h"
#include "MapPublish.h"
#include "ORBmatcher.h"
#include "converter.h"
#include "cvutil.h"
#include "optimizer.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>

namespace se2lam
{

#define DO_LOCAL_BA 1
#define DO_GLOBAL_BA 0

using namespace std;
using namespace cv;
using namespace Eigen;

typedef std::unique_lock<std::mutex> locker;

string g_configPath = "/home/vance/dataset/rk/dibeaDataSet/se2_config/";
size_t g_matchToRefSum = 0;
const int EDGE = 12;


TestTrackKLT::TestTrackKLT()
    : mState(cvu::NO_READY_YET), mLastState(cvu::NO_READY_YET), mpReferenceKF(nullptr),
      mpNewKF(nullptr), mpLoopKF(nullptr), mnMPsCandidate(0), mnKPMatches(0), mnKPsInline(0),
      mnKPMatchesGood(0), mnMPsTracked(0), mnMPsNewAdded(0), mnMPsInline(0), mnLostFrames(0),
      mLoopScore(0)
{
    mpORBextractor = new ORBextractor(Config::MaxFtrNumber, Config::ScaleFactor, Config::MaxLevel);
    mpORBmatcher = new ORBmatcher(0.9, true);
    mpORBvoc = new ORBVocabulary();
    string strVocFile = Config::DataPath + "../se2_config/ORBvoc.bin";
    bool bVocLoad = mpORBvoc->loadFromBinaryFile(strVocFile);
    if (!bVocLoad)
        cerr << "[Track][Error] Wrong path to vocabulary, Falied to open it." << endl;

    nMinFrames = min(2, cvCeil(0.25 * Config::FPS));  // 上溢
    nMaxFrames = cvFloor(5 * Config::FPS);            // 下溢
    nMinMatches = std::min(cvFloor(0.1 * Config::MaxFtrNumber), 40);
    mMaxAngle = static_cast<float>(g2o::deg2rad(80.));
    mMaxDistance = 0.3f * Config::UpperDepth;

    mAffineMatrix = Mat::eye(2, 3, CV_64FC1);  // double
    preSE2.meas.setZero();
    preSE2.cov.setZero();

    fprintf(stderr, "[Track][Info ] 相关参数如下: \n - 最小/最大KF帧数: %d/%d\n"
                    " - 最大移动距离/角度: %.0fmm/%.0fdeg\n - 最少匹配数量: %d\n",
            nMinFrames, nMaxFrames, mMaxDistance, g2o::rad2deg(mMaxAngle), nMinMatches);

    mImageRows = Config::ImgSize.height;  // 240
    mImageCols = Config::ImgSize.width;   // 320

    mCellHeight = 48;  // 分块尺寸
    mCellWidth = 64;
    mnCellsW = mImageRows / mCellHeight;     // 240/60=4
    mnCellsH = mImageCols / mCellWidth;      // 320/80=4
    mnCells = mnCellsH * mnCellsW;           // 16
    mnMaxNumPtsInCell = Config::MaxFtrNumber / mnCells;  // 分块检点的最大点数
    mvNumPtsInCell.resize(mnCells, 0);
    mnMaskRadius = 2;  // mask建立时的特征点周边半径
    fprintf(stderr, "[Track][Info ] KLT相关参数如下:\n"
                    " - Cell Size: %d x %d\n - Cells: %d x %d = %d\n - Max points in cell: %d\n",
            mCellWidth, mCellHeight, mnCellsW, mnCellsH, mnCells, mnMaxNumPtsInCell);
}

TestTrackKLT::~TestTrackKLT()
{
    delete mpORBextractor;
    delete mpORBmatcher;
    delete mpORBvoc;
}

bool TestTrackKLT::checkReady()
{
    if (!mpMap || !mpMapPublisher)
        return false;
    return true;
}

void TestTrackKLT::run(const cv::Mat& img, const Se2& odo, const double time)
{
    if (mState == cvu::NO_READY_YET)
        mState = cvu::FIRST_FRAME;

    WorkTimer timer;
    mLastState = mState;
    bool bOK = false;
    double t1 = 0, t2 = 0, t3 = 0;

    {
        locker lock(mMutexForPub);

        Mat imgUn, imgClahed;
        Ptr<CLAHE> clahe = createCLAHE(3.0, Size(8, 8));
        clahe->apply(img, imgClahed);
        undistort(imgClahed, imgUn, Config::Kcam, Config::Dcam);
        double t1 = timer.count();

        timer.start();
        if (mState == cvu::FIRST_FRAME) {
            processFirstFrame(imgUn, odo, time);
            return;
        } else if (mState == cvu::OK) {
            mLastRefKFid = mpReferenceKF->id;
            bOK = trackReferenceKF(imgUn, odo, time);
        } else if (mState == cvu::LOST) {
            bOK = doRelocalization(imgUn, odo, time);
        }
        t2 = timer.count();
    }


    if (bOK) {
        //? TODO 更新一下MPCandidates里面Tc2w?
        mnLostFrames = 0;
        mState = cvu::OK;
    } else {
        mnLostFrames++;
        mState = cvu::LOST;
        // if (mnLostFrames > 50)
        //    startNewTrack();
    }

    resetLocalTrack();  // KF判断在这里

    t3 = t1 + t2;
    trackTimeTatal += t3;
    printf("[Track][Timer] #%ld-#%ld 前端构建Frame/追踪/总耗时为: %.2f/%.2f/%.2fms, 平均耗时: %.2fms\n",
           mCurrentFrame.id, mLastRefKFid, t1, t2, t3, trackTimeTatal * 1.f / mCurrentFrame.id);
}

void TestTrackKLT::processFirstFrame(const Mat& img, const Se2& odo, double time)
{
    mForwImg = img;

    // 直线掩模上特征点提取
    vector<Keyline> lines;
    Mat lineMask = getLineMask(mForwImg, lines, false);
    detectFeaturePointsCell(mForwImg, lineMask);
    mvForwPts = mvNewPts;

    // 创建当前帧
    if (mvForwPts.size() > Config::MaxFtrNumber * 0.5) {
        vector<KeyPoint> vKPs;
        KeyPoint::convert(mvForwPts, vKPs);
        mCurrentFrame = Frame(mForwImg, odo, vKPs, mpORBextractor, time);

        cout << "========================================================" << endl;
        cout << "[Track][Info ] Create first frame with " << mCurrentFrame.N << " features. "
             << "And the start odom is: " << odo << endl;
        cout << "========================================================" << endl;

        mCurrentFrame.setPose(Se2(0.f, 0.f, 0.f));
        mpReferenceKF = make_shared<KeyFrame>(mCurrentFrame);  // 首帧为关键帧
        mpMap->insertKF(mpReferenceKF);  // 首帧的KF直接给Map,没有给LocalMapper
        mpMap->updateLocalGraph();       // 首帧添加到LocalMap里

        mCurrImg = mpReferenceKF->mImage.clone();
        mvCurrPts = mvForwPts;
        mLastFrame = mCurrentFrame;
        mState = cvu::OK;
    } else {
        cout << "[Track][Warni] Failed to create first frame for too less keyPoints: "
             << mvForwPts.size() << endl;

        Frame::nextId = 0;
        mState = cvu::FIRST_FRAME;
    }
}

bool TestTrackKLT::trackReferenceKF(const Mat& img, const Se2& odo, double time)
{
    size_t n1 = 0, n2 = 0, n3 = 0, n4 = 0;
    Mat show1, show2, show3, show4;

    mForwImg = img;
    mnKPsInline = 0;
    std::fill(mvKPMatchIdx.begin(), mvKPMatchIdx.end(), -1);
    mvKPMatchIdx.resize(mpReferenceKF->N, -1);

    // 1.预测上一帧img和KP旋转后的值
    const Se2 dOdo = odo - mpReferenceKF->odom;  // A21
    predictPointsAndImage(dOdo);  // 得到旋转后的预测点mvPrevPts

    // 2.光流追踪上一帧的KP
    vector<uchar> status;
    vector<float> err;
    calcOpticalFlowPyrLK(mPrevImg, mForwImg, mvPrevPts, mvForwPts, status, err, Size(17, 17), 0);
    assert(mpReferenceKF->N == mvForwPts.size());
    mvPrevPts.clear();
    mnKPMatches = mvForwPts.size();

    // 3.更新和参考帧的匹配关系
    size_t j = 0;
    mvIdxToFirstAdded.clear();
    mvIdxToFirstAdded.resize(mvForwPts.size(), 0);
    for (size_t i = 0, iend = mpReferenceKF->N; i < iend; i++) {
        mvIdxToFirstAdded[i] = i;
        if (status[i] && inBorder(mvForwPts[i])) {
            mvCurrPts[j] = mvCurrPts[i];
            mvForwPts[j] = mvForwPts[i];
            mvCellLabel[j] = mvCellLabel[i];
            mvIdxToFirstAdded[j] = mvIdxToFirstAdded[i];
            mvKPMatchIdx[i] = j; // mvKPMatchIdx 对应RefKF的KPs
            j++;
        } else {
            mvNumPtsInCell[mvCellLabel[i]]--; // 数量先扣, 回头再把消减mvForwPts[i]给去掉
        }
    }
    mvCurrPts.resize(j);
    mvForwPts.resize(j);
    mvCellLabel.resize(j);
    mvIdxToFirstAdded.resize(j);
    mnKPMatches = mnKPsInline = j;
    if (mnKPMatches >= 10) {
        vector<uchar> inliersMask(mvCurrPts.size());
//        mAffineMatrix = estimateAffinePartial2D(mvCurrPts, mvForwPts, inliersMask, RANSAC, 2.0); // 更新A
//        inliersMask.clear();
        Mat H = findHomography(mvCurrPts, mvForwPts, RANSAC, 3.0, inliersMask);

        // 去掉外点再一次更新匹配关系!
        size_t k = 0;
        for (size_t j = 0, iend = inliersMask.size(); j < iend; ++j) {
            if (inliersMask[j]) {
                mvCurrPts[k] = mvCurrPts[j];
                mvForwPts[k] = mvForwPts[j];
                mvCellLabel[k] = mvCellLabel[j];
                mvIdxToFirstAdded[k] = mvIdxToFirstAdded[j];
                mvKPMatchIdx[mvIdxToFirstAdded[j]] = k;
                k++;
            } else {
                mvNumPtsInCell[mvCellLabel[j]]--;
                mvKPMatchIdx[mvIdxToFirstAdded[j]] = -1;
                mnKPsInline--;
            }
        }

        mvCurrPts.resize(k);
        mvForwPts.resize(k);
        mvCellLabel.resize(k);
        mvIdxToFirstAdded.resize(k);
        assert(mnKPsInline == k);
    }
    updateAffineMatix();
    mvCurrPts.clear();
    printf("[Track][Info ] #%ld-#%ld 光流追踪上Ref的内点数/总点数 = %d/%ld\n",
           Frame::nextId, mLastRefKFid, mnKPsInline, mpReferenceKF->N);

    // 4.增加新点 TODO 放到生成KF后去做
    vector<Keyline> lines;
    mMask = getLineMask(mForwImg, lines, false);
    const Mat borderMaskRow(EDGE, mImageCols, CV_8UC1, Scalar(0));
    const Mat borderMaskCol(mImageRows, EDGE, CV_8UC1, Scalar(0));
    borderMaskRow.copyTo(mMask.rowRange(0, EDGE));
    borderMaskRow.copyTo(mMask.rowRange(mImageRows - EDGE, mImageRows));
    borderMaskCol.copyTo(mMask.colRange(0, EDGE));
    borderMaskCol.copyTo(mMask.colRange(mImageCols - EDGE, mImageCols));
    imshow("mask", mMask);
    waitKey(30);
    for (const auto& p : mvForwPts)
        cv::circle(mMask, p, mnMaskRadius, Scalar(0), -1);  // 标记掩模
    detectFeaturePointsCell(mForwImg, mMask);
    const size_t n = mvForwPts.size();
    const size_t m = mvNewPts.size();
    mvForwPts.resize(n + m);
    mvIdxToFirstAdded.resize(n + m);
    for (size_t i = 0; i < m; ++i) {
        mvForwPts[n + i] = mvNewPts[i];
        mvIdxToFirstAdded[n + i] = n + i;
    }

    // 5.更新Frame
    vector<KeyPoint> vKPsCurFrame(mvForwPts.size());
    cv::KeyPoint::convert(mvForwPts, vKPsCurFrame);  //! NOTE 金字塔层数没了
    mCurrentFrame = Frame(mForwImg, odo, vKPsCurFrame, mpORBextractor, time);
    updateFramePoseFromRef();

    // 6.三角化
    mvKPMatchIdx.resize(mCurrentFrame.N, -1); // 保证不越界
    mnMPsInline = doTriangulate(mpReferenceKF, &mCurrentFrame);

    // reset data
    cv::KeyPoint::convert(mpReferenceKF->mvKeyPoints, mvCurrPts);

    return true;
}

void TestTrackKLT::updateFramePoseFromRef()
{
    const Se2 Tb1b2 = mCurrentFrame.odom - mpReferenceKF->odom;
    const Mat Tc2c1 = Config::Tcb * Tb1b2.inv().toCvSE3() * Config::Tbc;
    mCurrentFrame.setTrb(Tb1b2);
    mCurrentFrame.setTcr(Tc2c1);
    mCurrentFrame.setPose(Tc2c1 * mpReferenceKF->getPose());  // 位姿的预测初值用odo进行更新
    // mCurrentFrame.setPose(mpReferenceKF->getTwb() + Tb1b2);

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
}

int TestTrackKLT::doTriangulate(PtrKeyFrame& pKF, Frame* frame)
{
    if (mvKPMatchIdx.empty() || mCurrentFrame.id - mpReferenceKF->id < nMinFrames)
        return 0;

    // 以下成员变量在这个函数中会被修改
    mnMPsCandidate = mMPCandidates.size();
    mvKPMatchIdxGood = mvKPMatchIdx;
    mnMPsInline = 0;
    mnMPsTracked = 0;
    mnMPsNewAdded = 0;
    mnKPMatchesGood = 0;
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
        PtrMapPoint pObservedMP = nullptr;                     // 对于的MP观测
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

        // 如果参考帧KP没有对应的MP, 或者对应MP视差不好,
        // 则为此匹配对KP三角化计算深度(相对参考帧的坐标)
        // 由于两个投影矩阵是两KF之间的相对投影, 故三角化得到的坐标是相对参考帧的坐标, 即Pc1
        const Point2f pt1 = pKF->mvKeyPoints[i].pt;
        const Point2f pt2 = frame->mvKeyPoints[mvKPMatchIdx[i]].pt;
        const Point3f Pc1 = cvu::triangulate(pt1, pt2, Proj1, Proj2);
        const Point3f Pw = cvu::se3map(cvu::inv(Tc1w), Pc1);

        // Pc2用Tcr计算出来的是预测值, 故这里用Pc1的深度判断即可
        const bool bAccepDepth = Config::acceptDepth(Pc1.z);  // 深度是否符合
        const bool bCandidated = mMPCandidates.count(i);      // 是否有对应的MP候选
        assert(!(bObserved && bCandidated));                  // 不能即有MP观测又是MP候选

        if (bAccepDepth) {  // 如果深度计算符合预期
            const bool bGoodPrl = cvu::checkParallax(Ocam1, Ocam2, Pc1, 2);  // 视差是否良好
            if (bGoodPrl) {                                                  // 如果视差好
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
            } else {              // 如果视差不好
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
                    MPCandidate MPcan(pKF, Pc1, frame->id, mvKPMatchIdx[i], Tc2w);
                    mMPCandidates.emplace(i, MPcan);
                    mnMPsCandidate++;
                    n32++;
                }
            }
        } else {                               // 如果深度计算不符合预期
            if (!bObserved && !bCandidated) {  // 情况3.3
                n33++;
                mnKPMatchesGood--;
                mvKPMatchIdxGood[i] = -1;
            }
            // 情况1.2.3和2.3不处理
        }
    }

    //    printf("[Track][Info ] #%ld-#%ld KP匹配结果: KP好匹配数/总内点数/总匹配数: %d/%d/%d, "
    //           "MP总观测数(Ref)/关联数/视差好的/更新到好的: %d/%d/%d/%d\n", frame->id, pKF->id,
    //           mnKPMatchesGood, mnKPsInline, mnKPMatches, nObsRef, mnMPsTracked, n11, n121);
    //    printf("[Track][Info ] #%ld-#%ld 三角化结果: 候选MP原总数/转正数/更新数/增加数/现总数:
    //    %d/%d/%d/%d/%d, "
    //           "三角化新增MP数/新增候选数/剔除匹配数: %d/%d/%d, 新生成MP数: %d\n",
    //           frame->id, pKF->id, n2, n21, n22, n32, mnMPsCandidate, n31, n32, n33,
    //           mnMPsNewAdded);

    assert(n11 + n121 == mnMPsTracked);
    assert(n21 + n31 == mnMPsNewAdded);
    assert(n33 + mnKPMatchesGood == mnKPsInline);
    assert(mnMPsTracked + mnMPsNewAdded == mnMPsInline);
    assert((n2 - n21 + n32 == mnMPsCandidate) && (mnMPsCandidate == mMPCandidates.size()));
    assert(nObsCur + mnMPsTracked + mnMPsNewAdded == frame->countObservations());
    assert(nObsRef + mnMPsNewAdded == pKF->countObservations());

    return mnMPsNewAdded;
}

int TestTrackKLT::removeOutliers(const PtrKeyFrame& pKFRef, const Frame* pFCur,
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

void TestTrackKLT::resetLocalTrack()
{
    // 正常状态进行关键帧判断
    if (mState == cvu::OK) {
        if (needNewKF()) {
            PtrKeyFrame pNewKF = make_shared<KeyFrame>(mCurrentFrame);
            mpReferenceKF->preOdomFromSelf = make_pair(pNewKF, preSE2);
            pNewKF->preOdomToSelf = make_pair(mpReferenceKF, preSE2);

            // 重定位成功将当前帧变成KF
            //! TODO 重定位的预积分信息怎么解决??
            if (mState == cvu::OK && mLastState == cvu::LOST) {
                pNewKF->addCovisibleKF(mpLoopKF);
                mpLoopKF->addCovisibleKF(pNewKF);
                printf("[Track][Info ] #%ld 成为了新的KF(#%ld), 因为刚刚重定位成功!\n", pNewKF->id,
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
            preSE2.meas.setZero();
            preSE2.cov.setZero();

#if USE_KLT
            cv::KeyPoint::convert(mpReferenceKF->mvKeyPoints, mvCurrPts);
            mCurrImg = mpReferenceKF->mImage.clone();  //! NOTE 需要深拷贝
#endif
        } else {
            copyForPub();
        }
        mLastFrame = mCurrentFrame;
        mnKPMatches = 0;
        mnKPsInline = 0;
        mnKPMatchesGood = 0;
        mnMPsTracked = 0;
        mnMPsNewAdded = 0;
        mnMPsInline = 0;
        mpLoopKF.reset();
        return;
    }

    // 当前帧刚刚丢失要确保上一帧是最新的KF
    assert(mState == cvu::LOST);
    if (mState == cvu::LOST && mLastState == cvu::OK && !mLastFrame.isNull() &&
        mLastFrame.id != mpReferenceKF->id) {
        printf("[Track][Info ] #%ld-#%ld 上一帧(#%ld)成为了新的KF(#%ld)! 因为当前帧刚刚丢失!\n",
               mCurrentFrame.id, mLastRefKFid, mLastFrame.id, KeyFrame::mNextIdKF);

        PtrKeyFrame pNewKF = make_shared<KeyFrame>(mLastFrame);
        //! TODO TODO TODO 这里要扣掉当前帧的预积分信息
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
        preSE2.meas.setZero();
        preSE2.cov.setZero();
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
    preSE2.meas.setZero();
    preSE2.cov.setZero();
}

bool TestTrackKLT::needNewKF()
{
    // 刚重定位成功需要建立新的KF
    if (mState == cvu::OK && mLastState == cvu::LOST) {
        printf("[Track][Info ] #%ld 刚重定位成功! 需要加入新的KF! \n", mCurrentFrame.id);
        return true;
    }

    int deltaFrames = static_cast<int>(mCurrentFrame.id - mpReferenceKF->id);

    // 必要条件
    bool c0 = deltaFrames >= nMinFrames;  // 下限

    // 充分条件
    bool c1 = mnKPMatches < 50 ||
              (mnKPsInline < 20 &&
               mnKPMatchesGood < 10);  // 和参考帧匹配的内点数太少. 这里顺序很重要! 30/20/15
    bool c2 = mnMPsInline > 170;  // 关联/新增MP数量够多, 说明这时候比较理想 60
    bool bNeedKFByVO = c0 && (c1 || c2);

    // 1.跟踪要跪了必须要加入新的KF
    if (bNeedKFByVO) {
        printf("[Track][Info ] #%ld-#%ld 成为了新的KF(#%ld), 其KF条件满足情况: 内点少(%d, "
               "%d/%d/%d)/新增多(%d, %d+%d=%d)\n",
               mCurrentFrame.id, mLastRefKFid, KeyFrame::mNextIdKF, c1, mnKPMatches, mnKPsInline,
               mnKPMatchesGood, c2, mnMPsTracked, mnMPsNewAdded, mnMPsInline);
        return true;
    }

    // 上限/旋转/平移
    bool c4 = deltaFrames > nMaxFrames;  // 上限
    Se2 dOdo = mCurrentFrame.odom - mpReferenceKF->odom;
    bool c5 = static_cast<double>(abs(dOdo.theta)) >= mMaxAngle;  // 旋转量超过50°
    cv::Mat cTc = Config::Tcb * dOdo.toCvSE3() * Config::Tbc;
    cv::Mat xy = cTc.rowRange(0, 2).col(3);
    bool c6 = cv::norm(xy) >= mMaxDistance;  // 相机的平移量足够大
    bool bNeedKFByOdo = (c5 || c6) || c4;  // 相机移动取决于深度上限,考虑了不同深度下视野的不同

    // 2.如果跟踪效果还可以, 就看旋转平移条件
    if (bNeedKFByOdo) {
        printf("[Track][Info ] #%ld-#%ld 成为了新的KF(#%ld), 其KF条件满足情况: "
               "大旋转(%d)/大平移(%d)/达上限(%d)"
               ", 匹配情况: 内点(%d/%d/%d)/新增(%d+%d=%d)\n",
               mCurrentFrame.id, mLastRefKFid, KeyFrame::mNextIdKF, c5, c6, c4, mnKPMatches,
               mnKPsInline, mnKPMatchesGood, mnMPsTracked, mnMPsNewAdded, mnMPsInline);
        return true;
    }

    return false;
}

void TestTrackKLT::copyForPub()
{
    locker lock1(mMutexForPub);
    locker lock2(mpMapPublisher->mMutexPub);

    mpMapPublisher->mnCurrentFrameID = mCurrentFrame.id;
    mpMapPublisher->mCurrentFramePose = mCurrentFrame.getPose() /*.clone()*/;
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
            std::snprintf(strMatches, 64,
                          "F: %ld->%ld, LoopKF: %ld(%ld)->%ld, Score: %.3f, M: %d/%d/%d/%d",
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

bool TestTrackKLT::detectIfLost(Frame& f, const Mat& Tcw_opt)
{
    Se2 Twb_opt;
    Twb_opt.fromCvSE3(Tcw_opt);
    Se2 Twb_ori = f.getTwb();
    Se2 dSe2 = Twb_opt - Twb_ori;
    const double dt = sqrt(dSe2.x * dSe2.x + dSe2.y * dSe2.y);
    const double da = abs(dSe2.theta);
    if (dt > mMaxDistance) {
        fprintf(stderr,
                "[Track][Warni] #%ld\t 位姿优化结果相差太大: dt(%.2f) > mMaxDistance(%.2f)\n", f.id,
                dt, mMaxDistance);
        return true;
    }
    if (da > mMaxAngle) {
        fprintf(stderr, "[Track][Warni] #%ld\t 位姿优化结果相差太大: da(%.2f) < mMaxAngle(%.2f)\n",
                f.id, da, mMaxAngle);
        return true;
    }
    return false;
}

Mat TestTrackKLT::getAffineMatrix(const Se2& dOdo)
{
    Point2f rotationCenter;
    rotationCenter.x = Config::cx - Config::Tbc.at<float>(1, 3) * Config::fx / 3000.f;
    rotationCenter.y = Config::cy - Config::Tbc.at<float>(0, 3) * Config::fy / 3000.f;
    Mat R0 = getRotationMatrix2D(rotationCenter, dOdo.theta * 180.f / CV_PI, 1);

    Mat Tc1c2 = Config::Tcb * dOdo.inv().toCvSE3() * Config::Tbc;  // 4x4
    Mat Rc1c2 = Tc1c2.rowRange(0, 3).colRange(0, 3).clone();       // 3x3
    Mat tc1c2 = Tc1c2.rowRange(0, 3).col(3).clone();               // 3x1
    Mat R = Config::Kcam * Rc1c2 * (Config::Kcam).inv();  // 3x3 相当于A, 但少了部分平移信息
    Mat t = Config::Kcam * tc1c2 / 3000.f;                // 3x1 假设MP平均深度3m

    Mat A;
    R.rowRange(0, 2).convertTo(A, CV_64FC1);
    R0.colRange(0, 2).copyTo(A.colRange(0, 2));       // 去掉尺度变换
    A.at<double>(0, 2) += (double)t.at<float>(0, 0);  // 加上平移对图像造成的影响
    A.at<double>(1, 2) += (double)t.at<float>(1, 0);

    return A.clone();
}

bool TestTrackKLT::doRelocalization(const Mat& img, const Se2& odo, double time)
{
    if (mCurrentFrame.isNull())  // 图像挂掉的时候Frame会变成null.
        return false;
    if (mCurrentFrame.mTimeStamp < mLastFrame.mTimeStamp)
        return false;

    mnMPsNewAdded = mnMPsCandidate = mnKPMatchesGood = 0;
    mnKPsInline = mnMPsInline = mnMPsTracked = 0;

    updateFramePoseFromRef();

    fprintf(stderr, "[Track][Info ] #%ld-#%ld 正在进行重定位...\n", mCurrentFrame.id, mLastRefKFid);
    const bool bDetected = detectLoopClose(&mCurrentFrame);
    if (bDetected) {
        const bool bVerified = verifyLoopClose(&mCurrentFrame);
        if (bVerified)
            return true;
    }
    return false;
}

void TestTrackKLT::predictPointsAndImage(const Se2& dOdo)
{
    assert(!mvCurrPts.empty());
    assert(!mCurrImg.empty());

    mvPrevPts.clear();
    Mat R = getRotationMatrix2D(Point2f(0, 0), dOdo.theta * 180.f / CV_PI, 1.).colRange(0, 2);
    R.copyTo(mAffineMatrix.colRange(0, 2));
    warpAffine(mCurrImg, mPrevImg, mAffineMatrix, mCurrImg.size());
    transform(mvCurrPts, mvPrevPts, mAffineMatrix);
}

bool TestTrackKLT::inBorder(const Point2f& pt)
{
    const int minBorderX = EDGE;
    const int minBorderY = minBorderX;
    const int maxBorderX = mImageCols - EDGE;
    const int maxBorderY = mImageRows - EDGE;

    const int x = cvRound(pt.x);
    const int y = cvRound(pt.y);

    return minBorderX <= x && x < maxBorderX && minBorderY <= y && y < maxBorderY;
}

void TestTrackKLT::updateAffineMatix()
{
    assert(mvCurrPts.size() == mvForwPts.size());
    assert(mvKPMatchIdx.size() == mpReferenceKF->N);

    vector<Point2f> ptRef, ptCur;
    ptRef.reserve(mvForwPts.size());
    ptCur.reserve(mvForwPts.size());
    for (int i = 0, iend = mvForwPts.size(); i < iend; ++i) {
        if (mvKPMatchIdx[mvIdxToFirstAdded[i]] < 0)
            continue;
        ptRef.push_back(mvCurrPts[i]);
        ptCur.push_back(mvForwPts[i]);
    }

    const size_t N = ptRef.size();
    const Mat R = mAffineMatrix.colRange(0, 2); // R21
    const Mat J = -R;

    Mat t = Mat::zeros(2, 1, CV_64FC1);
    double errorSumLast = 9999999;
    for (int it = 0; it < 10; ++it) {
        Mat H = Mat::zeros(2, 2, CV_64FC1);
        Mat b = Mat::zeros(2, 1, CV_64FC1);
        double errorSum = 0;
        for (size_t i = 0; i < N; ++i) {
            Mat x1, x2;
            Mat(ptRef[i]).convertTo(x1, CV_64FC1);
            Mat(ptCur[i]).convertTo(x2, CV_64FC1);
            Mat e = x2 - R * x1 - t;
            H += J.t() * J;
            b += J.t() * e;
            errorSum += norm(e);
        }

        Mat dt = -H.inv(SVD) * b;
        t += dt;

        //cout << "iter = " << it << ", ave chi = " << errorSum / N << ", t = " << t.t() << endl;
        if (errorSumLast < errorSum) {
            t -= dt;
            break;
        }
        if (errorSumLast - errorSum < 1e-6)
            break;
        errorSumLast = errorSum;
    }

    t.copyTo(mAffineMatrix.col(2));
}

void TestTrackKLT::detectFeaturePointsCell(const Mat& image, const Mat& mask)
{
    vector<Mat> cellImgs, cellMasks;
    segImageToCells(image, cellImgs);
    segImageToCells(mask, cellMasks);

    mvNewPts.clear();
    mvNewPts.reserve(mnMaxNumPtsInCell * mnCells);
    const int th = mnMaxNumPtsInCell * 0.1;
    for (int i = 0; i < mnCells; ++i) {
        int newPtsToAdd = mnMaxNumPtsInCell - mvNumPtsInCell[i];
        newPtsToAdd = newPtsToAdd > mnMaxNumPtsInCell ? mnMaxNumPtsInCell : newPtsToAdd;
        if (newPtsToAdd > th) {
            vector<Point2f> ptsInThisCell;
            ptsInThisCell.reserve(newPtsToAdd);
            goodFeaturesToTrack(cellImgs[i], ptsInThisCell, newPtsToAdd, 0.05, mnMaskRadius,
                                cellMasks[i], 3);
            mvNumPtsInCell[i] += static_cast<int>(ptsInThisCell.size());

            // 获得特征点在图像上的实际坐标
            for (size_t j = 0, jend = ptsInThisCell.size(); j < jend; j++) {
                int cellIndexX = i % mnCellsW;
                int cellIndexY = i / mnCellsW;

                Point2f& thisPoint = ptsInThisCell[j];
                thisPoint.x += mCellWidth * cellIndexX;
                thisPoint.y += mCellHeight * cellIndexY;
                mvNewPts.push_back(thisPoint);
                mvCellLabel.push_back(i);
            }
        }
    }
}

void TestTrackKLT::segImageToCells(const Mat& image, vector<Mat>& cellImgs)
{
    Mat imageCut;
    int m = image.cols / mCellWidth;
    int n = image.rows / mCellHeight;
    cellImgs.reserve(m * n);
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            Rect rect(i * mCellWidth, j * mCellHeight, mCellWidth, mCellHeight);
            imageCut = Mat(image, rect);
            cellImgs.push_back(imageCut.clone());
        }
    }
}

}  // namespace se2lam

/*
int main(int argc, char* argv[])
{
    //! check input
    if (argc < 2) {
        fprintf(stderr, "Usage: test_kltTracking <dataPath> [number_frames_to_process]");
        exit(-1);
    }
    int num = INT_MAX;
    if (argc == 3) {
        num = atoi(argv[2]);
        cout << " - set number_frames_to_process = " << num << endl << endl;
    }

    //! initialization
    Config::readConfig(g_configPath);
    Mat K = Config::Kcam, D = Config::Dcam;

    string dataFolder = string(argv[1]) + "slamimg";
    vector<RK_IMAGE> imgFiles;
    readImagesRK(dataFolder, imgFiles);

    string odomRawFile = string(argv[1]) + "odo_raw.txt";  // [mm]
    ifstream rec(odomRawFile);
    if (!rec.is_open()) {
        cerr << "[Main ][Error] Please check if the file exists! " << odomRawFile << endl;
        rec.close();
        ros::shutdown();
        exit(-1);
    }
    float x, y, theta;
    string line;

    //! main loop
    TestTrackKLT klt;
    Mat imgGray, imgUn, imgClahe;
    Ptr<CLAHE> clahe = createCLAHE(3.0, cv::Size(8, 8));
    num = std::min(num, static_cast<int>(imgFiles.size()));
    int skipFrames = 30;
    WorkTimer timer;
    for (int i = 0; i < num; ++i) {
        if (i < skipFrames) {
            std::getline(rec, line);
            continue;
        }

        std::getline(rec, line);
        istringstream iss(line);
        iss >> x >> y >> theta;
        Se2 odo(x, y, normalizeAngle(theta));

        imgGray = imread(imgFiles[i].fileName, CV_LOAD_IMAGE_GRAYSCALE);
        if (imgGray.data == nullptr)
            continue;
        clahe->apply(imgGray, imgClahe);
        cv::undistort(imgClahe, imgUn, K, D);

        klt.trackCellToRef(imgUn, odo);
    }

    return 0;
}
*/
