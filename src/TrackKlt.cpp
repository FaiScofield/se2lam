//
// Created by lmp on 19-10-28.
//
#include "TrackKlt.h"
#include "Config.h"
#include "LineDetector.h"
#include "LocalMapper.h"
#include "Map.h"
#include "ORBmatcher.h"
#include "converter.h"
#include "cvutil.h"
#include "optimizer.h"
#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <ros/ros.h>

namespace se2lam
{

using namespace std;
using namespace cv;
using namespace Eigen;

typedef unique_lock<mutex> locker;

// setMask用到的结构体
struct PairMask {
    cv::Point2f ptInForw;    // 点在当前帧下的像素坐标
    cv::Point2f ptInCurr;    // 对应上一帧的点的像素坐标
    cv::Point2f ptInPrev;
    unsigned long firstAdd;  // 所在图像的id
    size_t idxToAdd;         // 与生成帧的匹配点索引
    int cellLabel;           // 特征点所在块的lable
    int trackCount;          // 被追踪上的次数
};


bool TrackKlt::mbUseOdometry = true;

TrackKlt::TrackKlt()
    : mState(cvu::NO_READY_YET), mLastState(cvu::NO_READY_YET), mbPrint(true),
      mbNeedVisualization(false), mnGoodPrl(0), mnInliers(0), mnMatchSum(0), mnTrackedOld(0),
      mnLostFrames(0), mbFinishRequested(false), mbFinished(false)
{
    mpORBextractor = new ORBextractor(Config::MaxFtrNumber, Config::ScaleFactor, Config::MaxLevel);
    mpORBmatcher = new ORBmatcher(0.9, true);

    mLocalMPs = vector<Point3f>(Config::MaxFtrNumber, Point3f(-1.f, -1.f, -1.f));
    nMinFrames = min(2, cvCeil(0.25 * Config::FPS));  // 上溢
    nMaxFrames = cvFloor(5 * Config::FPS);            // 下溢
    mMaxAngle = g2o::deg2rad(90.);
    mMaxDistance = 0.5 * Config::UpperDepth * 0.15;

    mK = Config::Kcam;
    mD = Config::Dcam;
    mAffineMatrix = Mat::eye(3, 2, CV_64FC1);  // double

    mbPrint = Config::GlobalPrint;
    mbNeedVisualization = Config::NeedVisualization;

    // klt跟踪添加
    mnImgRows = Config::ImgSize.height;  // 240
    mnImgCols = Config::ImgSize.width;   // 320
    mnMaxCnt = Config::MaxFtrNumber;     // 最大特征点数量, 非分块时使用

    mnCellHeight = 60;                  // 分块尺寸
    mnCellWidth = 80;
    mnCellsRow = mnImgRows / mnCellHeight;  // 240/60=4
    mnCellsCol = mnImgCols / mnCellWidth;  // 320/80=4
    mnCells = mnCellsRow * mnCellsCol;       // 16
    mnMaxNumPtsInCell = mnMaxCnt / mnCells;                 // 分块检点的最大点数
    mvNumPtsInCell.resize(mnCells, 0);
    mnMaskRadius = 3;                        // mask建立时的特征点周边半径
    mbKFinserted = false;
}

TrackKlt::~TrackKlt()
{
    delete mpORBextractor;
    delete mpORBmatcher;
}

void TrackKlt::run()
{
    if (Config::LocalizationOnly)
        return;

    if (mState == cvu::NO_READY_YET)
        mState = cvu::FIRST_FRAME;

    WorkTimer timer;

    ros::Rate rate(Config::FPS * 5);
    while (ros::ok()) {
        if (checkFinish())
            break;

        bool sensorUpdated = mpSensors->update();
        if (sensorUpdated) {
            timer.start();

            mLastState = mState;

            Mat img;
            Se2 odo;
            double dTheta = 0;
            mpSensors->readData(odo, img, dTheta);

            Mat imgUn, imgClahed;
            Ptr<CLAHE> clahe = createCLAHE(3.0, Size(8, 8));
            clahe->apply(img, imgClahed);
            undistort(imgClahed, imgUn, mK, mD);

            double t1 = timer.count();
            timer.start();
            {
                // KLT跟踪
                locker lock(mMutexForPub);
                if (mState == cvu::FIRST_FRAME) {
                    // 为初始帧创建帧信息, 分块模式
                    createFrameFirstKlt(imgUn, odo);
                } else {
                    // 非初始帧则进行tracking,计算位姿trackKltcell为分块模式，trackKlt为全局模式
                    trackKltCell(imgUn, odo, dTheta);
                    // trackKlt(img, odo, theta);
                }
                mpMap->setCurrentFramePose(mCurrentFrame.getPose());
            }

            mLastFrame = mCurrentFrame;
            mLastOdom = odo;

            double t2 = timer.count();
            fprintf(stdout, "[Track][Timer] #%ld 当前帧前端读取数据耗时: %.2fms\n",
                    mCurrentFrame.id, t1);
            fprintf(stdout, "[Track][Timer] #%ld 当前帧前端取点追踪耗时: %.2fms\n",
                    mCurrentFrame.id, t2);
            fprintf(stdout, "[Track][Timer] #%ld 当前帧前端进程总耗时: %.2fms\n", mCurrentFrame.id,
                    t1 + t2);
        }

        rate.sleep();
    }

    cerr << "[Track][Info ] Exiting tracking .." << endl;
    setFinish();
}


/**
 * @brief   klt跟踪模式下创建首帧
 * @param img   输入图像(灰度图), 且已经过畸变校正
 * @param odo   里程计信息
 *
 * @author  Maple.Liu
 * @date    2019.10.28
 */
void TrackKlt::createFrameFirstKlt(const Mat& img, const Se2& odo)
{
    mForwImg = img;

    // 直线掩模上特征点提取
    vector<Keyline> lines;
    Mat lineMask = getLineMask(mForwImg, lines, false);
    detectFeaturePointsCell(mForwImg, lineMask);// 更新mvNewPts, mvCellPointsNum, mvKPCellLable
    addNewPoints(); // 将新检测到的特征点mvNewPts添加到mForwPts中

    // 创建当前帧
    if (mvForwPts.size() > 200) {
        vector<KeyPoint> vKPs;
        vKPs.resize(mvForwPts.size());
        for (int i = 0, iend = mvForwPts.size(); i < iend; ++i) {
            vKPs[i].pt = mvForwPts[i];
            vKPs[i].octave = 0;
        }
        mCurrentFrame = Frame(mForwImg, odo, vKPs, mpORBextractor);

        cout << "========================================================" << endl;
        cout << "[Track][Info ] Create first frame with " << mCurrentFrame.N << " features. "
             << "And the start odom is: " << odo << endl;
        cout << "========================================================" << endl;

        mCurrentFrame.setPose(Se2(0.f, 0.f, 0.f));
        mpReferenceKF = make_shared<KeyFrame>(mCurrentFrame);  // 首帧为关键帧
        mpMap->insertKF(mpReferenceKF);  // 首帧的KF直接给Map,没有给LocalMapper
        mpMap->updateLocalGraph();       // 首帧添加到LocalMap里

        resetLocalTrack();
        resetKltData();

        mState = cvu::OK;
    } else {
        cout << "[Track][Warni] Failed to create first frame for too less keyPoints: "
             << mvForwPts.size() << endl;
        Frame::nextId = 1;

        mState = cvu::FIRST_FRAME;
    }
}

/*Klt跟踪模式下实现特征点跟踪
 * img：输入图像
 * odo：IMU信息
 * imuTheta:相邻两帧的陀螺仪旋转角度
 * @Maple
 * 2019.10.28
 */
void TrackKlt::trackKlt(const Mat& img, const Se2& odo, double imuTheta)
{
    mForwImg = img;

    size_t n1 = 0, n2 = 0, n3 = 0, n4 = 0;
    Mat show1, show2, show3, show4, show5;

    //! 1.预测上一帧img和KP旋转后的值
    predictPointsAndImage(imuTheta);  // 得到旋转后的预测点mvPrevPts

    //! 2.光流追踪上一帧的KP
    if (!mvCurrPts.empty()) {
        vector<uchar> status;
        vector<float> err;
        calcOpticalFlowPyrLK(mPrevImg, mForwImg, mvPrevPts, mvForwPts, status, err, Size(21, 21), 0);
        n1 = mvForwPts.size();

        for (size_t i = 0, iend = mvForwPts.size(); i < iend; i++) {
            if (status[i] && !inBorder(mvForwPts[i]))
                status[i] = 0;
        }

        reduceVectorCell(status);   // 把所有跟踪失败和图像外的点剔除
        n2 = mvForwPts.size();
//        show1 = drawMachesPointsToLastFrame("Match Predict");

        rejectWithRansac(true); // Ransac匹配剔除outliers, 通过未旋转的ptsCurr
        n3 = mvForwPts.size();
//        show2 = drawMachesPointsToLastFrame("Match Ransac ");

        for (auto& n : mvTrackCount) // 光流追踪成功, 特征点被成功跟中的次数就加一
            n++;
    }

    //! 3.更新当前帧与参考关键帧的匹配关系
    // 统计当前帧与关键帧匹配点的个数, 更新匹配关系
//    mMatchIdxWithRefKF.clear();
//    for (size_t i = 0, iend = ptsForw.size(); i < iend; ++i) {
//        if (mvIdFirstAdded[i] == mpReferenceKF->id)
//            mMatchIdxWithRefKF.emplace(i, mvIdFirstAdded[i]);
//    }
//    g_matchToRefSum += mMatchIdxWithRefKF.size();
//    show3 = drawMachesPointsToRefFrame("Match Ref KF");
//    printf("#%ld 光流追踪, 从参考帧追踪上的点数/平均追踪点数 = %ld/%.2f\n", idCurr,
//           forwMatchIdxWithRefKF.size(), g_matchToRefSum / (idCurr - 1.0));

    //! 4.有必要的话提取新的特征点
    setMask();  // 设置mask, 去除密集点
    n4 = mvForwPts.size();
//    show4 = drawMachesPointsToLastFrame("Masked & Added");
    printf("#%ld 光流追踪, 去密后剩点数/内点数/追踪上的点数/总点数 = %ld/%ld/%ld/%ld\n", Frame::nextId,
           n4, n3, n2, n1);

    int needPoints = mnMaxCnt - static_cast<int>(mvForwPts.size());
    if (needPoints > 50) {
        vector<Keyline> klTmp;
        Mat maskline = getLineMask(mForwImg, klTmp, false);
        mMask = mMask.mul(maskline);
        goodFeaturesToTrack(mForwImg, mvNewPts, needPoints, 0.05, mnMaskRadius, mMask);
    } else {
        mvNewPts.clear();
    }
//    show5 = drawNewAddPointsInMatch(show4);

//    Mat showMatchs;
//    vconcat(show1, show2, showMatchs);
//    vconcat(showMatchs, show5, showMatchs);
//    vconcat(showMatchs, show3, showMatchs);
//    imshow("KLT Matches To Last & Ransac & Ref", showMatchs);

    printf("#%ld 光流追踪, 新增点数为: %ld, ", Frame::nextId, mvNewPts.size());
    addNewPoints();
    printf("目前共有点数为: %ld\n", mvForwPts.size());

    //! 更新Frame
    vector<KeyPoint> vKPsCurFrame(mvForwPts.size());
    cv::KeyPoint::convert(mvForwPts, vKPsCurFrame);
    mCurrentFrame = Frame(mForwImg, odo, vKPsCurFrame, mpORBextractor);

    // 更新当前帧与关键帧的匹配关系
//    mvMatchIdx.resize(mpReferenceKF->N, -1);  // 与关键帧匹配关系
//    for (int i = 0; i < matchedPtsWithRef; i++) {
//        if (mvIdxToFirstAdded[i] < 0)
//            cout << "IDError" << endl;
//        else
//            mvMatchIdx[mvIdxToFirstAdded[i]] = i;
//    }


    updateFramePose();

    mnTrackedOld = doTriangulate();  // 更新viewMPs

    N1 += mnTrackedOld;
    N2 += mnInliers;
    N3 += mnMatchSum;
    if (mbPrint && mCurrentFrame.id % 50 == 0) {  // 每隔50帧输出一次平均匹配情况
        float sum = mCurrentFrame.id - 1.0;
        printf("[Track][Info ] #%ld 与参考帧匹配平均点数: tracked/matched/matchedSum = "
               "%.2f/%.2f/%.2f .\n",
               mCurrentFrame.id, N1 * 1.f / sum, N2 * 1.f / sum, N3 * 1.f / sum);
    }

    // Need new KeyFrame decision
    if (needNewKF()) {
        PtrKeyFrame pKF = make_shared<KeyFrame>(mCurrentFrame);
        assert(mpMap->getCurrentKF()->mIdKF == mpReferenceKF->mIdKF);

        //! TODO 预计分量，这里暂时没用上
        mpReferenceKF->preOdomFromSelf = make_pair(pKF, preSE2);
        pKF->preOdomToSelf = make_pair(mpReferenceKF, preSE2);

        // 添加给LocalMapper，LocalMapper会根据mLocalMPs生成真正的MP
        // LocalMapper会在更新完共视关系和MPs后把当前KF交给Map
        mpLocalMapper->addNewKF(pKF, mLocalMPs, mvMatchIdx, mvbGoodPrl);

        mpReferenceKF = pKF;
        resetLocalTrack();
    }

    resetKltData();
}


/**
 * @brief   klt分块提点跟踪模式下实现特征点跟踪
 * @param img       输入图像
 * @param odo       里程计信息
 * @param imuTheta  角度差值
 *
 * @author  Maple.Liu
 * @date    2019.10.28
 */
void TrackKlt::trackKltCell(const Mat& img, const se2lam::Se2& odo, double imuTheta)
{
    mForwImg = img;

    // 如果添加了关键帧则重置一些变量
    if (mbKFinserted) {
        mbKFinserted = false;
        mvIdFirstAdded.clear();

        mvIdxToFirstAdded.resize(mpReferenceKF->N);  // 重置跟踪标签关键帧与当前帧的匹配关系
        for (size_t i = 0; i < mpReferenceKF->N; ++i)
            mvIdxToFirstAdded[i] = i;
        mvTrackCount.clear();
        mvTrackCount.resize(mpReferenceKF->N, 1);  //重置跟踪次数
    }

    size_t n1 = 0, n2 = 0, n3 = 0, n4 = 0;
    Mat show1, show2, show3, show4, show5;

    //! 1.预测上一帧img和KP旋转后的值
    predictPointsAndImage(imuTheta);  // 得到旋转后的预测点mvPrevPts

    //! 2.光流追踪上一帧的KP
    if (!mvCurrPts.empty()) {
        vector<uchar> status;
        vector<float> err;
        calcOpticalFlowPyrLK(mPrevImg, mForwImg, mvPrevPts, mvForwPts, status, err, Size(21, 21), 0);

        // 将位于图像边界外的点标记为0
        for (size_t i = 0, iend = mvForwPts.size(); i < iend; i++) {
            if (status[i] && !inBorder(mvForwPts[i]))
                status[i] = 0;
        }

        reduceVectorCell(status);  // 把所有跟踪失败和图像外的点剔除
        n2 = mvForwPts.size();
//        show1 = drawMachesPointsToLastFrame("Match Predict");

        rejectWithRansac(true);  // Ransac匹配剔除outliers, 通过未旋转的ptsCurr
        n3 = mvForwPts.size();
//        show2 = drawMachesPointsToLastFrame("Match Ransac ");

        for (auto& n : mvTrackCount)  // 光流追踪成功, 特征点被成功跟中的次数就加一
            n++;

    }

    //! 3.更新当前帧与参考关键帧的匹配关系
    // 统计当前帧与关键帧匹配点的个数, 更新匹配关系
//    forwMatchIdxWithRefKF.clear();
//    for (size_t i = 0, iend = ptsForw.size(); i < iend; ++i) {
//        if (idFirstAdd[i] == KFRef->id)
//            forwMatchIdxWithRefKF.emplace(i, idxToFirstAdd[i]);
//    }
//    g_matchToRefSum += forwMatchIdxWithRefKF.size();
//    show3 = drawMachesPointsToRefFrame("Match Ref KF");
//    printf("#%ld 光流追踪, 从参考帧追踪上的点数/平均追踪点数 = %ld/%.2f\n", idCurr,
//           forwMatchIdxWithRefKF.size(), g_matchToRefSum / (idCurr - 1.0));

    //! 4.有必要的话提取新的特征点
    setMaskCell();  // 设置mask, 去除密集点
    n4 = mvForwPts.size();
//    show4 = drawMachesPointsToLastFrame("Masked & Added");
    printf("#%ld 光流追踪, 去密后剩点数/内点数/追踪上的点数/总点数 = %ld/%ld/%ld/%ld\n", Frame::nextId,
           n4, n3, n2, n1);

    // 分块情况下最好不要借助直线掩模
//    vector<Keyline> lines;
//    Mat maskline = getLineMask(mForwImg, lines, false);
//    mMask = mMask.mul(maskline);
    detectFeaturePointsCell(mForwImg, mMask);

//    show5 = drawNewAddPointsInMatch(show4);

//    Mat showMatchs;
//    vconcat(show1, show2, showMatchs);
//    vconcat(showMatchs, show5, showMatchs);
//    vconcat(showMatchs, show3, showMatchs);
//    imshow("KLT Matches To Last & Ransac & Ref", showMatchs);

    printf("#%ld 光流追踪, 新增点数为: %ld, ", Frame::nextId, mvNewPts.size());
    addNewPoints();
    printf("目前共有点数为: %ld\n", mvForwPts.size());

    // 统计当前帧与关键帧匹配点的个数
//    int matchedPtsWithRef = 0;
//    for (auto& n : mvIdFirstAdded) {
//        if (n == mpReferenceKF->id)
//            matchedPtsWithRef++;
//    }

    //! 5.更新Frame
    vector<KeyPoint> vKPsCurFrame(mvForwPts.size());
    cv::KeyPoint::convert(mvForwPts, vKPsCurFrame);
    mCurrentFrame = Frame(mForwImg, odo, vKPsCurFrame, mpORBextractor);

    // 更新当前帧与关键帧的匹配关系
//    mvMatchIdx.resize(mpReferenceKF->N, -1);  // 与关键帧匹配关系
//    for (int i = 0; i < matchedPtsWithRef; i++) {
//        if (mvIdxToFirstAdded[i] < 0)
//            cout << "IDError" << endl;
//        else
//            mvMatchIdx[mvIdxToFirstAdded[i]] = i;
//    }

//    mnMatchSum = matchedPtsWithRef;

//    mnInliers = mnMatchSum;
    // 利用基础矩阵F计算匹配内点，内点数大于10才能继续
//    mnInliers = removeOutliers();  // H更新
//    cout << "[Track] ORBmatcher get valid/tatal matches " << mnInliers << "/" << mRef_kpt <<
//    endl;
    updateFramePose();

    drawMatchesForPub(false);
    mnTrackedOld = doTriangulate();  // 更新viewMPs
    printf("[Track][Info ] #%ld 与参考帧匹配获得的点数: trackedOld/inliers/matchedSum = %d/%d/%d.\n",
           mCurrentFrame.id, mnTrackedOld, mnInliers, mnMatchSum);

    N1 += mnTrackedOld;
    N2 += mnInliers;
    N3 += mnMatchSum;
    if (mbPrint && mCurrentFrame.id % 20 == 0) {  // 每隔50帧输出一次平均匹配情况
        float sum = mCurrentFrame.id - 1.0;
        printf("[Track][Info ] #%ld 与参考帧匹配平均点数: tracked/matched/matchedSum = "
               "%.2f/%.2f/%.2f .\n",
               mCurrentFrame.id, N1 * 1.f / sum, N2 * 1.f / sum, N3 * 1.f / sum);
    }

    // Need new KeyFrame decision
    if (needNewKF()) {
        mbKFinserted = true;

        PtrKeyFrame pKF = make_shared<KeyFrame>(mCurrentFrame);

        assert(mpMap->getCurrentKF()->mIdKF == mpReferenceKF->mIdKF);

        //! TODO 预计分量，这里暂时没用上
        mpReferenceKF->preOdomFromSelf = make_pair(pKF, preSE2);
        pKF->preOdomToSelf = make_pair(mpReferenceKF, preSE2);

        // 添加给LocalMapper，LocalMapper会根据mLocalMPs生成真正的MP
        // LocalMapper会在更新完共视关系和MPs后把当前KF交给Map
        mpLocalMapper->addNewKF(pKF, mLocalMPs, mvMatchIdx, mvbGoodPrl);

        mpReferenceKF = pKF;
        resetLocalTrack();
    }
}

//! 根据RefKF和当前帧的里程计更新先验位姿和变换关系, odom是se2绝对位姿而非增量
//! 如果当前帧不是KF，则当前帧位姿由RefK叠加上里程计数据计算得到
//! 这里默认场景是工业级里程计，精度比较准
void TrackKlt::updateFramePose()
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
    Se2 odok = mCurFrame.odom - mLastOdom;
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

//! 当前帧设为KF时才执行，将当前KF变参考帧，将当前帧的KP转到mPrevMatched中.
void TrackKlt::resetLocalTrack()
{
    KeyPoint::convert(mCurrentFrame.mvKeyPoints, mPrevMatched);  // KeyPoint转Point2f

    // 更新当前Local MP为参考帧观测到的MP
    mLocalMPs = mpReferenceKF->mvViewMPs;
    mnGoodPrl = 0;
    mvMatchIdx.clear();

    for (int i = 0; i < 3; ++i)
        preSE2.meas[i] = 0;
    for (int i = 0; i < 9; ++i)
        preSE2.cov[i] = 0;

    mAffineMatrix = Mat::eye(3, 2, CV_64FC1);

    // 重置klt相关变量
//    size_t n = mvCurrPts.size();
//    mvTrackCount.resize(n, 1);
//    mvIdFirstAdded.resize(n, mpReferenceKF->id);
//    mvIdxToFirstAdded.clear();
//    mvIdxToFirstAdded.reserve(n);
//    for (size_t i = 0; i < n; ++i)
//        mvIdxToFirstAdded.push_back(i);
}

void TrackKlt::resetKltData()
{
    mCurrImg = mForwImg.clone();  //! NOTE 需要深拷贝
    mvCurrPts = mvForwPts;
    mLastFrame = mCurrentFrame;
}

//! 可视化用，数据拷贝
size_t TrackKlt::copyForPub(Mat& img1, Mat& img2, vector<KeyPoint>& kp1, vector<KeyPoint>& kp2,
                            vector<int>& vMatches12)
{
    if (!mbNeedVisualization)
        return 0;
    if (mvMatchIdx.empty())
        return 0;

    locker lock(mMutexForPub);

    mpReferenceKF->copyImgTo(img1);
    mCurrentFrame.copyImgTo(img2);

    kp1 = mpReferenceKF->mvKeyPoints;
    kp2 = mCurrentFrame.mvKeyPoints;
    vMatches12 = mvMatchIdx;

    return mvMatchIdx.size();
}

void TrackKlt::drawFrameForPub(Mat& imgLeft)
{
    if (!mbNeedVisualization)
        return;

    locker lock(mMutexForPub);

    //! 画左侧两幅图
    Mat imgUp = mCurrentFrame.mImage.clone();
    Mat imgDown = mpReferenceKF->mImage.clone();
    if (imgUp.channels() == 1)
        cvtColor(imgUp, imgUp, CV_GRAY2BGR);
    if (imgDown.channels() == 1)
        cvtColor(imgDown, imgDown, CV_GRAY2BGR);

    for (size_t i = 0, iend = mvMatchIdx.size(); i != iend; ++i) {
        Point2f ptRef = mpReferenceKF->mvKeyPoints[i].pt;
        if (mvMatchIdx[i] < 0) {
            circle(imgDown, ptRef, 3, Scalar(255, 0, 0), 1);  // 未匹配上的为蓝色
            continue;
        } else {
            circle(imgDown, ptRef, 3, Scalar(0, 255, 0), 1);  // 匹配上的为绿色

            Point2f ptCurr = mCurrentFrame.mvKeyPoints[mvMatchIdx[i]].pt;
            circle(imgUp, ptCurr, 3, Scalar(0, 255, 0), 1);  // 当前KP为绿色
            circle(imgUp, ptRef, 3, Scalar(0, 0, 255), 1);   // 参考KP为红色
            line(imgUp, ptRef, ptCurr, Scalar(0, 255, 0));
        }
    }
    vconcat(imgUp, imgDown, imgLeft);

    char strMatches[64];
    std::snprintf(strMatches, 64, "F: %ld-%ld, M: %d/%d", mpReferenceKF->id, mCurrentFrame.id,
                  mnInliers, mnMatchSum);
    putText(imgLeft, strMatches, Point(50, 15), 1, 1, Scalar(0, 0, 255), 2);
}

void TrackKlt::drawMatchesForPub(bool warp)
{
    if (!mbNeedVisualization)
        return;
    if (mCurrentFrame.id == mpReferenceKF->id)
        return;

    Mat imgWarp, A12, A21;
    Mat imgCur = mCurrentFrame.mImage.clone();
    Mat imgRef = mpReferenceKF->mImage.clone();
    if (imgCur.channels() == 1)
        cvtColor(imgCur, imgCur, CV_GRAY2BGR);
    if (imgRef.channels() == 1)
        cvtColor(imgRef, imgRef, CV_GRAY2BGR);

    // 所有KP先上蓝色
    drawKeypoints(imgCur, mCurrentFrame.mvKeyPoints, imgCur, Scalar(255, 0, 0),
                  DrawMatchesFlags::DRAW_OVER_OUTIMG);
    drawKeypoints(imgRef, mpReferenceKF->mvKeyPoints, imgRef, Scalar(255, 0, 0),
                  DrawMatchesFlags::DRAW_OVER_OUTIMG);

    if (warp) {
        A12 = Mat::eye(3, 3, CV_64FC1);
        mAffineMatrix.copyTo(A12.rowRange(0, 2));
        A21 = A12.inv(DECOMP_SVD);
        warpPerspective(imgCur, imgWarp, A21, imgRef.size());
        hconcat(imgWarp, imgRef, mImgOutMatch);
    } else {
        hconcat(imgCur, imgRef, mImgOutMatch);
    }

    char strMatches[64];
    std::snprintf(strMatches, 64, "KF: %ld-%ld, M: %d/%d", mpReferenceKF->mIdKF,
                  mCurrentFrame.id - mpReferenceKF->id, mnInliers, mnMatchSum);
    putText(mImgOutMatch, strMatches, Point(240, 15), 1, 1, Scalar(0, 0, 255), 2);

    for (size_t i = 0, iend = mvMatchIdx.size(); i != iend; ++i) {
        if (mvMatchIdx[i] < 0) {
            continue;
        } else {
            Point2f ptRef = mpReferenceKF->mvKeyPoints[i].pt;
            Point2f ptCur = mCurrentFrame.mvKeyPoints[mvMatchIdx[i]].pt;

            Point2f ptl, ptr;
            if (warp) {
                Mat pt1 = (Mat_<double>(3, 1) << ptCur.x, ptCur.y, 1);
                Mat pt1_warp = A21 * pt1;
                pt1_warp /= pt1_warp.at<double>(2);
                ptl = Point2f(pt1_warp.at<double>(0), pt1_warp.at<double>(1));
            } else {
                ptl = ptCur;
            }
            ptr = ptRef + Point2f(imgCur.cols, 0);

            // 匹配上KP的为绿色
            circle(mImgOutMatch, ptl, 3, Scalar(0, 255, 0));
            circle(mImgOutMatch, ptr, 3, Scalar(0, 255, 0));
            line(mImgOutMatch, ptl, ptr, Scalar(255, 255, 0, 0.5));
        }
    }
}

Mat TrackKlt::getImageMatches()
{
    locker lock(mMutexForPub);
    return mImgOutMatch;
}

/**
 * @brief 计算KF之间的残差和信息矩阵, 后端优化用
 * @param dOdo      [Input ]后一帧与前一帧之间的里程计差值
 * @param Tc1c2     [Output]残差
 * @param Info_se3  [Output]信息矩阵
 */
void TrackKlt::calcOdoConstraintCam(const Se2& dOdo, Mat& cTc, g2o::Matrix6d& Info_se3)
{
    const Mat bTc = Config::Tbc;
    const Mat cTb = Config::Tcb;

    const Mat bTb = Se2(dOdo.x, dOdo.y, dOdo.theta).toCvSE3();

    cTc = cTb * bTb * bTc;

    float dx = dOdo.x * Config::OdoUncertainX + Config::OdoNoiseX;
    float dy = dOdo.y * Config::OdoUncertainY + Config::OdoNoiseY;
    float dtheta = dOdo.theta * Config::OdoUncertainTheta + Config::OdoNoiseTheta;

    g2o::Matrix6d Info_se3_bTb = g2o::Matrix6d::Zero();
    //    float data[6] = { 1.f/(dx*dx), 1.f/(dy*dy), 1, 1e4, 1e4,
    //    1.f/(dtheta*dtheta) };
    float data[6] = {1.f / (dx * dx), 1.f / (dy * dy), 1e-4, 1e-4, 1e-4, 1.f / (dtheta * dtheta)};
    for (int i = 0; i < 6; ++i)
        Info_se3_bTb(i, i) = data[i];
    Info_se3 = Info_se3_bTb;


    //    g2o::Matrix6d J_bTb_cTc = toSE3Quat(bTc).adj();
    //    J_bTb_cTc.block(0,3,3,3) = J_bTb_cTc.block(3,0,3,3);
    //    J_bTb_cTc.block(3,0,3,3) = g2o::Matrix3D::Zero();

    //    Info_se3 = J_bTb_cTc.transpose() * Info_se3_bTb * J_bTb_cTc;

    //    for (int i = 0; i < 6; ++i)
    //        for (int j = 0; j < i; ++j)
    //            Info_se3(i,j) = Info_se3(j,i);

    // assert(verifyInfo(Info_se3));
}

/**
 * @brief 计算KF与MP之间的误差不确定度，计算约束(R^t)*∑*R
 * @param Pc1   [Input ]MP在KF1相机坐标系下的坐标
 * @param Tc1w  [Input ]KF1相机到世界坐标系的变换
 * @param Tc2w  [Input ]KF2相机到世界坐标系的变换
 * @param info1 [Output]MP在KF1中投影误差的信息矩阵
 * @param info2 [Output]MP在KF2中投影误差的信息矩阵
 */
void TrackKlt::calcSE3toXYZInfo(Point3f Pc1, const Mat& Tc1w, const Mat& Tc2w,
                                Eigen::Matrix3d& info1, Eigen::Matrix3d& info2)
{
    Point3f O1 = Point3f(cvu::inv(Tc1w).rowRange(0, 3).col(3));
    Point3f O2 = Point3f(cvu::inv(Tc2w).rowRange(0, 3).col(3));
    Point3f Pw = cvu::se3map(cvu::inv(Tc1w), Pc1);
    Point3f vO1 = Pw - O1;
    Point3f vO2 = Pw - O2;
    float sinParallax = norm(vO1.cross(vO2)) / (norm(vO1) * norm(vO2));

    Point3f Pc2 = cvu::se3map(Tc2w, Pw);
    float length1 = norm(Pc1);
    float length2 = norm(Pc2);
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
    Point3f k1 = Pc1.cross(z1);
    float normk1 = norm(k1);
    float sin1 = normk1 / (norm(z1) * norm(Pc1));
    k1 = k1 * (std::asin(sin1) / normk1);
    Point3f k2 = Pc2.cross(z2);
    float normk2 = norm(k2);
    float sin2 = normk2 / (norm(z2) * norm(Pc2));
    k2 = k2 * (std::asin(sin2) / normk2);

    Mat R1, R2;
    Mat k1mat = (Mat_<float>(3, 1) << k1.x, k1.y, k1.z);
    Mat k2mat = (Mat_<float>(3, 1) << k2.x, k2.y, k2.z);
    Rodrigues(k1mat, R1);
    Rodrigues(k2mat, R2);

    info1 = toMatrix3d(R1.t() * info_xyz1 * R1);
    info2 = toMatrix3d(R2.t() * info_xyz2 * R2);
}

/**
* @brief   根据本质矩阵H剔除外点，利用了RANSAC算法
* @return  返回内点数
*/
int TrackKlt::removeOutliers()
{
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
        return 0;

    vector<unsigned char> mask;
    mAffineMatrix = estimateAffine2D(ptRef, ptCur, mask, RANSAC, 3.0);
//   Mat F = findFundamentalMat(pt1, pt2, FM_RANSAC, 3, 0.99, mask);
//   Mat H = findHomography(pt1, pt2, RANSAC, 3, mask);  // 朝天花板摄像头应该用H矩阵, F会退化

    int nInlier = 0;
    for (size_t i = 0, iend = mask.size(); i < iend; ++i) {
        if (!mask[i])
            mvMatchIdx[idxRef[i]] = -1;
        else
            nInlier++;
    }

// If too few match inlier, discard all matches. The enviroment might not be
// suitable for image tracking.
//    if (nInlier < 10) {
//        nInlier = 0;
//        std::fill(mvMatchIdx.begin(), mvMatchIdx.end(), -1);
//    }

    return nInlier;
}

bool TrackKlt::needNewKF()
{
    int nOldKP = mpReferenceKF->countObservations();
    bool c0 = static_cast<int>(mCurrentFrame.id - mpReferenceKF->id) > nMinFrames;
    bool c1 = static_cast<float>(mnTrackedOld) <= nOldKP * 0.5f;
    bool c2 = mnGoodPrl > 40;
    bool c3 = static_cast<int>(mCurrentFrame.id - mpReferenceKF->id) > nMaxFrames;
    bool c4 = mnMatchSum < 0.1f * Config::MaxFtrNumber || mnMatchSum < 20;
    bool bNeedNewKF = c0 && ((c1 && c2) || c3 || c4);

    bool bNeedKFByOdo = false;
    if (mbUseOdometry) {
        Se2 dOdo = mCurrentFrame.odom - mpReferenceKF->odom;
        bool c5 = static_cast<double>(abs(dOdo.theta)) >= mMaxAngle;  // 旋转量超过20°
        Mat cTc = Config::Tcb * dOdo.toCvSE3() * Config::Tbc;
        Mat xy = cTc.rowRange(0, 2).col(3);
        bool c6 = norm(xy) >= mMaxDistance;  // 相机的平移量足够大
        bool c7 = mnMatchSum < 10;           //

        bNeedKFByOdo = c5 || c6 || c7;  // 相机移动取决于深度上限,考虑了不同深度下视野的不同
    }
    bNeedNewKF = bNeedNewKF && bNeedKFByOdo;  // 加上odom的移动条件, 把与改成了或

    // 最后还要看LocalMapper准备好了没有，LocalMapper正在执行优化的时候是不接收新KF的
    if (mpLocalMapper->acceptNewKF()) {
        return bNeedNewKF;
    } else if (c0 && (c4 || c3) && bNeedKFByOdo) {
        printf("[Track][Info ] #%ld 强制添加KF\n", mCurrentFrame.id);
        mpLocalMapper->setAbortBA();  // 如果必须要加入关键帧,则终止LocalMap的优化
        return bNeedNewKF;
    }

/*
    int nMPObs = mpReferenceKF->countObservation();  // 注意初始帧观测数为0
    bool c1 = (float)mnTrackedOld > nMPObs * 0.5f;   // 2.关联MP数不能太多(要小于50%)
    if (nMPObs && c1) {
        printf("[Track][Info ] #%ld 不是KF, 因为关联MP数超过了50%%(%d)\n", mCurrentFrame.id,
               mnTrackedOld);
        return false;
    }

    bool c4 = mnInliers < 0.1f * Config::MaxFtrNumber;  // 3.3 或匹配内点数小于1/10最大特征点数
    if (c4 && mpLocalMapper->acceptNewKF()) {
        printf("[Track][Info ] #%ld 成为了KF, 因为匹配内点数小于10%%\n", mCurrentFrame.id);
        return true;
    } else if (c4) {
        printf("[Track][Info ] #%ld 不是KF, 虽然匹配内点数小于10%%, 但局部地图正在工作!\n",
               mCurrentFrame.id);
        return false;
    }
*/

    return false;
}


/**
* @brief 关键函数, 与参考帧匹配内点数大于10则进行三角化获取深度
*  mLocalMPs, mvbGoodPrl, mnGoodPrl 在此更新
* @return  返回匹配上参考帧的MP的数量
*/
int TrackKlt::doTriangulate()
{
    if (mCurrentFrame.id - mpReferenceKF->id < nMinFrames)
        return 0;

    Mat Tcr = mCurrentFrame.getTcr();
    Mat Tc1c2 = cvu::inv(Tcr);
    Point3f Ocam1 = Point3f(0.f, 0.f, 0.f);
    Point3f Ocam2 = Point3f(Tc1c2.rowRange(0, 3).col(3));
    mvbGoodPrl = vector<bool>(mpReferenceKF->N, false);
    mnGoodPrl = 0;

    // 相机1和2的投影矩阵
    Mat Proj1 = Config::PrjMtrxEye;                 // P1 = K * Mat::eye(3, 4, CV_32FC1)
    Mat Proj2 = Config::Kcam * Tcr.rowRange(0, 3);  // P2 = K * Tc2c1(3*4)
    // 1.遍历参考帧的KP
    int nTrackedOld(0), nGoodDepth(0), nBadDepth(0);
    for (size_t i = 0, iend = mpReferenceKF->N; i < iend; ++i) {
        if (mvMatchIdx[i] < 0)
            continue;

        // 2.如果参考帧的KP与当前帧的KP有匹配,且参考帧KP已经有对应的MP观测了，则可见地图点更新为此MP
        if (mpReferenceKF->hasObservation(i)) {
            mLocalMPs[i] = mpReferenceKF->mvViewMPs[i];
            //  mLocalMPs[i] = mpReferenceKF->getViewMPPoseInCamareFrame(i);
            nTrackedOld++;
            continue;
        }

        // 3.如果参考帧KP没有对应的MP，则为此匹配对KP三角化计算深度(相对参考帧的坐标)
        // 由于两个投影矩阵是两KF之间的相对投影, 故三角化得到的坐标是相对参考帧的坐标, 即Pc1
        Point2f pt1 = mpReferenceKF->mvKeyPoints[i].pt;
        Point2f pt2 = mCurrentFrame.mvKeyPoints[mvMatchIdx[i]].pt;
        Point3f Pc1 = cvu::triangulate(pt1, pt2, Proj1, Proj2);

        // 3.如果深度计算符合预期，就将有深度的KP更新到LocalMPs里, 其中视差较好的会被标记
        if (Config::acceptDepth(Pc1.z)) {
            nGoodDepth++;
            mLocalMPs[i] = Pc1;
            // 检查视差
            if (cvu::checkParallax(Ocam1, Ocam2, Pc1, 2)) {
                mnGoodPrl++;
                mvbGoodPrl[i] = true;
            }
        } else {  // 3.如果深度计算不符合预期，就剔除此匹配点对
            nBadDepth++;
            mvMatchIdx[i] = -1;
        }
    }
    printf("[Track][Info ] #%ld 1.三角化生成了%d个点且%d个具有良好视差, "
           "因深度不符合预期而剔除的匹配点对有%d个\n",
           mCurrentFrame.id, nGoodDepth, mnGoodPrl, nBadDepth);

    return nTrackedOld;
}

void TrackKlt::requestFinish()
{
    locker lock(mMutexFinish);
    mbFinishRequested = true;
}

bool TrackKlt::checkFinish()
{
    locker lock(mMutexFinish);
    return mbFinishRequested;
}

bool TrackKlt::isFinished()
{
    locker lock(mMutexFinish);
    return mbFinished;
}

void TrackKlt::setFinish()
{
    locker lock(mMutexFinish);
    mbFinished = true;
}


/*判断特征点是否超过边界，边界为3个像素
 * pt:需判断的点
 * @Maple
 * 2019.10.28
 */
bool TrackKlt::inBorder(const Point2f& pt)
{
    return Frame::minXUn <= pt.x && pt.x < Frame::maxXUn && Frame::minYUn <= pt.y &&
           pt.y < Frame::maxYUn;
}

/*更新容器
 * status:是否为误匹配的标志
 * @Maple
 * 2019.10.28
 */
void TrackKlt::reduceVector(const vector<uchar>& status)
{
    assert(mvPrevPts.size() == status.size());
    assert(mvCurrPts.size() == status.size());
    assert(mvForwPts.size() == status.size());

    int j = 0;
    for (size_t i = 0, iend = status.size(); i < iend; i++) {
        if (status[i]) {
            mvPrevPts[j] = mvPrevPts[i];
            mvCurrPts[j] = mvCurrPts[i];
            mvForwPts[j] = mvForwPts[i];
            mvIdFirstAdded[j] = mvIdFirstAdded[i];
            mvIdxToFirstAdded[j] = mvIdxToFirstAdded[i];
            mvTrackCount[j] = mvTrackCount[i];
            j++;
        }
    }
    mvPrevPts.resize(j);
    mvCurrPts.resize(j);
    mvForwPts.resize(j);
    mvIdFirstAdded.resize(j);
    mvIdxToFirstAdded.resize(j);
    mvTrackCount.resize(j);
}


/**
 * @brief 分块模式下根据内点标志更新当前帧(Forw)相关容器
 * @param status    内点标志
 *
 * @author  Maple.Liu
 * @date    2019.10.28
 */
void TrackKlt::reduceVectorCell(const vector<uchar>& status)
{
    assert(mvPrevPts.size() == status.size());
    assert(mvCurrPts.size() == status.size());
    assert(mvForwPts.size() == status.size());

    int j = 0;
    for (size_t i = 0, iend = status.size(); i < iend; i++) {
        if (status[i]) {
            mvPrevPts[j] = mvPrevPts[i];
            mvCurrPts[j] = mvCurrPts[i];
            mvForwPts[j] = mvForwPts[i];
            mvIdFirstAdded[j] = mvIdFirstAdded[i];
            mvIdxToFirstAdded[j] = mvIdxToFirstAdded[i];
            mvTrackCount[j] = mvTrackCount[i];
            mvCellLabel[j] = mvCellLabel[i];
            j++;
        } else {
            mvNumPtsInCell[mvCellLabel[i]]--;
        }
    }
    mvPrevPts.resize(j);
    mvCurrPts.resize(j);
    mvForwPts.resize(j);
    mvIdFirstAdded.resize(j);
    mvIdxToFirstAdded.resize(j);
    mvTrackCount.resize(j);
    mvCellLabel.resize(j);
}

/*利用Ransac筛选误匹配点
 * cell:是否分块
 * @Maple
 * 2019.10.28
 */
void TrackKlt::rejectWithRansac(bool isCell)
{
    if (mvForwPts.size() >= 8) {
        vector<unsigned char> inliersMask(mvCurrPts.size());
        findHomography(mvCurrPts, mvForwPts, RANSAC, 5, inliersMask);
        //  mAffineMatrix = estimateAffine2D(mvCurrPts, mvForwPts, inliersMask, RANSAC, 3.0);

        if (isCell)
            reduceVectorCell(inliersMask);
        else
            reduceVector(inliersMask);
    }
}

/*更新跟踪点
 * @Maple
 * 2019.10.28
 */
void TrackKlt::addNewPoints()
{
    size_t n = mvForwPts.size();
    size_t m = mvNewPts.size();
    mvForwPts.resize(n + m);
    mvTrackCount.resize(n + m);
    mvIdFirstAdded.resize(n + m);
    mvIdxToFirstAdded.resize(n + m);
    for (size_t i = 0; i < m; ++i) {
        mvForwPts[n + i] = mvNewPts[i];
        mvTrackCount[n + i] =  1;
        mvIdFirstAdded[n + i] =  Frame::nextId;  // Frame还未生成
        mvIdxToFirstAdded[n + i] = i + n;
    }
    mvNewPts.clear();
}

/**
 * @brief   获取预测点的坐标
 * @param rotImg    旋转后的图像
 * @param ptsPrev   预测后的特征点坐标
 * @param angle     相邻两帧的选择角度
 *
 * @author  Maple.Liu
 * @date    2019.10.28
 */
void TrackKlt::predictPointsAndImage(double angle)
{
    mvPrevPts.clear();
    if (abs(angle) > 0.01) {  // 0.001
        Point rotationCenter;
        rotationCenter.x = 160.5827 - 0.01525;  //! TODO
        rotationCenter.y = 117.7329 - 3.6984;

        getRotatedPoint(mvCurrPts, mvPrevPts, rotationCenter, angle);
        Mat rotationMatrix = getRotationMatrix2D(rotationCenter, angle * 180 / CV_PI, 1.);
        warpAffine(mCurrImg, mPrevImg, rotationMatrix, mCurrImg.size());
    } else {
        mPrevImg = mCurrImg.clone();
        mvPrevPts = mvCurrPts;
    }
}

/*绘制相邻两帧匹配点对
 * @Maple
 * 2019.10.28
 */
void TrackKlt::drawMachesPoints()
{
    vector<DMatch> goodmatches;
    vector<KeyPoint> laskeypoints, currentkeypoints;
    for (size_t i = 0, iend = mvCurrPts.size(); i < iend; ++i) {
        KeyPoint ls, cur;
        ls.pt.x = mvCurrPts[i].x;
        ls.pt.y = mvCurrPts[i].y;
        cur.pt.x = mvForwPts[i].x;
        cur.pt.y = mvForwPts[i].y;
        DMatch dmatches(i, i, 1);
        goodmatches.push_back(dmatches);
        laskeypoints.push_back(ls);
        currentkeypoints.push_back(cur);
    }
    Mat matcheImage;
    drawMatches(mCurrImg, laskeypoints, mForwImg, currentkeypoints, goodmatches, matcheImage);
    imshow("matches", matcheImage);
    waitKey(30);
}

/*绘制相邻两帧预测点对
 * @Maple
 * 2019.10.28
 */
void TrackKlt::drawPredictPoints()
{
    vector<DMatch> goodmatches;
    vector<KeyPoint> laskeypoints, currentkeypoints;
    for (size_t i = 0, iend = mvCurrPts.size(); i < iend; ++i) {
        KeyPoint ls, cur;
        ls.pt.x = mvCurrPts[i].x;
        ls.pt.y = mvCurrPts[i].y;
        cur.pt.x = mvPrevPts[i].x;
        cur.pt.y = mvPrevPts[i].y;
        DMatch dmatches(i, i, 1);
        goodmatches.push_back(dmatches);
        laskeypoints.push_back(ls);
        currentkeypoints.push_back(cur);
    }
    Mat matcheImage;
    drawMatches(mCurrImg, laskeypoints, mForwImg, currentkeypoints, goodmatches, matcheImage);
    imshow("matches", matcheImage);
    waitKey();
}


/**
 * @brief 根据旋转中心确定特征点预测位置
 * @note  旋转中心在里程计的坐标原点, 对于图像的旋转中心则需要利用外参对其进行变换后求得
 * @param srcPoints 待旋转的点
 * @param dstPoints 旋转后的点
 * @param center    旋转中心
 * @param angle     旋转角度
 *
 * @author Maple.Liu
 * @date   2019.10.28
 */
void TrackKlt::getRotatedPoint(const vector<Point2f>& srcPoints, vector<Point2f>& dstPoints,
                               const Point& center, double angle)
{
    dstPoints.resize(srcPoints.size());

    double row = static_cast<double>(mnImgRows);
    for (size_t i = 0, iend = srcPoints.size(); i < iend; i++) {
        double x1 = srcPoints[i].x;
        double y1 = row - srcPoints[i].y;
        double x2 = center.x;
        double y2 = row - center.y;
        double x = cvRound((x1 - x2) * cos(angle) - (y1 - y2) * sin(angle) + x2);
        double y = cvRound((x1 - x2) * sin(angle) + (y1 - y2) * cos(angle) + y2);
        y = row - y;

        dstPoints[i] = Point2f(x, y);
    }
}

/*将图像分块
 * frame:输入图像
 * cellImage：分块后图像存储位置
 * @Maple
 * 2019.10.28
 */
void TrackKlt::segImageToCells(const Mat& image, vector<Mat>& cellImgs)
{
    Mat imageCut;
    int m = image.cols / mnCellWidth;
    int n = image.rows / mnCellHeight;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            Rect rect(i * mnCellWidth, j * mnCellHeight, mnCellWidth, mnCellHeight);
            imageCut = Mat(image, rect);
            cellImgs.push_back(imageCut.clone());
        }
    }
}


/**
 * @brief 图像分块内进行特征点提取
 * @param image 输入图像
 * @param mask  输入掩模
 *
 * @author  Maple.Liu
 * @date    2019.10.28
 */
void TrackKlt::detectFeaturePointsCell(const Mat& image, const Mat& mask)
{
    vector<Mat> cellImgs, cellMasks;
    segImageToCells(image, cellImgs);
    segImageToCells(mask, cellMasks);

    mvNewPts.clear();
    mvNewPts.reserve(mnMaxNumPtsInCell * mnCells);
    int th = mnMaxNumPtsInCell * 0.2;
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
                int cellIndexX = i % mnCellsCol;
                int cellIndexY = i / mnCellsCol;

                Point2f& thisPoint = ptsInThisCell[j];
                thisPoint.x += mnCellWidth * cellIndexX;
                thisPoint.y += mnCellHeight * cellIndexY;

                mvNewPts.push_back(thisPoint);
                mvCellLabel.push_back(i);
            }
        }
    }
}


/**
 * @brief   设置一个掩模, 对跟踪点进行排序(连续追踪次数从大到小)并去除密集点
 *
 * @author  Maple.Liu
 * @date    2019.10.28
 */
void TrackKlt::setMask()
{
    mMask = Mat(mnImgRows, mnImgCols, CV_8UC1, Scalar(255));
    PairMask filtMask;
    vector<pair<int, PairMask>> vCountPtsId;

    for (size_t i = 0, iend = mvForwPts.size(); i < iend; ++i) {
        filtMask.firstAdd = mvIdFirstAdded[i];
        filtMask.idxToAdd = mvIdxToFirstAdded[i];
        filtMask.ptInForw = mvForwPts[i];
        filtMask.ptInCurr = mvCurrPts[i];
        filtMask.ptInPrev = mvPrevPts[i];
        vCountPtsId.emplace_back(mvTrackCount[i], filtMask);
    }
    sort(vCountPtsId.begin(), vCountPtsId.end(),
         [](const pair<int, PairMask>& a, const pair<int, PairMask>& b) {
             return a.first > b.first;
         });

//    size_t n = mvForwPts.size() * 0.75;
    mvCurrPts.clear();
    mvForwPts.clear();
    mvPrevPts.clear();
    mvTrackCount.clear();
    mvIdFirstAdded.clear();
    mvIdxToFirstAdded.clear();
//    mvCurrPts.reserve(n);
//    mvForwPts.reserve(n);
//    mvPrevPts.reserve(n);
//    mvTrackCount.reserve(n);
//    mvIdFirstAdded.reserve(n);
//    mvIdxToFirstAdded.reserve(n);
    for (const auto& it : vCountPtsId) {
        if (mMask.at<uchar>(it.second.ptInForw) == 255) {
            mvTrackCount.push_back(it.first);
            mvForwPts.push_back(it.second.ptInForw);
            mvCurrPts.push_back(it.second.ptInCurr);
            mvPrevPts.push_back(it.second.ptInPrev);
            mvIdFirstAdded.push_back(it.second.firstAdd);
            mvIdxToFirstAdded.push_back(it.second.idxToAdd);
            cv::circle(mMask, it.second.ptInForw, mnMaskRadius, Scalar(0), -1);  // 标记掩模
        }
    }
}

/**
 * @brief 对于klt跟踪成功的点进行排序(连续追踪次数从大到小)并去除密集点(分块模式下)
 *
 * @author  Maple.Liu
 * @date    2019.10.28
 *
 * @bug     最后一行代码存在bug
 * @note    bug已修改. 2019.11.07. Vance.Wu
 */
void TrackKlt::setMaskCell()
{
    mMask = Mat(mnImgRows, mnImgCols, CV_8UC1, Scalar(255));
    PairMask filtMask;
    vector<pair<int, PairMask>> vCountPtsId;

    for (size_t i = 0, iend = mvForwPts.size(); i < iend; ++i) {
        filtMask.firstAdd = mvIdFirstAdded[i];
        filtMask.idxToAdd = mvIdxToFirstAdded[i];
        filtMask.ptInForw = mvForwPts[i];
        filtMask.ptInCurr = mvCurrPts[i];
        filtMask.ptInPrev = mvPrevPts[i];
        filtMask.cellLabel = mvCellLabel[i];
        vCountPtsId.emplace_back(mvTrackCount[i], filtMask);
    }
    sort(vCountPtsId.begin(), vCountPtsId.end(),
         [](const pair<int, PairMask>& a, const pair<int, PairMask>& b) {
             return a.first > b.first;
         });

//    size_t n = mvForwPts.size() * 0.75;
    mvCurrPts.clear();
    mvForwPts.clear();
    mvPrevPts.clear();
    mvTrackCount.clear();
    mvIdFirstAdded.clear();
    mvIdxToFirstAdded.clear();
    mvCellLabel.clear();
//    mvCurrPts.reserve(n);
//    mvForwPts.reserve(n);
//    mvPrevPts.reserve(n);
//    mvTrackCount.reserve(n);
//    mvIdFirstAdded.reserve(n);
//    mvIdxToFirstAdded.reserve(n);
//    mvCellLabel.reserve(n);
    for (const auto& it : vCountPtsId) {
        if (mMask.at<uchar>(it.second.ptInForw) == 255) {
            mvTrackCount.push_back(it.first);
            mvForwPts.push_back(it.second.ptInForw);
            mvCurrPts.push_back(it.second.ptInCurr);
            mvPrevPts.push_back(it.second.ptInPrev);
            mvIdFirstAdded.push_back(it.second.firstAdd);
            mvIdxToFirstAdded.push_back(it.second.idxToAdd);
            mvCellLabel.push_back(it.second.cellLabel);
            cv::circle(mMask, it.second.ptInForw, mnMaskRadius, Scalar(0), -1);  // 标记掩模
        } else {
            //! bug 已改. @Vance - 20191107
            mvNumPtsInCell[it.second.cellLabel]--;
        }
    }
}

}  // namespace se2lam
