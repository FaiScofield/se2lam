//
// Created by lmp on 19-10-28.
//
#include "TrackKlt.h"
#include "Config.h"
#include "LocalMapper.h"
#include "Map.h"
#include "ORBmatcher.h"
#include "converter.h"
#include "cvutil.h"
#include "optimizer.h"
#include <ros/ros.h>
#include "lineDetection.h"
#include "Preprocess.h"

namespace se2lam
{
using namespace std;
using namespace cv;
using namespace Eigen;

typedef unique_lock<mutex> locker;

bool TrackKlt::mbUseOdometry = true;

TrackKlt::TrackKlt()
        : mState(cvu::NO_READY_YET), mLastState(cvu::NO_READY_YET), mbPrint(true),
          mbNeedVisualization(false), mnGoodPrl(0), mnInliers(0), mnMatchSum(0), mnTrackedOld(0),
          mnLostFrames(0), mbFinishRequested(false), mbFinished(false)
{
    mpORBextractor = new ORBextractor(Config::MaxFtrNumber, Config::ScaleFactor, Config::MaxLevel);

    mLocalMPs = vector<Point3f>(Config::MaxFtrNumber, Point3f(-1.f, -1.f, -1.f));

    nMinFrames = min(1, cvCeil(0.25 * Config::FPS));  // 上溢
    nMaxFrames = cvFloor(2 * Config::FPS);            // 下溢
    mMaxAngle = g2o::deg2rad(5.);//origen20
    mMaxDistance = 0.5 * Config::UpperDepth * 0.15;//0.5

    mK = Config::Kcam;
    mD = Config::Dcam;
    mHomography = Mat::eye(3, 3, CV_64FC1);  // float

    mbPrint = Config::GlobalPrint;
    mbNeedVisualization = Config::NeedVisualization;

    //klt跟踪添加
    mnImgRows = 240;//图像尺寸
    mnImgCols = 320;
    mbInterKeyFrame = false;
    mnMinDist = 0;//mask建立时的特征点周边半径
    mnMaxCnt = 1000; //最大特征点数量
    ceilColSize = 80;//分块尺寸
    ceilRowSize = 60;
    mnMaxCntCeil = 80;//分块检点的最大点数

}

TrackKlt::~TrackKlt()
{
    delete mpORBextractor;
}

//setMask用到的结构体
struct pair_mask
{
    cv::Point2f pt;//跟踪的点
    int id;//所在图像的id
    int index;//与关键帧的匹配对序列
};

//setMask用到的结构体(ceil模式下)
struct pair_mask_ceil
{
    cv::Point2f pt;//跟踪的点
    int id;//所在图像的id
    int index;//与关键帧的匹配对序列
    int clable;//特征点所在块的lable
};

void TrackKlt::run()
{
    if (Config::LocalizationOnly)
        return;

    if (mState == cvu::NO_READY_YET)
        mState = cvu::FIRST_FRAME;

    WorkTimer timer;
    double imgTime = 0.;
    ros::Rate rate(Config::FPS * 5);
    while (ros::ok()) {
        bool sensorUpdated = mpSensors->update();
        if (sensorUpdated) {
            timer.start();

            mLastState = mState;

            Mat img;
            Se2 odo;
            double Imu_theta = 0;
            bool useCeil;
            mpSensors->readData(odo,img,Imu_theta,useCeil);
            {
                 //KLT跟踪
                locker lock(mMutexForPub);
                if (mState == cvu::FIRST_FRAME)
                    createFrameFirstKlt(img, useCeil, odo); // 为初始帧创建帧信息true为分块模式，false为全局模式
                else
                {
                    if(useCeil)
                        trackKltCeil(img, odo, Imu_theta);       // 非初始帧则进行tracking,计算位姿trackKltCeil为分块模式，trackKlt为全局模式
                    else
                        trackKlt(img, odo, Imu_theta);       // 非初始帧则进行tracking,计算位姿trackKltCeil为分块模式，trackKlt为全局模式
                }

            }
            mpMap->setCurrentFramePose(mCurrentFrame.getPose());
            mLastOdom = odo;

            timer.stop();
            fprintf(stdout, "[Track][Timer] #%ld 当前帧前段追踪总耗时: %.2fms\n", mCurrentFrame.id,
                    timer.time);
        }

        if (checkFinish())
            break;

        rate.sleep();
    }

    cerr << "[Track][Info ] Exiting tracking .." << endl;
    setFinish();
}

/*Klt跟踪模式下创建首帧
 * img：输入图像
 * ceil:是否执行分块提点模式
 * odo：IMU信息
 * @Maple
 * 2019.10.28
 */
void TrackKlt::createFrameFirstKlt(const Mat& img, const bool ceil, const Se2& odo)
{
    mCurrentFrame = Frame(img, odo, ceil, mpORBextractor, mK, mD);

    if (mCurrentFrame.mvKeyPoints.size() > 100) {
        cout << "========================================================" << endl;
        cout << "[Track][Info ] Create first frame with " << mCurrentFrame.N << " features. "
             << "And the start odom is: " << odo << endl;
        cout << "========================================================" << endl;

        mCurrentFrame.setPose(Se2(0, 0, 0));
        mpReferenceKF = make_shared<KeyFrame>(mCurrentFrame);  // 首帧为关键帧
        mpMap->insertKF(mpReferenceKF);  // 首帧的KF直接给Map,没有给LocalMapper
        mpMap->updateLocalGraph();       // 首帧添加到LocalMap里

        resetLocalTrack();

        mState = cvu::OK;
    } else {
        cout << "[Track][Warni] Failed to create first frame for too less keyPoints: "
             << mCurrentFrame.mvKeyPoints.size() << endl;
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
void TrackKlt::trackKlt(const Mat &img, const Se2& odo, double imuTheta){

    WorkTimer timer;
    timer.start();

    cv::Mat gray,unimg;
    //如果图像是RGB将其转换成灰度图
    if (img.channels() != 1)
        cv::cvtColor(img, unimg, cv::COLOR_RGB2GRAY);
    else
    {
        unimg = img.clone();
    }

    undistort(unimg, gray, Config::Kcam, Config::Dcam);
    //如果是第一帧图像，将其赋值给第一帧图像
    if (mForwImg.empty())
    {
        //cur和forw分别是LK光流跟踪的前后两帧，forw才是真正的＂当前＂帧，cur实际上是上一帧，而pre是上一次发布的帧
        //如果当前帧的图像数据flow_img为空，说明当前是第一次读入图像数据
        //将读入的图像赋给当前帧flow_img
        //同时，还将读入的图像赋给mPrevImg,mCurImg,这是为了避免后面使用到这些数据时，它们是空的
        mPrevImg = mRefFrame.mPrevImg;
        mCurImg = mRefFrame.mCurImg;
        mForwImg = gray;
        mvNewPts = mRefFrame.mvNewPts;
        mvPrevPts = mRefFrame.mvPrevPts;
        mvCurPts = mRefFrame.mvCurPts;
        mvForwPts = mRefFrame.mvForwPts;
        mvIds = mRefFrame.mvIds;
        mvTrackNum = mRefFrame.mvTrackNum;
        mvTrackIdx = mRefFrame.mvTrackIdx;
    }
    else
    {
        //否则，说明之前就已经有图像读入
        //所以只需要更新当前帧mForwImg的数据
        mForwImg = gray;
        //如果添加了关键帧
        if(mbInterKeyFrame)
        {
            mbInterKeyFrame = false;
            mvIds.clear();
            mvIds.resize(mRefFrame.keyPoints.size(),mRefFrame.id);//重置mvIds
            mvTrackIdx.clear();//重置跟踪标签关键帧与当前帧的匹配关系
            for(int i = 0; i<mRefFrame.N;i++)
                mvTrackIdx.push_back(i);
            mvTrackNum.clear();
            mvTrackNum.resize(mRefFrame.keyPoints.size(),1);//重置跟踪次数
        }
    }

    cv::Mat rot_img;
    vector<cv::Point2f> points_prev;
    predictPointsAndImag(imuTheta, rot_img, points_prev);
    mvPrevPts = points_prev;//预测点
    if (mvCurPts.size() > 0)
    {
        vector<uchar> status;
        vector<float> err;
        cv::calcOpticalFlowPyrLK(rot_img, mForwImg, points_prev, mvForwPts, status, err, cv::Size(21, 21), 0);

        //将位于图像边界外的点标记为0
        for (int i = 0; i < int(mvForwPts.size()); i++)
            if (status[i] && !inBorder(mvForwPts[i]))
                status[i] = 0;
        //根据status,把跟踪失败的点剔除
        reduceVector(status);
    }

    //通过Ransac剔除outliers
    rejectWithRansac(false);

    //光流追踪成功.特征点被成功跟中的次数就加一
    //数值代表被追中的次数．数值越大，说明被追踪的就越久
    for (auto &n : mvTrackNum)
        n++;

    //绘制匹配图
//    drawMachesPoints();
//    drawPredictPoints();

    //设置mask
    setMask();//保证相邻的特征点之间要相隔３０个像素，设置mask
    //计算是否需要提取新的特征点
    int needPoints = mnMaxCnt - static_cast<int>(mvForwPts.size());
    if (needPoints > 50)
    {
        if(mMask.empty())
            cout << "mMask is empty " << endl;
        if (mMask.type() != CV_8UC1)
            cout << "mMask type wrong " << endl;
        if (mMask.size() != mForwImg.size())
            cout << "wrong size " << endl;
        Preprocess pre;
        Mat sharpImage1 = pre.sharpping(mForwImg,11);
        Mat gammaImage1 = pre.GaMma(sharpImage1,0.7);
        Mat sharpImage = pre.sharpping(img,8);
        Mat gammaImage = pre.GaMma(sharpImage,1.8);

//        cv::Mat imgGray;
//        //!  限制对比度自适应直方图均衡
//        Ptr<CLAHE> clahe = createCLAHE(3.0, cv::Size(8, 8));
//        clahe->apply(mForwImg, imgGray);
        Mat maskline = getLineMask(gammaImage, false);
        mMask = mMask.mul(maskline);
        cv::goodFeaturesToTrack(gammaImage, mvNewPts, needPoints, 0.01, mnMinDist, mMask);
    }
    else
        mvNewPts.clear();

    //将新检测到的特征点mvNewPts添加到mvForwPts中，id初始化－１,mvTrackNum初始化为１
    addPoints();
    //当下一帧图像到来时，当前帧数据就成为了上一帧发布的数据
    mPrevImg = mCurImg;
    //把当前帧的数据mForwImg,mvForwPts赋给上一帧mCurImg,mvCurPts
    mCurImg = mForwImg;
    mvCurPts = mvForwPts;

    //统计当前帧与关键帧匹配点的个数
    int mRef_kpt = 0;
    for(auto &n : mvIds)
        if(n == mRefFrame.id)
            mRef_kpt++;
        else
            continue;

    //获取当前帧的特征点
    vector<cv::KeyPoint> mckeyPoints;
    mckeyPoints.resize(mvForwPts.size());
    for (int i = 0;i<mvForwPts.size();i++)
    {
        mckeyPoints[i].pt = mvForwPts[i];
        mckeyPoints[i].octave = 0;
    }
    //更行当前帧与关键帧的匹配关系
    mvMatchIdx.resize(mRefFrame.keyPoints.size(),-1); //与关键帧匹配关系
    for(int i = 0; i< mRef_kpt; i++)
    {
        if(mvTrackIdx[i] < 0)
            cout<<"IDError"<<endl;
        else
            mvMatchIdx[mvTrackIdx[i]] = i;
    }
    //跟新Frame
    mCurrentFrame = Frame(img, odo, mckeyPoints, mpORBextractor, Config::Kcam, Config::Dcam);
    mnMatchSum = mRef_kpt;

    // 利用基础矩阵F计算匹配内点，内点数大于10才能继续
    mnInliers = removeOutliers();        // H更新
    cout << "[Track] ORBmatcher get valid/tatal matches " << mnInliers
         << "/" << mRef_kpt << endl;
    updateFramePose();
    if (mnInliers>10) {                     // 内点数大于10则三角化计算MP
        mnTrackedOld = doTriangulate();  // 更新viewMPs
        drawMatchesForPub(true);
        if (mbPrint)
            printf("[Track][Info ] #%ld 与参考帧匹配获得的点数: trackedOld/inliers/matchedSum = %d/%d/%d.\n",
                   mCurrentFrame.id, mnTrackedOld, mnInliers, mnMatchSum);
    } else {
        drawMatchesForPub(false);
        if (mbPrint)
            fprintf(stderr, "[Track][Warni] #%ld 与参考帧匹配内点数少于10! trackedOld/matchedSum = %d/%d.\n",
                    mCurrentFrame.id, mnTrackedOld, mnMatchSum);
    }

    N1 += mnTrackedOld;
    N2 += mnInliers;
    N3 += mnMatchSum;
    if (mbPrint && mCurrentFrame.id % 5 == 0) {  // 每隔50帧输出一次平均匹配情况
        float sum = mCurrentFrame.id - 1.0;
        printf("[Track][Info ] #%ld 与参考帧匹配平均点数: tracked/matched/matchedSum = "
               "%.2f/%.2f/%.2f .\n",
               mCurrentFrame.id, N1 * 1.f / sum, N2 * 1.f / sum, N3 * 1.f / sum);
    }

    // Need new KeyFrame decision
    if (needNewKF()) {
        mbInterKeyFrame = true;
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

    timer.stop();
}


/*Klt分块提点跟踪模式下实现特征点跟踪
 * img：输入图像
 * odo：IMU信息
 * imuTheta:相邻两帧的陀螺仪旋转角度
 * @Maple
 * 2019.10.28
 */
void TrackKlt::trackKltCeil(const Mat &img, const se2lam::Se2 &odo, double imuTheta) {

    WorkTimer timer;
    timer.start();
    //mRefFrame = mFrame;

    cv::Mat gray,unimg;
    //如果图像是RGB将其转换成灰度图
    if (img.channels() != 1)
        cv::cvtColor(img, unimg, cv::COLOR_RGB2GRAY);
    else
    {
        unimg = img.clone();
    }

    undistort(unimg, gray, Config::Kcam, Config::Dcam);
    //如果是第一帧图像，将其赋值给第一帧图像
    if (mForwImg.empty())
    {
        //cur和forw分别是LK光流跟踪的前后两帧，forw才是真正的＂当前＂帧，cur实际上是上一帧，而pre是上一次发布的帧
        //如果当前帧的图像数据flow_img为空，说明当前是第一次读入图像数据
        //将读入的图像赋给当前帧flow_img
        //同时，还将读入的图像赋给mPrevImg,mCurImg,这是为了避免后面使用到这些数据时，它们是空的
        mPrevImg = mRefFrame.mPrevImg;
        mCurImg = mRefFrame.mCurImg;
        mForwImg = gray;
        mvNewPts = mRefFrame.mvNewPts;
        mvPrevPts = mRefFrame.mvPrevPts;
        mvCurPts = mRefFrame.mvCurPts;
        mvForwPts = mRefFrame.mvForwPts;
        mvIds = mRefFrame.mvIds;
        mvTrackNum = mRefFrame.mvTrackNum;
        mvTrackIdx = mRefFrame.mvTrackIdx;
        mvCeilPointsNum = mRefFrame.mvCeilPointsNum;
        mvCeilLable = mRefFrame.mvCeilLable;
    }
    else
    {
        //否则，说明之前就已经有图像读入
        //所以只需要更新当前帧mForwImg的数据
        mForwImg = gray;
        //如果添加了关键帧
        if(mbInterKeyFrame)
        {
            mbInterKeyFrame = false;
            mvIds.clear();
            mvIds.resize(mRefFrame.keyPoints.size(),mRefFrame.id);//重置mvIds
            mvTrackIdx.clear();//重置跟踪标签关键帧与当前帧的匹配关系
            for(int i = 0; i<mRefFrame.N;i++)
                mvTrackIdx.push_back(i);
            mvTrackNum.clear();
            mvTrackNum.resize(mRefFrame.keyPoints.size(),1);//重置跟踪次数
        }
    }

    cv::Mat rot_img;
    vector<cv::Point2f> points_prev;
    predictPointsAndImag(imuTheta, rot_img, points_prev);
    mvPrevPts = points_prev;//预测点
    if (mvCurPts.size() > 0)
    {
        vector<uchar> status;
        vector<float> err;
        cv::calcOpticalFlowPyrLK(rot_img, mForwImg, points_prev, mvForwPts, status, err, cv::Size(20, 20), 0);

        //将位于图像边界外的点标记为0
        for (int i = 0; i < int(mvForwPts.size()); i++)
            if (status[i] && !inBorder(mvForwPts[i]))
                status[i] = 0;
        //根据status,把跟踪失败的点剔除
        reduceVectorCeil(status);
    }

    //通过Ransac剔除outliers
    rejectWithRansac(true);

    //光流追踪成功.特征点被成功跟中的次数就加一
    //数值代表被追中的次数．数值越大，说明被追踪的就越久
    for (auto &n : mvTrackNum)
        n++;

    //绘制匹配图
//    drawMachesPoints();
//    drawPredictPoints();

    //设置mask
    setMaskCeil();//保证相邻的特征点之间要相隔３０个像素，设置mask
    //计算是否需要提取新的特征点
    if(mMask.empty())
        cout << "mask is empty " << endl;
    if (mMask.type() != CV_8UC1)
        cout << "mask type wrong " << endl;
    if (mMask.size() != mForwImg.size())
        cout << "wrong size " << endl;

    Preprocess pre;
    Mat sharpImage1 = pre.sharpping(mForwImg,11);
    Mat gammaImage1 = pre.GaMma(sharpImage1,0.7);

    Mat sharpImage = pre.sharpping(img,8);
    Mat gammaImage = pre.GaMma(sharpImage,1.8);
//    cv::Mat imgGray;
//    //!  限制对比度自适应直方图均衡
//    Ptr<CLAHE> clahe = createCLAHE(3.0, cv::Size(8, 8));
//    clahe->apply(mForwImg, imgGray);
    Mat maskline = getLineMask(gammaImage, false);
    mMask = mMask.mul(maskline);
    detectFeaturePointsCeil(gammaImage1,mMask);

    //将新检测到的特征点mvNewPts添加到mvForwPts中，id初始化－１,mvTrackNum初始化为１
    addPoints();
    //当下一帧图像到来时，当前帧数据就成为了上一帧发布的数据
    mPrevImg = mCurImg;
    //把当前帧的数据mForwImg,mvForwPts赋给上一帧mCurImg,mvCurPts
    mCurImg = mForwImg;
    mvCurPts = mvForwPts;


    //统计当前帧与关键帧匹配点的个数
    int mRef_kpt = 0;
    for(auto &n : mvIds)
        if(n == mRefFrame.id)
            mRef_kpt++;
        else
            continue;

    //获取当前帧的特征点
    vector<cv::KeyPoint> mckeyPoints;
    mckeyPoints.resize(mvForwPts.size());
    for (int i = 0;i<mvForwPts.size();i++)
    {
        mckeyPoints[i].pt = mvForwPts[i];
        mckeyPoints[i].octave = 0;
    }
    //更行当前帧与关键帧的匹配关系
    mvMatchIdx.resize(mRefFrame.keyPoints.size(),-1); //与关键帧匹配关系
    for(int i = 0; i< mRef_kpt; i++)
    {
        if(mvTrackIdx[i] < 0)
            cout<<"IDError"<<endl;
        else
            mvMatchIdx[mvTrackIdx[i]] = i;
    }
    //跟新Frame
    mCurrentFrame = Frame(img, odo, mckeyPoints, mpORBextractor, Config::Kcam, Config::Dcam);
    mnMatchSum = mRef_kpt;

//    mnInliers = mnMatchSum;
    // 利用基础矩阵F计算匹配内点，内点数大于10才能继续
    mnInliers = removeOutliers();        // H更新
    cout << "[Track] ORBmatcher get valid/tatal matches " << mnInliers
         << "/" << mRef_kpt << endl;
    updateFramePose();
    if (mnInliers>10) {                     // 内点数大于10则三角化计算MP
        mnTrackedOld = doTriangulate();  // 更新viewMPs
        drawMatchesForPub(true);
        if (mbPrint)
            printf("[Track][Info ] #%ld 与参考帧匹配获得的点数: trackedOld/inliers/matchedSum = %d/%d/%d.\n",
                   mCurrentFrame.id, mnTrackedOld, mnInliers, mnMatchSum);
    } else {
        drawMatchesForPub(false);
        if (mbPrint)
            fprintf(stderr, "[Track][Warni] #%ld 与参考帧匹配内点数少于10! trackedOld/matchedSum = %d/%d.\n",
                    mCurrentFrame.id, mnTrackedOld, mnMatchSum);
    }

    N1 += mnTrackedOld;
    N2 += mnInliers;
    N3 += mnMatchSum;
    if (mbPrint && mCurrentFrame.id % 5 == 0) {  // 每隔50帧输出一次平均匹配情况
        float sum = mCurrentFrame.id - 1.0;
        printf("[Track][Info ] #%ld 与参考帧匹配平均点数: tracked/matched/matchedSum = "
               "%.2f/%.2f/%.2f .\n",
               mCurrentFrame.id, N1 * 1.f / sum, N2 * 1.f / sum, N3 * 1.f / sum);
    }

    // Need new KeyFrame decision
    if (needNewKF()) {
        mbInterKeyFrame = true;
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

    timer.stop();
}

//! mpKF是Last KF，根据Last KF和当前帧的里程计更新先验位姿和变换关系, odom是se2绝对位姿而非增量
//! 注意相机的平移量不等于里程的平移量!!!
//! 由于相机和里程计的坐标系不一致，在没有旋转时这两个平移量会相等，但是一旦有旋转，这两个量就不一致了!
//! 如果当前帧不是KF，则当前帧位姿由Last KF叠加上里程计数据计算得到
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
    //    Eigen::Map<Vector3d> meas(preSE2.meas);
    //    Se2 odok = mCurFrame.odom - mLastOdom;
    //    Vector2d odork(odok.x, odok.y);
    //    Matrix2d Phi_ik = Rotation2Dd(meas[2]).toRotationMatrix();
    //    meas.head<2>() += Phi_ik * odork;
    //    meas[2] += odok.theta;

    //    Matrix3d Ak = Matrix3d::Identity();
    //    Matrix3d Bk = Matrix3d::Identity();
    //    Ak.block<2, 1>(0, 2) = Phi_ik * Vector2d(-odork[1], odork[0]);
    //    Bk.block<2, 2>(0, 0) = Phi_ik;
    //    Eigen::Map<Matrix3d, RowMajor> Sigmak(preSE2.cov);
    //    Matrix3d Sigma_vk = Matrix3d::Identity();
    //    Sigma_vk(0, 0) = (Config::OdoNoiseX * Config::OdoNoiseX);
    //    Sigma_vk(1, 1) = (Config::OdoNoiseY * Config::OdoNoiseY);
    //    Sigma_vk(2, 2) = (Config::OdoNoiseTheta * Config::OdoNoiseTheta);
    //    Matrix3d Sigma_k_1 = Ak * Sigmak * Ak.transpose() + Bk * Sigma_vk * Bk.transpose();
    //    Sigmak = Sigma_k_1;
}

//! 当前帧设为KF时才执行，将当前KF变参考帧，将当前帧的KP转到mPrevMatched中.
void TrackKlt::resetLocalTrack()
{
    KeyPoint::convert(mCurrentFrame.mvKeyPoints, mPrevMatched);  // cv::KeyPoint转cv::Point2f

    mRefFrame = mCurrentFrame;
    // 更新当前Local MP为参考帧观测到的MP
    mLocalMPs = mpReferenceKF->mvViewMPs;
    mnGoodPrl = 0;
    mvMatchIdx.clear();

    for (int i = 0; i < 3; ++i)
        preSE2.meas[i] = 0;
    for (int i = 0; i < 9; ++i)
        preSE2.cov[i] = 0;

    mHomography = Mat::eye(3, 3, CV_64FC1);
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

    Mat imgWarp, H21;
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
        H21 = mHomography.inv(DECOMP_SVD);
        warpPerspective(imgCur, imgWarp, H21, imgRef.size());
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
                Mat pt2 = H21 * pt1;
                pt2 /= pt2.at<double>(2);
                ptl = Point2f(pt2.at<double>(0), pt2.at<double>(1));
            } else {
                ptl = ptCur;
            }
            ptr = ptRef + Point2f(imgCur.cols, 0);

            // 匹配上KP的为绿色
            circle(mImgOutMatch, ptl, 3, Scalar(0, 255, 0));
            circle(mImgOutMatch, ptr, 3, Scalar(0, 255, 0));
            line(mImgOutMatch, ptl, ptr, Scalar(255, 255, 0, 0.6));
        }
    }
}

cv::Mat TrackKlt::getImageMatches()
{
    locker lock(mMutexForPub);
    return mImgOutMatch;
}

//! 后端优化用，信息矩阵
//! TODO 公式待推导
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

//! 后端优化用，计算约束(R^t)*∑*R
//! TODO 公式待推导
void TrackKlt::calcSE3toXYZInfo(Point3f Pc1, const Mat& Tc1w, const Mat& Tc2w, Eigen::Matrix3d& info1,
                             Eigen::Matrix3d& info2)
{
    Point3f O1 = Point3f(cvu::inv(Tc1w).rowRange(0, 3).col(3));
    Point3f O2 = Point3f(cvu::inv(Tc2w).rowRange(0, 3).col(3));
    Point3f Pw = cvu::se3map(cvu::inv(Tc1w), Pc1);
    Point3f vO1 = Pw - O1;
    Point3f vO2 = Pw - O2;
    float sinParallax = cv::norm(vO1.cross(vO2)) / (cv::norm(vO1) * cv::norm(vO2));

    Point3f Pc2 = cvu::se3map(Tc2w, Pw);
    float length1 = cv::norm(Pc1);
    float length2 = cv::norm(Pc2);
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
    float normk1 = cv::norm(k1);
    float sin1 = normk1 / (cv::norm(z1) * cv::norm(Pc1));
    k1 = k1 * (std::asin(sin1) / normk1);
    Point3f k2 = Pc2.cross(z2);
    float normk2 = cv::norm(k2);
    float sin2 = normk2 / (cv::norm(z2) * cv::norm(Pc2));
    k2 = k2 * (std::asin(sin2) / normk2);

    Mat R1, R2;
    Mat k1mat = (Mat_<float>(3, 1) << k1.x, k1.y, k1.z);
    Mat k2mat = (Mat_<float>(3, 1) << k2.x, k2.y, k2.z);
    cv::Rodrigues(k1mat, R1);
    cv::Rodrigues(k2mat, R2);

    info1 = toMatrix3d(R1.t() * info_xyz1 * R1);
    info2 = toMatrix3d(R2.t() * info_xyz2 * R2);
}

/**
* @brief   根据本质矩阵H剔除外点，利用了RANSAC算法
* @return  返回内点数
*/
int TrackKlt::removeOutliers()
{
    vector<Point2f> pt1, pt2;
    vector<size_t> idx;
//    pt1.reserve(mpReferenceKF->N);
//    pt2.reserve(mCurrentFrame.N);
//    idx.reserve(mpReferenceKF->N);

    for (int i = 0, iend = mpReferenceKF->N; i < iend; i++) {
        if (mvMatchIdx[i] < 0)
            continue;
        idx.push_back(i);
        pt1.push_back(mpReferenceKF->mvKeyPoints[i].pt);
        pt2.push_back(mCurrentFrame.mvKeyPoints[mvMatchIdx[i]].pt);
    }

    if (pt1.size() == 0)
        return 0;

    vector<unsigned char> mask;
//   Mat F = findFundamentalMat(pt1, pt2, FM_RANSAC, 3, 0.99, mask);
    mHomography = findHomography(pt1, pt2, RANSAC, 5, mask);  // 朝天花板摄像头应该用H矩阵, F会退化

    int nInlier = 0;
    for (size_t i = 0, iend = mask.size(); i < iend; ++i) {
        if (!mask[i])
            mvMatchIdx[idx[i]] = -1;
        else
            nInlier++;
    }

    // If too few match inlier, discard all matches. The enviroment might not be
    // suitable for image tracking.
    if (nInlier < 10) {
        nInlier = 0;
        std::fill(mvMatchIdx.begin(), mvMatchIdx.end(), -1);
    }

    return nInlier;
}

bool TrackKlt::needNewKF()
{
    int nOldKP = mpReferenceKF->countObservations();
    bool c0 = mCurrentFrame.id - mpReferenceKF->id > nMinFrames;
    bool c1 = static_cast<float>(mnTrackedOld) <= nOldKP * 0.5f;
    bool c2 = mnGoodPrl > 40;
    bool c3 = mCurrentFrame.id - mpReferenceKF->id > nMaxFrames;
    bool c4 = mnMatchSum < 0.1f * Config::MaxFtrNumber || mnMatchSum < 20;
    bool bNeedNewKF = c0 && ( (c1 && c2) || c3 || c4 );

    bool bNeedKFByOdo = false;
    if (mbUseOdometry) {
        Se2 dOdo = mCurrentFrame.odom - mpReferenceKF->odom;
        bool c5 = static_cast<double>(abs(dOdo.theta)) >= mMaxAngle;  // 旋转量超过20°
        cv::Mat cTc = Config::Tcb * dOdo.toCvSE3() * Config::Tbc;
        cv::Mat xy = cTc.rowRange(0, 2).col(3);
        bool c6 = cv::norm(xy) >= mMaxDistance;  // 相机的平移量足够大
        bool c7 = mnMatchSum < 10;  //

        bNeedKFByOdo = c5 || c6 || c7;  // 相机移动取决于深度上限,考虑了不同深度下视野的不同
    }
    bNeedNewKF = bNeedNewKF && bNeedKFByOdo;  // 加上odom的移动条件, 把与改成了或

    // 最后还要看LocalMapper准备好了没有，LocalMapper正在执行优化的时候是不接收新KF的
    if (mpLocalMapper->acceptNewKF()) {
        return bNeedNewKF;
    } else if (c0 && (c4 || c3) && bNeedKFByOdo) {
        printf("[Track][Info ] #%ld 强制添加KF\n", mCurrentFrame.id, mnTrackedOld);
        mpLocalMapper->setAbortBA();  // 如果必须要加入关键帧,则终止LocalMap的优化,下一帧进来时就可以变成KF了
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
    //    if (mCurrentFrame.id - mpReferenceKF->id < nMinFrames)
    //        return 0;

    Mat Tcr = mCurrentFrame.getTcr();
    Mat Tc1c2 = cvu::inv(Tcr);
    Point3f Ocam1 = Point3f(0.f, 0.f, 0.f);
    Point3f Ocam2 = Point3f(Tc1c2.rowRange(0, 3).col(3));
    mvbGoodPrl = vector<bool>(mpReferenceKF->N, false);
    mnGoodPrl = 0;

    // 相机1和2的投影矩阵
    cv::Mat Proj1 = Config::PrjMtrxEye;                 // P1 = K * cv::Mat::eye(3, 4, CV_32FC1)
    cv::Mat Proj2 = Config::Kcam * Tcr.rowRange(0, 3);  // P2 = K * Tc2c1(3*4)
    // 1.遍历参考帧的KP
    int nTrackedOld(0), nGoodDepth(0), nBadDepth(0);
    for (size_t i = 0, iend = mpReferenceKF->N; i < iend; ++i) {
        if (mvMatchIdx[i] < 0)
            continue;

        // 2.如果参考帧的KP与当前帧的KP有匹配,且参考帧KP已经有对应的MP观测了，则可见地图点更新为此MP
        if (mpReferenceKF->hasObservation(i)) {
            mLocalMPs[i] = mpReferenceKF->mvViewMPs[i];
            //            mLocalMPs[i] = mpReferenceKF->getViewMPPoseInCamareFrame(i);
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
//    printf("[Track][Info ] #%ld 1.三角化生成了%d个点且%d个具有良好视差, "
//           "因深度不符合预期而剔除的匹配点对有%d个\n",
//           mCurrentFrame.id, nGoodDepth, mnGoodPrl, nBadDepth);

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

Se2 TrackKlt::dataAlignment(std::vector<Se2>& dataOdoSeq, double& timeImg)
{
    Se2 res;
    size_t n = dataOdoSeq.size();
    if (n < 2) {
        cerr << "[Track][Warni] Less odom sequence input!" << endl;
        return res;
    }

    //! 计算单帧图像时间内的平均速度
    Se2 tranSum = dataOdoSeq[n - 1] - dataOdoSeq[0];
    double dt = dataOdoSeq[n - 1].timeStamp - dataOdoSeq[0].timeStamp;
    double r = (timeImg - dataOdoSeq[n - 1].timeStamp) / dt;

    assert(r >= 0.f);

    Se2 transDelta(tranSum.x * r, tranSum.y * r, tranSum.theta * r);
    res = dataOdoSeq[n - 1] + transDelta;

    return res;
}

//! TODO 完成重定位功能
void TrackKlt::relocalization(const cv::Mat& img, const double& imgTime, const Se2& odo)
{
    return;
}

/*判断特征点是否超过边界，边界为3个像素
 * pt:需判断的点
 * @Maple
 * 2019.10.28
 */
bool TrackKlt::inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 3;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < mnImgCols - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < mnImgRows - BORDER_SIZE;
}

/*更新容器
 * status:是否为误匹配的标志
 * @Maple
 * 2019.10.28
 */
void TrackKlt::reduceVector(vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(mvIds.size()); i++)
        if (status[i])
        {
            mvPrevPts[j] = mvPrevPts[i];
            mvCurPts[j] = mvCurPts[i];
            mvForwPts[j] = mvForwPts[i];
            mvIds[j] = mvIds[i];
            mvTrackIdx[j] = mvTrackIdx[i];
            mvTrackNum[j++] = mvTrackNum[i];
        }
    mvPrevPts.resize(j);
    mvCurPts.resize(j);
    mvForwPts.resize(j);
    mvIds.resize(j);
    mvTrackIdx.resize(j);
    mvTrackNum.resize(j);
}

/*分块模式下更新容器
 * status:是否为误匹配的标志
 * @Maple
 * 2019.10.28
 */
void TrackKlt::reduceVectorCeil(vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(mvIds.size()); i++)
        if (status[i])
        {
            mvCeilLable[j] = mvCeilLable[i];
            mvPrevPts[j] = mvPrevPts[i];
            mvCurPts[j] = mvCurPts[i];
            mvForwPts[j] = mvForwPts[i];
            mvIds[j] = mvIds[i];
            mvTrackIdx[j] = mvTrackIdx[i];
            mvTrackNum[j++] = mvTrackNum[i];
        } else
            mvCeilPointsNum[mvCeilLable[i]] --;
    mvCeilLable.resize(j);
    mvPrevPts.resize(j);
    mvCurPts.resize(j);
    mvForwPts.resize(j);
    mvIds.resize(j);
    mvTrackIdx.resize(j);
    mvTrackNum.resize(j);
}

/*利用Ransac筛选误匹配点
 * ceil:是否分块
 * @Maple
 * 2019.10.28
 */
void TrackKlt::rejectWithRansac(bool isCeil)
{
    if (mvForwPts.size() >= 8)
    {
        cv::Mat homography;
        vector<unsigned char> inliersMask(mvCurPts.size());
        homography = cv::findHomography(mvCurPts, mvForwPts, cv::RANSAC, 5, inliersMask);

        if(isCeil)
            reduceVectorCeil(inliersMask);
        else
            reduceVector(inliersMask);
    }
}


/*对于跟踪点进行排序并去除密集点
 * @Maple
 * 2019.10.28
 */
void TrackKlt::setMask()
{
    mMask = cv::Mat(mnImgRows, mnImgCols, CV_8UC1, cv::Scalar(255));
    pair_mask filt_mask;
    vector<pair<int, pair_mask>> cnt_pts_id;

    for (unsigned int i = 0; i < mvForwPts.size(); i++)
    {
        filt_mask.id = mvIds[i];
        filt_mask.pt = mvForwPts[i];
        filt_mask.index = mvTrackIdx[i];
        cnt_pts_id.push_back(make_pair(mvTrackNum[i], filt_mask));
    }
    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair_mask> &a, const pair<int, pair_mask> &b)
    {
        return a.first > b.first;
    });

    mvForwPts.clear();
    mvIds.clear();
    mvTrackNum.clear();
    mvTrackIdx.clear();
    for (auto &it : cnt_pts_id)
    {
        if (mMask.at<uchar>(it.second.pt) == 255)
        {
            mvForwPts.push_back(it.second.pt);
            mvIds.push_back(it.second.id);
            mvTrackNum.push_back(it.first);
            mvTrackIdx.push_back(it.second.index);
            cv::circle(mMask, it.second.pt, mnMinDist, 0, -1);
        }
    }
}

/*更新跟踪点
 * @Maple
 * 2019.10.28
 */
void TrackKlt::addPoints()
{
    for (auto &p : mvNewPts)
    {
        mvForwPts.push_back(p);
        mvIds.push_back(mRefFrame.nextId);
        mvTrackNum.push_back(1);
        mvTrackIdx.push_back(-1);
    }
}

/*获取预测点的坐标
 * Ww：相邻两帧的选择角度
 * rotImg:旋转后的图像
 * points_prev：预测后的特征点坐标
 * @Maple
 * 2019.10.28
 */
void TrackKlt::predictPointsAndImag(double Ww,cv::Mat &rotImg,vector<cv::Point2f> &points_prev)
{
    double theta = Ww;
    if (abs(theta) > 0.001)
    {
        points_prev.clear();
        cv::Point rotate_center;
        rotate_center.x = 160.5827 - 0.01525;
        rotate_center.y = 117.7329 - 3.6984;
        getRotatePoint(mvCurPts, points_prev, rotate_center, theta);
        cv::Mat affine_matrix = getRotationMatrix2D(rotate_center, theta * 180 / CV_PI, 1.0);
        warpAffine(mCurImg, rotImg, affine_matrix, mCurImg.size());
    }
    else
    {
        rotImg = mCurImg.clone();
        points_prev = mvCurPts;
    }

}

/*绘制相邻两帧匹配点对
 * @Maple
 * 2019.10.28
 */
void TrackKlt::drawMachesPoints()
{
    vector<cv::DMatch> goodmatches;
    vector<cv::KeyPoint>laskeypoints, currentkeypoints;
    for (int i = 0; i < mvCurPts.size(); i++)
    {
        cv::KeyPoint ls, cur;
        ls.pt.x = mvCurPts[i].x;
        ls.pt.y = mvCurPts[i].y;
        cur.pt.x = mvForwPts[i].x;
        cur.pt.y = mvForwPts[i].y;
        cv::DMatch dmatches(i, i, 1);
        goodmatches.push_back(dmatches);
        laskeypoints.push_back(ls);
        currentkeypoints.push_back(cur);
    }
    cv::Mat matcheImage;
    drawMatches(mCurImg, laskeypoints, mForwImg, currentkeypoints, goodmatches, matcheImage);
    cv::imshow("matches", matcheImage);
    cv::waitKey();
}

/*绘制相邻两帧预测点对
 * @Maple
 * 2019.10.28
 */
void TrackKlt::drawPredictPoints()
{
    vector<cv::DMatch> goodmatches;
    vector<cv::KeyPoint>laskeypoints, currentkeypoints;
    for (int i = 0; i < mvCurPts.size(); i++)
    {
        cv::KeyPoint ls, cur;
        ls.pt.x = mvCurPts[i].x;
        ls.pt.y = mvCurPts[i].y;
        cur.pt.x = mvPrevPts[i].x;
        cur.pt.y = mvPrevPts[i].y;
        cv::DMatch dmatches(i, i, 1);
        goodmatches.push_back(dmatches);
        laskeypoints.push_back(ls);
        currentkeypoints.push_back(cur);
    }
    cv::Mat matcheImage;
    drawMatches(mCurImg, laskeypoints, mForwImg, currentkeypoints, goodmatches, matcheImage);
    cv::imshow("matches", matcheImage);
    cv::waitKey();
}

/*根据旋转中心确定特征点预测位置
 * Points:原始点
 * dstPoints：预测点位子
 * rotate_center：旋转中心
 * angle:旋转角度
 * @Maple
 * 2019.10.28
 */
void TrackKlt::getRotatePoint(vector<cv::Point2f> Points, vector<cv::Point2f>& dstPoints, const cv::Point rotate_center, double angle) {
    cv::Point2f prpoint;
    double x1 = 0, y1 = 0;
    double row = double(mnImgRows);
    for (size_t i = 0; i < Points.size(); i++)
    {
        x1 = Points[i].x;
        y1 = row - Points[i].y;
        double x2 = rotate_center.x;
        double y2 = row - rotate_center.y;
        double x = cvRound((x1 - x2) * cos(angle) - (y1 - y2) * sin(angle) + x2);
        double y = cvRound((x1 - x2) * sin(angle) + (y1 - y2) * cos(angle) + y2);
        y = row - y;
        prpoint.x = x;
        prpoint.y = y;
        dstPoints.push_back(prpoint);
    }
    return;
}

/*将图像分块
 * frame:输入图像
 * ceilImage：分块后图像存储位置
 * @Maple
 * 2019.10.28
 */
void TrackKlt::ceilImage(cv::Mat frame, vector<cv::Mat>& ceilImage)
{
    cv::Mat image_cut, roi_img;
    int m = frame.cols / ceilColSize;
    int n = frame.rows / ceilRowSize;
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < m; i++)
        {
            cv::Rect rect(i * ceilColSize, j * ceilRowSize, ceilColSize, ceilRowSize);
            image_cut = cv::Mat(frame, rect);
            roi_img = image_cut.clone();
            ceilImage.push_back(roi_img);
        }
    }
}

/*图像分块内进行特征点提取
 * frame:待提点图像
 * mask：掩模
 * @Maple
 * 2019.10.28
 */
void TrackKlt::detectFeaturePointsCeil(cv::Mat frame, cv::Mat mask)
{
    vector<cv::Mat> ceil_Image,ceil_mask;
    vector<vector<cv::Point2f>> ceilFeature(16);
    ceilImage(frame, ceil_Image);
    ceilImage(mask, ceil_mask);
    mvNewPts.clear();
    for (int i = 0; i < 16; i++)
    {
        ceilFeature[i].clear();
        int dceilNum = mnMaxCntCeil - mvCeilPointsNum[i];
        if(dceilNum > 10)
        {
            cv::goodFeaturesToTrack(ceil_Image[i], ceilFeature[i], dceilNum, 0.01, mnMinDist, ceil_mask[i],3);
            mvCeilPointsNum[i] = mvCeilPointsNum[i] + ceilFeature[i].size();
            for (int j = 0; j < ceilFeature[i].size(); j++)
            {
                if (i < 4)
                {
                    ceilFeature[i][j].x = ceilFeature[i][j].x + ceilColSize * i;
                }
                else if (i >= 4 && i < 8)
                {
                    ceilFeature[i][j].x = ceilFeature[i][j].x + ceilColSize * (i - 4);
                    ceilFeature[i][j].y = ceilFeature[i][j].y + ceilRowSize;
                }
                else if (i >= 8 && i < 12)
                {
                    ceilFeature[i][j].x = ceilFeature[i][j].x + ceilColSize * (i - 8);
                    ceilFeature[i][j].y = ceilFeature[i][j].y + ceilRowSize * 2;
                }
                else
                {
                    ceilFeature[i][j].x = ceilFeature[i][j].x + ceilColSize * (i - 12);
                    ceilFeature[i][j].y = ceilFeature[i][j].y + ceilRowSize * 3;
                }
                mvNewPts.push_back(ceilFeature[i][j]);
                mvCeilLable.push_back(i);
            }
        }
    }
}

/*对于跟踪点进行排序并去除密集点(分块模式下)
 * @Maple
 * 2019.10.28
 */
void TrackKlt::setMaskCeil()
{
    mMask = cv::Mat(mnImgRows, mnImgCols, CV_8UC1, cv::Scalar(255));
    pair_mask_ceil filt_mask;
    vector<pair<int, pair_mask_ceil>> cnt_pts_id;
    for (unsigned int i = 0; i < mvForwPts.size(); i++)
    {
        filt_mask.id = mvIds[i];
        filt_mask.pt = mvForwPts[i];
        filt_mask.index = mvTrackIdx[i];
        filt_mask.clable = mvCeilLable[i];
        cnt_pts_id.push_back(make_pair(mvTrackNum[i], filt_mask));
    }
    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair_mask_ceil> &a, const pair<int, pair_mask_ceil> &b)
    {
        return a.first > b.first;
    });
    mvForwPts.clear();
    mvIds.clear();
    mvTrackNum.clear();
    mvTrackIdx.clear();
    for (auto &it : cnt_pts_id)
    {
        if (mMask.at<uchar>(it.second.pt) == 255)
        {
            mvCeilLable.push_back(it.second.clable);
            mvForwPts.push_back(it.second.pt);
            mvIds.push_back(it.second.id);
            mvTrackNum.push_back(it.first);
            mvTrackIdx.push_back(it.second.index);
            cv::circle(mMask, it.second.pt, mnMinDist, 0, -1);
        } else
            mvCeilPointsNum[mvCeilLable[it.second.clable]] --;
    }
}

}  // namespace se2lam
