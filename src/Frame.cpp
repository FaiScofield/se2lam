/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG
* (github.com/hbtang)
*/

#include "Frame.h"
#include "KeyFrame.h"
#include "LineDetection.h"
#include "Track.h"
#include "converter.h"
#include "cvutil.h"

namespace se2lam
{
using namespace cv;
using namespace std;

typedef unique_lock<mutex> locker;

unsigned long Frame::nextId = 1;
bool Frame::bIsInitialComputations = true;
bool Frame::bNeedVisualization = true;
float Frame::minXUn, Frame::minYUn, Frame::maxXUn, Frame::maxYUn;
float Frame::gridElementWidthInv, Frame::gridElementHeightInv;

Frame::Frame()
{
}

Frame::~Frame()
{
}

/*
* @跟踪过程中建立Frame
* @Maple_liu
* @Email:mingpei.liu@rock-chips.com
* @2019.10.23
*/
Frame::Frame(const Mat& imgGray, const Se2& odo, vector<cv::KeyPoint> mckeyPoints,
             ORBextractor* extractor, const Mat& K, const Mat& distCoef)
    : mpORBExtractor(extractor), mTimeStamp(0.f), odom(odo)
{

    //! 首帧会根据k和d重新计算图像边界, 只计算一次
    if (bIsInitialComputations) {
        //! 计算去畸变后的图像边界
        computeBoundUn(K, distCoef);

        gridElementWidthInv = FRAME_GRID_COLS / (maxXUn - minXUn);
        gridElementHeightInv = FRAME_GRID_ROWS / (maxYUn - minYUn);

        bIsInitialComputations = false;
    }

    id = nextId++;

    Mat imgUn, imgClahed;

    //! 输入图像去畸变
//    assert(imgGray.channels() == 1);
//    undistort(imgGray, imgGray, K, distCoef);

    //!  限制对比度自适应直方图均衡
    Ptr<CLAHE> clahe = createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(imgGray, imgClahed);

    mImg = imgClahed.clone();
    keyPoints = mckeyPoints;
    mvKeyPoints = keyPoints;

    //! 计算描述子
    mpORBExtractor->getDescriptor(mImg, keyPoints, mDescriptors);

    N = mvKeyPoints.size();
    if (mvKeyPoints.empty())
        return;

    if (bNeedVisualization)
        mImg.copyTo(mImage);

    //! Scale Levels Info
    mnScaleLevels = mpORBExtractor->getLevels();       // default 1
    mfScaleFactor = mpORBExtractor->getScaleFactor();  // default 1.2

    //! 计算金字塔每层尺度和高斯噪声
    mvScaleFactors.resize(mnScaleLevels);
    mvLevelSigma2.resize(mnScaleLevels);
    mvScaleFactors[0] = 1.0f;
    mvLevelSigma2[0] = 1.0f;
    for (int i = 1; i < mnScaleLevels; i++) {
        mvScaleFactors[i] = mvScaleFactors[i - 1] * mfScaleFactor;
        mvLevelSigma2[i] = mvScaleFactors[i] * mvScaleFactors[i];
    }

    mvInvLevelSigma2.resize(mvLevelSigma2.size());
    for (int i = 0; i < mnScaleLevels; i++)
        mvInvLevelSigma2[i] = 1.0 / mvLevelSigma2[i];
}

/*
* @第一帧图像建立Frame
* @Maple_liu
* @Email:mingpei.liu@rock-chips.com
* @2019.10.23
*/
Frame::Frame(const Mat& imgGray, const Se2& odo, bool Ceil, ORBextractor* extractor, const Mat& K,
             const Mat& distCoef)
    : mpORBExtractor(extractor), mTimeStamp(0.f), odom(odo)
{

    //! 首帧会根据k和d重新计算图像边界, 只计算一次
    if (bIsInitialComputations) {
        //! 计算去畸变后的图像边界
        computeBoundUn(K, distCoef);

        assert(maxXUn != minXUn && maxYUn != minYUn);
        gridElementWidthInv = FRAME_GRID_COLS / (maxXUn - minXUn);
        gridElementHeightInv = FRAME_GRID_ROWS / (maxYUn - minYUn);

        bIsInitialComputations = false;
        bNeedVisualization = Config::NeedVisualization;
    }

    id = nextId++;

    Mat imgUn, imgClahed;

    //! 输入图像去畸变
    assert(imgGray.channels() == 1);
    undistort(imgGray, imgUn, K, distCoef);

    //!  限制对比度自适应直方图均衡
    Ptr<CLAHE> clahe = createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(imgUn, imgClahed);

    //! 光流相关参数设置
    mnMinDist = 0;       // mask建立时的特征点周边半径
    mnMaxCnt = 1000;     //最大特征点数量
    mnCeilColSize = 80;  //分块尺寸
    mnCeilRowSize = 60;
    mvCeilPointsNum.resize(16, 0);
    mImg = imgClahed;

    //! 第一帧图像，将其赋值给第一帧图像
    // cur和forw分别是LK光流跟踪的前后两帧，forw才是真正的＂当前＂帧，cur实际上是上一帧，而pre是上一次发布的帧，也就是rejectWithF()函数
    //如果当前帧的图像数据flow_img为空，说明当前是第一次读入图像数据
    //将读入的图像赋给当前帧flow_img
    //同时，还将读入的图像赋给mPrevImg,mCurImg,这是为了避免后面使用到这些数据时，它们是空的
    mPrevImg = mCurImg = mForwImg = mImg;


//    Preprocess pre;
//    Mat sharpImage1 = pre.sharpping(mForwImg, 11);
//    imgGray = pre.GaMma(sharpImage1, 0.7);
//    Mat sharpImage = pre.sharpping(mImg, 8);
//    Mat imgLine = pre.GaMma(sharpImage, 1.8);

    //! 直线上特征点提取
    Mat mask = getLineMask(imgClahed, false);
    if (Ceil)
        detectFeaturePointsCeil(imgClahed, mask);  // mask_ceil提点
    else
        cv::goodFeaturesToTrack(imgClahed, mvNewPts, mnMaxCnt, 0.01, mnMinDist, mask);  // mask提点

    //! 将新检测到的特征点mvNewPts添加到mForwPts中，id初始化－１,mvTrackNum初始化为１
    addPoints();

    keyPoints.resize(mvForwPts.size());
    for (int i = 0; i < mvForwPts.size(); i++) {
        keyPoints[i].pt = mvForwPts[i];
        keyPoints[i].octave = 0;
        mvTrackIdx.push_back(i);
    }
    mvKeyPoints = keyPoints;

    //! 计算描述子
    mpORBExtractor->getDescriptor(mImg, keyPoints, mDescriptors);

    N = mvKeyPoints.size();
    if (mvKeyPoints.empty())
        return;

    if (bNeedVisualization)
        mImg.copyTo(mImage);

    //! Scale Levels Info
    mnScaleLevels = mpORBExtractor->getLevels();       // default 1
    mfScaleFactor = mpORBExtractor->getScaleFactor();  // default 1.2

    //! 计算金字塔每层尺度和高斯噪声
    mvScaleFactors.resize(mnScaleLevels);
    mvLevelSigma2.resize(mnScaleLevels);
    mvScaleFactors[0] = 1.0f;
    mvLevelSigma2[0] = 1.0f;
    for (int i = 1; i < mnScaleLevels; i++) {
        mvScaleFactors[i] = mvScaleFactors[i - 1] * mfScaleFactor;
        mvLevelSigma2[i] = mvScaleFactors[i] * mvScaleFactors[i];
    }

    mvInvLevelSigma2.resize(mvLevelSigma2.size());
    for (int i = 0; i < mnScaleLevels; i++)
        mvInvLevelSigma2[i] = 1.0 / mvLevelSigma2[i];
}

Frame::Frame(const Mat& imgGray, const Se2& odo, ORBextractor* extractor, const Mat& K,
             const Mat& distCoef)
    : mpORBExtractor(extractor), mTimeStamp(0.f), odom(odo)
{
    //! 首帧会根据k和d重新计算图像边界, 只计算一次
    if (bIsInitialComputations) {
        //! 计算去畸变后的图像边界
        computeBoundUn(K, distCoef);

        assert(maxXUn != minXUn && maxYUn != minYUn);
        gridElementWidthInv = FRAME_GRID_COLS / (maxXUn - minXUn);
        gridElementHeightInv = FRAME_GRID_ROWS / (maxYUn - minYUn);

        bIsInitialComputations = false;
        bNeedVisualization = Config::NeedVisualization;
    }

    id = nextId++;

    Mat imgUn, imgClahed;

    //! 输入图像去畸变
    assert(imgGray.channels() == 1);
    undistort(imgGray, imgUn, K, distCoef);

    //!  限制对比度自适应直方图均衡
    Ptr<CLAHE> clahe = createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(imgUn, imgClahed);

    //! 提取特征点
    (*mpORBExtractor)(imgClahed, cv::Mat(), mvKeyPoints, mDescriptors);

    N = mvKeyPoints.size();
    if (mvKeyPoints.empty())
        return;

    if (bNeedVisualization)
        imgClahed.copyTo(mImage);

    //    mvpMapPoints = vector<PtrMapPoint>(N, static_cast<PtrMapPoint>(nullptr));
    //    mvbOutlier = vector<bool>(N, false);

    //! Scale Levels Info
    mnScaleLevels = mpORBExtractor->getLevels();       // default 1
    mfScaleFactor = mpORBExtractor->getScaleFactor();  // default 1.2

    //! 计算金字塔每层尺度和高斯噪声
    mvScaleFactors.resize(mnScaleLevels);
    mvLevelSigma2.resize(mnScaleLevels);
    mvScaleFactors[0] = 1.0f;
    mvLevelSigma2[0] = 1.0f;
    for (int i = 1; i != mnScaleLevels; ++i) {
        mvScaleFactors[i] = mvScaleFactors[i - 1] * mfScaleFactor;
        mvLevelSigma2[i] = mvScaleFactors[i] * mvScaleFactors[i];
    }

    mvInvLevelSigma2.resize(mvLevelSigma2.size());
    for (int i = 0; i != mnScaleLevels; ++i)
        mvInvLevelSigma2[i] = 1.0 / mvLevelSigma2[i];

    //! Assign Features to Grid Cells. 将特征点按照cell存放
    int nReserve = 0.5 * N / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
    for (int i = 0; i != FRAME_GRID_COLS; ++i)
        for (int j = 0; j < FRAME_GRID_ROWS; ++j)
            mGrid[i][j].reserve(nReserve);

    for (size_t i = 0; i != N; ++i) {
        cv::KeyPoint& kp = mvKeyPoints[i];

        int nGridPosX, nGridPosY;
        if (PosInGrid(kp, nGridPosX, nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

Frame::Frame(const Mat& imgGray, const double& time, const Se2& odo, ORBextractor* extractor,
             const Mat& K, const Mat& distCoef)
    : mpORBExtractor(extractor), mTimeStamp(time), odom(odo)
{
    //! 首帧会根据k和d重新计算图像边界, 只计算一次
    if (bIsInitialComputations) {
        //! 计算去畸变后的图像边界
        computeBoundUn(K, distCoef);

        gridElementWidthInv = FRAME_GRID_COLS / (maxXUn - minXUn);
        gridElementHeightInv = FRAME_GRID_ROWS / (maxYUn - minYUn);

        bIsInitialComputations = false;
    }

    id = nextId++;

    Mat imgUn, imgClahed;

    //! 输入图像去畸变
    assert(imgGray.channels() == 1);
    undistort(imgGray, imgUn, K, distCoef);

    //!  限制对比度自适应直方图均衡
    Ptr<CLAHE> clahe = createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(imgUn, imgClahed);

    //! 提取特征点
    (*mpORBExtractor)(imgClahed, cv::Mat(), mvKeyPoints, mDescriptors);

    N = mvKeyPoints.size();
    if (mvKeyPoints.empty())
        return;

    if (bNeedVisualization)
        imgClahed.copyTo(mImage);

    //    mvpMapPoints = vector<PtrMapPoint>(N, static_cast<PtrMapPoint>(nullptr));
    //    mvbOutlier = vector<bool>(N, false);

    //! Scale Levels Info
    mnScaleLevels = mpORBExtractor->getLevels();       // default 5
    mfScaleFactor = mpORBExtractor->getScaleFactor();  // default 1.2

    //! 计算金字塔每层尺度和高斯噪声
    mvScaleFactors.resize(mnScaleLevels);
    mvLevelSigma2.resize(mnScaleLevels);
    mvScaleFactors[0] = 1.0f;
    mvLevelSigma2[0] = 1.0f;
    for (int i = 1; i != mnScaleLevels; ++i) {
        mvScaleFactors[i] = mvScaleFactors[i - 1] * mfScaleFactor;
        mvLevelSigma2[i] = mvScaleFactors[i] * mvScaleFactors[i];
    }

    mvInvLevelSigma2.resize(mvLevelSigma2.size());
    for (int i = 0; i != mnScaleLevels; ++i)
        mvInvLevelSigma2[i] = 1.0 / mvLevelSigma2[i];

    //! Assign Features to Grid Cells. 将特征点按照cell存放
    int nReserve = 0.5 * N / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
    for (int i = 0; i != FRAME_GRID_COLS; ++i)
        for (int j = 0; j < FRAME_GRID_ROWS; ++j)
            mGrid[i][j].reserve(nReserve);

    for (size_t i = 0; i != N; ++i) {
        cv::KeyPoint& kp = mvKeyPoints[i];

        int nGridPosX, nGridPosY;
        if (PosInGrid(kp, nGridPosX, nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

Frame::Frame(const Frame& f)
    : mpORBExtractor(f.mpORBExtractor), mTimeStamp(f.mTimeStamp), id(f.id), odom(f.odom), N(f.N),
      mDescriptors(f.mDescriptors.clone()), mvKeyPoints(f.mvKeyPoints),
      /*mvpMapPoints(f.mvpMapPoints), mvbOutlier(f.mvbOutlier),*/ mnScaleLevels(f.mnScaleLevels),
      mfScaleFactor(f.mfScaleFactor), mvScaleFactors(f.mvScaleFactors),
      mvLevelSigma2(f.mvLevelSigma2), mvInvLevelSigma2(f.mvInvLevelSigma2), Tcr(f.Tcr.clone()),
      Trb(f.Trb)
{
    // klt
    mPrevImg = f.mPrevImg;
    mCurImg = f.mCurImg;
    mvNewPts = f.mvNewPts;
    mvPrevPts = f.mvPrevPts;
    mvCurPts = f.mvCurPts;
    mvForwPts = f.mvForwPts;
    mvIds = f.mvIds;
    mvTrackNum = f.mvTrackNum;
    mvTrackIdx = f.mvTrackIdx;
    mvCeilPointsNum = f.mvCeilPointsNum;
    mvCeilLable = f.mvCeilLable;

    f.mImg.copyTo(mImg);
    keyPoints = f.keyPoints;
    keyPointsUn = f.keyPointsUn;

    if (bNeedVisualization)
        f.mImage.copyTo(mImage);

    for (int i = 0; i != FRAME_GRID_COLS; ++i)
        for (int j = 0; j < FRAME_GRID_ROWS; ++j)
            mGrid[i][j] = f.mGrid[i][j];

    if (!f.Tcw.empty())
        setPose(f.Tcw);
}

Frame& Frame::operator=(const Frame& f)
{
    // klt
    mPrevImg = f.mPrevImg;
    mCurImg = f.mCurImg;
    mvNewPts = f.mvNewPts;
    mvPrevPts = f.mvPrevPts;
    mvCurPts = f.mvCurPts;
    mvForwPts = f.mvForwPts;
    mvIds = f.mvIds;
    mvTrackNum = f.mvTrackNum;
    mvTrackIdx = f.mvTrackIdx;
    mvCeilPointsNum = f.mvCeilPointsNum;
    mvCeilLable = f.mvCeilLable;

    f.mImg.copyTo(mImg);
    keyPoints = f.keyPoints;
    keyPointsUn = f.keyPointsUn;


    mpORBExtractor = f.mpORBExtractor;

    if (bNeedVisualization)
        f.mImage.copyTo(mImage);

    mvKeyPoints = f.mvKeyPoints;
    f.mDescriptors.copyTo(mDescriptors);

    mTimeStamp = f.mTimeStamp;
    id = f.id;
    N = f.N;

    odom = f.odom;
    Trb = f.Trb;
    Tcr = f.Tcr;
    if (!f.Tcw.empty())
        setPose(f.Tcw);

//    mvpMapPoints = f.mvpMapPoints;
//    mvbOutlier = f.mvbOutlier;

    mnScaleLevels = f.mnScaleLevels;
    mfScaleFactor = f.mfScaleFactor;
    mvScaleFactors = f.mvScaleFactors;
    mvLevelSigma2 = f.mvLevelSigma2;
    mvInvLevelSigma2 = f.mvInvLevelSigma2;


    for (int i = 0; i != FRAME_GRID_COLS; ++i)
        for (int j = 0; j < FRAME_GRID_ROWS; ++j)
            mGrid[i][j] = f.mGrid[i][j];


    return *this;
}

Se2 Frame::getTwb()
{
    locker lock(mMutexPose);
    return Twb;
}

Mat Frame::getTcr()
{
    locker lock(mMutexPose);
    return Tcr.clone();
}

Mat Frame::getPose()
{
    locker lock(mMutexPose);
    return Tcw.clone();
}

Point3f Frame::getCameraCenter()
{
    locker lock(mMutexPose);
    Mat Ow = cvu::inv(Tcw).rowRange(0, 3).col(3);

    return Point3f(Ow);
}

void Frame::setPose(const Mat& _Tcw)
{
    locker lock(mMutexPose);
    _Tcw.copyTo(Tcw);
    Twb.fromCvSE3(cvu::inv(Tcw) * Config::Tcb);
}

void Frame::setPose(const Se2& _Twb)
{
    locker lock(mMutexPose);
    Twb = _Twb;
    Tcw = Config::Tcb * Twb.inv().toCvSE3();
}

void Frame::setTcr(const Mat& _Tcr)
{
    locker lock(mMutexPose);
    _Tcr.copyTo(Tcr);
}

void Frame::setTrb(const Se2& _Trb)
{
    locker lock(mMutexPose);
    Trb = _Trb;
}

//! 只执行一次
void Frame::computeBoundUn(const Mat& K, const Mat& D)
{
    float x = static_cast<float>(Config::ImgSize.width);
    float y = static_cast<float>(Config::ImgSize.height);
    assert(x > 0 && y > 0);
    if (D.at<float>(0) == 0.) {
        minXUn = 0.f;
        minYUn = 0.f;
        maxXUn = x;
        maxYUn = y;
        return;
    }
    Mat_<Point2f> mat(1, 4);
    mat << Point2f(0, 0), Point2f(x, 0), Point2f(0, y), Point2f(x, y);
    undistortPoints(mat, mat, K, D, Mat(), K);
    minXUn = std::min(mat(0).x, mat(2).x);
    minYUn = std::min(mat(0).y, mat(1).y);
    maxXUn = std::max(mat(1).x, mat(3).x);
    maxYUn = std::max(mat(2).y, mat(3).y);
}

bool Frame::inImgBound(Point2f pt)
{
    return (pt.x >= minXUn && pt.x <= maxXUn && pt.y >= minYUn && pt.y <= maxYUn);
}

// From ORB_SLAM
bool Frame::PosInGrid(cv::KeyPoint& kp, int& posX, int& posY)
{
    posX = round((kp.pt.x - minXUn) * gridElementWidthInv);
    posY = round((kp.pt.y - minYUn) * gridElementHeightInv);

    //! Keypoint's coordinates are undistorted, which could cause to go out of the image
    //! 特征点坐标是经过畸变矫正过的，可能会超出图像
    if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS)
        return false;

    return true;
}

/**
 * @brief 找到在 以x,y为中心,边长为2r的方形内且在[minLevel, maxLevel]的特征点
 * @return 返回区域内特征点在KF里的索引
 */
vector<size_t> Frame::GetFeaturesInArea(const float& x, const float& y, const float& r,
                                        int minLevel, int maxLevel) const
{
    assert(r > 0);

    vector<size_t> vIndices;
    vIndices.reserve(N);

    //! floor向下取整
    int nMinCellX = floor((x - minXUn - r) * gridElementWidthInv);
    nMinCellX = max(0, nMinCellX);
    if (nMinCellX >= FRAME_GRID_COLS)
        return vIndices;

    //! ceil向上取整
    int nMaxCellX = ceil((x - minXUn + r) * gridElementWidthInv);  // FIXME + ?
    nMaxCellX = min(FRAME_GRID_COLS - 1, nMaxCellX);
    if (nMaxCellX < 0)
        return vIndices;

    int nMinCellY = floor((y - minYUn - r) * gridElementHeightInv);
    nMinCellY = max(0, nMinCellY);
    if (nMinCellY >= FRAME_GRID_ROWS)
        return vIndices;

    int nMaxCellY = ceil((y - minYUn + r) * gridElementHeightInv);  // FIXME + ?
    nMaxCellY = min(FRAME_GRID_ROWS - 1, nMaxCellY);
    if (nMaxCellY < 0)
        return vIndices;

    assert(nMinCellX <= nMaxCellX);
    assert(nMinCellY <= nMaxCellY);

    const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);
    const bool bSameLevel = (minLevel == maxLevel);

    for (int ix = nMinCellX; ix <= nMaxCellX; ix++) {
        for (int iy = nMinCellY; iy <= nMaxCellY; iy++) {
            vector<size_t> vCell = mGrid[ix][iy];
            if (vCell.empty())
                continue;

            for (size_t j = 0, jend = vCell.size(); j < jend; ++j) {
                const cv::KeyPoint& kpUn = mvKeyPoints[vCell[j]];
                if (bCheckLevels && !bSameLevel) {
                    if (kpUn.octave < minLevel || kpUn.octave > maxLevel)
                        continue;
                } else if (bSameLevel) {
                    if (kpUn.octave != minLevel)
                        continue;
                }

                if (abs(kpUn.pt.x - x) > r || abs(kpUn.pt.y - y) > r)
                    continue;

                vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

/*
* @更新klt跟踪的特征点及标签
* @Maple_liu
* @Email:mingpei.liu@rock-chips.com
* @2019.10.23
*/
void Frame::addPoints()
{
    for (auto& p : mvNewPts) {
        mvForwPts.push_back(p);
        mvPrevPts.push_back(p);
        mvCurPts.push_back(p);
        mvIds.push_back(id);
        mvTrackNum.push_back(1);
    }
}

//将图像分块
void Frame::ceilImage(const cv::Mat& frame, vector<cv::Mat>& ceil_Image)
{
    cv::Mat image_cut, roi_img;
    int m = frame.cols / mnCeilColSize;
    int n = frame.rows / mnCeilRowSize;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            cv::Rect rect(i * mnCeilColSize, j * mnCeilRowSize, mnCeilColSize, mnCeilRowSize);
            image_cut = cv::Mat(frame, rect);
            roi_img = image_cut.clone();
            ceil_Image.push_back(roi_img);
        }
    }
}

//图像分块内进行特征点提取
void Frame::detectFeaturePointsCeil(const cv::Mat& frame, const cv::Mat& mask)
{
    vector<cv::Mat> ceil_Image, ceil_mask;
    vector<vector<cv::Point2f>> ceilFeature(16);
    ceilImage(frame, ceil_Image);
    ceilImage(mask, ceil_mask);
    for (int i = 0; i < 16; i++) {
        ceilFeature[i].clear();
        if (mvCeilPointsNum[i] < 50) {
            cv::goodFeaturesToTrack(ceil_Image[i], ceilFeature[i], 100, 0.01, mnMinDist,
                                    ceil_mask[i]);
            mvCeilPointsNum[i] = mvCeilPointsNum[i] + ceilFeature[i].size();
            // num = num + num_ceil_points[i];
            for (int j = 0; j < ceilFeature[i].size(); j++) {
                if (i < 4) {
                    ceilFeature[i][j].x = ceilFeature[i][j].x + mnCeilColSize * i;
                } else if (i >= 4 && i < 8) {
                    ceilFeature[i][j].x = ceilFeature[i][j].x + mnCeilColSize * (i - 4);
                    ceilFeature[i][j].y = ceilFeature[i][j].y + mnCeilRowSize;
                } else if (i >= 8 && i < 12) {
                    ceilFeature[i][j].x = ceilFeature[i][j].x + mnCeilColSize * (i - 8);
                    ceilFeature[i][j].y = ceilFeature[i][j].y + mnCeilRowSize * 2;
                } else {
                    ceilFeature[i][j].x = ceilFeature[i][j].x + mnCeilColSize * (i - 12);
                    ceilFeature[i][j].y = ceilFeature[i][j].y + mnCeilRowSize * 3;
                }
                mvNewPts.push_back(ceilFeature[i][j]);
                mvCeilLable.push_back(i);
            }
        }
    }
}

}  // namespace se2lam
