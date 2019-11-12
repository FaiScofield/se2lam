/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG
* (github.com/hbtang)
*/

#include "Frame.h"
#include "KeyFrame.h"
#include "LineDetector.h"
#include "Track.h"
#include "converter.h"
#include "cvutil.h"

namespace se2lam
{
using namespace cv;
using namespace std;

typedef unique_lock<mutex> locker;

unsigned long Frame::nextId = 1;
bool Frame::bIsInitialComputation = true;
bool Frame::bNeedVisualization = true;
float Frame::minXUn, Frame::minYUn, Frame::maxXUn, Frame::maxYUn;
float Frame::gridElementWidthInv, Frame::gridElementHeightInv;

Frame::Frame()
{}

Frame::~Frame()
{}


/**
 * @brief   klt的Frame构造
 * @param imgGray   已矫正过畸变的灰度图
 * @param odo       里程计信息
 * @param vKPs      特征点
 *
 * @author  Maple.Liu
 * @date    2019.10.23
 */
Frame::Frame(const Mat& imgGray, const Se2& odo, const vector<KeyPoint>& vKPs,
             ORBextractor* extractor)
    : mpORBExtractor(extractor), mTimeStamp(0.f), odom(odo), mvKeyPoints(vKPs)
{
    //! 首帧会根据k和d重新计算图像边界, 只计算一次
    if (bIsInitialComputation) {
        //! 计算去畸变后的图像边界
        computeBoundUn(Config::Kcam, Config::Dcam);

        gridElementWidthInv = GRID_COLS / (maxXUn - minXUn);
        gridElementHeightInv = GRID_ROWS / (maxYUn - minYUn);

        bIsInitialComputation = false;
        bNeedVisualization = Config::NeedVisualization;
        fprintf(stderr, "\n[Frame][Info ] 去畸变的图像边界为: X: %.1f - %.1f, Y: %.1f - %.1f\n",
                minXUn, maxXUn, minYUn, maxYUn);
    }

    id = nextId++;

    //! 计算描述子
    mpORBExtractor->getDescriptors(imgGray, mvKeyPoints, mDescriptors);
    N = mvKeyPoints.size();
    if (mvKeyPoints.empty())
        return;

    if (bNeedVisualization)
        imgGray.copyTo(mImage);

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

    //! Assign Features to Grid Cells. 将特征点按照cell存放
    int nReserve = 0.5 * N / (GRID_COLS * GRID_ROWS);
    for (int i = 0; i != GRID_COLS; ++i)
        for (int j = 0; j < GRID_ROWS; ++j)
            mGrid[i][j].reserve(nReserve);

    for (size_t i = 0; i != N; ++i) {
        KeyPoint& kp = mvKeyPoints[i];

        int nGridPosX, nGridPosY;
        if (posInGrid(kp, nGridPosX, nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

/**
 * @brief   构造函数, 对图像去畸变并应用直方图均衡, 提取特征点
 * @param imgGray   灰度图
 * @param odo       里程计信息
 * @param extractor ORB特征提取类的指针
 * @param K         相机内参
 * @param distCoef  相机畸变参数
 */
Frame::Frame(const Mat& imgGray, const Se2& odo, ORBextractor* extractor, const Mat& K,
             const Mat& distCoef)
    : mpORBExtractor(extractor), mTimeStamp(0.f), odom(odo)
{
    //! 首帧会根据k和d重新计算图像边界, 只计算一次
    if (bIsInitialComputation) {
        //! 计算去畸变后的图像边界
        computeBoundUn(K, distCoef);

        gridElementWidthInv = GRID_COLS / (maxXUn - minXUn);
        gridElementHeightInv = GRID_ROWS / (maxYUn - minYUn);

        bIsInitialComputation = false;
        bNeedVisualization = Config::NeedVisualization;
        fprintf(stderr, "\n[Frame] 去畸变的图像边界为: X: %.1f - %.1f, Y: %.1f - %.1f\n", minXUn,
                maxXUn, minYUn, maxYUn);
    }

    id = nextId++;

    Mat imgUn, imgClahed;

    //! 输入图像去畸变
    assert(imgGray.channels() == 1);
    undistort(imgGray, imgUn, K, distCoef);

    //!  限制对比度自适应直方图均衡
    Ptr<CLAHE> clahe = createCLAHE(3.0, Size(8, 8));
    clahe->apply(imgUn, imgClahed);

    //! 提取特征点
    (*mpORBExtractor)(imgClahed, Mat(), mvKeyPoints, mDescriptors);

    N = mvKeyPoints.size();
    if (mvKeyPoints.empty())
        return;

    if (bNeedVisualization)
        imgClahed.copyTo(mImage);

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
    int nReserve = 0.5 * N / (GRID_COLS * GRID_ROWS);
    for (int i = 0; i != GRID_COLS; ++i)
        for (int j = 0; j < GRID_ROWS; ++j)
            mGrid[i][j].reserve(nReserve);

    for (size_t i = 0; i != N; ++i) {
        KeyPoint& kp = mvKeyPoints[i];

        int nGridPosX, nGridPosY;
        if (posInGrid(kp, nGridPosX, nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

/**
 * @brief   带时间戳信息的构造函数
 * @author  Vance.Wu
 */
Frame::Frame(const Mat& imgGray, const double& time, const Se2& odo, ORBextractor* extractor,
             const Mat& K, const Mat& distCoef)
    : mpORBExtractor(extractor), mTimeStamp(time), odom(odo)
{
    //! 首帧会根据k和d重新计算图像边界, 只计算一次
    if (bIsInitialComputation) {
        //! 计算去畸变后的图像边界
        computeBoundUn(K, distCoef);

        gridElementWidthInv = GRID_COLS / (maxXUn - minXUn);
        gridElementHeightInv = GRID_ROWS / (maxYUn - minYUn);

        bIsInitialComputation = false;
        bNeedVisualization = Config::NeedVisualization;
        fprintf(stderr, "\n[Frame] 去畸变的图像边界为: X: %.1f - %.1f, Y: %.1f - %.1f\n", minXUn,
                maxXUn, minYUn, maxYUn);
    }

    id = nextId++;

    Mat imgUn, imgClahed;

    //! 输入图像去畸变
    assert(imgGray.channels() == 1);
    undistort(imgGray, imgUn, K, distCoef);

    //!  限制对比度自适应直方图均衡
    Ptr<CLAHE> clahe = createCLAHE(3.0, Size(8, 8));
    clahe->apply(imgUn, imgClahed);

    //! 提取特征点
    (*mpORBExtractor)(imgClahed, Mat(), mvKeyPoints, mDescriptors);

    N = mvKeyPoints.size();
    if (mvKeyPoints.empty())
        return;

    if (bNeedVisualization)
        imgClahed.copyTo(mImage);

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
    int nReserve = 0.5 * N / (GRID_COLS * GRID_ROWS);
    for (int i = 0; i != GRID_COLS; ++i)
        for (int j = 0; j < GRID_ROWS; ++j)
            mGrid[i][j].reserve(nReserve);

    for (size_t i = 0; i != N; ++i) {
        KeyPoint& kp = mvKeyPoints[i];

        int nGridPosX, nGridPosY;
        if (posInGrid(kp, nGridPosX, nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

//! 复制构造函数
Frame::Frame(const Frame& f)
    : mpORBExtractor(f.mpORBExtractor), mTimeStamp(f.mTimeStamp), id(f.id), odom(f.odom), N(f.N),
      mDescriptors(f.mDescriptors.clone()), mvKeyPoints(f.mvKeyPoints),
      mnScaleLevels(f.mnScaleLevels), mfScaleFactor(f.mfScaleFactor),
      mvScaleFactors(f.mvScaleFactors), mvLevelSigma2(f.mvLevelSigma2),
      mvInvLevelSigma2(f.mvInvLevelSigma2), Tcr(f.Tcr.clone()), Trb(f.Trb)
{
    if (bNeedVisualization)
        f.mImage.copyTo(mImage);

    for (int i = 0; i != GRID_COLS; ++i)
        for (int j = 0; j < GRID_ROWS; ++j)
            mGrid[i][j] = f.mGrid[i][j];

    if (!f.Tcw.empty())
        setPose(f.Tcw);
}

//! 赋值拷贝
Frame& Frame::operator=(const Frame& f)
{
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

    mnScaleLevels = f.mnScaleLevels;
    mfScaleFactor = f.mfScaleFactor;
    mvScaleFactors = f.mvScaleFactors;
    mvLevelSigma2 = f.mvLevelSigma2;
    mvInvLevelSigma2 = f.mvInvLevelSigma2;

    for (int i = 0; i != GRID_COLS; ++i)
        for (int j = 0; j < GRID_ROWS; ++j)
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
bool Frame::posInGrid(KeyPoint& kp, int& posX, int& posY)
{
    posX = round((kp.pt.x - minXUn) * gridElementWidthInv);
    posY = round((kp.pt.y - minYUn) * gridElementHeightInv);

    //! Keypoint's coordinates are undistorted, which could cause to go out of the image
    //! 特征点坐标是经过畸变矫正过的，可能会超出图像
    if (posX < 0 || posX >= GRID_COLS || posY < 0 || posY >= GRID_ROWS)
        return false;

    return true;
}

/**
 * @brief 找到在 以x,y为中心,边长为2r的方形内且在[minLevel, maxLevel]的特征点
 * @return 返回区域内特征点在KF里的索引
 */
vector<size_t> Frame::getFeaturesInArea(const float& x, const float& y, const float& r,
                                        int minLevel, int maxLevel) const
{
    assert(r > 0);

    vector<size_t> vIndices;
    vIndices.reserve(N);

    //! floor向下取整
    int nMinCellX = floor((x - minXUn - r) * gridElementWidthInv);
    nMinCellX = max(0, nMinCellX);
    if (nMinCellX >= GRID_COLS)
        return vIndices;

    //! ceil向上取整
    int nMaxCellX = ceil((x - minXUn + r) * gridElementWidthInv);
    nMaxCellX = min(GRID_COLS - 1, nMaxCellX);
    if (nMaxCellX < 0)
        return vIndices;

    int nMinCellY = floor((y - minYUn - r) * gridElementHeightInv);
    nMinCellY = max(0, nMinCellY);
    if (nMinCellY >= GRID_ROWS)
        return vIndices;

    int nMaxCellY = ceil((y - minYUn + r) * gridElementHeightInv);
    nMaxCellY = min(GRID_ROWS - 1, nMaxCellY);
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
                const KeyPoint& kpUn = mvKeyPoints[vCell[j]];
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


}  // namespace se2lam
