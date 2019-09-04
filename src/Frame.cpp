/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG
* (github.com/hbtang)
*/

#include "Frame.h"
#include "KeyFrame.h"
#include "Track.h"
#include "converter.h"
#include "cvutil.h"

#include "Preprocess.h"

namespace se2lam
{
using namespace cv;
using namespace std;

bool Frame::mbInitialComputations = true;
int Frame::nextId = 0;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;
float Frame::minXUn, Frame::minYUn, Frame::maxXUn, Frame::maxYUn;


Frame::Frame()
{}

Frame::Frame(const Mat &im, const Se2 &odo, ORBextractor *extractor, const Mat &K, const Mat &distCoef)
{
    mpORBExtractor = extractor;

    undistort(im, img, Config::Kcam, Config::Dcam); //! 输入图像去畸变

    //!  限制对比度自适应直方图均衡
    Ptr<CLAHE> clahe = createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(img, img);

    (*mpORBExtractor)(img, cv::Mat(), keyPoints, descriptors);

    N = keyPoints.size();
    if (keyPoints.empty())
        return;

    keyPointsUn = keyPoints;
    mvpMapPoints = vector<PtrMapPoint>(N, static_cast<PtrMapPoint>(NULL));
    mvbOutlier = vector<bool>(N, false);

    if (mbInitialComputations) {
        //! 计算去畸变后的图像边界
        computeBoundUn(K, distCoef);

        mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / (maxXUn - minXUn);
        mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / (maxYUn - minYUn);

        mbInitialComputations = false;
    }

    id = nextId;
    nextId++;

    //! Scale Levels Info
    mnScaleLevels = mpORBExtractor->GetLevels();       // default 5
    mfScaleFactor = mpORBExtractor->GetScaleFactor();  // default 1.2

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
    int nReserve = 0.5 * N / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
    for (unsigned int i = 0; i < FRAME_GRID_COLS; i++)
        for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++)
            mGrid[i][j].reserve(nReserve);

    for (size_t i = 0, iend = keyPointsUn.size(); i < iend; i++) {
        cv::KeyPoint &kp = keyPointsUn[i];

        int nGridPosX, nGridPosY;
        if (PosInGrid(kp, nGridPosX, nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }

    odom = odo;
// Tcw = cv::Mat::eye(4,4,CV_32FC1);
// Tcr = cv::Mat::eye(4,4,CV_32FC1);
}

Frame::Frame(const Frame &f)
{
    f.img.copyTo(img);
    keyPoints = f.keyPoints;
    keyPointsUn = f.keyPointsUn;
    f.descriptors.copyTo(descriptors);
    N = f.N;
    mvpMapPoints = f.mvpMapPoints;
    mvbOutlier = f.mvbOutlier;

    minXUn = f.minXUn;
    minYUn = f.minYUn;
    maxXUn = f.maxXUn;
    maxYUn = f.maxYUn;

    mnScaleLevels = f.mnScaleLevels;
    mfScaleFactor = f.mfScaleFactor;
    mvScaleFactors = f.mvScaleFactors;
    mvLevelSigma2 = f.mvLevelSigma2;
    mvInvLevelSigma2 = f.mvInvLevelSigma2;
    for (int i = 0; i < FRAME_GRID_COLS; i++)
        for (int j = 0; j < FRAME_GRID_ROWS; j++)
            mGrid[i][j] = f.mGrid[i][j];

    f.Tcw.copyTo(Tcw);
    f.Tcr.copyTo(Tcr);

    mTime = f.getTime();

    id = f.id;
    odom = f.odom;
    Twb = f.Twb;
    Trb = f.Trb;

    lineFeature = f.lineFeature;
    pointAndLineLable = f.pointAndLineLable;
    lineIncludePoints = f.lineIncludePoints;
}

Frame &Frame::operator=(const Frame &f)
{
    f.img.copyTo(img);
    keyPoints = f.keyPoints;
    keyPointsUn = f.keyPointsUn;
    f.descriptors.copyTo(descriptors);
    N = f.N;
    mvpMapPoints = f.mvpMapPoints;
    mvbOutlier = f.mvbOutlier;

    minXUn = f.minXUn;
    minYUn = f.minYUn;
    maxXUn = f.maxXUn;
    maxYUn = f.maxYUn;

    mnScaleLevels = f.mnScaleLevels;
    mfScaleFactor = f.mfScaleFactor;
    mvScaleFactors = f.mvScaleFactors;
    mvLevelSigma2 = f.mvLevelSigma2;
    mvInvLevelSigma2 = f.mvInvLevelSigma2;
    for (int i = 0; i < FRAME_GRID_COLS; i++)
        for (int j = 0; j < FRAME_GRID_ROWS; j++)
            mGrid[i][j] = f.mGrid[i][j];

    f.Tcw.copyTo(Tcw);
    f.Tcr.copyTo(Tcr);

    mTime = f.getTime();

    id = f.id;
    odom = f.odom;
    Twb = f.Twb;
    Trb = f.Trb;

    lineFeature = f.lineFeature;
    pointAndLineLable = f.pointAndLineLable;
    lineIncludePoints = f.lineIncludePoints;

    return *this;
}

float Frame::getTime() const
{
    return mTime;
}

void Frame::setTime(float time)
{
    mTime = time;
}

//! 这个函数没用了
void Frame::undistortKeyPoints(const Mat &K, const Mat &D)
{
    keyPointsUn = keyPoints;
    if (D.at<float>(0) == 0.) {
        return;
    }
    Mat_<Point2f> mat(1, keyPoints.size());
    for (size_t i = 0; i < keyPoints.size(); i++) {
        mat(i) = keyPoints[i].pt;
    }
    undistortPoints(mat, mat, K, D, Mat(), K);
    for (size_t i = 0; i < keyPoints.size(); i++) {
        keyPointsUn[i].pt = mat(i);
    }
    assert(keyPoints.size() == keyPointsUn.size());
}

void Frame::computeBoundUn(const Mat &K, const Mat &D)
{
    float x = (float)img.cols;
    float y = (float)img.rows;
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
bool Frame::PosInGrid(cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x - minXUn) * mfGridElementWidthInv);
    posY = round((kp.pt.y - minYUn) * mfGridElementHeightInv);

    //! Keypoint's coordinates are undistorted, which could cause to go out of the image
    //! 特征点坐标是经过畸变矫正过的，可能会超出图像
    if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS)
        return false;

    return true;
}

/**
 * @brief 找到在 以x,y为中心,边长为2r的方形内且在[minLevel, maxLevel]的特征点
 * @param x        图像坐标u
 * @param y        图像坐标v
 * @param r        边长
 * @param minLevel 最小尺度
 * @param maxLevel 最大尺度
 * @return         满足条件的特征点的序号
 */
vector<size_t> Frame::GetFeaturesInArea(const float &x, const float &y, const float &r,
                                        int minLevel, int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(keyPointsUn.size());

    //! floor向下取整
    int nMinCellX = floor((x - minXUn - r) * mfGridElementWidthInv);
    nMinCellX = max(0, nMinCellX);
    if (nMinCellX >= FRAME_GRID_COLS)
        return vIndices;

    //! ceil向上取整
    int nMaxCellX = ceil((x - minXUn + r) * mfGridElementWidthInv);
    nMaxCellX = min(FRAME_GRID_COLS - 1, nMaxCellX);
    if (nMaxCellX < 0)
        return vIndices;

    int nMinCellY = floor((y - minYUn - r) * mfGridElementHeightInv);
    nMinCellY = max(0, nMinCellY);
    if (nMinCellY >= FRAME_GRID_ROWS)
        return vIndices;

    int nMaxCellY = ceil((y - minYUn + r) * mfGridElementHeightInv);
    nMaxCellY = min(FRAME_GRID_ROWS - 1, nMaxCellY);
    if (nMaxCellY < 0)
        return vIndices;

    const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);
    const bool bSameLevel = (minLevel == maxLevel);

    for (int ix = nMinCellX; ix <= nMaxCellX; ix++) {
        for (int iy = nMinCellY; iy <= nMaxCellY; iy++) {
            vector<size_t> vCell = mGrid[ix][iy];
            if (vCell.empty())
                continue;

            for (size_t j = 0, jend = vCell.size(); j < jend; j++) {
                const cv::KeyPoint &kpUn = keyPointsUn[vCell[j]];
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

void Frame::copyImgTo(cv::Mat &imgRet)
{
    lock_guard<mutex> lock(mMutexImg);
    img.copyTo(imgRet);
}

void Frame::copyDesTo(cv::Mat &desRet)
{
    lock_guard<mutex> lock(mMutexDes);
    descriptors.copyTo(desRet);
}

Frame::~Frame()
{}

}  // namespace se2lam
