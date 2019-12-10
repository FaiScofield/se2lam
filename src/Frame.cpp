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

namespace se2lam
{
using namespace cv;
using namespace std;

typedef unique_lock<mutex> locker;

unsigned long Frame::nextId = 0;
bool Frame::bIsInitialComputation = true;
bool Frame::bNeedVisualization = true;
float Frame::minXUn, Frame::minYUn, Frame::maxXUn, Frame::maxYUn;
float Frame::gridElementWidthInv, Frame::gridElementHeightInv;

Frame::Frame()
{}

Frame::~Frame()
{}

//! 特征点法的Frame构造
Frame::Frame(const Mat& im, const Se2& odo, ORBextractor* extractor, double time)
    : mpORBExtractor(extractor), mbBowVecExist(false), mTimeStamp(time), odom(odo), mbNull(false)
{
    //! 首帧会根据k和d重新计算图像边界, 只计算一次
    if (bIsInitialComputation) {
        //! 计算去畸变后的图像边界
        computeBoundUn(Config::Kcam, Config::Dcam);

        gridElementWidthInv = static_cast<float>(GRID_COLS) / (maxXUn - minXUn);
        gridElementHeightInv = static_cast<float>(GRID_ROWS) / (maxYUn - minYUn);

        bIsInitialComputation = false;
        bNeedVisualization = Config::NeedVisualization;
        fprintf(stderr, "\n[Frame][Info ] 去畸变的图像边界为: X: [%.1f, %.1f], Y: [%.1f, %.1f]\n",
                minXUn, maxXUn, minYUn, maxYUn);
    }

    id = nextId++;

    Mat imgGray, imgUn, imgClahed;

    //! 输入图像去畸变
    if (im.channels() != 1)
        cvtColor(im, imgGray, CV_BGR2GRAY);
    else
        imgGray = im;
    undistort(imgGray, imgUn, Config::Kcam, Config::Dcam);

    //!  限制对比度自适应直方图均衡
    Ptr<CLAHE> clahe = createCLAHE(3.0, Size(8, 8));
    clahe->apply(imgUn, imgClahed);

    //! 提取特征点
    (*mpORBExtractor)(imgClahed, Mat(), mvKeyPoints, mDescriptors);
    N = mvKeyPoints.size();
    if (N == 0) {
        cerr << "[Frame][Error] #" << id << " No features in this frame!" << endl;
        this->setNull();
        return;
    }
    mvbMPOutlier.resize(N, true);
    mvpMapPoints.resize(N, static_cast<PtrMapPoint>(nullptr));

    if (bNeedVisualization)
        imgClahed.copyTo(mImage);

    //! 计算金字塔每层尺度和高斯噪声
    mnScaleLevels = mpORBExtractor->getLevels();       // default 5
    mfScaleFactor = mpORBExtractor->getScaleFactor();  // default 1.2
    mvScaleFactors.resize(mnScaleLevels);
    mvLevelSigma2.resize(mnScaleLevels);
    mvInvLevelSigma2.resize(mnScaleLevels);
    mvScaleFactors[0] = 1.0f;
    mvLevelSigma2[0] = 1.0f;
    for (int i = 1; i != mnScaleLevels; ++i) {
        mvScaleFactors[i] = mvScaleFactors[i - 1] * mfScaleFactor;
        mvLevelSigma2[i] = mvScaleFactors[i] * mvScaleFactors[i];
        mvInvLevelSigma2[i] = 1.0 / mvLevelSigma2[i];
    }

    //! Assign Features to Grid Cells. 将特征点按照cell存放
    int nReserve = N / (GRID_COLS * GRID_ROWS);
    for (int i = 0; i != GRID_COLS; ++i) {
        for (int j = 0; j < GRID_ROWS; ++j)
            mGrid[i][j].reserve(nReserve);
    }
    for (size_t i = 0; i != N; ++i) {
        const KeyPoint& kp = mvKeyPoints[i];
        int nGridPosX, nGridPosY;
        if (posInGrid(kp, nGridPosX, nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

//! 光流法中的Frame构造
Frame::Frame(const Mat& im, const Se2& odo, const vector<KeyPoint>& vKPs, ORBextractor* extractor, double time)
    : mpORBExtractor(extractor), mbBowVecExist(false), mTimeStamp(time), odom(odo),
      mvKeyPoints(vKPs), mbNull(false)
{
    //! 首帧会根据k和d重新计算图像边界, 只计算一次
    if (bIsInitialComputation) {
        //! 计算去畸变后的图像边界
        computeBoundUn(Config::Kcam, Config::Dcam);

        gridElementWidthInv = static_cast<float>(GRID_COLS) / (maxXUn - minXUn);
        gridElementHeightInv = static_cast<float>(GRID_ROWS) / (maxYUn - minYUn);

        bIsInitialComputation = false;
        bNeedVisualization = Config::NeedVisualization;
        fprintf(stderr, "\n[Frame][Info ] 去畸变后的图像边界为: X: [%.1f, %.1f], Y: [%.1f, %.1f]\n",
                minXUn, maxXUn, minYUn, maxYUn);
    }

    id = nextId++;

    //! 计算描述子
    mpORBExtractor->getDescriptors(im, mvKeyPoints, mDescriptors);
    N = mvKeyPoints.size();
    if (N == 0) {
        cerr << "[Frame][Error] #" << id << " No features in this frame!" << endl;
        this->setNull();
        return;
    }
    mvbMPOutlier.resize(N, true);
    mvpMapPoints.resize(N, static_cast<PtrMapPoint>(nullptr));

    if (bNeedVisualization)
        im.copyTo(mImage);

    //! 计算金字塔每层尺度和高斯噪声
    mnScaleLevels = mpORBExtractor->getLevels();       // default 1
    mfScaleFactor = mpORBExtractor->getScaleFactor();  // default 1.2
    mvScaleFactors.resize(mnScaleLevels);
    mvLevelSigma2.resize(mnScaleLevels);
    mvInvLevelSigma2.resize(mnScaleLevels);
    mvScaleFactors[0] = 1.0f;
    mvLevelSigma2[0] = 1.0f;
    for (int i = 1; i != mnScaleLevels; ++i) {
        mvScaleFactors[i] = mvScaleFactors[i - 1] * mfScaleFactor;
        mvLevelSigma2[i] = mvScaleFactors[i] * mvScaleFactors[i];
        mvInvLevelSigma2[i] = 1.0 / mvLevelSigma2[i];
    }

    //! Assign Features to Grid Cells. 将特征点按照cell存放
    int nReserve = N / (GRID_COLS * GRID_ROWS);
    for (int i = 0; i != GRID_COLS; ++i) {
        for (int j = 0; j < GRID_ROWS; ++j)
            mGrid[i][j].reserve(nReserve);
    }
    for (size_t i = 0; i != N; ++i) {
        KeyPoint& kp = mvKeyPoints[i];
        int nGridPosX, nGridPosY;
        if (posInGrid(kp, nGridPosX, nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

//! 复制构造函数, 初始化KF用
Frame::Frame(const Frame& f)
    : mpORBExtractor(f.mpORBExtractor), mbBowVecExist(f.mbBowVecExist), mTimeStamp(f.mTimeStamp),
      id(f.id), odom(f.odom), N(f.N), mDescriptors(f.mDescriptors.clone()), mvKeyPoints(f.mvKeyPoints),
      mvbMPOutlier(f.mvbMPOutlier), mnScaleLevels(f.mnScaleLevels), mfScaleFactor(f.mfScaleFactor),
      mvScaleFactors(f.mvScaleFactors), mvLevelSigma2(f.mvLevelSigma2), mvInvLevelSigma2(f.mvInvLevelSigma2),
      mbNull(f.mbNull), Tcr(f.Tcr), Tcw(f.Tcw), Trb(f.Trb), Twb(f.Twb), mvpMapPoints(f.mvpMapPoints)
{
    if (bNeedVisualization)
        f.mImage.copyTo(mImage);

    for (int i = 0; i != GRID_COLS; ++i)
        for (int j = 0; j < GRID_ROWS; ++j)
            mGrid[i][j] = f.mGrid[i][j];
}

//! 赋值拷贝, Track中交换前后帧数据使用
Frame& Frame::operator=(const Frame& f)
{
    mpORBExtractor = f.mpORBExtractor;

    mBowVec = f.mBowVec;
    mFeatVec = f.mFeatVec;
    mbBowVecExist = f.mbBowVecExist;

    mTimeStamp = f.mTimeStamp;
    id = f.id;
    odom = f.odom;

    if (bNeedVisualization)
        f.mImage.copyTo(mImage);

    N = f.N;
    mvKeyPoints = f.mvKeyPoints;
    mvbMPOutlier = f.mvbMPOutlier;
    f.mDescriptors.copyTo(mDescriptors);
    for (int i = 0; i != GRID_COLS; ++i)
        for (int j = 0; j < GRID_ROWS; ++j)
            mGrid[i][j] = f.mGrid[i][j];

    mnScaleLevels = f.mnScaleLevels;
    mfScaleFactor = f.mfScaleFactor;
    mvScaleFactors = f.mvScaleFactors;
    mvLevelSigma2 = f.mvLevelSigma2;
    mvInvLevelSigma2 = f.mvInvLevelSigma2;

    mbNull = f.mbNull;
    mvpMapPoints = f.mvpMapPoints;
    Trb = f.Trb;
    Twb = f.Twb;
    Tcr = f.Tcr.clone();
    Tcw = f.Tcw.clone();

    return *this;
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

Se2 Frame::getTwb()
{
    locker lock(mMutexPose);
    return Twb;
}

Se2 Frame::getTrb()
{
    locker lock(mMutexPose);
    return Trb;
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

// 只执行一次
void Frame::computeBoundUn(const Mat& K, const Mat& D)
{
    float x = static_cast<float>(Config::ImgSize.width);
    float y = static_cast<float>(Config::ImgSize.height);
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

bool Frame::inImgBound(const cv::Point2f& pt) const
{
    return (pt.x >= minXUn && pt.x <= maxXUn && pt.y >= minYUn && pt.y <= maxYUn);
}

// From ORB_SLAM
bool Frame::posInGrid(const KeyPoint& kp, int& posX, int& posY) const
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
 * @brief 找到在 以x,y为中心,边长为2r的方形内且在[minLevel, maxLevel]的特征点. 花销较大
 * @return 返回区域内特征点在KF里的索引
 */
vector<size_t> Frame::getFeaturesInArea(const float x, const float y, const float r,
                                        const int minLevel, const int maxLevel) const
{
    assert(r > 0);

    vector<size_t> vIndices;
    vIndices.reserve(0.1 * N);

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

void Frame::computeBoW(const ORBVocabulary* _pVoc)
{
    if (mBowVec.empty() || mFeatVec.empty()) {
        vector<cv::Mat> vCurrentDesc = toDescriptorVector(mDescriptors);
        // Feature vector associate features with nodes in the 4th level (from leaves up)
        // We assume the vocabulary tree has 6 levels, change the 4 otherwise
        _pVoc->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
    }
    mbBowVecExist = true;
}


Point3f Frame::getMPPoseInCamareFrame(size_t idx)
{
    locker lock(mMutexObs);
    Point3f ret(-1.f, -1.f, -1.f);
    if (mvpMapPoints[idx]) {
        const Point3f Pw = mvpMapPoints[idx]->getPos();
        ret = cvu::se3map(Tcw, Pw);
    }
    return ret;
}

PtrMapPoint Frame::getObservation(size_t idx)
{
    locker lock(mMutexObs);
    return mvpMapPoints[idx];
}

vector<PtrMapPoint> Frame::getObservations(bool checkValid, bool checkParallax)
{
    locker lock(mMutexObs);
    if (!checkValid) {
        return mvpMapPoints;
    } else {
        vector<PtrMapPoint> vValidMPs;
        vValidMPs.reserve(N >> 1);
        for (size_t i = 0; i < N; ++i) {
            const PtrMapPoint& pMP = mvpMapPoints[i];
            if (!pMP || pMP->isNull())
                continue;
            if (checkParallax && !pMP->isGoodPrl())
                continue;
            vValidMPs.push_back(pMP);
        }
        return vValidMPs;
    }
}

size_t Frame::countObservations()
{
    locker lock(mMutexObs);
    size_t count = 0;
    for (size_t i = 0; i < N; ++i) {
        const PtrMapPoint& pMP = mvpMapPoints[i];
        if (!pMP || pMP->isNull())
            continue;
        count++;
    }
    return count;
}

void Frame::setObservation(const PtrMapPoint& pMP, size_t idx)
{
    locker lock(mMutexObs);
    if (idx >= 0 && idx < N)
        mvpMapPoints[idx] = pMP;
}

void Frame::clearObservations()
{
    locker lock(mMutexObs);
    std::fill(mvpMapPoints.begin(), mvpMapPoints.end(), nullptr);
}

bool Frame::hasObservationByIndex(size_t idx)
{
    locker lock(mMutexObs);
    if (idx >= N)
        return false;

    if (mvpMapPoints[idx])
        return true;
    else
        return false;
}

void Frame::eraseObservationByIndex(size_t idx)
{
    locker lock(mMutexObs);
    if (idx >= 0 && idx < N)
        mvpMapPoints[idx] = nullptr;
}

bool Frame::hasObservationByPointer(const PtrMapPoint& pMP)
{
    locker lock(mMutexObs);
    for (size_t i = 0; i < N; ++i) {
        if (mvpMapPoints[i] == pMP)
            return true;
    }
    return false;
}

void Frame::eraseObservationByPointer(const PtrMapPoint& pMP)
{
    locker lock(mMutexObs);
    for (size_t i = 0; i < N; ++i) {
        if (mvpMapPoints[i] == pMP) {
            mvpMapPoints[i] == nullptr;
            break;
        }
    }
}

void Frame::updateObservationsAfterOpt()
{
    locker lock(mMutexObs);
    for (size_t i = 0; i < N; ++i) {
        if (mvpMapPoints[i] && mvbMPOutlier[i])
            mvpMapPoints[i] = nullptr;
    }
}

void Frame::setNull()
{
    locker lock(mMutexObs);
    mDescriptors.release();
    mvKeyPoints.clear();
    mvpMapPoints.clear();
    mpORBExtractor = nullptr;
    mbNull = true;
    if (bNeedVisualization)
        mImage.release();
}

}  // namespace se2lam
