/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include "Map.h"
#include "Config.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "ORBmatcher.h"
#include "Track.h"
#include "cvutil.h"
#include "converter.h"

namespace se2lam
{

using namespace cv;
using namespace std;
typedef unique_lock<mutex> locker;

unsigned long MapPoint::mNextId = 0;


MapPoint::MapPoint()
    : mId(0), mpMap(nullptr), mbNull(false), mbGoodParallax(false), mMinDist(0.f), mMaxDist(10.f),
      mNormalVector(0.f, 0.f, 0.f), mMainKF(nullptr), mMainOctave(0),
      mLevelScaleFactor(Config::ScaleFactor)
{
    mvPosTmp.reserve(6);
}

//! 构造后请立即为其添加观测
MapPoint::MapPoint(const cv::Point3f& pos, bool goodPrl)
    : mId(mNextId++), mpMap(nullptr), mbNull(false), mbGoodParallax(goodPrl), mMinDist(0.f),
      mMaxDist(10.f), mPos(pos), mNormalVector(Point3f(0.f, 0.f, 0.f)), mMainKF(nullptr),
      mMainOctave(0), mLevelScaleFactor(Config::ScaleFactor)
{

    mvPosTmp.reserve(6);
}

MapPoint::~MapPoint()
{
    fprintf(stderr, "[MapPoint] MP#%ld 已被析构!\n", mId);
}

// 外部调用, 加锁
void MapPoint::setNull()
{
    locker lock(mMutexObs);
    setNullSelf();
}

// 内部调用, 不加锁
void MapPoint::setNullSelf()
{
    mbNull = true;
    mbGoodParallax = false;
    mMainKF = nullptr;

    PtrMapPoint pThis = shared_from_this();
    for (auto it = mObservations.begin(), iend = mObservations.end(); it != iend; ++it) {
        PtrKeyFrame pKF = it->first;
        if (pKF->hasObservationByPointer(pThis))
            pKF->eraseObservationByPointer(pThis);
    }
    mObservations.clear();
    mMainDescriptor.release();

    if (mpMap != nullptr)
        mpMap->eraseMP(pThis);
}

// 可以保证返回的KF都是有效的
vector<PtrKeyFrame> MapPoint::getObservations()
{
    locker lock(mMutexObs);

    vector<PtrKeyFrame> pKFs;
    pKFs.reserve(mObservations.size());
    for (auto iter = mObservations.begin(), iend = mObservations.end(); iter != iend; ++iter) {
        PtrKeyFrame pKF = iter->first;
        if (!pKF || pKF->isNull())
            continue;
        pKFs.push_back(pKF);
    }
    return pKFs;
}

// Do pKF.setViewMP() before using this
void MapPoint::addObservation(const PtrKeyFrame& pKF, size_t idx)
{
    locker lock(mMutexObs);

    float oldObsSize = static_cast<float>(mObservations.size());
    // 插入时insert()的效率高, 更新时operator[]的效率高, emplace()不需要额外构造
    mObservations.emplace(pKF, idx);

    updateMainKFandDescriptor();
    updateParallax(pKF);

    // 更新平均观测方向
    Point3f newObs = pKF->getMPPoseInCamareFrame(idx);
    Point3f newNorm = newObs * (1.f / cv::norm(newObs));
    mNormalVector = (mNormalVector * oldObsSize + newNorm) / (oldObsSize + 1.f);
}

void MapPoint::eraseObservation(const PtrKeyFrame& pKF)
{
    locker lock(mMutexObs);

    Point3f viewPos = pKF->getMPPoseInCamareFrame(mObservations[pKF]);
    Point3f normPos = viewPos * (1.f / cv::norm(viewPos));

    mObservations.erase(pKF);
    if (mObservations.empty()) {
        fprintf(stderr, "[MapPoint] MP#%ld 因为没有观测而被自己设置为null\n", mId);
        setNullSelf();  // 这个函数不能加锁mMutexObs
    } else {
        updateMainKFandDescriptor();
        // 更新平均观测方向
        float size = static_cast<float>(mObservations.size());
        mNormalVector = (mNormalVector * (size + 1.f) - normPos) / size;
    }
}

bool MapPoint::hasObservation(const PtrKeyFrame& pKF)
{
    locker lock(mMutexObs);
    return (mObservations.count(pKF));
}

size_t MapPoint::countObservations()
{
    locker lock(mMutexObs);
    return mObservations.size();
}


Point3f MapPoint::getNormalVector()
{
    locker lock(mMutexObs);
    return mNormalVector;
}

Point2f MapPoint::getMainMeasureProjection()
{
    locker lock(mMutexObs);
    assert(mMainKF != nullptr);
    size_t idx = mObservations[mMainKF];
    return mMainKF->mvKeyPoints[idx].pt;
}

Mat MapPoint::getDescriptor()
{
    locker lock(mMutexObs);
    return mMainDescriptor.clone();
}

PtrKeyFrame MapPoint::getMainKF()
{
    locker lock(mMutexObs);
    return mMainKF;
}

int MapPoint::getOctave(const PtrKeyFrame& pKF)
{
    locker lock(mMutexObs);
    if (mObservations.count(pKF))
        return pKF->mvKeyPoints[mObservations[pKF]].octave;
    else
        return -1;
}

int MapPoint::getKPIndexInKF(PtrKeyFrame pKF)
{
    locker lock(mMutexObs);
    if (mObservations.count(pKF))
        return mObservations[pKF];
    else
        return -1;
}

int MapPoint::getMainOctave()
{
    locker lock(mMutexObs);
    return mMainOctave;
}

Point3f MapPoint::getPos()
{
    locker lock(mMutexPos);
    return mPos;
}

void MapPoint::setPos(const Point3f& pt3f)
{
    locker lock(mMutexPos);
    mPos = pt3f;
//    updateMeasureInKFs();
}

//! MP在优化更新坐标后要更新其他KF对其的观测值(Pc), setPos()后需要执行!
//void MapPoint::updateMeasureInKFs()
//{
//    for (auto it = mObservations.begin(), iend = mObservations.end(); it != iend; ++it) {
//        PtrKeyFrame pKF = it->first;
//        assert(pKF != nullptr);
//        assert(!pKF->isNull());

//        Mat Tcw = pKF->getPose();
//        Point3f posKF = cvu::se3map(Tcw, mPos);

//        pKF->setObservation([it->second] = posKF;
//    }
//}


/**
 * @brief 根据观测方向及坐标合理性, 判断此MP是不是KF的一个合理观测, LocalMap里关联MP时调用
 * @param posKF  此MP在KF相机坐标系下的坐标, 即Pc
 * @param kp     此MP投影到KF后匹配上的特征点
 * @return
 */
bool MapPoint::acceptNewObserve(const Point3f& posKF, const KeyPoint& kp)
{
    locker lock(mMutexObs);

    float dist = cv::norm(posKF);
    float cosAngle = cv::norm(posKF.dot(mNormalVector)) / (dist * cv::norm(mNormalVector));
    bool c1 = std::abs(mMainKF->mvKeyPoints[mObservations[mMainKF]].octave - kp.octave) <= 2;
    bool c2 = cosAngle >= 0.866f;  // no larger than 30 degrees
    bool c3 = dist >= mMinDist && dist <= mMaxDist;
    return c1 && c2 && c3;
}

// This MP would be replaced and abandoned later
void MapPoint::mergedInto(const shared_ptr<MapPoint>& pMP)
{
    locker lock(mMutexObs);
    for (auto it = mObservations.begin(), iend = mObservations.end(); it != iend; ++it) {
        PtrKeyFrame pKF = it->first;
        assert(pKF != nullptr);
        assert(!pKF->isNull());
        size_t idx = it->second;
        pKF->setObservation(pMP, idx);
        pMP->addObservation(pKF, idx);
        mObservations.erase(pKF);
    }
    //! NOTE 不能直接setNullSelf(), 此函数会删除mObservations里KF的观测, 观测已经更新, 不可删除
    //! 故要在上面的循环体中先删除mObservations的元素
    fprintf(stderr, "[MapPoint] MP#%ld 正在被 MP#%ld 所替换, 即将析构!\n", mId, pMP->mId);
    setNullSelf();
}


/**
 * @brief 更新MP的 mainKF和描述子
 * 在addObservation()和eraseObservation()后需要调用! 注意不要加锁mMutexObs
 * mMainKF只有在这个函数里才更新!
 */
void MapPoint::updateMainKFandDescriptor()
{
    if (mObservations.empty())
        return;

    vector<Mat> vDes;
    vector<PtrKeyFrame> vKFs;
    vDes.reserve(mObservations.size());
    vKFs.reserve(mObservations.size());
    for (auto i = mObservations.begin(), iend = mObservations.end(); i != iend; ++i) {
        PtrKeyFrame pKF = i->first;
        if (!pKF)
            continue;
        if (!pKF->isNull()) {
            vKFs.push_back(pKF);
            vDes.push_back(pKF->mDescriptors.row(i->second));
        }
    }

    if (vDes.empty()) {
        fprintf(stderr, "[MapPoint] MP#%ld 被设为null因为它没有描述子!\n", mId);
        setNullSelf();
        return;
    }

    // Compute distances between them
    const size_t N = vDes.size();
    int Distances[N][N];
    for (size_t i = 0; i < N; ++i) {
        Distances[i][i] = 0;
        for (size_t j = i + 1; j < N; ++j) {
            int distij = ORBmatcher::DescriptorDistance(vDes[i], vDes[j]);
            Distances[i][j] = distij;
            Distances[j][i] = distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    int bestMedian = INT_MAX;
    size_t bestIdx = 0;
    for (size_t i = 0; i < N; ++i) {
        vector<int> vDists(Distances[i], Distances[i] + N);  // Distances[i]行由int[]转vector<int>
        std::sort(vDists.begin(), vDists.end());
        int median = vDists[0.5 * (N - 1)];
        if (median < bestMedian) {  // 每行描述子到其他汉明距离的中位数为这一行最好的描述子
            bestMedian = median;
            bestIdx = i;
        }
    }

    mMainDescriptor = vDes[bestIdx].clone();                           // 更新最佳描述子
    if (mMainKF != nullptr && mMainKF->mIdKF == vKFs[bestIdx]->mIdKF)  // 更新mMainKF
        return;
    mMainKF = vKFs[bestIdx];

    // 更新其他参数
    size_t idx = mObservations[mMainKF];
    mMainOctave = mMainKF->mvKeyPoints[idx].octave;
    mLevelScaleFactor = mMainKF->mvScaleFactors[mMainOctave];  // 第0层值为1
    float dist = cv::norm(mMainKF->getMPPoseInCamareFrame(idx));
    int nlevels = mMainKF->mnScaleLevels;  // 金字塔总层数

    //! 金字塔为1层时这里mMinDist和mMinDist会相等! 程序错误!
    //! 已修正 20191126
    if (mLevelScaleFactor - 1.0 < 1e-6) {
        mMaxDist = dist * Config::ScaleFactor;  // 1.2
        mMinDist = dist;
    } else {
        mMaxDist = dist * mLevelScaleFactor;
        mMinDist = mMaxDist / mMainKF->mvScaleFactors[nlevels - 1];
    }
}

/**
 * @brief 视差不好的MP增加新的观测后会更新视差
 * 一旦MP视差合格后就不会再更新, 如果连续6帧KF后视差仍不好, 会抛弃该MP
 * @param pKF   能观测到此MP的KF指针
 * TODO 有改进空间, 可以用深度滤波器更新视差
 */
void MapPoint::updateParallax(const PtrKeyFrame& pKF)
{
    if (mbGoodParallax || mObservations.size() <= 2)
        return;
//    if (mvPosTmp.size() > 5) {
//        mbGoodParallax = true;
//        mvPosTmp.clear();
//        return;
//    }

    // Get the oldest KF in the last 6 KFs. 由从此KF往前最多10帧KF和此KF做三角化更新MP坐标
    for (auto it = mObservations.begin(), iend = mObservations.end(); it != iend; ++it) {
        PtrKeyFrame pKFj = it->first;
        assert(pKFj != nullptr);
        if (pKF->mIdKF - pKFj->mIdKF > 10)  // 6
            continue;
        if (updateParallaxCheck(pKFj, pKF))  // 和往前10帧内的KF都进行一次三角化, 直至更新成为止
            break; // 一次更新完成就退出
    }

//    size_t n = mvPosTmp.size();
//    if (n > 0) {
//        cerr << "MP#" << mId << " 的临时视差: ";
//        Point3f poseTatal(0.f, 0.f, 0.f);
//        for (size_t i = 0; i < n; ++i) {
//            poseTatal += mvPosTmp[i];
//            cout << mvPosTmp[i] << ", ";
//        }
//        cout << endl;

//        locker lock(mMutexPos);
//        mPos = poseTatal * (1.f / n);
//        mbGoodParallax = true;
////        updateMeasureInKFs();
//        fprintf(stderr, "[MapPoint] MP#%ld 的视差被更新为good.\n", mId);
//    }

    // 如果和前面10帧的三角化后视差仍然没有更新到好则抛弃此MP
    if (!mbGoodParallax && mObservations.size() > 11)
        setNullSelf();
}

//! Update measurements in KFs. 更新约束和信息矩阵
bool MapPoint::updateParallaxCheck(const PtrKeyFrame& pKF1, const PtrKeyFrame& pKF2)
{
    if (pKF1->mIdKF == pKF2->mIdKF)
        return false;

    // Do triangulation
    const Mat Tc1w = pKF1->getPose();
    const Mat Tc2w = pKF2->getPose();
    const Mat Proj1 = Config::Kcam * Tc1w.rowRange(0, 3);
    const Mat Proj2 = Config::Kcam * Tc2w.rowRange(0, 3);
    const Point2f pt1 = pKF1->mvKeyPoints[mObservations[pKF1]].pt;
    const Point2f pt2 = pKF2->mvKeyPoints[mObservations[pKF2]].pt;
    const Point3f posW = cvu::triangulate(pt1, pt2, Proj1, Proj2);

    const Point3f Pc1 = cvu::se3map(Tc1w, posW);
    const Point3f Pc2 = cvu::se3map(Tc2w, posW);
    if (!Config::acceptDepth(Pc1.z) || !Config::acceptDepth(Pc2.z))
        return false;

    const Point3f Ocam1 = pKF1->getCameraCenter();
    const Point3f Ocam2 = pKF2->getCameraCenter();
    if (!cvu::checkParallax(Ocam1, Ocam2, posW, 2))
        return false;

    // 视差良好, 则更新此点的三维坐标
    {
        locker lock(mMutexPos);
         mPos = posW;
         mbGoodParallax = true;
         fprintf(stderr, "[MapPoint] MP#%ld 的视差被更新为good.\n", mId);
         // mvPosTmp.push_back(posW);
    }

    // Update measurements in KFs. 更新约束和信息矩阵
    Eigen::Matrix3d xyzinfo1, xyzinfo2;
    calcSE3toXYZInfo(Pc1, Tc1w, Tc2w, xyzinfo1, xyzinfo2);
    pKF1->setObsAndInfo(shared_from_this(), mObservations[pKF1], xyzinfo1);
    pKF2->setObsAndInfo(shared_from_this(), mObservations[pKF2], xyzinfo2);

//    Mat Rc1w = Tc1w.rowRange(0, 3).colRange(0, 3);
//    Mat xyzinfoW = Rc1w.t() * toCvMat(xyzinfo1) * Rc1w;

    // Update measurements in other KFs. 位姿变后要更新其他KF对其的观测坐标值
//    for (auto it = mObservations.begin(), iend = mObservations.end(); it != iend; ++it) {
//        PtrKeyFrame pKFi = it->first;
//        if (!pKFi)
//            continue;
//        if (pKFi->mIdKF == pKF2->mIdKF || pKFi->mIdKF == pKF1->mIdKF)
//            continue;
//        Mat Tciw = pKFi->getPose();
//        // Point3f Pci = cvu::se3map(Tciw, posW);
//        Mat Rciw = Tciw.rowRange(0, 3).colRange(0, 3);
//        Mat xyzinfoi = Rciw * xyzinfoW * Rciw.t();
//        pKFi->setObsAndInfo(shared_from_this(), mObservations[pKFi], toMatrix3d(xyzinfoi));
//    }
    return true;
}


}  // namespace se2lam
