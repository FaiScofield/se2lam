/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include "MapPoint.h"
#include "Config.h"
#include "KeyFrame.h"
#include "ORBmatcher.h"
#include "Track.h"
#include "cvutil.h"

namespace se2lam
{

using namespace cv;
using namespace std;
typedef unique_lock<mutex> locker;


//bool MPIdLessThan::operator()(const std::shared_ptr<MapPoint>& lhs, const std::shared_ptr<MapPoint>& rhs) const
//{
//    return lhs->mId < rhs->mId;
//}


unsigned long MapPoint::mNextId = 1;

MapPoint::MapPoint()
    : mMainOctave(0), mLevelScaleFactor(Config::ScaleFactor), mId(0), mMainKF(nullptr),
      mNormalVector(Point3f(0.f, 0.f, 0.f)), mbNull(false), mbGoodParallax(false)
{
    mObservations.clear();

    mMinDist = 0.f;
    mMaxDist = 0.f;
}

//! 构造后请立即为其添加观测
MapPoint::MapPoint(Point3f pos, bool goodPrl)
    : mMainOctave(0), mLevelScaleFactor(Config::ScaleFactor), mPos(pos), mMainKF(nullptr),
      mNormalVector(Point3f(0.f, 0.f, 0.f)), mbNull(false), mbGoodParallax(goodPrl)
{
    mId = mNextId++;
    mObservations.clear();

    mMinDist = 0.f;
    mMaxDist = 0.f;
}

MapPoint::~MapPoint()
{
    fprintf(stderr, "[MapPoint] MP#%ld 已被析构!\n", mId);
}

//! FIXME Count pointer = 4 时无法析构, Count pointer = 2时正常析构
void MapPoint::setNull(shared_ptr<MapPoint>& pThis)
{
    locker lock(mMutexObs);
    mbNull = true;
    mbGoodParallax = false;

    fprintf(stderr, "[MapPoint] MP#%ld 处理KF观测前(%ld), 引用计数 = %ld\n",
            mId, mObservations.size(), pThis.use_count());
    for (auto it = mObservations.begin(), iend = mObservations.end(); it != iend; ++it) {
        PtrKeyFrame pKF = it->first;
        if (pKF->hasObservation(pThis))
            pKF->eraseObservation(pThis);
    }
    fprintf(stderr, "[MapPoint] MP#%ld 处理KF观测后, 引用计数 = %ld\n",
            mId, mObservations.size(), pThis.use_count());

    mObservations.clear();
    mMainDescriptor.release();
    mMainKF = nullptr;

    mpMap->eraseMP(pThis);

    fprintf(stderr, "[MapPoint] MP#%ld 被Map设置为null. 引用计数 = %ld\n", pThis->mId,
            pThis.use_count());
}

//! Abandon a MP as an outlier, only for internal use. 注意不要加锁mMutexObs
void MapPoint::setNull()
{
    mbNull = true;
    mbGoodParallax = false;
    for (auto it = mObservations.begin(), iend = mObservations.end(); it != iend; ++it) {
        PtrKeyFrame pKF = it->first;
        if (pKF->hasObservation(it->second))
            pKF->eraseObservation(it->second);
    }
    mObservations.clear();
    mMainDescriptor.release();
    mMainKF = nullptr;

    auto ptr = shared_from_this();
    mpMap->eraseMP(ptr);
    fprintf(stderr, "[MapPoint] MP#%ld 被自己设置为null. 引用计数 = %ld\n", this->mId,
            ptr.use_count());
}

size_t MapPoint::countObservation()
{
    locker lock(mMutexObs);
    return mObservations.size();
}

bool MapPoint::hasObservation(const PtrKeyFrame& pKF)
{
    locker lock(mMutexObs);
    return (mObservations.find(pKF) != mObservations.end());
}

//! 可以保证返回的KF都是有效的
set<PtrKeyFrame> MapPoint::getObservations()
{
    locker lock(mMutexObs);

    set<PtrKeyFrame> pKFs;
    for (auto i = mObservations.begin(), iend = mObservations.end(); i != iend; ++i) {
        PtrKeyFrame pKF = i->first;
        if (!pKF)
            continue;
        if (pKF->isNull())
            continue;
        pKFs.insert(i->first);
    }
    return pKFs;
}

void MapPoint::eraseObservation(const PtrKeyFrame& pKF)
{
    locker lock(mMutexObs);

    mObservations.erase(pKF);
    if (mObservations.size() == 0) {
        fprintf(stderr, "[MapPoint] MP#%ld 因为没有观测而被设置为null\n", mId);
        setNull();  // 这个函数不能加锁mMutexObs
    }
    else
        updateMainKFandDescriptor();
}

void MapPoint::eraseObservations(const map<PtrKeyFrame, size_t>& obsCandidates)
{
    locker lock(mMutexObs);

    //! map.erase()足够安全不需要提前检查元素是否在字典里
    for (auto& oc : obsCandidates)
        mObservations.erase(oc.first);

    if (mObservations.size() == 0) {
        setNull();
    } else {
        updateMainKFandDescriptor();
    }
}

// Do pKF.setViewMP() before using this
void MapPoint::addObservation(const PtrKeyFrame& pKF, size_t idx)
{
    locker lock(mMutexObs);

    //! 插入时insert()的效率高, 更新时operator[]的效率高
    mObservations.insert(make_pair(pKF, idx));

    WorkTimer timer;
    updateMainKFandDescriptor();
    updateParallax(pKF);
//    printf("[MapPoint][Timer] MP#%ld 添加观测后更新描述子和视差共耗时: %.2fms\n", mId, timer.count());

    if (mbNull && mObservations.size() > 0)
        mbNull = false;
}

//! 多次观测一次性添加, 并更新相关变量, 防止多次计算
void MapPoint::addObservations(const map<PtrKeyFrame, size_t>& obsCandidates)
{
    locker lock(mMutexObs);

    for (auto& oc : obsCandidates) {
        mObservations.insert(oc);
        updateParallax(oc.first);
    }
    updateMainKFandDescriptor();

    if (mbNull && mObservations.size() > 0)
        mbNull = false;
}

// float MapPoint::getInvLevelSigma2(const PtrKeyFrame& pKF)
//{
//    int index = mObservations[pKF];
//    return pKF->mvInvLevelSigma2[index];
//}

Point3f MapPoint::getPos()
{
    locker lock(mMutexPos);
    return mPos;
}

void MapPoint::setPos(const Point3f& pt3f)
{
    locker lock(mMutexPos);
    mPos = pt3f;
}

/**
 * @brief   根据观测方向及坐标合理性, 判断此MP是不是KF的一个合理观测, LocalMap里关联MP时调用
 * @param posKF  此MP在KF相机坐标系下的坐标, 即Pc
 * @param kp     此MP投影到KF后匹配上的特征点
 * @return
 */
bool MapPoint::acceptNewObserve(Point3f posKF, const KeyPoint kp)
{
    locker lock(mMutexObs);

    float dist = cv::norm(posKF);
    float cosAngle = cv::norm(posKF.dot(mNormalVector)) / (dist * cv::norm(mNormalVector));
    bool c1 = std::abs(mMainKF->mvKeyPoints[mObservations[mMainKF]].octave - kp.octave) <= 2;
    bool c2 = cosAngle >= 0.866f;  // no larger than 30 degrees
//    bool c3 = dist >= mMinDist && dist <= mMaxDist;
    bool c3 = true;
    return c1 && c2 && c3;
}

Point3f MapPoint::getNormalVector()
{
    locker lock(mMutexObs);
    return mNormalVector;
}

Point2f MapPoint::getMainMeasure()
{
    locker lock(mMutexObs);
    if (!mMainKF) {
        fprintf(stderr, "[MapPoint] Error for MainKF doesn't exist!\n");
    }
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


/**
 * @brief 更新MP的mainKF,描述子以及平均观测方向mNormalVector
 * 在addObservation()和eraseObservation()后需要调用! 注意不要加锁mMutexObs
 * mMainKF只有在这个函数里才更新!
 */
void MapPoint::updateMainKFandDescriptor()
{
    if (mObservations.empty())
        return;

    Point3f pose;
    {
        locker lock1(mMutexPos);
        pose = mPos;  // 3d点在世界坐标系中的位置
    }

    vector<Mat> vDes;
    vector<PtrKeyFrame> vKFs;
    vDes.reserve(mObservations.size());
    vKFs.reserve(mObservations.size());

    Point3f normal(0.f, 0.f, 0.f);
    int n = 0;
    for (auto i = mObservations.begin(), iend = mObservations.end(); i != iend; ++i) {
        PtrKeyFrame pKF = i->first;
        if (!pKF)
            continue;
        if (!pKF->isNull()) {
            vKFs.push_back(pKF);
            vDes.push_back(pKF->mDescriptors.row(i->second));

            Point3f Owi = pKF->getCameraCenter();
            Point3f normali = pose - Owi;
            normal += normali / cv::norm(normali);  // 所有KF对该点的观测方向归一化为单位向量再求和
            n++;
        }
    }

    if (vDes.empty()) {
        fprintf(stderr, "[MapPoint] Set this MP#%ld to null because no desciptors in updateMainKFandDescriptor()\n", mId);
        setNull();
        return;
    }

    // Compute distances between them
    const size_t N = vDes.size();
    float Distances[N][N];
    for (size_t i = 0; i != N; ++i) {
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
    for (size_t i = 0; i != N; ++i) {
        vector<int> vDists(Distances[i], Distances[i] + N);
        std::sort(vDists.begin(), vDists.end());
        int median = vDists[0.5 * (N - 1)];
        if (median < bestMedian) {
            bestMedian = median;
            bestIdx = i;
        }
    }

    mNormalVector = normal / n;  // 获得平均的观测方向
    mMainDescriptor = vDes[bestIdx].clone();

    if (mMainKF != nullptr && mMainKF->mIdKF == vKFs[bestIdx]->mIdKF)
        return;

    mMainKF = vKFs[bestIdx];  //! mMainKF 只有在这里才更新

    size_t idx = mObservations[mMainKF];
    mMainOctave = mMainKF->mvKeyPoints[idx].octave;
//    mLevelScaleFactor = mMainKF->mvScaleFactors[mMainOctave];
    float dist = cv::norm(mMainKF->mvViewMPs[idx]);
    int nlevels = mMainKF->mnScaleLevels;

    mMinDist = dist / mLevelScaleFactor;
    mMaxDist = dist * mLevelScaleFactor;
//    mMinDist = mMaxDist / mMainKF->mvScaleFactors[nlevels - 1];
}

/**
 * @brief 视差不好的MP增加新的观测后会更新视差
 * 一旦MP视差合格后就不会再更新, 如果连续6帧KF后视差仍不好, 会抛弃该MP
 * @param pKF   能观测到此MP的KF指针
 * TODO 待更新为深度滤波器
 */
void MapPoint::updateParallax(const PtrKeyFrame& pKF)
{
    if (mbGoodParallax || mObservations.size() <= 2)
        return;

    // Get the oldest KF in the last 6 KFs. 由从此KF往前最多10帧KF和此KF做三角化更新MP坐标
    PtrKeyFrame pKF0 = mObservations.rbegin()->first;
    for (auto it = ++mObservations.rbegin(), iend = mObservations.rend(); it != iend; ++it) {
        PtrKeyFrame pKFj = it->first;
        if (!pKFj)
            continue;
        if (pKF->mIdKF - pKFj->mIdKF > 10)   // 6
            break;
        if (updateParallaxCheck(pKFj, pKF))
            break;
    }

    // If still no good parallax after more than 10 KFs, abandon this MP
    if (!mbGoodParallax && mObservations.size() > 10)
        setNull();
}

//! Update measurements in KFs. 更新约束和信息矩阵
bool MapPoint::updateParallaxCheck(const PtrKeyFrame &pKF1, const PtrKeyFrame &pKF2)
{
    if (pKF1->mIdKF == pKF2->mIdKF)
        return false;

    // Do triangulation
    Mat Tc1w = pKF1->getPose();
    Mat Tc2w = pKF2->getPose();
    Mat Proj1 = Config::Kcam * Tc1w.rowRange(0, 3);
    Mat Proj2 = Config::Kcam * Tc2w.rowRange(0, 3);
    Point2f pt1 = pKF1->mvKeyPoints[mObservations[pKF1]].pt;
    Point2f pt2 = pKF2->mvKeyPoints[mObservations[pKF2]].pt;
    Point3f posW = cvu::triangulate(pt1, pt2, Proj1, Proj2);

    Point3f Pc1 = cvu::se3map(Tc1w, posW);
    Point3f Pc2 = cvu::se3map(Tc2w, posW);
    if (!Config::acceptDepth(Pc1.z) || !Config::acceptDepth(Pc2.z))
        return false;

    Point3f Ocam1 = Point3f(cvu::inv(Tc1w).rowRange(0, 3).col(3));
    Point3f Ocam2 = Point3f(cvu::inv(Tc2w).rowRange(0, 3).col(3));
    if (!cvu::checkParallax(Ocam1, Ocam2, posW, 2))
        return false;

    // 视差良好, 则更新此点的三维坐标
    {
        locker lock(mMutexPos);
        mPos = posW;
        mbGoodParallax = true;
        fprintf(stderr, "[MapPoint] MP#%ld 的视差被更新为good.\n", mId);
    }

    // Update measurements in KFs. 更新约束和信息矩阵
    Eigen::Matrix3d xyzinfo1, xyzinfo2;
    Track::calcSE3toXYZInfo(Pc1, Tc1w, Tc2w, xyzinfo1, xyzinfo2);
    pKF1->setViewMP(Pc1, mObservations[pKF1], xyzinfo1);
    pKF2->setViewMP(Pc2, mObservations[pKF2], xyzinfo2);

    Mat Rc1w = Tc1w.rowRange(0, 3).colRange(0, 3);
    Mat xyzinfoW = Rc1w.t() * toCvMat(xyzinfo1) * Rc1w;

    // Update measurements in other KFs
    for (auto it = mObservations.begin(), iend = mObservations.end(); it != iend; ++it) {
        PtrKeyFrame pKFi = it->first;
        if (!pKFi)
            continue;
        if (pKFi->mIdKF == pKF2->mIdKF || pKFi->mIdKF == pKF1->mIdKF)
            continue;
        Mat Tciw = pKFi->getPose();
        Point3f Pci = cvu::se3map(Tciw, posW);
        Mat Rciw = Tciw.rowRange(0, 3).colRange(0, 3);
        Mat xyzinfoi = Rciw * xyzinfoW * Rciw.t();
        pKFi->setViewMP(Pci, mObservations[pKFi], toMatrix3d(xyzinfoi));
    }
    return true;
}

//! MP在优化更新坐标后要更新其他KF对其的观测值. 在Map里调用
void MapPoint::updateMeasureInKFs()
{
    locker lock(mMutexObs);
    for (auto it = mObservations.begin(), iend = mObservations.end(); it != iend; ++it) {
        PtrKeyFrame pKF = it->first;
        if (!pKF)
            continue;
        if (pKF->isNull())
            continue;

        Mat Tcw = pKF->getPose();
        Point3f posKF = cvu::se3map(Tcw, mPos);
        pKF->mvViewMPs[it->second] = posKF;
    }
}


// This MP would be replaced and abandoned later
void MapPoint::mergedInto(const shared_ptr<MapPoint>& pMP)
{
    locker lock(mMutexObs);
    for (auto it = mObservations.begin(), iend = mObservations.end(); it != iend; ++it) {
        PtrKeyFrame pKF = it->first;
        if (!pKF)
            continue;
        if (pKF->isNull())
            continue;
        size_t idx = it->second;
        pKF->setObservation(pMP, idx);
        pMP->addObservation(pKF, idx);
        mObservations.erase(pKF);
    }
    //! NOTE 不能直接setNull(), 此函数会删除mObservations里KF的观测, 观测已经更新, 不可删除
    //! 故要在上面的循环体中先删除mObservations的元素
    fprintf(stderr, "[MapPoint] A MP#%ld is merged by #%ld, now it's obervations count = %ld\n",
           mId, pMP->mId, mObservations.size());
    setNull();
}

int MapPoint::getOctave(const PtrKeyFrame& pKF)
{
    locker lock(mMutexObs);
    if (mObservations.count(pKF))
        return pKF->mvKeyPoints[mObservations[pKF]].octave;
    else
        return -1;
}

int MapPoint::getIndexInKF(const PtrKeyFrame& pKF)
{
    locker lock(mMutexObs);
    if (mObservations.count(pKF))
        return mObservations[pKF];
    else
        return -1;
}


}  // namespace se2lam
