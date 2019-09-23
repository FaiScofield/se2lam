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

unsigned long MapPoint::mNextId = 1;

MapPoint::MapPoint()
    : mMainKF(static_cast<PtrKeyFrame>(nullptr)), mMainOctave(0),
      mLevelScaleFactor(Config::ScaleFactor), mId(0), mNormalVector(Point3f(0, 0, 0)),
      mbNull(false), mbGoodParallax(false)
{
    mObservations.clear();

    mMinDist = 0.f;
    mMaxDist = 0.f;
}

//! 构造后请立即为其添加观测
MapPoint::MapPoint(Point3f pos, bool goodPrl)
    : mMainKF(static_cast<PtrKeyFrame>(nullptr)), mMainOctave(0),
      mLevelScaleFactor(Config::ScaleFactor), mPos(pos), mNormalVector(Point3f(0, 0, 0)),
      mbNull(false), mbGoodParallax(goodPrl)
{
    mId = mNextId++;
    mObservations.clear();

    mMinDist = 0.f;
    mMaxDist = 0.f;
}

MapPoint::~MapPoint()
{}

bool MapPoint::isNull()
{
    return mbNull;
}

bool MapPoint::isGoodPrl()
{
    return mbGoodParallax;
}

void MapPoint::setNull(const shared_ptr<MapPoint>& pThis)
{
    locker lock(mMutexObs);
    mbNull = true;
    mbGoodParallax = false;
    for (auto it = mObservations.begin(), iend = mObservations.end(); it != iend; it++) {
        PtrKeyFrame pKF = it->first;
        if (pKF->hasObservation(pThis))
            pKF->eraseObservation(pThis);
    }
    mObservations.clear();
    mMainDescriptor.release();
    mMainKF = nullptr;
}

// Abandon a MP as an outlier, only for internal use
void MapPoint::setNull()
{
    locker lock(mMutexObs);
    mbNull = true;
    mbGoodParallax = false;
    for (auto it = mObservations.begin(), iend = mObservations.end(); it != iend; it++) {
        PtrKeyFrame pKF = it->first;
        if (pKF->hasObservation(it->second))
            pKF->eraseObservation(it->second);
    }
    mObservations.clear();
    mMainDescriptor.release();
    mMainKF = nullptr;
}


Point3f MapPoint::getPos()
{
    locker lock(mMutexPos);
    return mPos;
}

/**
 * @brief 取消对KF的关联,在删减冗余KF时会用到
 * @param pKF   取消关联的KF对象
 */
void MapPoint::eraseObservation(const PtrKeyFrame& pKF)
{
    locker lock(mMutexObs);

    if (mbNull)
        return;

    mObservations.erase(pKF);

    if (mObservations.size() == 0) {
        setNull();
    } else {
        updateMainKFandDescriptor();
    }
}

void MapPoint::eraseObservations(const std::vector<std::pair<PtrKeyFrame, size_t> > &obsCandidates)
{
    locker lock(mMutexObs);

    if (mbNull)
        return;

    //! map.erase()足够安全不需要检查元素是否在字典里
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

    //! map.insert()会保证元素唯一性. operator[]返回引用,则会更新
//    mObservations.insert(make_pair(pKF, idx));
    mObservations[pKF] = idx;

    updateMainKFandDescriptor();

    updateParallax(pKF);

    if (mbNull) {
        mbNull = false;
    }
}

/**
 * @brief 多次观测一次性添加, 并更新相关变量, 防止多次计算
 * @param obsCandidates 待添加的多个观测
 */
void MapPoint::addObservations(const vector<pair<PtrKeyFrame, size_t>>& obsCandidates)
{
    locker lock(mMutexObs);

    for (auto& oc : obsCandidates) {
        mObservations.insert(oc);
        updateParallax(oc.first);
    }

    updateMainKFandDescriptor();

    if (mbNull) {
        mbNull = false;
    }
}

/**
 * @brief 视差不好的MP增加新的观测后会更新视差
 * 一旦MP视差合格后就不会再更新, 如果连续6帧KF后视差仍不好, 会抛弃该MP
 * @param pKF   能观测到此MP的KF指针
 */
void MapPoint::updateParallax(const PtrKeyFrame& pKF)
{
    if (mbGoodParallax || mObservations.size() <= 2)
        return;

    // Get the oldest KF in the last 6 KFs. 由从此KF往前的第6帧KF和此KF做三角化更新MP坐标
    PtrKeyFrame pKF0 = mObservations.begin()->first;
    for (auto it = mObservations.begin(), iend = mObservations.end(); it != iend; it++) {
        PtrKeyFrame pKF_ = it->first;
        if (!pKF_)
            continue;
        if (pKF->mIdKF - pKF_->mIdKF > 6)
            continue;
//        if (pKF_->mIdKF < pKF_->mIdKF) {  //! NOTE 这里写错了!
        if (pKF_->mIdKF < pKF0->mIdKF) {  //! @Vance: 20190918改
            pKF0 = pKF_;
        }
    }

    // Do triangulation
    Mat Tcw0 = pKF0->getPose();
    Mat Tcw = pKF->getPose();
    Mat P0 = Config::Kcam * Tcw0.rowRange(0, 3);
    Mat P1 = Config::Kcam * Tcw.rowRange(0, 3);
    Point2f pt0 = pKF0->mvKeyPoints[mObservations[pKF0]].pt;
    Point2f pt1 = pKF->mvKeyPoints[mObservations[pKF]].pt;
    Point3f posW = cvu::triangulate(pt0, pt1, P0, P1);

    Point3f pos0 = cvu::se3map(Tcw0, posW);
    Point3f pos1 = cvu::se3map(Tcw, posW);
    if (Config::acceptDepth(pos0.z) && Config::acceptDepth(pos1.z)) {
        Point3f Ocam0 = Point3f(cvu::inv(Tcw0).rowRange(0, 3).col(3));
        Point3f Ocam1 = Point3f(cvu::inv(Tcw).rowRange(0, 3).col(3));
        //! 视差良好, 则更新此点的三维坐标
        if (cvu::checkParallax(Ocam0, Ocam1, posW, 2)) {
            {
                locker lock(mMutexPos);
                mPos = posW;
                mbGoodParallax = true;
            }

            // Update measurements in KFs. 更新约束和信息矩阵
            Eigen::Matrix3d xyzinfo0, xyzinfo1;
            Track::calcSE3toXYZInfo(pos0, Tcw0, Tcw, xyzinfo0, xyzinfo1);
            pKF0->setViewMP(pos0, mObservations[pKF0], xyzinfo0);
            pKF->setViewMP(pos1, mObservations[pKF], xyzinfo1);

            Mat Rcw0 = Tcw0.rowRange(0, 3).colRange(0, 3);
            Mat xyzinfoW = Rcw0.t() * toCvMat(xyzinfo0) * Rcw0;

            // Update measurements in other KFs
            for (auto it = mObservations.begin(), iend = mObservations.end(); it != iend; it++) {
                PtrKeyFrame pKF_k = it->first;
                if (!pKF_k)
                    continue;
                if (pKF_k->mIdKF == pKF->mIdKF || pKF_k->mIdKF == pKF0->mIdKF) {
                    continue;
                }
                Mat Tcw_k = pKF_k->getPose();
                Point3f posk = cvu::se3map(Tcw_k, posW);
                Mat Rcwk = Tcw_k.rowRange(0, 3).colRange(0, 3);
                Mat xyzinfok = Rcwk * xyzinfoW * Rcwk.t();
                pKF_k->setViewMP(posk, mObservations[pKF_k], toMatrix3d(xyzinfok));
            }
        }
    }

    // If no good parallax after more than 6 KFs, abandon this MP
    if (pKF->mIdKF - pKF0->mIdKF >= 6 && !mbGoodParallax) {
        setNull();
    }
}

float MapPoint::getInvLevelSigma2(const PtrKeyFrame& pKF)
{
    int index = mObservations[pKF];
    return pKF->mvInvLevelSigma2[index];
}

int MapPoint::getOctave(const PtrKeyFrame pKF)
{
    int index = mObservations[pKF];
    return pKF->mvKeyPoints[index].octave;
}

void MapPoint::setPos(const Point3f& pt3f)
{
    locker lock(mMutexPos);
    mPos = pt3f;
}

/**
 * @brief 判断KF观测是否合理, LocalMap里关联MP时调用
 * @param posKF  能观测到此MP的KF
 * @param kp     KF对应的特征点
 * @return
 */
bool MapPoint::acceptNewObserve(Point3f posKF, const KeyPoint kp)
{
    float dist = cv::norm(posKF);
    float cosAngle = cv::norm(posKF.dot(mNormalVector)) / (dist * cv::norm(mNormalVector));
    bool c1 = std::abs(mMainKF->mvKeyPoints[mObservations[mMainKF]].octave - kp.octave) <= 2;
    bool c2 = cosAngle >= 0.866f;  // no larger than 30 degrees
    bool c3 = dist >= mMinDist && dist <= mMaxDist;
    return c1 && c2 && c3;
}

std::set<PtrKeyFrame> MapPoint::getObservations()
{
    locker lock(mMutexObs);
    std::set<PtrKeyFrame> pKFs;
    for (auto i = mObservations.begin(), iend = mObservations.end(); i != iend; i++) {
        PtrKeyFrame pKF = i->first;
        if (!pKF)
            continue;
        if (pKF->isNull())
            continue;
        pKFs.insert(i->first);
    }
    return pKFs;
}

Point2f MapPoint::getMainMeasure()
{
    size_t idx = mObservations[mMainKF];
    return mMainKF->mvKeyPoints[idx].pt;
}

/**
 * @brief 更新MP的mainKF,描述子以及平均观测方向mNormalVector
 * 在addObservation()和eraseObservation()后需要调用!
 */
void MapPoint::updateMainKFandDescriptor()
{
    Point3f pose;
    {
        locker lock1(mMutexPos);

        if (mbNull || mObservations.empty())
            return;
        pose = mPos;  // 3d点在世界坐标系中的位置
    }

    vector<Mat> vDes;
    vector<PtrKeyFrame> vKFs;
    vDes.reserve(mObservations.size());
    vKFs.reserve(mObservations.size());

    Point3f normal(0.f, 0.f, 0.f);
    int n = 0;
    for (auto i = mObservations.begin(), iend = mObservations.end(); i != iend; i++) {
        PtrKeyFrame pKF = i->first;
        if (!pKF)
            continue;
        if (!pKF->isNull()) {
            vKFs.push_back(pKF);
            vDes.push_back(pKF->mDescriptors.row(i->second));

            Point3f Owi = pKF->getCameraCenter();
            Point3f normali = pose - Owi;
            normal += normali / cv::norm(normali);  // 对所有关键帧对该点的观测方向归一化为单位向量进行求和
            n++;
        }
    }


    if (vDes.empty())
        return;

    // Compute distances between them
    const size_t N = vDes.size();
    float Distances[N][N];
    for (size_t i = 0; i < N; i++) {
        Distances[i][i] = 0;
        for (size_t j = i + 1; j < N; j++) {
            int distij = ORBmatcher::DescriptorDistance(vDes[i], vDes[j]);
            Distances[i][j] = distij;
            Distances[j][i] = distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    int bestMedian = INT_MAX;
    size_t bestIdx = 0;
    for (size_t i = 0; i < N; i++) {
        vector<int> vDists(Distances[i], Distances[i] + N);
        std::sort(vDists.begin(), vDists.end());
        int median = vDists[0.5 * (N - 1)];
        if (median < bestMedian) {
            bestMedian = median;
            bestIdx = i;
        }
    }

    {
        locker lock2(mMutexPos);
        mNormalVector = normal / n;               // 获得平均的观测方向
    }
    mMainKF = vKFs[bestIdx];
    mMainDescriptor = vDes[bestIdx].clone();

    size_t idx = mObservations[mMainKF];
    mMainOctave = mMainKF->mvKeyPoints[idx].octave;
    mLevelScaleFactor = mMainKF->mvScaleFactors[mMainOctave];
    float dist = cv::norm(mMainKF->mViewMPs[idx]);
    int nlevels = mMainKF->mnScaleLevels;

    mMaxDist = dist * mLevelScaleFactor;
    mMinDist = mMaxDist / mMainKF->mvScaleFactors[nlevels - 1];
}

/**
 * @brief MP在优化更新坐标后要更新其他KF对其的观测值. 在Map里调用
 */
void MapPoint::updateMeasureInKFs()
{
    locker lock(mMutexObs);
    for (auto it = mObservations.begin(), iend = mObservations.end(); it != iend; it++) {
        PtrKeyFrame pKF = it->first;
        if (!pKF)
            continue;
        if (pKF->isNull())
            continue;

        Mat Tcw = pKF->getPose();
        Point3f posKF = cvu::se3map(Tcw, mPos);
        pKF->mViewMPs[it->second] = posKF;
    }
}

int MapPoint::countObservation()
{
    locker lock(mMutexObs);
    return mObservations.size();
}


// This MP would be replaced and abandoned later
void MapPoint::mergedInto(const shared_ptr<MapPoint>& pMP)
{
    locker lock(mMutexObs);
    for (auto it = mObservations.begin(), iend = mObservations.end(); it != iend; it++) {
        PtrKeyFrame pKF = it->first;
        if (!pKF)
            continue;
        if (pKF->isNull())
            continue;
        size_t idx = it->second;
        pKF->setObservation(pMP, idx);
        pMP->addObservation(pKF, idx);
    }
}

size_t MapPoint::getFtrIdx(const PtrKeyFrame& pKF)
{
    locker lock(mMutexObs);
    return mObservations[pKF];
}

void MapPoint::setGoodPrl(bool value)
{
    mbGoodParallax = value;
}


bool MapPoint::hasObservation(const PtrKeyFrame& pKF)
{
    locker lock(mMutexObs);
    return (mObservations.find(pKF) != mObservations.end());
}

Point3f MapPoint::getNormalVector()
{
    locker lock(mMutexPos);
    return mNormalVector;
}

}  // namespace se2lam
