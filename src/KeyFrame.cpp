/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include "KeyFrame.h"
#include "Frame.h"
#include "MapPoint.h"
#include "cvutil.h"

namespace se2lam
{
using namespace cv;
using namespace std;

typedef unique_lock<mutex> locker;

unsigned long KeyFrame::mNextIdKF = 1;    //! F,KF和MP的编号都是从1开始

KeyFrame::KeyFrame() : mIdKF(0), mbBowVecExist(false), mbNull(false)
{
    PtrKeyFrame pKF = static_cast<PtrKeyFrame>(nullptr);
    mOdoMeasureFrom = make_pair(pKF, SE3Constraint());
    mOdoMeasureTo = make_pair(pKF, SE3Constraint());

    preOdomFromSelf = make_pair(pKF, PreSE2());
    preOdomToSelf = make_pair(pKF, PreSE2());

    // Scale Levels Info
    mnScaleLevels = Config::MaxLevel;
    mfScaleFactor = Config::ScaleFactor;

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
}


KeyFrame::KeyFrame(const Frame &frame) : Frame(frame), mbBowVecExist(false), mbNull(false)
{
    size_t n = frame.N;
    mViewMPs = vector<Point3f>(n, Point3f(-1.f, -1.f, -1.f));
    mViewMPsInfo = vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>(
        n, Eigen::Matrix3d::Identity() * -1);

    mIdKF = mNextIdKF++;

    PtrKeyFrame pKF = static_cast<PtrKeyFrame>(nullptr);
    mOdoMeasureFrom = make_pair(pKF, SE3Constraint());
    mOdoMeasureTo = make_pair(pKF, SE3Constraint());

    preOdomFromSelf = make_pair(pKF, PreSE2());
    preOdomToSelf = make_pair(pKF, PreSE2());
}

KeyFrame::~KeyFrame()
{
    fprintf(stderr, "[KeyFrame] A KF#%ld(#%ld) is decontructed!\n", mIdKF, id);
}

// Please handle odometry based constraints after calling this function
//! FIXME KF在setNull()后仍然有6-8个引用计数!!!无法析构!
void KeyFrame::setNull(const shared_ptr<KeyFrame> &pThis)
{
    locker lckPose(mMutexPose);
    locker lckObs(mMutexObs);
    locker lckCov(mMutexCovis);

    if (mIdKF == 1)
        return;

    mImage.release();
    mDescriptors.release();
    mvKeyPoints.clear();

    // Handle Feature based constraints
    fprintf(stderr, "[KeyFrame] KF#%ld before Handling constraints, Count pointer = %ld\n",
           mIdKF, pThis.use_count());
    for (auto it = mFtrMeasureFrom.begin(), iend = mFtrMeasureFrom.end(); it != iend; ++it) {
        it->first->mFtrMeasureTo.erase(pThis);
    }
    for (auto it = mFtrMeasureTo.begin(), iend = mFtrMeasureTo.end(); it != iend; ++it) {
        it->first->mFtrMeasureFrom.erase(pThis);
    }
    fprintf(stderr, "[KeyFrame] KF#%ld after Handling constraints, Count pointer = %ld\n",
           mIdKF, pThis.use_count());
    mFtrMeasureFrom.clear();
    mFtrMeasureTo.clear();

    // Handle observations in MapPoints, 取消MP对此KF的关联
    fprintf(stderr, "[KeyFrame] KF#%ld before Handling MP observations(%ld), Count pointer = %ld\n",
           mIdKF, mObservations.size(), pThis.use_count());
    for (auto it = mObservations.begin(), iend = mObservations.end(); it != iend; ++it) {
        PtrMapPoint pMP = it->first;
        pMP->eraseObservation(pThis);
    }
    fprintf(stderr, "[KeyFrame] KF#%ld after Handling MP observations, Count pointer = %ld\n",
           mIdKF, pThis.use_count());

    // Handle Covisibility, 取消其他KF对此KF的共视关系
    fprintf(stderr, "[KeyFrame] KF#%ld before Handling Covisibility(%ld), Count pointer = %ld\n",
           mIdKF, mCovisibleKFs.size(), pThis.use_count());
    for (auto it = mCovisibleKFs.begin(), iend = mCovisibleKFs.end(); it != iend; ++it) {
        (*it)->eraseCovisibleKF(pThis);
    }
    fprintf(stderr, "[KeyFrame] KF#%ld after Handling Covisibility, Count pointer = %ld\n",
           mIdKF, pThis.use_count());

    mObservations.clear();
    mDualObservations.clear();
    mCovisibleKFs.clear();
    mViewMPs.clear();
    mViewMPsInfo.clear();

    fprintf(stderr, "[KeyFrame] A KF#%ld is set to null, Count pointer = %ld\n",
           mIdKF, pThis.use_count());

    mbNull = true;
    mpORBExtractor = nullptr;
}

size_t KeyFrame::countObservation()
{
    locker lock(mMutexObs);
    return mObservations.size();
}

void KeyFrame::setViewMP(Point3f pt3f, int idx, Eigen::Matrix3d info)
{
    locker lock(mMutexObs);
    mViewMPs[idx] = pt3f;
    mViewMPsInfo[idx] = info;
}

void KeyFrame::eraseCovisibleKF(const shared_ptr<KeyFrame> pKF)
{
    locker lock(mMutexCovis);
    mCovisibleKFs.erase(pKF);
}

void KeyFrame::addCovisibleKF(const shared_ptr<KeyFrame> pKF)
{
    locker lock(mMutexCovis);
    mCovisibleKFs.insert(pKF);
}

set<PtrKeyFrame> KeyFrame::getAllCovisibleKFs()
{
    locker lock(mMutexCovis);
    return mCovisibleKFs;
}

/**
 * @brief KeyFrame::getAllObsMPs 获取KF关联的MP
 * 在LocalMap调用Map::updateLocalGraph()时 checkParallax = false
 * @param checkParallax 是否得到具有良好视差的MP, 默认开起
 * @return
 */
set<PtrMapPoint> KeyFrame::getAllObsMPs(bool checkParallax)
{
    locker lock(mMutexObs);
    set<PtrMapPoint> spMP;
    for (auto i = mObservations.begin(), iend = mObservations.end(); i != iend; ++i) {
        PtrMapPoint pMP = i->first;
        if (!pMP)
            continue;
        if (pMP->isNull())
            continue;
        if (checkParallax && !pMP->isGoodPrl())
            continue;
        spMP.insert(pMP);
    }
    return spMP;
}

bool KeyFrame::isNull()
{
    return mbNull;
}

bool KeyFrame::hasObservation(const PtrMapPoint &pMP)
{
    locker lock(mMutexObs);
    map<PtrMapPoint, size_t>::iterator it = mObservations.find(pMP);

    return (it != mObservations.end());
}

bool KeyFrame::hasObservation(size_t idx)
{
    locker lock(mMutexObs);
    auto it = mDualObservations.find(idx);

    return (it != mDualObservations.end());
}

//Mat KeyFrame::getPose()
//{
//    locker lock(mMutexPose);
//    return Tcw.clone();
//}

//void KeyFrame::setPose(const Mat &_Tcw)
//{
//    locker lock(mMutexPose);
//    _Tcw.copyTo(Tcw);
//    Twb.fromCvSE3(cvu::inv(Tcw) * Config::Tcb);
//}

//void KeyFrame::setPose(const Se2 &_Twb)
//{
//    locker lock(mMutexPose);
//    Twb = _Twb;
//    Tcw = Config::Tcb * Twb.inv().toCvSE3();
//}

void KeyFrame::addObservation(PtrMapPoint pMP, size_t idx)
{
    locker lock(mMutexObs);
    if (!pMP)
        return;
    if (pMP->isNull())
        return;
    mObservations[pMP] = idx;
    mDualObservations[idx] = pMP;
}

map<PtrMapPoint, size_t> KeyFrame::getObservations()
{
    locker lock(mMutexObs);
    return mObservations;
}

void KeyFrame::eraseObservation(const PtrMapPoint pMP)
{
    locker lock(mMutexObs);
    size_t idx = mObservations[pMP];
    mObservations.erase(pMP);
    mDualObservations.erase(idx);
}

void KeyFrame::eraseObservation(size_t idx)
{
    locker lock(mMutexObs);
    mObservations.erase(mDualObservations[idx]);
    mDualObservations.erase(idx);
}

void KeyFrame::addFtrMeasureFrom(shared_ptr<KeyFrame> pKF, const Mat &_mea, const Mat &_info)
{
    mFtrMeasureFrom.insert(make_pair(pKF, SE3Constraint(_mea, _info)));
}

void KeyFrame::eraseFtrMeasureFrom(shared_ptr<KeyFrame> pKF)
{
    mFtrMeasureFrom.erase(pKF);
}

void KeyFrame::addFtrMeasureTo(shared_ptr<KeyFrame> pKF, const Mat &_mea, const Mat &_info)
{
    mFtrMeasureTo.insert(make_pair(pKF, SE3Constraint(_mea, _info)));
}

void KeyFrame::eraseFtrMeasureTo(shared_ptr<KeyFrame> pKF)
{
    mFtrMeasureTo.erase(pKF);
}

void KeyFrame::setOdoMeasureFrom(shared_ptr<KeyFrame> pKF, const Mat &_mea, const Mat &_info)
{
    mOdoMeasureFrom = make_pair(pKF, SE3Constraint(_mea, _info));
}

void KeyFrame::setOdoMeasureTo(shared_ptr<KeyFrame> pKF, const Mat &_mea, const Mat &_info)
{
    mOdoMeasureTo = make_pair(pKF, SE3Constraint(_mea, _info));
}

void KeyFrame::ComputeBoW(ORBVocabulary *_pVoc)
{
    if (mBowVec.empty() || mFeatVec.empty()) {
        vector<cv::Mat> vCurrentDesc = toDescriptorVector(mDescriptors);
        // Feature vector associate features with nodes in the 4th level (from leaves up)
        // We assume the vocabulary tree has 6 levels, change the 4 otherwise
        _pVoc->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
    }
    mbBowVecExist = true;
}

DBoW2::FeatureVector KeyFrame::GetFeatureVector()
{
//    boost::mutex::scoped_lock lock(mMutexFeatures);
    return mFeatVec;
}

DBoW2::BowVector KeyFrame::GetBowVector()
{
//    boost::mutex::scoped_lock lock(mMutexFeatures);
    return mBowVec;
}

//! 找到特征点对应的MPs，关键函数，后续应该有一个观测更新的函数需要被调用
vector<PtrMapPoint> KeyFrame::GetMapPointMatches()
{
//    size_t N = mvKeyPoints.size();
    vector<PtrMapPoint> ret(N, nullptr);

    std::map<size_t, PtrMapPoint>::iterator iter;
    for (size_t i = 0; i != N; ++i) {
        iter = mDualObservations.find(i);
        if (iter != mDualObservations.end())
            ret[i] = iter->second;
    }

    return ret;
}

/**
 * @brief KeyFrame::setObservation 设置MP的观测，在MP被merge的时候调用
 * @param pMP   观测上要设置的MP
 * @param idx   特征点的编号
 */
void KeyFrame::setObservation(const PtrMapPoint &pMP, size_t idx)
{
    locker lock(mMutexObs);

    // 保证之前此索引位置有观测，然后替换掉
    if (mDualObservations.find(idx) == mDualObservations.end())
        return;

    mObservations.erase(mDualObservations[idx]);
    mObservations[pMP] = idx;
    mDualObservations[idx] = pMP;
}

PtrMapPoint KeyFrame::getObservation(size_t idx)
{
    locker lock(mMutexObs);

    if (!mDualObservations[idx]) {
        fprintf(stderr, "[KeyFrame] This is a NULL index in observation!\n");
    }
    return mDualObservations[idx];
}

int KeyFrame::getFeatureIndex(const PtrMapPoint &pMP)
{
    locker lock(mMutexObs);
    if (mObservations.count(pMP))
        return mObservations[pMP];
    else
        return -1;
}

}  // namespace se2lam
