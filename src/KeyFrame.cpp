/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include "KeyFrame.h"
#include "Frame.h"
#include "MapPoint.h"
#include "Map.h"
#include "cvutil.h"

namespace se2lam
{

using namespace cv;
using namespace std;
typedef unique_lock<mutex> locker;


//bool KFIdLessThan::operator()(const std::shared_ptr<KeyFrame>& lhs, const std::shared_ptr<KeyFrame>& rhs) const
//{
//    return lhs->mIdKF < rhs->mIdKF;
//}


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
    mvViewMPs = vector<Point3f>(n, Point3f(-1.f, -1.f, -1.f));
    mvViewMPsInfo = vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>(
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
    fprintf(stderr, "[KeyFrame] KF#%ld(#%ld) 已被析构!\n", mIdKF, id);
}

// Please handle odometry based constraints after calling this function
//! 加入成员变量map的指针后通过调用map删除其相关容器中的指针, 现已可以正常析构. 20191015
void KeyFrame::setNull(const shared_ptr<KeyFrame> &pThis)
{
    locker lckPose(mMutexPose);
    locker lckObs(mMutexObs);
    locker lckCov(mMutexCovis);

    if (mIdKF == 1)
        return;

    mbNull = true;
    mpORBExtractor = nullptr;

    mvKeyPoints.clear();
    mDescriptors.release();
    if (bNeedVisulization)
        mImage.release();

    // Handle Feature based constraints
    size_t n1 = mFtrMeasureFrom.size(), n2 = mFtrMeasureTo.size();
    if (n1 || n2)
        fprintf(stderr, "[KeyFrame] KF#%ld 取消特征约束之前引用计数 = %ld\n", mIdKF, pThis.use_count());
    if (n1 > 0)
        for (auto it = mFtrMeasureFrom.begin(), iend = mFtrMeasureFrom.end(); it != iend; ++it)
            it->first->mFtrMeasureTo.erase(pThis);
    if (n2 > 0)
        for (auto it = mFtrMeasureTo.begin(), iend = mFtrMeasureTo.end(); it != iend; ++it)
            it->first->mFtrMeasureFrom.erase(pThis);
    if (n1 || n2)
        fprintf(stderr, "[KeyFrame] KF#%ld 取消特征约束之后引用计数 = %ld\n", mIdKF, pThis.use_count());
    mFtrMeasureFrom.clear();
    mFtrMeasureTo.clear();

    // Handle observations in MapPoints, 取消MP对此KF的关联
    fprintf(stderr, "[KeyFrame] KF#%ld 取消MP观测前(%ld)引用计数 = %ld\n",
           mIdKF, mObservations.size(), pThis.use_count());
    for (auto it = mObservations.begin(), iend = mObservations.end(); it != iend; ++it) {
        PtrMapPoint pMP = it->first;
        pMP->eraseObservation(pThis);
    }
    fprintf(stderr, "[KeyFrame] KF#%ld 取消MP观测后引用计数 = %ld\n", mIdKF, pThis.use_count());

    // Handle Covisibility, 取消其他KF对此KF的共视关系
    fprintf(stderr, "[KeyFrame] KF#%ld 取消共视关系前(%ld)引用计数 = %ld\n",
           mIdKF, mspCovisibleKFs.size(), pThis.use_count());
    for (auto it = mspCovisibleKFs.begin(), iend = mspCovisibleKFs.end(); it != iend; ++it) {
        (*it)->eraseCovisibleKF(pThis);
    }
    fprintf(stderr, "[KeyFrame] KF#%ld 取消共视关系后引用计数 = %ld\n", mIdKF, pThis.use_count());

    mObservations.clear();
    mDualObservations.clear();
    mspCovisibleKFs.clear();
    mvViewMPs.clear();
    mvViewMPsInfo.clear();

    mpMap->eraseKF(pThis);

    fprintf(stderr, "[KeyFrame] KF#%ld 被Map设置为null, 引用计数 = %ld\n",
           mIdKF, pThis.use_count());
}

size_t KeyFrame::countObservation()
{
    locker lock(mMutexObs);
    return mObservations.size();
}

void KeyFrame::setViewMP(Point3f pt3f, size_t idx, Eigen::Matrix3d info)
{
    locker lock(mMutexObs);
    mvViewMPs[idx] = pt3f;
    mvViewMPsInfo[idx] = info;
}

Point3f KeyFrame::getViewMPPoseInCamareFrame(size_t idx)
{
    locker lock(mMutexObs);

    Point3f ret(-1.f, -1.f, -1.f);

    auto it = mDualObservations.find(idx);
    if (it != mDualObservations.end()) {
        Point3f Pw = mDualObservations[idx]->getPos();
        ret = cvu::se3map(Tcw, Pw);
    }
    return ret;
}

void KeyFrame::eraseCovisibleKF(const shared_ptr<KeyFrame> pKF)
{
    locker lock(mMutexCovis);
    mspCovisibleKFs.erase(pKF);
}

void KeyFrame::addCovisibleKF(const shared_ptr<KeyFrame> pKF)
{
    locker lock(mMutexCovis);
    mspCovisibleKFs.insert(pKF);
}

set<PtrKeyFrame> KeyFrame::getAllCovisibleKFs()
{
    locker lock(mMutexCovis);
    return mspCovisibleKFs;
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
    auto it = mObservations.find(pMP);

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
    mObservations.insert(make_pair(pMP, idx));
    mDualObservations.insert(make_pair(idx, pMP));
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

vector<PtrMapPoint> KeyFrame::GetMapPointMatches()
{
    vector<PtrMapPoint> ret(N, nullptr);
    for (auto iter = mDualObservations.begin(), iend = mDualObservations.end(); iter != iend; ++iter)
        ret[iter->first] = iter->second;

    return ret;
}

/**
 * @brief 设置MP的观测，在MP被merge的时候调用
 * @param pMP   观测上要设置的MP
 * @param idx   特征点的编号
 */
void KeyFrame::setObservation(const PtrMapPoint &pMP, size_t idx)
{
    locker lock(mMutexObs);

    // 保证之前此索引位置有观测，然后替换掉
    if (mDualObservations.find(idx) == mDualObservations.end())
        return;

//    mObservations.erase(mDualObservations[idx]);
    mObservations[pMP] = idx;
    mDualObservations[idx] = pMP;
}

PtrMapPoint KeyFrame::getObservation(size_t idx)
{
    locker lock(mMutexObs);

    if (mDualObservations[idx] == nullptr)
        fprintf(stderr, "[KeyFrame] This is a NULL index in observation!\n");

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
