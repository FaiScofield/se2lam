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

int KeyFrame::mNextIdKF = 0;    //! F,KF和MP的编号都是从0开始

KeyFrame::KeyFrame() : mIdKF(-1), mbBowVecExist(false), mbNull(false)
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
    for (int i = 1; i < mnScaleLevels; i++) {
        mvScaleFactors[i] = mvScaleFactors[i - 1] * mfScaleFactor;
        mvLevelSigma2[i] = mvScaleFactors[i] * mvScaleFactors[i];
    }

    mvInvLevelSigma2.resize(mvLevelSigma2.size());
    for (int i = 0; i < mnScaleLevels; i++)
        mvInvLevelSigma2[i] = 1.0 / mvLevelSigma2[i];
}


KeyFrame::KeyFrame(const Frame &frame) : Frame(frame), mbBowVecExist(false), mbNull(false)
{
    size_t n = frame.N;
    mViewMPs = vector<Point3f>(n, Point3f(-1, -1, -1));
    mViewMPsInfo = vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>(
        n, Eigen::Matrix3d::Identity() * -1);

    mIdKF = mNextIdKF;
    mNextIdKF++;

    PtrKeyFrame pKF = static_cast<PtrKeyFrame>(nullptr);
    mOdoMeasureFrom = make_pair(pKF, SE3Constraint());
    mOdoMeasureTo = make_pair(pKF, SE3Constraint());

    preOdomFromSelf = make_pair(pKF, PreSE2());
    preOdomToSelf = make_pair(pKF, PreSE2());
}

KeyFrame::~KeyFrame()
{
}


// Please handle odometry based constraints after calling this function
void KeyFrame::setNull(const shared_ptr<KeyFrame> &pThis)
{
    locker lckImg(mMutexImg);
    locker lckPose(mMutexPose);
    locker lckObs(mMutexObs);
    locker lckDes(mMutexDes);
    locker lckCov(mMutexCovis);

    mbNull = true;
    mpORBExtractor = nullptr;
    mIdKF = -1;
    mImage.release();
    mDescriptors.release();
    mvKeyPoints.clear();
    mvKeyPoints.clear();
    mvpMapPoints.clear();
    mvbOutlier.clear();

    // Handle Feature based constraints
    for (auto it = mFtrMeasureFrom.begin(), iend = mFtrMeasureFrom.end(); it != iend; it++) {
        it->first->mFtrMeasureTo.erase(pThis);
    }
    for (auto it = mFtrMeasureTo.begin(), iend = mFtrMeasureTo.end(); it != iend; it++) {
        it->first->mFtrMeasureFrom.erase(pThis);
    }
    mFtrMeasureFrom.clear();
    mFtrMeasureTo.clear();

    // Handle observations in MapPoints, 取消MP对此KF的关联
    for (auto it = mObservations.begin(), iend = mObservations.end(); it != iend; it++) {
        PtrMapPoint pMP = it->first;
        pMP->eraseObservation(pThis);
    }

    // Handle Covisibility, 取消其他KF对此KF的共视关系
    for (auto it = mCovisibleKFs.begin(), iend = mCovisibleKFs.end(); it != iend; it++) {
        (*it)->eraseCovisibleKF(pThis);
    }
    mObservations.clear();
    mDualObservations.clear();
    mCovisibleKFs.clear();
    mViewMPs.clear();
    mViewMPsInfo.clear();
}

int KeyFrame::getSizeObsMP()
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
    auto i = mObservations.begin(), iend = mObservations.end();
    for (; i != iend; i++) {
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
    map<PtrMapPoint, int>::iterator it = mObservations.find(pMP);
    return (it != mObservations.end());

//    return mObservations.count(pMP);
}

bool KeyFrame::hasObservation(int idx)
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

void KeyFrame::addObservation(PtrMapPoint pMP, int idx)
{
    locker lock(mMutexObs);
    if (pMP->isNull())
        return;
    mObservations[pMP] = idx;
    mDualObservations[idx] = pMP;
}

map<PtrMapPoint, int> KeyFrame::getObservations()
{
    locker lock(mMutexObs);
    return mObservations;
}

void KeyFrame::eraseObservation(const PtrMapPoint pMP)
{
    locker lock(mMutexObs);
    int idx = mObservations[pMP];
    mObservations.erase(pMP);
    mDualObservations.erase(idx);
}

void KeyFrame::eraseObservation(int idx)
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
    lock_guard<mutex> lck(mMutexDes);
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

/**
 * @brief KeyFrame::GetMapPointMatches 找到特征点对应的MPs，关键函数，后续应该有一个观测更新的函数需要被调用
 * @return
 */
vector<PtrMapPoint> KeyFrame::GetMapPointMatches()
{
    vector<PtrMapPoint> ret;
    int numKPs = mvKeyPoints.size();
    for (int i = 0; i < numKPs; i++) {
        PtrMapPoint pMP = nullptr;
        std::map<int, PtrMapPoint>::iterator iter;
        iter = mDualObservations.find(i);

        if (iter == mDualObservations.end()) {
            ret.push_back(pMP);  //!@Vance: 此特征点没有match上的MP
        } else {
            pMP = iter->second;
            ret.push_back(pMP);
        }
    }

    return ret;
}

/**
 * @brief KeyFrame::setObservation 设置MP的观测，在MP被merge的时候调用
 * @param pMP   观测上要设置的MP
 * @param idx   特征点的编号
 */
void KeyFrame::setObservation(const PtrMapPoint &pMP, int idx)
{
    locker lock(mMutexObs);

    if (mDualObservations.find(idx) == mDualObservations.end())
        return;

    mObservations.erase(mDualObservations[idx]);
    mObservations[pMP] = idx;
    mDualObservations[idx] = pMP;
}

PtrMapPoint KeyFrame::getObservation(int idx)
{
    locker lock(mMutexObs);

    if (!mDualObservations[idx]) {
        fprintf(stderr, "[KeyFrame] This is a NULL index in feature!\n");
    }
    return mDualObservations[idx];
}

int KeyFrame::getFtrIdx(const PtrMapPoint &pMP)
{
    locker lock(mMutexObs);
    if (mObservations.find(pMP) == mObservations.end())
        return -1;
    return mObservations[pMP];
}

}  // namespace se2lam
