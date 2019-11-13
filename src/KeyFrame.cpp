/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include "KeyFrame.h"
#include "Map.h"
#include "MapPoint.h"
#include "cvutil.h"

namespace se2lam
{

using namespace cv;
using namespace std;

typedef unique_lock<mutex> locker;

unsigned long KeyFrame::mNextIdKF = 1;  //! F,KF和MP的编号都是从1开始


KeyFrame::KeyFrame() : mIdKF(0), mbBowVecExist(false), mbNull(false)
{
    PtrKeyFrame pKF = static_cast<PtrKeyFrame>(nullptr);
    mOdoMeasureFrom = make_pair(pKF, SE3Constraint());
    mOdoMeasureTo = make_pair(pKF, SE3Constraint());
    preOdomFromSelf = make_pair(pKF, PreSE2());
    preOdomToSelf = make_pair(pKF, PreSE2());
}


KeyFrame::KeyFrame(const Frame& frame) : Frame(frame), mbBowVecExist(false), mbNull(false)
{
    mIdKF = mNextIdKF++;

    const size_t& n = frame.N;
    mvViewMPs = vector<Point3f>(n, Point3f(-1.f, -1.f, -1.f));
    mvViewMPsInfo = vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>(
        n, Eigen::Matrix3d::Identity() * -1);

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
void KeyFrame::setNull()
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
    if (bNeedVisualization)
        mImage.release();

    PtrKeyFrame pThis = shared_from_this();

    // Handle Feature based constraints, 取消特征约束
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
    mOdoMeasureFrom.first = nullptr;
    mOdoMeasureTo.first = nullptr;
    preOdomFromSelf.first = nullptr;
    preOdomToSelf.first = nullptr;

    // Handle observations in MapPoints, 取消MP对此KF的关联
    fprintf(stderr, "[KeyFrame] KF#%ld 取消MP观测前(%ld)引用计数 = %ld\n", mIdKF,
            mObservations.size(), pThis.use_count());
    for (auto it = mObservations.begin(), iend = mObservations.end(); it != iend; ++it) {
        PtrMapPoint pMP = it->first;
        pMP->eraseObservation(pThis);
    }
    mObservations.clear();
    mDualObservations.clear();
    fprintf(stderr, "[KeyFrame] KF#%ld 取消MP观测后引用计数 = %ld\n", mIdKF, pThis.use_count());

    // Handle Covisibility, 取消其他KF对此KF的共视关系
    fprintf(stderr, "[KeyFrame] KF#%ld 取消共视关系前(%ld)引用计数 = %ld\n", mIdKF,
            mspCovisibleKFs.size(), pThis.use_count());
    for (auto it = mspCovisibleKFs.begin(), iend = mspCovisibleKFs.end(); it != iend; ++it) {
        (*it)->eraseCovisibleKF(pThis);
    }
    mspCovisibleKFs.clear();
    mCovisibleKFsWeight.clear();
    mvpCovisibleKFsSorted.clear();
    fprintf(stderr, "[KeyFrame] KF#%ld 取消共视关系后引用计数 = %ld\n", mIdKF, pThis.use_count());

    mvViewMPs.clear();
    mvViewMPsInfo.clear();

    mpMap->eraseKF(pThis);
    fprintf(stderr, "[KeyFrame] KF#%ld 被Map设置为null, 处理好各种变量后的引用计数 =  %ld\n", mIdKF,
            pThis.use_count());
}

set<PtrKeyFrame> KeyFrame::getAllCovisibleKFs()
{
    locker lock(mMutexCovis);
    return mspCovisibleKFs;
}

vector<PtrKeyFrame> KeyFrame::getBestCovisibleKFs(size_t n)
{
    locker lock(mMutexCovis);
    if (n > 0)
        return vector<PtrKeyFrame>(mvpCovisibleKFsSorted.begin(),
                                   mvpCovisibleKFsSorted.begin() + n);
    else
        return vector<PtrKeyFrame>(mvpCovisibleKFsSorted.begin(), mvpCovisibleKFsSorted.end());
}

void KeyFrame::addCovisibleKF(const shared_ptr<KeyFrame>& pKF)
{
    locker lock(mMutexCovis);
    mspCovisibleKFs.insert(pKF);
}

void KeyFrame::addCovisibleKF(const shared_ptr<KeyFrame>& pKF, int weight)
{
    locker lock(mMutexCovis);
    mCovisibleKFsWeight.emplace(pKF, weight);
    mspCovisibleKFs.insert(pKF);
}

void KeyFrame::eraseCovisibleKF(const shared_ptr<KeyFrame>& pKF)
{
    locker lock(mMutexCovis);
    mspCovisibleKFs.erase(pKF);
    mCovisibleKFsWeight.erase(pKF);
}

void KeyFrame::sortCovisibleKFs()
{
    locker lock(mMutexCovis);

    vector<pair<PtrKeyFrame, int>> vpCovisbleKFsWeight(mCovisibleKFsWeight.begin(),
                                                       mCovisibleKFsWeight.end());
    std::sort(vpCovisbleKFsWeight.begin(), vpCovisbleKFsWeight.end(), SortByValueGreater());

    size_t n = vpCovisbleKFsWeight.size();
    mvpCovisibleKFsSorted.clear();
    mvpCovisibleKFsSorted.resize(n);
    for (size_t i = 0; i < n; ++i)
        mvpCovisibleKFsSorted[i] = vpCovisbleKFsWeight[i].first;
}

//! TODO  test funciton. 不要加锁
void KeyFrame::updateCovisibleKFs()
{
    map<PtrKeyFrame, int> KFCounter;
    set<PtrMapPoint> spMP = getAllObsMPs(false);  // mMutexObs

    for (auto iter = spMP.begin(), iend = spMP.end(); iter != iend; iter++) {
        PtrMapPoint pMP = *iter;
        if (!pMP || pMP->isNull())
            continue;

        set<PtrKeyFrame> sKFObs = pMP->getObservations();
        for (auto mit = sKFObs.begin(), mend = sKFObs.end(); mit != mend; mit++) {
            PtrKeyFrame pKFObs = *mit;
            if (pKFObs->mIdKF == mIdKF)
                continue;  // 除去自身，自己与自己不算共视
            KFCounter[pKFObs]++;
        }
    }

    if (KFCounter.empty())
        return;  // This should not happen

    vector<pair<PtrKeyFrame, int>> vpKFsWeight(KFCounter.begin(), KFCounter.end());
    std::sort(vpKFsWeight.begin(), vpKFsWeight.end(), SortByValueGreater());

    int th = 5;
    if (vpKFsWeight[0].second < th) {
        addCovisibleKF(vpKFsWeight[0].first, vpKFsWeight[0].second);  // mMutexCovis
        (vpKFsWeight[0].first)->addCovisibleKF(shared_from_this(), vpKFsWeight[0].second);
    } else {
        for (auto mit = vpKFsWeight.begin(), mend = vpKFsWeight.end(); mit != mend; mit++) {
            if (mit->second >= th) {
                addCovisibleKF(mit->first, mit->second);  // mMutexCovis
                (mit->first)->addCovisibleKF(shared_from_this(), mit->second);
            }
        }
    }

    sortCovisibleKFs();  // mMutexCovis
}

size_t KeyFrame::countCovisibleKFs()
{
    locker lock(mMutexCovis);
    return mspCovisibleKFs.size();
    // return mCovisibleKFsWeight.size();
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
        if (!pMP || pMP->isNull())
            continue;
        if (checkParallax && !pMP->isGoodPrl())
            continue;
        spMP.insert(pMP);
    }
    return spMP;
}

map<PtrMapPoint, size_t> KeyFrame::getObservations()
{
    locker lock(mMutexObs);
    return mObservations;
}

vector<PtrMapPoint> KeyFrame::getMapPointMatches()
{
    locker lock(mMutexObs);
    vector<PtrMapPoint> ret(N, nullptr);
    for (auto iter = mDualObservations.begin(), iend = mDualObservations.end(); iter != iend;
         ++iter)
        ret[iter->first] = iter->second;

    return ret;
}

PtrMapPoint KeyFrame::getObservation(size_t idx)
{
    locker lock(mMutexObs);

    if (mDualObservations.count(idx))
        return mDualObservations[idx];
    else
        return nullptr;
}

int KeyFrame::getFeatureIndex(const PtrMapPoint& pMP)
{
    locker lock(mMutexObs);
    if (mObservations.count(pMP))
        return mObservations[pMP];
    else
        return -1;
}

void KeyFrame::addObservation(const PtrMapPoint& pMP, size_t idx)
{
    locker lock(mMutexObs);
    mObservations.emplace(pMP, idx);
    mDualObservations.emplace(idx, pMP);
}

void KeyFrame::eraseObservation(const PtrMapPoint& pMP)
{
    locker lock(mMutexObs);
    mDualObservations.erase(mObservations[pMP]);
    mObservations.erase(pMP);
}

void KeyFrame::eraseObservation(size_t idx)
{
    locker lock(mMutexObs);
    mObservations.erase(mDualObservations[idx]);
    mDualObservations.erase(idx);
}

bool KeyFrame::hasObservation(const PtrMapPoint& pMP)
{
    locker lock(mMutexObs);
    return mObservations.count(pMP);
}

bool KeyFrame::hasObservation(size_t idx)
{
    locker lock(mMutexObs);
    return mDualObservations.count(idx);
}

void KeyFrame::setObservation(const PtrMapPoint& pMP, size_t idx)
{
    locker lock(mMutexObs);

    // 保证之前此索引位置有观测，然后替换掉
    if (mDualObservations.find(idx) == mDualObservations.end())
        return;

    mObservations[pMP] = idx;
    mDualObservations[idx] = pMP;
}

size_t KeyFrame::countObservations()
{
    locker lock(mMutexObs);
    return mObservations.size();
}

void KeyFrame::setViewMP(const Point3f& pt3f, size_t idx, const Eigen::Matrix3d& info)
{
    locker lock(mMutexObs);
    if (idx < N) {
        mvViewMPs[idx] = pt3f;
        mvViewMPsInfo[idx] = info;
    }
}

Point3f KeyFrame::getViewMPPoseInCamareFrame(size_t idx)
{
    locker lock(mMutexObs);

    Point3f ret(-1.f, -1.f, -1.f);
    if (mDualObservations.count(idx)) {
        Point3f Pw = mDualObservations[idx]->getPos();
        ret = cvu::se3map(Tcw, Pw);
    }
    return ret;
}


void KeyFrame::addFtrMeasureFrom(const shared_ptr<KeyFrame>& pKF, const Mat& _mea, const Mat& _info)
{
    mFtrMeasureFrom.emplace(pKF, SE3Constraint(_mea, _info));
}

void KeyFrame::eraseFtrMeasureFrom(const shared_ptr<KeyFrame>& pKF)
{
    mFtrMeasureFrom.erase(pKF);
}

void KeyFrame::addFtrMeasureTo(const shared_ptr<KeyFrame>& pKF, const Mat& _mea, const Mat& _info)
{
    mFtrMeasureTo.emplace(pKF, SE3Constraint(_mea, _info));
}

void KeyFrame::eraseFtrMeasureTo(const shared_ptr<KeyFrame>& pKF)
{
    mFtrMeasureTo.erase(pKF);
}

void KeyFrame::setOdoMeasureFrom(const shared_ptr<KeyFrame>& pKF, const Mat& _mea, const Mat& _info)
{
    mOdoMeasureFrom = make_pair(pKF, SE3Constraint(_mea, _info));
}

void KeyFrame::setOdoMeasureTo(const shared_ptr<KeyFrame>& pKF, const Mat& _mea, const Mat& _info)
{
    mOdoMeasureTo = make_pair(pKF, SE3Constraint(_mea, _info));
}


void KeyFrame::computeBoW(const ORBVocabulary* _pVoc)
{
    if (mBowVec.empty() || mFeatVec.empty()) {
        vector<cv::Mat> vCurrentDesc = toDescriptorVector(mDescriptors);
        // Feature vector associate features with nodes in the 4th level (from leaves up)
        // We assume the vocabulary tree has 6 levels, change the 4 otherwise
        _pVoc->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
    }
    mbBowVecExist = true;
}

}  // namespace se2lam
