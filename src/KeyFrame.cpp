/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include "KeyFrame.h"
#include "Config.h"
#include "Map.h"
#include "MapPoint.h"
#include "converter.h"
#include "cvutil.h"

namespace se2lam
{

using namespace cv;
using namespace std;
using namespace Eigen;

typedef unique_lock<mutex> locker;

unsigned long KeyFrame::mNextIdKF = 0;  //! F,KF和MP的编号都是从1开始

KeyFrame::KeyFrame() : mIdKF(0), mbBowVecExist(false), mpMap(nullptr), mbNull(false)
{
    PtrKeyFrame pKF = static_cast<PtrKeyFrame>(nullptr);
    mOdoMeasureFrom = make_pair(pKF, SE3Constraint());
    mOdoMeasureTo = make_pair(pKF, SE3Constraint());
    preOdomFromSelf = make_pair(pKF, PreSE2());
    preOdomToSelf = make_pair(pKF, PreSE2());
}

KeyFrame::KeyFrame(const Frame& frame) : Frame(frame), mIdKF(mNextIdKF++), mpMap(nullptr)
{
    const size_t& n = frame.N;
    mvViewMPsInfo = vector<Matrix3d, aligned_allocator<Matrix3d>>(n, Matrix3d::Identity() * -1);

    PtrKeyFrame nullKF = static_cast<PtrKeyFrame>(nullptr);
    mOdoMeasureFrom = make_pair(nullKF, SE3Constraint());
    mOdoMeasureTo = make_pair(nullKF, SE3Constraint());
    preOdomFromSelf = make_pair(nullKF, PreSE2());
    preOdomToSelf = make_pair(nullKF, PreSE2());
}

KeyFrame::~KeyFrame()
{
    fprintf(stderr, "[KeyFrame] KF#%ld(#%ld) 已被析构!\n", mIdKF, id);
}

// Please handle odometry based constraints after calling this function
//! 加入成员变量map的指针后通过调用map删除其相关容器中的指针, 现已可以正常析构. 20191015
void KeyFrame::setNull()
{
    locker lckObs(mMutexObs);
    locker lckCov(mMutexCovis);

    if (mIdKF == 0)
        return;

    mbNull = true;
    mpORBExtractor = nullptr;

    mvKeyPoints.clear();
    mvViewMPsInfo.clear();
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
    if (mOdoMeasureFrom.first != nullptr) {
        if (mOdoMeasureFrom.first->mOdoMeasureTo.first->mIdKF == mIdKF)
            mOdoMeasureFrom.first->mOdoMeasureTo.first = nullptr;
        mOdoMeasureFrom.first = nullptr;
    }
    if (mOdoMeasureTo.first != nullptr) {
        if (mOdoMeasureTo.first->mOdoMeasureFrom.first->mIdKF == mIdKF)
            mOdoMeasureTo.first->mOdoMeasureFrom.first = nullptr;
        mOdoMeasureTo.first = nullptr;
    }
    if (preOdomFromSelf.first != nullptr) {
        if (preOdomFromSelf.first->preOdomToSelf.first->mIdKF == mIdKF)
            preOdomFromSelf.first->preOdomToSelf.first = nullptr;
        preOdomFromSelf.first = nullptr;
    }
    if (preOdomToSelf.first != nullptr) {
        if (preOdomToSelf.first->preOdomFromSelf.first->mIdKF == mIdKF)
            preOdomToSelf.first->preOdomFromSelf.first = nullptr;
        preOdomToSelf.first = nullptr;
    }

    // Handle observations in MapPoints, 取消MP对此KF的关联
    fprintf(stderr, "[KeyFrame] KF#%ld 取消MP观测前(%ld)引用计数 = %ld\n", mIdKF,
            countObservations(), pThis.use_count());
    for (size_t i = 0; i < N; ++i) {
        const PtrMapPoint& pMP = mvpMapPoints[i];
        if (pMP)
            pMP->eraseObservation(pThis);
    }
    mvpMapPoints.clear();
    fprintf(stderr, "[KeyFrame] KF#%ld 取消MP观测后引用计数 = %ld\n", mIdKF, pThis.use_count());

    // Handle Covisibility, 取消其他KF对此KF的共视关系
    fprintf(stderr, "[KeyFrame] KF#%ld 取消共视关系前(%ld)引用计数 = %ld\n", mIdKF,
            mCovisibleKFsWeight.size(), pThis.use_count());
    for (auto it = mCovisibleKFsWeight.begin(), iend = mCovisibleKFsWeight.end(); it != iend; ++it) {
        it->first->eraseCovisibleKF(pThis);
    }
    mCovisibleKFsWeight.clear();
    mvpCovisibleKFsSorted.clear();
    fprintf(stderr, "[KeyFrame] KF#%ld 取消共视关系后引用计数 = %ld\n", mIdKF, pThis.use_count());

    if (mpMap != nullptr) {
        assert(mIdKF != mpMap->getCurrentKF()->mIdKF);
        mpMap->eraseKF(pThis);
    }

    fprintf(stderr, "[KeyFrame] KF#%ld 被Map设置为null, 处理好各种变量后的引用计数 =  %ld\n", mIdKF,
            pThis.use_count());
}

vector<PtrKeyFrame> KeyFrame::getAllCovisibleKFs()
{
    locker lock(mMutexCovis);
    return mvpCovisibleKFsSorted;
}

vector<PtrKeyFrame> KeyFrame::getBestCovisibleKFs(size_t n)
{
    locker lock(mMutexCovis);
    if (n > 0)
        return vector<PtrKeyFrame>(mvpCovisibleKFsSorted.begin(), mvpCovisibleKFsSorted.begin() + n);
    else
        return mvpCovisibleKFsSorted;
}

vector<PtrKeyFrame> KeyFrame::getCovisibleKFsByWeight(int w)
{
    locker lock(mMutexCovis);
    vector<int>::iterator it = upper_bound(mvOrderedWeights.begin(), mvOrderedWeights.end(), w,
                                           [](int a, int b) { return a > b; });
    if (it == mvOrderedWeights.end() && *mvOrderedWeights.rbegin() < w)
        return vector<PtrKeyFrame>();
    else {
        int n = it - mvOrderedWeights.begin();
        return vector<PtrKeyFrame>(mCovisibleKFsWeight.begin(), mCovisibleKFsWeight.begin() + n);
    }
}

void KeyFrame::addCovisibleKF(const PtrKeyFrame& pKF, int weight)
{
    {
        locker lock(mMutexCovis);
        mCovisibleKFsWeight.emplace(pKF, weight);
    }
    updateCovisibleKFs();
}

void KeyFrame::eraseCovisibleKF(const shared_ptr<KeyFrame>& pKF)
{
    {
        locker lock(mMutexCovis);
        mCovisibleKFsWeight.erase(pKF);
    }
    updateCovisibleKFs();
}

void KeyFrame::updateCovisibleKFs()
{
    locker lock(mMutexCovis);

    vector<pair<PtrKeyFrame, int>> vpCovisbleKFsWeight(mCovisibleKFsWeight.begin(),
                                                       mCovisibleKFsWeight.end());
    std::sort(vpCovisbleKFsWeight.begin(), vpCovisbleKFsWeight.end(), SortByValueGreater());

    size_t n = vpCovisbleKFsWeight.size();
    mvpCovisibleKFsSorted.clear();
    mvpCovisibleKFsSorted.resize(n);
    mvOrderedWeights.clear();
    mvOrderedWeights.resize(n);
    for (size_t i = 0; i < n; ++i) {
        mvpCovisibleKFsSorted[i] = vpCovisbleKFsWeight[i].first;
        mvOrderedWeights[i] = vpCovisbleKFsWeight[i].second;
    }
}

size_t KeyFrame::countCovisibleKFs()
{
    locker lock(mMutexCovis);
    return mCovisibleKFsWeight.size();
}


int KeyFrame::getFeatureIndex(const PtrMapPoint& pMP)
{
    locker lock(mMutexObs);
    const int idx = pMP->getKPIndexInKF(shared_from_this());
    if (idx >= 0 && idx < N)
        return mvpMapPoints[idx] == pMP ? idx : -1;
    else
        return -1;
}

bool KeyFrame::hasObservation(const PtrMapPoint& pMP)
{
    locker lock(mMutexObs);
    const int idx = pMP->getKPIndexInKF(shared_from_this());
    if (idx < 0 || idx >= N)
        return false;

    if (mvpMapPoints[idx] == pMP)
        return true;
    else {
        cerr << "[KeyFrame][Warni] MP#" << pMP->mId << " 错误! idx = " << idx << endl;
        return false;
    }
}

void KeyFrame::eraseObservation(const PtrMapPoint& pMP)
{
    locker lock(mMutexObs);
    const int idx = pMP->getKPIndexInKF(shared_from_this());
    if (idx < 0 || idx >= N)
        return;

    if (mvpMapPoints[idx] == pMP)
        mvpMapPoints[idx] = nullptr;
    else
        cerr << "[KeyFrame][Warni] MP#" << pMP->mId << " 错误! idx = " << idx << endl;
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


}  // namespace se2lam
