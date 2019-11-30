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

KeyFrame::KeyFrame() : mIdKF(0), mpMap(nullptr)
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
    mvbViewMPsInfoExist = vector<bool>(n, false);
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
    size_t nObs = countObservations();

    locker lckObs(mMutexObs);
    locker lckCov(mMutexCovis);

    if (mIdKF == 0)
        return;

    mbNull = true;
    mpORBExtractor = nullptr;

    mvViewMPsInfo.clear();
    mvbViewMPsInfoExist.clear();
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
    fprintf(stderr, "[KeyFrame] KF#%ld 取消MP观测前(%ld)引用计数 = %ld\n", mIdKF, nObs, pThis.use_count());
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
    mvOrderedWeights.clear();
    fprintf(stderr, "[KeyFrame] KF#%ld 取消共视关系后引用计数 = %ld\n", mIdKF, pThis.use_count());

    if (mpMap != nullptr) {
        assert(mIdKF != mpMap->getCurrentKF()->mIdKF);
        mpMap->eraseKF(pThis);
    }

    fprintf(stderr, "[KeyFrame] KF#%ld 被Map设置为null, 处理好各种变量后的引用计数 =  %ld\n", mIdKF,
            pThis.use_count());
}

vector<shared_ptr<KeyFrame>> KeyFrame::getAllCovisibleKFs()
{
    locker lock(mMutexCovis);
    return mvpCovisibleKFsSorted;
}

vector<shared_ptr<KeyFrame>> KeyFrame::getBestCovisibleKFs(size_t n)
{
    locker lock(mMutexCovis);
    if (n > 0 && n <= mvpCovisibleKFsSorted.size())
        return vector<PtrKeyFrame>(mvpCovisibleKFsSorted.begin(), mvpCovisibleKFsSorted.begin() + n);
    else
        return mvpCovisibleKFsSorted;
}

vector<shared_ptr<KeyFrame>> KeyFrame::getCovisibleKFsByWeight(int w)
{
    locker lock(mMutexCovis);
    vector<int>::iterator it = upper_bound(mvOrderedWeights.begin(), mvOrderedWeights.end(), w,
                                           [](int a, int b) { return a > b; });
    if (it == mvOrderedWeights.end() && *mvOrderedWeights.rbegin() < w)
        return vector<PtrKeyFrame>();
    else {
        int n = it - mvOrderedWeights.begin();
        return vector<PtrKeyFrame>(mvpCovisibleKFsSorted.begin(), mvpCovisibleKFsSorted.begin() + n);
    }
}

map<shared_ptr<KeyFrame>, int> KeyFrame::getAllCovisibleKFsAndWeights()
{
    locker lock(mMutexCovis);
    return mCovisibleKFsWeight;
}

void KeyFrame::addCovisibleKF(const shared_ptr<KeyFrame>& pKF, int weight)
{
    {
        locker lock(mMutexCovis);
        // mCovisibleKFsWeight.emplace(pKF, weight);
        mCovisibleKFsWeight[pKF] = weight;
    }
    sortCovisibleKFs();
}

//! TODO  没啥必要
void KeyFrame::addCovisibleKF(const shared_ptr<KeyFrame>& pKF)
{
    vector<PtrMapPoint> vpMPs;
    {
        locker lockMPs(mMutexObs);
        if (mCovisibleKFsWeight.count(pKF))
            return;
        vpMPs = mvpMapPoints;
    }

    int weight;
    for (auto vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++) {
        PtrMapPoint pMP = (*vit);
        if (!pMP || pMP->isNull())
            continue;

        if (pMP->hasObservation(pKF))
            weight++;
    }

//    if (weight == 0)
//        return;

    addCovisibleKF(pKF, weight);
}

void KeyFrame::eraseCovisibleKF(const shared_ptr<KeyFrame>& pKF)
{
    {
        locker lock(mMutexCovis);
        mCovisibleKFsWeight.erase(pKF);
    }
    sortCovisibleKFs();
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
    mvOrderedWeights.clear();
    mvOrderedWeights.resize(n);
    for (size_t i = 0; i < n; ++i) {
        mvpCovisibleKFsSorted[i] = vpCovisbleKFsWeight[i].first;
        mvOrderedWeights[i] = vpCovisbleKFsWeight[i].second;
    }
}

void KeyFrame::updateCovisibleGraph()
{
    WorkTimer timer;

    map<PtrKeyFrame, int> KFcounter;  // TODO  这个变量待考究
    vector<PtrMapPoint> vpMPs = getObservations(true, false);  // lock

    // 1.通过3D点间接统计可以观测到这些3D点的所有关键帧之间的共视程度
    // 即统计每一个关键帧都有多少关键帧与它存在共视关系，统计结果放在KFcounter
    for (auto vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++) {
        PtrMapPoint pMP = (*vit);
        if (!pMP || pMP->isNull())
            continue;

        vector<PtrKeyFrame> thisObsKFs = pMP->getObservations();
        for (auto mit = thisObsKFs.begin(), mend = thisObsKFs.end(); mit != mend; mit++) {
            if ((*mit)->mIdKF == mIdKF)  // 除去自身，自己与自己不算共视
                continue;
            KFcounter[(*mit)]++;  // TODO. 待确认插入新的key时value值为1
        }
    }

    // This should not happen
    if (KFcounter.empty())
        return;

    // 2.次数超过阈值则添加共视关系, 防止次数整体过少, 将把次数最多的KF添加为共视关系
    int nmax = 0;
    int th = 0.3 * vpMPs.size();  // 共同观测MP点数阈值30%
    PtrKeyFrame pKFmax = nullptr;

    // vPairs记录与其它关键帧共视帧数大于th的关键帧
    vector<pair<int, PtrKeyFrame>> vPairs;
    vPairs.reserve(KFcounter.size());
    for (auto mit = KFcounter.begin(), mend = KFcounter.end(); mit != mend; mit++) {
        if (mit->second > nmax) {
            nmax = mit->second;
            pKFmax = mit->first;
        }
        if (mit->second >= th) {
            // 对应权重需要大于阈值，对这些关键帧建立连接
            vPairs.push_back(make_pair(mit->second, mit->first));
            (mit->first)->addCovisibleKF(shared_from_this(), mit->second);
        }
    }

    // 如果没有超过阈值的权重，则对权重最大的关键帧建立连接, 这是对之前th这个阈值可能过高的一个补丁
    if (vPairs.empty()) {
        vPairs.emplace_back(nmax, pKFmax);
        pKFmax->addCovisibleKF(shared_from_this(), nmax);
    }

    // vPairs里存的都是相互共视程度符合阈值(5)的关键帧和共视权重，由大到小
    sort(vPairs.begin(), vPairs.end(), [](const pair<int, PtrKeyFrame>& lhs, const pair<int, PtrKeyFrame>& rhs) {
        return lhs.first > rhs.first;
    });


    size_t n = vPairs.size();
    vector<PtrKeyFrame> vKFs(n);
    vector<int> vWs(n);
    map<PtrKeyFrame, int> mKFadnW;
    for (size_t i = 0; i < vPairs.size(); i++) {
        vKFs[i] = vPairs[i].second;
        vWs[i] = vPairs[i].first;
        mKFadnW.emplace(vPairs[i].second, vPairs[i].first);
    }

    {
        unique_lock<mutex> lockCon(mMutexCovis);
        mCovisibleKFsWeight = mKFadnW;
        mvpCovisibleKFsSorted = vKFs;
        mvOrderedWeights = vWs;
    }
    printf("[KeyFrame][Co] #%ld(KF#%ld) 更新共视关系成功, 针对%ld个MP观测, 更新了%ld个共视KF(共视超30%%), 共耗时%.2fms\n",
           id, mIdKF, vpMPs.size(), mKFadnW.size(), timer.count());
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
    if (idx >= 0 && idx < static_cast<int>(N))
        return mvpMapPoints[idx] == pMP ? idx : -1;
    else
        return -1;
}

bool KeyFrame::hasObservationByPointer(const PtrMapPoint& pMP)
{
    locker lock(mMutexObs);
    const int idx = pMP->getKPIndexInKF(shared_from_this());
    if (idx < 0 || idx >= static_cast<int>(N))
        return false;

    if (mvpMapPoints[idx] == pMP) {
        return true;
    } else {
        cerr << "[KeyFrame][Warni] MP#" << pMP->mId << " 错误! idx = " << idx << endl;
        return false;
    }
}

void KeyFrame::setObsAndInfo(const PtrMapPoint& pMP, size_t idx, const Matrix3d& info)
{
    locker lock(mMutexObs);
    if (idx >= 0 && idx < N) {
        mvpMapPoints[idx] = pMP;
        mvViewMPsInfo[idx] = info;
        mvbViewMPsInfoExist[idx] = true;
    } else {
        cerr << "[KeyFrame][Warni] 设置观测错误! 不存在的索引号 idx = " << idx << endl;
    }
}

void KeyFrame::eraseObservationByPointer(const PtrMapPoint& pMP)
{
    locker lock(mMutexObs);
    const int idx = pMP->getKPIndexInKF(shared_from_this());
    if (idx < 0 || idx >= static_cast<int>(N))
        return;

    if (mvpMapPoints[idx] == pMP) {
        mvpMapPoints[idx] = nullptr;
        mvbViewMPsInfoExist[idx] = false;
    } else {
        cerr << "[KeyFrame][Warni] MP#" << pMP->mId << " 错误! idx = " << idx << endl;
    }
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
