/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include "Map.h"
#include "GlobalMapper.h"
#include "LocalMapper.h"
#include "Track.h"
#include "converter.h"
#include "cvutil.h"
#include "optimizer.h"
#include <opencv2/flann.hpp>

namespace se2lam
{

using namespace cv;
using namespace std;
using namespace g2o;

typedef unique_lock<mutex> locker;

Map::Map() : mCurrentKF(nullptr), isEmpty(true), mpLocalMapper(nullptr)
{
    mCurrentFramePose = cv::Mat::eye(4, 4, CV_32FC1);
    mbNewKFInserted = false;
}

Map::~Map()
{}

void Map::insertKF(const PtrKeyFrame& pKF)
{
    locker lock(mMutexGlobalGraph);
    pKF->setMap(this);
    mspKFs.insert(pKF);
    mCurrentKF = pKF;
    isEmpty = false;
    mbNewKFInserted = true;
}

void Map::insertMP(const PtrMapPoint& pMP)
{
    locker lock(mMutexGlobalGraph);
    pMP->setMap(this);
    mspMPs.insert(pMP);
}

void Map::eraseKF(const PtrKeyFrame& pKF)
{
    locker lock(mMutexGlobalGraph);
    mspKFs.erase(pKF);

    locker lock2(mMutexLocalGraph);
    auto iter1 = find(mvLocalGraphKFs.begin(), mvLocalGraphKFs.end(), pKF);
    if (iter1 != mvLocalGraphKFs.end())
        mvLocalGraphKFs.erase(iter1);
    auto iter2 = find(mvLocalRefKFs.begin(), mvLocalRefKFs.end(), pKF);
    if (iter2 != mvLocalRefKFs.end())
        mvLocalRefKFs.erase(iter2);
}

void Map::eraseMP(const PtrMapPoint& pMP)
{
    locker lock(mMutexGlobalGraph);
    mspMPs.erase(pMP);

    locker lock2(mMutexLocalGraph);
    auto iter = find(mvLocalGraphMPs.begin(), mvLocalGraphMPs.end(), pMP);
    if (iter != mvLocalGraphMPs.end())
        mvLocalGraphMPs.erase(iter);
}

/**
 * @brief 回环检测成功后的地图点合并
 * FIXME 这里代码是两个KF只要能同时观测到两个MP中的一个就退出, 不能同时观测到就去merge, 感觉写反了？
 * 循环里面的renturn感觉应该改成break？
 * TODO 代码暂时没有执行到这里, 待验证!
 */
void Map::mergeMP(PtrMapPoint& toKeep, PtrMapPoint& toDelete)
{
    fprintf(stderr, "[ Map ][Info ] Merging MP between #%ld(keep) and #%ld(delete) MP.\n",
            toKeep->mId, toDelete->mId);
    vector<PtrKeyFrame> pKFs = toKeep->getObservations();
    for (auto it = pKFs.begin(), itend = pKFs.end(); it != itend; ++it) {
        PtrKeyFrame pKF = *it;
        //!@Vance: 有一KF能同时观测到这两个MP就返回，Why?
        if (pKF->hasObservationByPointer(toKeep) && pKF->hasObservationByPointer(toDelete)) {
            cerr << "[ Map ][Warni] Return for a KF(toKeep) has same observation." << endl;
            return;  //! TODO 应该是break？
        }
    }
    pKFs = toDelete->getObservations();
    for (auto it = pKFs.begin(), itend = pKFs.end(); it != itend; ++it) {
        PtrKeyFrame pKF = *it;
        if (pKF->hasObservationByPointer(toKeep) && pKF->hasObservationByPointer(toDelete)) {
            cerr << "[ Map ][Warni] Return for a KF(toDelete) has same observation." << endl;
            return;  //! TODO 应该是break？
        }
    }

    toDelete->mergedInto(toKeep);
    mspMPs.erase(toDelete);
    fprintf(stderr, "[ Map ][Info ] Have a merge between #%ld(keep) and #%ld(delete) MP.\n",
            toKeep->mId, toDelete->mId);

    // 还要更新局部的MP
    auto it = std::find(mvLocalGraphMPs.begin(), mvLocalGraphMPs.end(), toDelete);
    if (it != mvLocalGraphMPs.end())
        *it = toKeep;
}


size_t Map::countKFs()
{
    locker lock(mMutexGlobalGraph);
    return mspKFs.size();
}

size_t Map::countMPs()
{
    locker lock(mMutexGlobalGraph);
    return mspMPs.size();
}

size_t Map::countLocalKFs()
{
    locker lock(mMutexLocalGraph);
    return mvLocalGraphKFs.size();
}

size_t Map::countLocalMPs()
{
    locker lock(mMutexLocalGraph);
    return mvLocalGraphMPs.size();
}

size_t Map::countLocalRefKFs()
{
    locker lock(mMutexLocalGraph);
    return mvLocalRefKFs.size();
}

vector<PtrKeyFrame> Map::getAllKFs()
{
    locker lock(mMutexGlobalGraph);
    return vector<PtrKeyFrame>(mspKFs.begin(), mspKFs.end());
}

vector<PtrMapPoint> Map::getAllMPs()
{
    locker lock(mMutexGlobalGraph);
    return vector<PtrMapPoint>(mspMPs.begin(), mspMPs.end());
}

vector<PtrKeyFrame> Map::getLocalKFs()
{
    locker lock(mMutexLocalGraph);
    return vector<PtrKeyFrame>(mvLocalGraphKFs.begin(), mvLocalGraphKFs.end());
}

vector<PtrMapPoint> Map::getLocalMPs()
{
    locker lock(mMutexLocalGraph);
    return vector<PtrMapPoint>(mvLocalGraphMPs.begin(), mvLocalGraphMPs.end());
}

vector<PtrKeyFrame> Map::getRefKFs()
{
    locker lock(mMutexLocalGraph);
    return vector<PtrKeyFrame>(mvLocalRefKFs.begin(), mvLocalRefKFs.end());
}

void Map::clear()
{
    locker lock1(mMutexGlobalGraph);
    locker lock2(mMutexLocalGraph);
    locker lock3(mMutexCurrentKF);
    locker lock4(mMutexCurrentFrame);
    mspKFs.clear();
    mspMPs.clear();
    mvLocalGraphKFs.clear();
    mvLocalGraphMPs.clear();
    mvLocalRefKFs.clear();
    isEmpty = true;
    mCurrentKF = nullptr;
    mCurrentFramePose.release();
    KeyFrame::mNextIdKF = 0;
    MapPoint::mNextId = 0;
}

//! 没用到
void Map::setCurrentKF(const PtrKeyFrame& pKF)
{
    locker lock(mMutexCurrentKF);
    mCurrentKF = pKF;
}

PtrKeyFrame Map::getCurrentKF()
{
    locker lock(mMutexCurrentKF);
    return mCurrentKF;
}

void Map::setCurrentFramePose(const Mat& pose)
{
    locker lock(mMutexCurrentFrame);
    pose.copyTo(mCurrentFramePose);
}

Mat Map::getCurrentFramePose()
{
    locker lock(mMutexCurrentFrame);
    return mCurrentFramePose.clone();
}

/**
 * @brief 修剪冗余的KF, 如果LocalKFs ≤ 3帧，就不会修剪
 * 此函数在LocalMapper::pruneRedundantKFinMap()函数中调用
 * 在共视KF中观测到本KF中MP两次以上的比例大于80%则视此KF为冗余的
 *
 * @return 返回是否有经过修剪的标志
 */
int Map::pruneRedundantKF()
{
    vector<PtrMapPoint> vLocalGraphMPs;
    vector<PtrKeyFrame> vLocalGraphKFs;

    {
        locker lock1(mMutexLocalGraph);
        vLocalGraphMPs = mvLocalGraphMPs;
        vLocalGraphKFs = mvLocalGraphKFs;
    }

    if (vLocalGraphKFs.size() <= 3)
        return false;

    int nPruned = 0;
    const double theshl = 10000;                 // 10m
    const double thesht = 45 * 3.1415926 / 180;  // 45 degree to rad

    for (int i = 0, iend = vLocalGraphKFs.size(); i != iend; ++i) {
        PtrKeyFrame& thisKF = vLocalGraphKFs[i];

        // 首帧和当前帧不能被修剪
        if (thisKF->isNull() || mCurrentKF->mIdKF == thisKF->mIdKF || thisKF->mIdKF == 0)
            continue;

        // Count MPs in thisKF observed by covKFs 2 times
        //! 共视KF中观测到本KF中MP两次以上的比例大于80%则视此KF为冗余的
        set<PtrMapPoint> spMPs;
        const vector<PtrKeyFrame> covKFs = thisKF->getAllCovisibleKFs();
        float ratio = compareViewMPs(thisKF, covKFs, spMPs, 2);
        bool bIsThisKFRedundant = (ratio >= 0.7f);

        // Do Prune if pass threashold test
        //! Local KF中的一帧thisKF要被修剪，则需要更新它们的联接关系
        if (bIsThisKFRedundant) {
            PtrKeyFrame lastKF = thisKF->mOdoMeasureTo.first;
            PtrKeyFrame nextKF = thisKF->mOdoMeasureFrom.first;

            bool bHasFeatEdge = (thisKF->mFtrMeasureFrom.size() != 0);

            // Prune this KF and link a new odometry constrait
            if (lastKF && nextKF && !bHasFeatEdge) {
                Se2 dOdoLastThis = thisKF->odom - lastKF->odom;
                Se2 dOdoThisNext = nextKF->odom - thisKF->odom;

                double dl1 = sqrt(dOdoLastThis.x * dOdoLastThis.x + dOdoLastThis.y * dOdoLastThis.y);
                double dt1 = abs(dOdoLastThis.theta);
                double dl2 = sqrt(dOdoThisNext.x * dOdoThisNext.x + dOdoThisNext.y * dOdoThisNext.y);
                double dt2 = abs(dOdoThisNext.theta);

                //! 被修剪帧的前后帧之间位移不能超过10m，角度不能超过45°，防止修剪掉在大旋转大平移之间的KF
                if (dl1 < theshl && dl2 < theshl && dt1 < thesht && dt2 < thesht) {
                    // 给前后帧之间添加共视关联和约束
                    Mat measure;
                    g2o::Matrix6d info;
                    calcOdoConstraintCam(nextKF->odom - lastKF->odom, measure, info);
                    nextKF->setOdoMeasureTo(lastKF, measure, toCvMat6f(info));
                    lastKF->setOdoMeasureFrom(nextKF, measure, toCvMat6f(info));
                    nextKF->addCovisibleKF(lastKF);
                    lastKF->addCovisibleKF(nextKF);

                    thisKF->setNull();
                    fprintf(stderr, "[ Map ][Info ] KF#%ld 此KF被修剪, 引用计数 = %ld\n",
                            thisKF->mIdKF, thisKF.use_count());

                    nPruned++;
                }
            }
        }
    }  // KF被修剪(setNull)后, 局部地图会自动删除此KF

    return nPruned;
}

/**
 * @brief 更新局部地图的KFs与MPs, 以及参考KFs
 * 此函数在LocalMapper::run()函数中调用, mLocalGraphKFs, mRefKFs, mLocalGraphMPs在此更新
 */
void Map::updateLocalGraph(int maxLevel, int maxN, float searchRadius)
{
    WorkTimer timer;

    locker lock(mMutexLocalGraph);

    mvLocalGraphKFs.clear();
    mvLocalGraphMPs.clear();
    mvLocalRefKFs.clear();

    set<PtrKeyFrame, KeyFrame::IdLessThan> setLocalKFs;
    set<PtrKeyFrame, KeyFrame::IdLessThan> setLocalRefKFs;
    set<PtrMapPoint, MapPoint::IdLessThan> setLocalMPs;

    //! 1.获得LocalKFs
    if (static_cast<int>(countKFs()) <= maxN) {
        locker lock(mMutexGlobalGraph);
        setLocalKFs.insert(mspKFs.begin(), mspKFs.end());
    } else {
        setLocalKFs.insert(mCurrentKF);

        // kdtree查找, 利用几何关系找到当前KF附件的KF加入到localKFs中,
        // 否则经过同一个地方不会考虑到之前添加的KF
        int toAdd = maxN - setLocalKFs.size();
        addLocalGraphThroughKdtree(setLocalKFs, toAdd, searchRadius);  // lock global

        // 再根据共视关系, 获得当前KF附近的所有KF, 组成localKFs
        int searchLevel = maxLevel;  // 2
        while (searchLevel > 0) {
            set<PtrKeyFrame, KeyFrame::IdLessThan> currentLocalKFs = setLocalKFs;
            for (auto it = currentLocalKFs.begin(), iend = currentLocalKFs.end(); it != iend; ++it) {
                PtrKeyFrame pKF = (*it);
                if (pKF == nullptr || pKF->isNull())
                    continue;
                vector<PtrKeyFrame> pKFs = pKF->getAllCovisibleKFs();
                setLocalKFs.insert(pKFs.begin(), pKFs.end());
            }
            if (static_cast<int>(setLocalKFs.size()) >= maxN)
                break;
            searchLevel--;
        }
    }
    double t1 = timer.count();

    //! 2. 获得所有LocalMPs, 包括视差不好的
    timer.start();
    for (auto it = setLocalKFs.begin(), iend = setLocalKFs.end(); it != iend; ++it) {
        const PtrKeyFrame pKF = (*it);
        if (!pKF || pKF->isNull())
            continue;

        const vector<PtrMapPoint> pMPs = pKF->getObservations(true, false);
        setLocalMPs.insert(pMPs.begin(), pMPs.end());
    }
    double t2 = timer.count();

    //! 3.获得RefKFs, 在优化时会被固定.
    timer.start();
    for (auto it = setLocalMPs.begin(), iend = setLocalMPs.end(); it != iend; ++it) {
        const PtrMapPoint pMP = (*it);
        if (!pMP || pMP->isNull())
            continue;

        const vector<PtrKeyFrame> pKFs = pMP->getObservations();
        for (auto j = pKFs.begin(), jend = pKFs.end(); j != jend; ++j) {
            if (setLocalKFs.find((*j)) != setLocalKFs.end())
                continue;
            setLocalRefKFs.insert((*j));
        }
    }
    double t3 = timer.count();
    printf("[Local][ Map ] #%ld(KF#%ld) L2.更新局部地图, 总共得到了%ld个LocalKFs, "
           "%ld个LocalMPs和%ld个RefKFs, 共耗时%.2fms\n",
           mCurrentKF->id, mCurrentKF->mIdKF, setLocalKFs.size(), setLocalMPs.size(),
           setLocalRefKFs.size(), t1 + t2 + t3);

    mvLocalGraphKFs = vector<PtrKeyFrame>(setLocalKFs.begin(), setLocalKFs.end());
    mvLocalRefKFs = vector<PtrKeyFrame>(setLocalRefKFs.begin(), setLocalRefKFs.end());
    mvLocalGraphMPs = vector<PtrMapPoint>(setLocalMPs.begin(), setLocalMPs.end());
}

void Map::updateLocalGraph_new(const cv::Mat& pose, int maxLevel, int maxN, float searchRadius)
{
    WorkTimer timer;

    locker lock(mMutexLocalGraph);

    mvLocalGraphKFs.clear();
    mvLocalGraphMPs.clear();
    mvLocalRefKFs.clear();

    set<PtrKeyFrame, KeyFrame::IdLessThan> setLocalKFs;
    set<PtrKeyFrame, KeyFrame::IdLessThan> setLocalRefKFs;
    set<PtrMapPoint, MapPoint::IdLessThan> setLocalMPs;

    //! 1.获得LocalKFs
    if (static_cast<int>(countKFs()) <= maxN) {
        locker lock(mMutexGlobalGraph);
        setLocalKFs.insert(mspKFs.begin(), mspKFs.end());
    } else {
        // 邻域内的KF先加入
        addLocalGraphThroughKdtree_new(setLocalKFs, pose, maxN, searchRadius);  // lock
        // 再保证时间上最新的5帧KF是LocalKFs
        {
            locker lock(mMutexGlobalGraph);
            auto iter = mspKFs.rbegin();
            int k = 5;
            while (k-- > 0)
                setLocalKFs.insert(*iter++);
        }

        // LocalKF不够maxN个则补全
        int toAdd = maxN - setLocalKFs.size();
        if (toAdd > 0) {
            assert(static_cast<int>(setLocalKFs.size()) < maxN);
            int searchLevel = maxLevel;  // 2
            while (searchLevel > 0) {
                set<PtrKeyFrame, KeyFrame::IdLessThan> currentLocalKFs = setLocalKFs;
                for (auto it = currentLocalKFs.begin(), iend = currentLocalKFs.end(); it != iend; ++it) {
                    PtrKeyFrame pKF = (*it);
                    if (pKF == nullptr || pKF->isNull())
                        continue;
                    vector<PtrKeyFrame> pKFs = pKF->getAllCovisibleKFs();
                    setLocalKFs.insert(pKFs.begin(), pKFs.end());
                }
                if (static_cast<int>(setLocalKFs.size()) >= maxN)
                    break;
                searchLevel--;
            }
        }
    }
    double t1 = timer.count();

    //! 2. 获得LocalMPs
    timer.start();
    for (auto it = setLocalKFs.begin(), iend = setLocalKFs.end(); it != iend; ++it) {
        const PtrKeyFrame pKF = (*it);
        if (!pKF || pKF->isNull())
            continue;

        const vector<PtrMapPoint> vpMPs = pKF->getObservations(true, true); // 只要视差好的
        setLocalMPs.insert(vpMPs.begin(), vpMPs.end());
    }
    double t2 = timer.count();

    //! 3.获得RefKFs, 在优化时会被固定.
    timer.start();
    for (auto it = setLocalMPs.begin(), iend = setLocalMPs.end(); it != iend; ++it) {
        const PtrMapPoint pMP = (*it);
        if (!pMP || pMP->isNull())
            continue;

        const vector<PtrKeyFrame> vpKFs = pMP->getObservations();
        for (auto j = vpKFs.begin(), jend = vpKFs.end(); j != jend; ++j) {
            if (setLocalKFs.find((*j)) != setLocalKFs.end())
                continue;
            setLocalRefKFs.insert((*j));
        }
    }
    double t3 = timer.count();
    printf("[Local][ Map ] #%ld(KF#%ld) 更新局部地图, 共得到LocalKFs/LocalRefKFs/LocalMPs(视差良好)数量: "
           "%ld/%ld/%ld, 共耗时%.2fms\n",
           mCurrentKF->id, mCurrentKF->mIdKF, setLocalKFs.size(), setLocalRefKFs.size(),
           setLocalMPs.size(), t1 + t2 + t3);

    mvLocalGraphKFs = vector<PtrKeyFrame>(setLocalKFs.begin(), setLocalKFs.end());
    mvLocalRefKFs = vector<PtrKeyFrame>(setLocalRefKFs.begin(), setLocalRefKFs.end());
    mvLocalGraphMPs = vector<PtrMapPoint>(setLocalMPs.begin(), setLocalMPs.end());
}

/**
 * @brief   回环验证通过后, 将回环的两个关键帧的MP数据融合, 保留旧的MP
 *  此函数在GlobalMapper::VerifyLoopClose()中调用
 * @param mapMatchMP    回环两帧间匹配的MP点对，通过BoW匹配、RANSACN剔除误匹配得到
 * @param pKFCurr       当前KF
 * @param pKFLoop       与当前KF形成回环的KF
 */
void Map::mergeLoopClose(const std::map<int, int>& mapMatchMP, PtrKeyFrame& pKFCurr, PtrKeyFrame& pKFLoop)
{
    assert(pKFCurr->id != pKFLoop->id);

    {
        locker lock1(mMutexLocalGraph);
        locker lock2(mMutexGlobalGraph);

        pKFCurr->addCovisibleKF(pKFLoop);
        pKFLoop->addCovisibleKF(pKFCurr);
    }
    fprintf(stderr, "[ Map ][Info ] Merge loop close between KF#%ld and KF#%ld\n", pKFCurr->mIdKF,
            pKFLoop->mIdKF);

    for (auto iter = mapMatchMP.begin(), itend = mapMatchMP.end(); iter != itend; iter++) {
        const size_t idKPCurr = iter->first;
        const size_t idKPLoop = iter->second;

        if (pKFCurr->hasObservationByIndex(idKPCurr) && pKFLoop->hasObservationByIndex(idKPLoop)) {
            PtrMapPoint pMPCurr = pKFCurr->getObservation(idKPCurr);
            PtrMapPoint pMPLoop = pKFLoop->getObservation(idKPLoop);
            mergeMP(pMPLoop, pMPCurr);  // setnull 会锁住Map的mutex
        }
    }
}

/**
 * @brief 计算KF1观测中同时能被KF2观测到的MPs及相对比例, 主要用于共视KF的添加.
 * @param pKF1  当前KF
 * @param pKF2  LocalKF中的一帧
 * @param spMPs KF1的观测MPs中同时可以被KF2观测到的MPs
 * @return      返回共同观测在两个KF中的总观测的比例
 */
Point2f Map::compareViewMPs(const PtrKeyFrame& pKF1, const PtrKeyFrame& pKF2, set<PtrMapPoint>& spMPs)
{
    spMPs.clear();
    vector<PtrMapPoint> vMPs = pKF1->getObservations(true, false);

    int nSameMP = 0;
    for (auto i = vMPs.begin(), iend = vMPs.end(); i != iend; ++i) {
        PtrMapPoint pMP = *i;
        if (pKF2->hasObservationByPointer(pMP)) {
            spMPs.insert(pMP);
            nSameMP++;
        }
    }

    return Point2f(1.f * nSameMP / pKF1->countObservations(), 1.f * nSameMP / pKF2->countObservations());
}

/**
 * @brief 当前KF观测的MP可以被其它共视KFs同时观测次数>k次的比例, 主要用于冗余KF的判断.
 * @param pKFNow    当前帧KF
 * @param spKFsRef  参考KFs, 即LocalKFs(all covisible KFs)
 * @param spMPsRet  输出具有共同观测的MP集合[out]
 * @param k         共同观测次数, 默认为2
 * @return
 */
float Map::compareViewMPs(const PtrKeyFrame& pKFNow, const vector<PtrKeyFrame>& vpKFsRef,
                          set<PtrMapPoint>& spMPsRet, int k)
{
    if (vpKFsRef.empty())
        return -1.0f;

    const vector<PtrMapPoint> spMPsAll = pKFNow->getObservations(true, true); // TODO
    if (spMPsAll.empty()) {
        return -1.0f;
    }

    spMPsRet.clear();
    for (auto iter = spMPsAll.begin(); iter != spMPsAll.end(); iter++) {
        const PtrMapPoint pMP = *iter;
        if (!pMP || pMP->isNull())
            continue;

        int count = 0;
        for (auto iter2 = vpKFsRef.begin(); iter2 != vpKFsRef.end(); iter2++) {
            PtrKeyFrame pKFRef = *iter2;
            if (pKFRef->hasObservationByPointer(pMP)) {
                count++;

                if (count >= k) {
                    spMPsRet.insert(pMP);
                    break;  //! 在这里break可以减少循环执行次数，加速
                }
            }
        }
    }

    float ratio = spMPsRet.size() * 1.0f / spMPsAll.size();
    return ratio;
}


/**
 * @brief   加载局部地图做一次优化以移除MP外点
 *  在LocalMap::removeOutlierChi2()中调用
 * @param optimizer     优化求解器
 * @param vpEdgesAll    [out]针对每个LocalMP添加的边
 * @param vnAllIdx      [out]针对每个LocalMP添加的边对应的KF节点编号
 */
void Map::loadLocalGraph(SlamOptimizer& optimizer, vector<vector<EdgeProjectXYZ2UV*>>& vpEdgesAll,
                         vector<vector<int>>& vnAllIdx)
{
    WorkTimer timer;
    printf("[ Map ][Info ] #%ld(KF#%ld) 正在加载局部BA以移除离群MP. in LocalMapper::removeOutlierChi2()\n",
           mCurrentKF->id, mCurrentKF->mIdKF);

    int camParaId = 0;
    addCamPara(optimizer, Config::Kcam, camParaId);

    unsigned long maxKFid = 0;
    unsigned long minKFid = 0;

    // If no reference KF, the KF with minId should be fixed
    if (mvLocalRefKFs.empty()) {
        minKFid = (*(mvLocalGraphKFs.begin()))->mIdKF;
        for (auto i = mvLocalGraphKFs.begin(), iend = mvLocalGraphKFs.end(); i != iend; ++i) {
            PtrKeyFrame pKF = *i;
            if (!pKF && pKF->isNull())
                continue;
            if (pKF->mIdKF < minKFid)
                minKFid = pKF->mIdKF;
        }
    }

    const int nLocalKFs = mvLocalGraphKFs.size();
    const int nRefKFs = mvLocalRefKFs.size();

    // Add local graph KeyFrames
    for (int i = 0; i < nLocalKFs; ++i) {
        const PtrKeyFrame& pKF = mvLocalGraphKFs[i];
        assert(!pKF->isNull());

        const int vertexIdKF = i;
        const bool fixed = (pKF->mIdKF == minKFid) || (pKF->mIdKF == 0);

        // 添加位姿节点
        addVertexSE3Expmap(optimizer, toSE3Quat(pKF->getPose()), vertexIdKF, fixed);
        // 添加平面运动约束边
        addEdgeSE3ExpmapPlaneConstraint(optimizer, toSE3Quat(pKF->getPose()), vertexIdKF, Config::Tbc);
    }

    // Add odometry based constraints. 添加基于里程约束的相机位姿边
    for (int idi = 0; idi < nLocalKFs; ++idi) {
        const PtrKeyFrame& pKF = mvLocalGraphKFs[idi];
        assert(!pKF->isNull());

        const PtrKeyFrame pKF1 = pKF->mOdoMeasureFrom.first;
        if (!pKF1 || pKF1->isNull())
            continue;

        auto it = std::find(mvLocalGraphKFs.begin(), mvLocalGraphKFs.end(), pKF1);
        if (it == mvLocalGraphKFs.end())
            continue;

        const int idj = it - mvLocalGraphKFs.begin();  // 这里id1倒是可以跟上面的i对应上
        g2o::Matrix6d info = toMatrix6d(pKF->mOdoMeasureFrom.second.info);
        // id1 - from, i - to
        addEdgeSE3Expmap(optimizer, toSE3Quat(pKF->mOdoMeasureFrom.second.measure), idj, idi, info);
        //addEdgeSE3Expmap(optimizer, toSE3Quat(pKF->mOdoMeasureFrom.second.measure), i, id1, info);
    }

    // Add Reference KeyFrames as fixed. 将RefKFs固定
    for (int i = 0; i < nRefKFs; ++i) {
        const PtrKeyFrame& pKF = mvLocalRefKFs[i];
        assert(!pKF->isNull());

        const int vertexIdKF = i + nLocalKFs;

        addVertexSE3Expmap(optimizer, toSE3Quat(pKF->getPose()), vertexIdKF, true);
    }

    //! 接下来给MP添加节点和边
    maxKFid = nLocalKFs + nRefKFs /* + 1*/;

    // Store xyz2uv edges
    const int N = mvLocalGraphMPs.size();
    vpEdgesAll.clear();
    vpEdgesAll.reserve(N);
    vnAllIdx.clear();
    vnAllIdx.reserve(N);

    const float delta = Config::ThHuber;

    // Add local graph MapPoints
    for (int i = 0; i < N; ++i) {
        const PtrMapPoint& pMP = mvLocalGraphMPs[i];  // FIXME 第0个MP指针为null
        if (!pMP || pMP->isNull())
            continue;

        // 添加MP的节点Vertex
        const int vertexIdMP = maxKFid + i;  // MP的节点编号
        addVertexSBAXYZ(optimizer, toVector3d(pMP->getPos()), vertexIdMP);

        vector<EdgeProjectXYZ2UV*> vpEdgesForMPi;
        vector<int> vNodeIdForMPi;
        if (!pMP->isGoodPrl()) {  // 如果MP视差不好则只加节点而不加边
            vpEdgesAll.push_back(vpEdgesForMPi);
            vnAllIdx.push_back(vNodeIdForMPi);
            continue;
        }

        // 添加MP的边, MP对KF有观测, 则增加一条边
        const vector<PtrKeyFrame> pObsKFs = pMP->getObservations();  // 已可保证得到的都是有效的KF
        for (auto j = pObsKFs.begin(), jend = pObsKFs.end(); j != jend; ++j) {
            PtrKeyFrame pObsKFj = (*j);

            // 确认一下联接关系没有错, 一旦对应关系错了, 优化就不准了
            if (checkAssociationErr(pObsKFj, pMP)) {
                fprintf(stderr, "[ Map ][Warni] removeOutlierChi2() Wrong Association! for KF#%ld "
                                "and MP#%ld \n",
                        pObsKFj->mIdKF, pMP->mId);
                continue;
            }

            // ObsKF必须也是图里的节点Vertex才能加边, 找到它在图里的节点编号
            int vertexIdKF = -1;
            auto it1 = std::find(mvLocalGraphKFs.begin(), mvLocalGraphKFs.end(), pObsKFj);
            if (it1 != mvLocalGraphKFs.end()) {
                vertexIdKF = it1 - mvLocalGraphKFs.begin();
            } else {
                auto it2 = std::find(mvLocalRefKFs.begin(), mvLocalRefKFs.end(), pObsKFj);
                if (it2 != mvLocalRefKFs.end())
                    vertexIdKF = it2 - mvLocalRefKFs.begin() + nLocalKFs;
            }
            if (vertexIdKF == -1)
                continue;

            const int ftrIdx = pMP->getKPIndexInKF(pObsKFj);
            const int octave = pMP->getOctave(pObsKFj);
            const float invSigma2 = pObsKFj->mvInvLevelSigma2[octave];
            const Vector2D uv = toVector2d(pObsKFj->mvKeyPoints[ftrIdx].pt);
            const Matrix2D info = Matrix2D::Identity() * invSigma2;
            EdgeProjectXYZ2UV* ej = addEdgeXYZ2UV(optimizer, uv, vertexIdMP, vertexIdKF, camParaId,
                                                  info, delta);
            ej->setLevel(0);

            // optimizer.addEdge(ei);  //! FIXME addEdgeXYZ2UV()函数里不是添加过了吗？
            vpEdgesForMPi.push_back(ej);
            vNodeIdForMPi.push_back(vertexIdKF);
        }

        vpEdgesAll.push_back(vpEdgesForMPi);
        vnAllIdx.push_back(vNodeIdForMPi);
    }
    printf("[ Map ][Info ] #%ld(KF#%ld) 加载局部BA成功, 耗时: %.2fms. in LocalMapper::removeOutlierChi2()\n",
           mCurrentKF->id, mCurrentKF->mIdKF, timer.count());
    assert(vpEdgesAll.size() == mvLocalGraphMPs.size());
    assert(vnAllIdx.size() == mvLocalGraphMPs.size());
}


/**
 * @brief   加载局部地图做优化, 在LocalMap::localBA()中调用
 *  - Vertices:
 *      - VertexSE2: 机器人位姿顶点, g2o的slam2d中标准类, _estimate为3维SE2类型
 *      - addVertexSBAXYZ: MP的坐标顶点, g2o的sba中标准类, _estimate为3维Vector3D类型
 *
 *  - Edges:
 *      - PreEdgeSE2: KF之间的预积分约束, 自定义类型
 *      - EdgeSE2XYZ: 计算重投影误差, 自定义类型
 *
 * @param optimizer 优化求解器
 */
void Map::loadLocalGraph(SlamOptimizer& optimizer)
{
    locker lock(mMutexLocalGraph);

    int camParaId = 0;
    CamPara* campr = addCamPara(optimizer, Config::Kcam, camParaId);

    const int nLocalKFs = mvLocalGraphKFs.size();
    const int nRefKFs = mvLocalRefKFs.size();

    const int maxVertexIdKF = nLocalKFs + nRefKFs;
    unsigned long minKFid = 0;  // LocalKFs里最小的KFid

    // If no reference KF, the KF with minId should be fixed
    if (nRefKFs == 0) {
//        minKFid = (*(mvLocalGraphKFs.begin()))->mIdKF;  //! FIXME? 原作为id
//        for (auto i = mvLocalGraphKFs.begin(), iend = mvLocalGraphKFs.end(); i != iend; ++i) {
//            PtrKeyFrame pKF = *i;
//            if (!pKF && pKF->isNull())
//                continue;
//            if (pKF->mIdKF < minKFid)
//                minKFid = pKF->mIdKF;
//        }
        if (mvLocalGraphKFs.size() >= 2)
            assert(mvLocalGraphKFs[0]->mIdKF < mvLocalGraphKFs[1]->mIdKF);

        // 目前Local容器都是有序的
        minKFid = mvLocalGraphKFs[0]->mIdKF;
    }

    // Add local graph KeyFrames
    for (int i = 0; i < nLocalKFs; ++i) {
        const PtrKeyFrame& pKFi = mvLocalGraphKFs[i];
        assert(!pKFi->isNull());

        const int vertexIdKF = i;
        const bool fixed = (pKFi->mIdKF == minKFid) || (pKFi->mIdKF == 0);

        const Se2 Twb = pKFi->getTwb();
        const g2o::SE2 pose(Twb.x, Twb.y, Twb.theta);
        addVertexSE2(optimizer, pose, vertexIdKF, fixed);
    }

    // Add odometry based constraints. 添加里程计约束边, 预积分信息
    for (int i = 0; i < nLocalKFs; ++i) {
        const PtrKeyFrame& pKFi = mvLocalGraphKFs[i];
        assert(!pKFi->isNull());

        const PtrKeyFrame pKFj = pKFi->preOdomFromSelf.first;
        if (!pKFj || pKFj->isNull())
            continue;
        assert(pKFj->mIdKF > pKFi->mIdKF); // from是指自己的下一KF

        PreSE2 meas = pKFi->preOdomFromSelf.second;
        auto it = std::find(mvLocalGraphKFs.begin(), mvLocalGraphKFs.end(), pKFj);
        if (it == mvLocalGraphKFs.end())
            continue;

        const int id1 = distance(mvLocalGraphKFs.begin(), it);
        addPreEdgeSE2(optimizer, meas.meas, i, id1, meas.cov.inverse());
    }

    // Add Reference KeyFrames as fixed. 将RefKFs固定
    for (int i = 0; i < nRefKFs; ++i) {
        const PtrKeyFrame& pKF = mvLocalRefKFs[i];
        assert(!pKF->isNull());

        const int vertexIdKF = i + nLocalKFs;

        const Se2 Twb = pKF->getTwb();
        const g2o::SE2 pose(Twb.x, Twb.y, Twb.theta);
        addVertexSE2(optimizer, pose, vertexIdKF, true);
    }


    // Store xyz2uv edges
    const int nMPs = mvLocalGraphMPs.size();
    const float delta = Config::ThHuber;

    // Add local graph MapPoints
    for (int i = 0; i < nMPs; ++i) {
        const PtrMapPoint& pMP = mvLocalGraphMPs[i];
        assert(!pMP->isNull());

        const int vertexIdMP = i + maxVertexIdKF;
        const Vector3D lw = toVector3d(pMP->getPos());
        addVertexSBAXYZ(optimizer, lw, vertexIdMP);

        const vector<PtrKeyFrame> pKFs = pMP->getObservations();
        for (auto j = pKFs.begin(), jend = pKFs.end(); j != jend; ++j) {
            const PtrKeyFrame pKFj = (*j);
            if (checkAssociationErr(pKFj, pMP)) {
                fprintf(stderr, "[ Map ][Warni] localBA() 索引错误! For KF#%ld-%d and MP#%ld-%d\n",
                        pKFj->mIdKF, pKFj->getFeatureIndex(pMP), pMP->mId, pMP->getKPIndexInKF(pKFj));
                continue;
            }

            const int octave = pMP->getOctave(pKFj);
            const int ftrIdx = pMP->getKPIndexInKF(pKFj);
            const float Sigma2 = pKFj->mvLevelSigma2[octave];  // 单层时都是1.0
            const Vector2D uv = toVector2d(pKFj->mvKeyPoints[ftrIdx].pt);

            // 针对当前MPi的某一个观测KFj, 如果KFj在图里(是一个顶点)则给它加上边
            int vertexIdKF = -1;
            auto it1 = std::find(mvLocalGraphKFs.begin(), mvLocalGraphKFs.end(), pKFj);
            if (it1 != mvLocalGraphKFs.end()) {
                vertexIdKF = it1 - mvLocalGraphKFs.begin();
            } else {
                auto it2 = std::find(mvLocalRefKFs.begin(), mvLocalRefKFs.end(), pKFj);
                if (it2 != mvLocalRefKFs.end())
                    vertexIdKF = it2 - mvLocalRefKFs.begin() + nLocalKFs;
            }
            if (vertexIdKF == -1)
                continue;

            // compute covariance/information
            const Matrix2D Sigma_u = Matrix2D::Identity() * Sigma2;
            const Vector3D lc = toVector3d(pKFj->getMPPoseInCamareFrame(ftrIdx));

            const double zc = lc(2);
            const double zc_inv = 1. / zc;
            const double zc_inv2 = zc_inv * zc_inv;
            const float fx = Config::fx;
            Matrix23D J_pi;
            J_pi << fx * zc_inv, 0, -fx * lc(0) * zc_inv2, 0, fx * zc_inv, -fx * lc(1) * zc_inv2;
            const Matrix3D Rcw = toMatrix3d(pKFj->getPose().rowRange(0, 3).colRange(0, 3));
            const Se2 Twb = pKFj->getTwb();
            const Vector3D pi(Twb.x, Twb.y, 0);

            const Matrix23D J_pi_Rcw = J_pi * Rcw;

            const Matrix2D J_rotxy = (J_pi_Rcw * skew(lw - pi)).block<2, 2>(0, 0);
            const Vector2D J_z = -J_pi_Rcw.block<2, 1>(0, 2);
            const float Sigma_rotxy = 1.f / Config::PlaneMotionInfoXrot;
            const float Sigma_z = 1.f / Config::PlaneMotionInfoZ;
            const Matrix2D Sigma_all =
                Sigma_rotxy * J_rotxy * J_rotxy.transpose() + Sigma_z * J_z * J_z.transpose() + Sigma_u;

            addEdgeSE2XYZ(optimizer, uv, vertexIdKF, vertexIdMP, Config::Kcam, toSE3Quat(Config::Tbc),
                          Sigma_all.inverse(), delta);
        }
    }

    size_t nVertices = optimizer.vertices().size();
    size_t nEdges = optimizer.edges().size();
    printf("[ Map ][Info ] #%ld(KF#%ld) 加载LocalGraph: 边数为%ld, 节点数为%ld: LocalKFs/nRefKFs/nMPs = %d/%d/%d\n",
           mCurrentKF->id, mCurrentKF->mIdKF, nEdges, nVertices, nLocalKFs, nRefKFs, nMPs);
}

void Map::loadLocalGraph_test(SlamOptimizer& optimizer)
{
    locker lock(mMutexLocalGraph);

    int camParaId = 0;
    CamPara* campr = addCamPara(optimizer, Config::Kcam, camParaId);

    const int nLocalKFs = mvLocalGraphKFs.size();
    const int nRefKFs = mvLocalRefKFs.size();

    const int maxVertexIdKF = nLocalKFs + nRefKFs;
    unsigned long minKFid = 0;  // LocalKFs里最小的KFid

    // If no reference KF, the KF with minId should be fixed
    if (nRefKFs == 0) {
        if (nLocalKFs >= 2)
            assert(mvLocalGraphKFs[nLocalKFs-2]->mIdKF < mvLocalGraphKFs[nLocalKFs-1]->mIdKF);

        // 目前Local容器都是有序的
        minKFid = mvLocalGraphKFs[0]->mIdKF;
    }

    // Add local graph KeyFrames
    for (int i = 0; i < nLocalKFs; ++i) {
        const PtrKeyFrame& pKFi = mvLocalGraphKFs[i];
        assert(!pKFi->isNull());

        const int vertexIdKF = i;
        const bool fixed = (pKFi->mIdKF == minKFid) || (pKFi->mIdKF == 0);

        const Se2 Twb = pKFi->getTwb();
        const g2o::SE2 pose(Twb.x, Twb.y, Twb.theta);
        addVertexSE2(optimizer, pose, vertexIdKF, fixed);
    }

    // Add odometry based constraints. 添加里程计约束边, 预积分信息
    for (int i = 0; i < nLocalKFs; ++i) {
        const PtrKeyFrame& pKFi = mvLocalGraphKFs[i];
        assert(!pKFi->isNull());

        const PtrKeyFrame pKFj = pKFi->preOdomFromSelf.first;
        if (!pKFj || pKFj->isNull())
            continue;
        assert(pKFj->mIdKF > pKFi->mIdKF); // from是指自己的下一KF

        PreSE2 meas = pKFi->preOdomFromSelf.second;
        auto it = std::find(mvLocalGraphKFs.begin(), mvLocalGraphKFs.end(), pKFj);
        if (it == mvLocalGraphKFs.end())
            continue;

        const int id1 = distance(mvLocalGraphKFs.begin(), it);
        addEdgeSE2(optimizer, meas.meas, i, id1, meas.cov.inverse());
    }

    // Add Reference KeyFrames as fixed. 将RefKFs固定
    for (int i = 0; i < nRefKFs; ++i) {
        const PtrKeyFrame& pKF = mvLocalRefKFs[i];
        assert(!pKF->isNull());

        const int vertexIdKF = i + nLocalKFs;

        const Se2 Twb = pKF->getTwb();
        const g2o::SE2 pose(Twb.x, Twb.y, Twb.theta);
        addVertexSE2(optimizer, pose, vertexIdKF, true);
    }


    // Store xyz2uv edges
    const int nMPs = mvLocalGraphMPs.size();
    const float delta = Config::ThHuber;

    // Add local graph MapPoints
    for (int i = 0; i < nMPs; ++i) {
        const PtrMapPoint& pMP = mvLocalGraphMPs[i];
        assert(!pMP->isNull());

        const int vertexIdMP = i + maxVertexIdKF;
        const Vector3D lw = toVector3d(pMP->getPos());
        addVertexSBAXYZ(optimizer, lw, vertexIdMP);

        const vector<PtrKeyFrame> pKFs = pMP->getObservations();
        for (auto j = pKFs.begin(), jend = pKFs.end(); j != jend; ++j) {
            const PtrKeyFrame pKFj = (*j);
            if (checkAssociationErr(pKFj, pMP)) {
                fprintf(stderr, "[ Map ][Warni] localBA() 索引错误! For KF#%ld-%d and MP#%ld-%d\n",
                        pKFj->mIdKF, pKFj->getFeatureIndex(pMP), pMP->mId, pMP->getKPIndexInKF(pKFj));
                continue;
            }

            const int octave = pMP->getOctave(pKFj);
            const int ftrIdx = pMP->getKPIndexInKF(pKFj);
            const float Sigma2 = pKFj->mvLevelSigma2[octave];  // 单层时都是1.0
            const Vector2D uv = toVector2d(pKFj->mvKeyPoints[ftrIdx].pt);

            // 针对当前MPi的某一个观测KFj, 如果KFj在图里(是一个顶点)则给它加上边
            int vertexIdKF = -1;
            auto it1 = std::find(mvLocalGraphKFs.begin(), mvLocalGraphKFs.end(), pKFj);
            if (it1 != mvLocalGraphKFs.end()) {
                vertexIdKF = it1 - mvLocalGraphKFs.begin();
            } else {
                auto it2 = std::find(mvLocalRefKFs.begin(), mvLocalRefKFs.end(), pKFj);
                if (it2 != mvLocalRefKFs.end())
                    vertexIdKF = it2 - mvLocalRefKFs.begin() + nLocalKFs;
            }
            if (vertexIdKF == -1)
                continue;

            // compute covariance/information
            const Matrix2D Sigma_u = Matrix2D::Identity() * Sigma2;
            const Vector3D lc = toVector3d(pKFj->getMPPoseInCamareFrame(ftrIdx));

            const double zc = lc(2);
            const double zc_inv = 1. / zc;
            const double zc_inv2 = zc_inv * zc_inv;
            const float fx = Config::fx;
            Matrix23D J_pi;
            J_pi << fx * zc_inv, 0, -fx * lc(0) * zc_inv2, 0, fx * zc_inv, -fx * lc(1) * zc_inv2;
            const Matrix3D Rcw = toMatrix3d(pKFj->getPose().rowRange(0, 3).colRange(0, 3));
            const Se2 Twb = pKFj->getTwb();
            const Vector3D pi(Twb.x, Twb.y, 0);

            const Matrix23D J_pi_Rcw = J_pi * Rcw;

            const Matrix2D J_rotxy = (J_pi_Rcw * skew(lw - pi)).block<2, 2>(0, 0);
            const Vector2D J_z = -J_pi_Rcw.block<2, 1>(0, 2);
            const float Sigma_rotxy = 1.f / Config::PlaneMotionInfoXrot;
            const float Sigma_z = 1.f / Config::PlaneMotionInfoZ;
            const Matrix2D Sigma_all =
                Sigma_rotxy * J_rotxy * J_rotxy.transpose() + Sigma_z * J_z * J_z.transpose() + Sigma_u;

            addEdgeSE2XYZ(optimizer, uv, vertexIdKF, vertexIdMP, Config::Kcam, toSE3Quat(Config::Tbc),
                          Sigma_all.inverse(), delta);
        }
    }

    size_t nVertices = optimizer.vertices().size();
    size_t nEdges = optimizer.edges().size();
    printf("[ Map ][Info ] #%ld(KF#%ld) 加载LocalGraph: 边数为%ld, 节点数为%ld: LocalKFs/nRefKFs/nMPs = %d/%d/%d\n",
           mCurrentKF->id, mCurrentKF->mIdKF, nEdges, nVertices, nLocalKFs, nRefKFs, nMPs);
}

//! 此函数没有被调用!
void Map::loadLocalGraphOnlyBa(SlamOptimizer& optimizer, vector<vector<EdgeProjectXYZ2UV*>>& vpEdgesAll,
                               vector<vector<int>>& vnAllIdx)
{
    locker lock(mMutexLocalGraph);

    int camParaId = 0;
    addCamPara(optimizer, Config::Kcam, camParaId);

    unsigned long maxKFid = 0;
    unsigned long minKFid = 0;
    // If no reference KF, the KF with minId should be fixed

    if (mvLocalRefKFs.empty()) {
        minKFid = (*(mvLocalGraphKFs.begin()))->mIdKF;
        for (auto i = mvLocalGraphKFs.begin(), iend = mvLocalGraphKFs.end(); i != iend; ++i) {
            PtrKeyFrame pKF = *i;
            if (pKF->isNull())
                continue;
            if (pKF->mIdKF < minKFid)
                minKFid = pKF->mIdKF;
        }
    }

    const int nLocalKFs = mvLocalGraphKFs.size();
    const int nRefKFs = mvLocalRefKFs.size();

    // Add local graph KeyFrames
    for (int i = 0; i != nLocalKFs; ++i) {
        PtrKeyFrame pKF = mvLocalGraphKFs[i];

        if (pKF->isNull())
            continue;

        int vertexIdKF = i;

        bool fixed = (pKF->mIdKF == minKFid) || pKF->mIdKF == 0;
        addVertexSE3Expmap(optimizer, toSE3Quat(pKF->getPose()), vertexIdKF, fixed);
    }

    // Add Reference KeyFrames as fixed
    for (int i = 0; i != nRefKFs; ++i) {
        PtrKeyFrame pKF = mvLocalRefKFs[i];

        if (pKF->isNull())
            continue;

        int vertexIdKF = i + nLocalKFs;

        addVertexSE3Expmap(optimizer, toSE3Quat(pKF->getPose()), vertexIdKF, true);
    }

    maxKFid = nLocalKFs + nRefKFs /* + 1*/;

    // Store xyz2uv edges
    const int N = mvLocalGraphMPs.size();
    vpEdgesAll.clear();
    vpEdgesAll.reserve(N);
    vnAllIdx.clear();
    vnAllIdx.reserve(N);

    const float delta = Config::ThHuber;

    // Add local graph MapPoints
    for (int i = 0; i != N; ++i) {
        PtrMapPoint pMP = mvLocalGraphMPs[i];
        assert(!pMP->isNull());

        int vertexIdMP = maxKFid + i;

        addVertexSBAXYZ(optimizer, toVector3d(pMP->getPos()), vertexIdMP);

        vector<PtrKeyFrame> pKFs = pMP->getObservations();
        vector<EdgeProjectXYZ2UV*> vpEdges;
        vector<int> vnIdx;

        if (!pMP->isGoodPrl()) {
            vpEdgesAll.push_back(vpEdges);
            vnAllIdx.push_back(vnIdx);
            continue;
        }

        for (auto j = pKFs.begin(), jend = pKFs.end(); j != jend; ++j) {
            PtrKeyFrame pKF = (*j);
            if (checkAssociationErr(pKF, pMP)) {
                fprintf(stderr, "[ Map ][Warni] loadLocalGraphOnlyBa() Wrong Association! for "
                                "KF#%ld and MP#%ld \n",
                        pKF->mIdKF, pMP->mId);
                continue;
            }

            const int ftrIdx = pMP->getKPIndexInKF(pKF);
            const int octave = pMP->getOctave(pKF);
            const float invSigma2 = pKF->mvInvLevelSigma2[octave];
            const Vector2D uv = toVector2d(pKF->mvKeyPoints[ftrIdx].pt);
            const Matrix2D info = Matrix2D::Identity() * invSigma2;

            int vertexIdKF = -1;

            auto it1 = std::find(mvLocalGraphKFs.begin(), mvLocalGraphKFs.end(), pKF);
            if (it1 != mvLocalGraphKFs.end()) {
                vertexIdKF = it1 - mvLocalGraphKFs.begin();
            } else {
                auto it2 = std::find(mvLocalRefKFs.begin(), mvLocalRefKFs.end(), pKF);
                if (it2 != mvLocalRefKFs.end())
                    vertexIdKF = it2 - mvLocalRefKFs.begin() + nLocalKFs;
            }

            if (vertexIdKF == -1)
                continue;

            EdgeProjectXYZ2UV* ei = addEdgeXYZ2UV(optimizer, uv, vertexIdMP, vertexIdKF, camParaId, info, delta);
            ei->setLevel(0);

            optimizer.addEdge(ei);
            vpEdges.push_back(ei);

            vnIdx.push_back(vertexIdKF);
        }

        vpEdgesAll.push_back(vpEdges);
        vnAllIdx.push_back(vnIdx);
    }
    assert(vpEdgesAll.size() == mvLocalGraphMPs.size());
    assert(vnAllIdx.size() == mvLocalGraphMPs.size());
}

/**
 * @brief   去除离群MP的联接关系，观测小于2的MP会被删除
 * 在LocalMapper::removeOutlierChi2()中调用，
 * @param vnOutlierIdxAll   所有离群MP的对应的边KF节点编号, 需要根据这个编号取消KF对此MP的观测
 * @return  小于2帧观测数的离群MP数量
 */
int Map::removeLocalOutlierMP(SlamOptimizer& optimizer)
{
    WorkTimer timer;
    std::vector<PtrKeyFrame> vLocalGraphKFs;
    std::vector<PtrMapPoint> vLocalGraphMPs;
    std::vector<PtrKeyFrame> vLocalRefKFs;

    // 所有离群MP边
    vector<vector<g2o::EdgeProjectXYZ2UV*>> vpEdgesAllForLocalMPs;
    // 所有离群MP的对应的边KF节点编号, 需要根据这个编号取消KF对此MP的观测
    vector<vector<int>> vNodeIdxAllForLocalMPs;

    {
        locker lock(mMutexLocalGraph);

        loadLocalGraph(optimizer, vpEdgesAllForLocalMPs, vNodeIdxAllForLocalMPs);
        optimizer.initializeOptimization(0);
        optimizer.optimize(Config::LocalIterNum);
        printf("[ Map ][Info ] #%ld(KF#%ld) 移除离群点前优化成功! 耗时%.2fms. 正在对%ld个局部MP计算内点...\n",
               mCurrentKF->id, mCurrentKF->mIdKF, timer.count(), vpEdgesAllForLocalMPs.size());

        vLocalGraphKFs = mvLocalGraphKFs;
        vLocalGraphMPs = mvLocalGraphMPs;
        vLocalRefKFs = mvLocalRefKFs;
    }
    assert(vNodeIdxAllForLocalMPs.size() == vLocalGraphMPs.size());


    // 2.提取离群MP
    timer.start();
    const double chi2 = 25;
    const size_t nAllMPs = vpEdgesAllForLocalMPs.size();
    vector<vector<int>> vnOutlierIdxAll;
    int nToRemoved = 0;
    for (size_t i = 0; i != nAllMPs; ++i) {
        vector<int> vnOutlierIdx;
        const size_t nEdgesMPi = vpEdgesAllForLocalMPs[i].size();
        vnOutlierIdx.reserve(nEdgesMPi);
        for (size_t j = 0; j < nEdgesMPi; ++j) {
            EdgeProjectXYZ2UV*& eij = vpEdgesAllForLocalMPs[i][j];

            if (eij->level() > 0)  // 已经处理过了跳过
                continue;

            eij->computeError();
            const bool chi2Bad = eij->chi2() > chi2;
            if (chi2Bad) {
                eij->setLevel(1);
                const int& idKF = vNodeIdxAllForLocalMPs[i][j];
                vnOutlierIdx.push_back(idKF);
                nToRemoved++;
            }
        }

        vnOutlierIdxAll.push_back(vnOutlierIdx);
    }
    printf("[ Map ][Info ] #%ld(KF#%ld) 提取离群MP数量: %d, 耗时%.2fms. 即将取消离群MP的观测关系...\n",
           mCurrentKF->id, mCurrentKF->mIdKF, nToRemoved, timer.count());

    // 3.去除离群MP
    const int nLocalKFs = vLocalGraphKFs.size();
    const int nLocalMPs = vLocalGraphMPs.size();
    int nBadMP = 0;
    for (int i = 0, iend = nLocalMPs; i != iend; ++i) {
        PtrMapPoint& pMP = vLocalGraphMPs[i];
        if (!pMP || pMP->isNull())
            continue;

        for (int j = 0, jend = vnOutlierIdxAll[i].size(); j < jend; ++j) {
            PtrKeyFrame pKF = nullptr;
            const int idxKF = vnOutlierIdxAll[i][j];

            if (idxKF < nLocalKFs) {
                pKF = vLocalGraphKFs[idxKF];
            } else {
                pKF = vLocalRefKFs[idxKF - nLocalKFs];
            }

            if (pKF == nullptr) {
                printf("!! ERR MP: KF In Outlier Edge Not In LocalKF or RefKF! at line number %d "
                       "in file %s\n",
                       __LINE__, __FILE__);
                exit(-1);
                continue;
            }

            assert(pKF->hasObservationByPointer(pMP));
            //if (!pKF->hasObservationByPointer(pMP))
            //    continue;


            if (checkAssociationErr(pKF, pMP)) {
                printf("!! ERR MP: Wrong Association [Outlier removal] ! at line number %d in file "
                       "%s\n",
                       __LINE__, __FILE__);
                exit(-1);
                continue;
            }

            pKF->eraseObservationByPointer(pMP);
            pMP->eraseObservation(pKF);  // 观测为0时会自动setNull(). lock Map::LocalMutex
        }

        if (pMP->isNull())
            nBadMP++;
    }

    printf("[INFO] #%ld(KF#%ld) MP外点提取数/实际移除(析构)数为: %d/%d个, 耗时: %.2fms\n",
           mCurrentKF->id, mCurrentKF->mIdKF, nToRemoved, nBadMP, timer.count());

    vpEdgesAllForLocalMPs.clear();
    vNodeIdxAllForLocalMPs.clear();

    return nBadMP;
}

/**
 * @brief   用优化后的结果更新KFs和MPs
 * 在LocalMapper::localBA()里调用
 * @param optimizer
 */
void Map::optimizeLocalGraph(SlamOptimizer& optimizer)
{
    locker lock1(mMutexLocalGraph);
    locker lock2(mMutexGlobalGraph);

    const size_t nLocalKFs = mvLocalGraphKFs.size();
    const size_t nRefKFs = mvLocalRefKFs.size();
    const size_t N = mvLocalGraphMPs.size();
    const size_t maxKFid = nLocalKFs + nRefKFs;

    for (size_t i = 0; i != nLocalKFs; ++i) {
        PtrKeyFrame& pKF = mvLocalGraphKFs[i];
        if (pKF->isNull())
            continue;
        const Vector3D vp = estimateVertexSE2(optimizer, i).toVector();
        pKF->setPose(Se2(vp(0), vp(1), vp(2)));
    }

    for (size_t j = 0; j != N; ++j) {
        PtrMapPoint& pMP = mvLocalGraphMPs[j];
        if (pMP->isNull() || !pMP->isGoodPrl())
            continue;

        Point3f pos = toCvPt3f(estimateVertexSBAXYZ(optimizer, j + maxKFid));
        pMP->setPos(pos);
        // pMP->updateMeasureInKFs(); // 现在在MapPoint::setPos()后会自动调用
    }
}

//! 新添的KF在关联MP和生成新的MP之后，如果与LocalKFs的共同MP观测数超过自身观测总数的30%，则为他们之间添加共视关系
//! 在LocalMap::addNewKF()里调用
void Map::updateCovisibility(const PtrKeyFrame& pNewKF)
{
    locker lock1(mMutexLocalGraph);
    locker lock2(mMutexGlobalGraph);

    //    printf("[ Map ][Info ] #%ld(KF#%ld) 正在更新共视关系... Local KFs数量为: %ld\n",
    //           pNewKF->id, pNewKF->mIdKF, mvLocalGraphKFs.size());
    for (auto i = mvLocalGraphKFs.begin(), iend = mvLocalGraphKFs.end(); i != iend; ++i) {
        set<PtrMapPoint> spMPs;
        PtrKeyFrame pKFi = *i;
        Point2f ratios = compareViewMPs(pNewKF, pKFi, spMPs);
        if (ratios.x > 0.25 || ratios.y > 0.25) {
            pNewKF->addCovisibleKF(pKFi);
            pKFi->addCovisibleKF(pNewKF);
            printf("[ Map ][Info ] #%ld(KF#%ld) 和(KF#%ld)添加了共视关系, 共视比例为%.2f, %.2f.\n",
                   pNewKF->id, pNewKF->mIdKF, pKFi->mIdKF, ratios.x, ratios.y);
        }
    }
}

bool Map::checkAssociationErr(const PtrKeyFrame& pKF, const PtrMapPoint& pMP)
{
    const int ftrIdx0 = pKF->getFeatureIndex(pMP);
    const int ftrIdx1 = pMP->getKPIndexInKF(pKF);
    if (ftrIdx0 != ftrIdx1 || ftrIdx0 == -1 || ftrIdx1 == -1) {
        fprintf(stderr, "!! checkAssociationErr: pKF->pMP / pMP->pKF: %d / %d\n", ftrIdx0, ftrIdx1);
        return true;
    }
    return false;
}

/**
 * @brief   提取特征图匹配对, 创建KF之间的约束
 * @param _pKF  当前帧
 * @return      返回当前帧和即在共视图又在特征图里的KF的匹配对
 */
vector<pair<PtrKeyFrame, PtrKeyFrame>> Map::SelectKFPairFeat(const PtrKeyFrame& _pKF)
{
    vector<pair<PtrKeyFrame, PtrKeyFrame>> _vKFPairs;

    // Smallest distance between KFs in covis-graph to create a new feature edge
    int threshCovisGraphDist = 5;

    set<PtrKeyFrame> sKFSelected;
    vector<PtrKeyFrame> vCovisKFs = _pKF->getAllCovisibleKFs();
    set<PtrKeyFrame> sKFLocal = GlobalMapper::getAllConnectedKFs_nLayers(_pKF, threshCovisGraphDist, sKFSelected);

    // 共视图里的KF如果在特征图里,就加入到sKFSelected里,然后会更新特征图
    for (auto iter = vCovisKFs.begin(); iter != vCovisKFs.end(); iter++) {
        const PtrKeyFrame _pKFCand = *iter;
        if (sKFLocal.count(_pKFCand) == 0) {
            sKFSelected.insert(*iter);
            sKFLocal = GlobalMapper::getAllConnectedKFs_nLayers(_pKF, threshCovisGraphDist, sKFSelected);
        } else {
            continue;
        }
    }

    _vKFPairs.clear();
    for (auto iter = sKFSelected.begin(); iter != sKFSelected.end(); iter++) {
        _vKFPairs.push_back(make_pair(_pKF, *iter));
    }

    return _vKFPairs;
}

/**
 * @brief 更新特征图
 * FIXME 目前特征图约束添加大部分是失败的, 待验证
 * @param _pKF  当前帧KF
 * @return      返回是否有更新的标志
 */
bool Map::updateFeatGraph(const PtrKeyFrame& _pKF)
{
    locker lock1(mMutexLocalGraph);
    locker lock2(mMutexGlobalGraph);

    vector<pair<PtrKeyFrame, PtrKeyFrame>> _vKFPairs = SelectKFPairFeat(_pKF);

    if (_vKFPairs.empty())
        return false;

    int numPairKFs = _vKFPairs.size();
    for (int i = 0; i != numPairKFs; ++i) {
        pair<PtrKeyFrame, PtrKeyFrame> pairKF = _vKFPairs[i];
        PtrKeyFrame ptKFFrom = pairKF.first;  //? 这里始终都是当前帧_pKF啊???
        PtrKeyFrame ptKFTo = pairKF.second;
        SE3Constraint ftrCnstr;

        assert(ptKFFrom->mIdKF == _pKF->mIdKF);

        //! NOTE 为KF对添加特征图约束,这里对应论文里的SE3XYZ约束??
        if (GlobalMapper::createFeatEdge(ptKFFrom, ptKFTo, ftrCnstr) == 0) {
            ptKFFrom->addFtrMeasureFrom(ptKFTo, ftrCnstr.measure, ftrCnstr.info);
            ptKFTo->addFtrMeasureTo(ptKFFrom, ftrCnstr.measure, ftrCnstr.info);
            if (Config::GlobalPrint) {
                fprintf(stderr, "\n\n[ Map ][Info ] #%ld(KF#%ld) Add feature constraint from "
                                "KF#%ld to KF#%ld!\n\n",
                        _pKF->id, _pKF->mIdKF, ptKFFrom->mIdKF, ptKFTo->mIdKF);
            }
        } else {
            if (Config::GlobalPrint)
                fprintf(stderr, "[ Map ][Error] #%ld(KF#%ld) Add feature constraint failed!\n",
                        _pKF->id, _pKF->mIdKF);
        }
    }

    return true;
}

size_t Map::addLocalGraphThroughKdtree(set<PtrKeyFrame, KeyFrame::IdLessThan>& setLocalKFs, int maxN,
                                     float searchRadius)
{
    size_t nNewAdd = 0;

    const vector<PtrKeyFrame> vKFsAll = getAllKFs();  // lock global
    vector<Point3f> vKFPoses(vKFsAll.size());
    for (size_t i = 0, iend = vKFsAll.size(); i != iend; ++i) {
        Mat Twc = cvu::inv(vKFsAll[i]->getPose());
        Point3f posei(Twc.at<float>(0, 3) * 0.001f, Twc.at<float>(1, 3) * 0.001f, Twc.at<float>(2, 3) * 0.001f);
        vKFPoses[i] = posei;
    }

    cv::flann::KDTreeIndexParams kdtreeParams;
    cv::flann::Index kdtree(Mat(vKFPoses).reshape(1), kdtreeParams);

    const Mat pose = cvu::inv(mCurrentKF->getPose());
    std::vector<float> query = {pose.at<float>(0, 3) * 0.001f, pose.at<float>(1, 3) * 0.001f,
                                pose.at<float>(2, 3) * 0.001f};
    std::vector<int> indices;
    std::vector<float> dists;
    kdtree.radiusSearch(query, indices, dists, searchRadius, maxN, cv::flann::SearchParams());
    for (size_t i = 0, iend = indices.size(); i != iend; ++i) {
        if (indices[i] > 0 && dists[i] < searchRadius) { // 距离在0.3m以内
            setLocalKFs.insert(vKFsAll[indices[i]]);
            nNewAdd++;
        }
    }
    return nNewAdd;
}

size_t Map::addLocalGraphThroughKdtree_new(set<PtrKeyFrame, KeyFrame::IdLessThan>& setLocalKFs,
                                           const Mat& pose, int maxN, float searchRadius)
{
    size_t nNewAdd = 0;

    const vector<PtrKeyFrame> vKFsAll = getAllKFs();  // lock global
    vector<Point3f> vKFPoses(vKFsAll.size());
    for (size_t i = 0, iend = vKFsAll.size(); i != iend; ++i) {
        Mat Twc = cvu::inv(vKFsAll[i]->getPose());
        Point3f posei(Twc.at<float>(0, 3) * 0.001f, Twc.at<float>(1, 3) * 0.001f,
                      Twc.at<float>(2, 3) * 0.001f);
        vKFPoses[i] = posei;
    }

    cv::flann::KDTreeIndexParams kdtreeParams;
    cv::flann::Index kdtree(Mat(vKFPoses).reshape(1), kdtreeParams);

    const Mat Twc = cvu::inv(pose);  // Twc
    std::vector<float> query = {Twc.at<float>(0, 3) * 0.001f, Twc.at<float>(1, 3) * 0.001f,
                                Twc.at<float>(2, 3) * 0.001f};
    std::vector<int> indices;
    std::vector<float> dists;
    kdtree.radiusSearch(query, indices, dists, searchRadius, maxN, cv::flann::SearchParams());
    for (size_t i = 0, iend = indices.size(); i != iend; ++i) {
        if (indices[i] > 0 && dists[i] < searchRadius) {
            setLocalKFs.insert(vKFsAll[indices[i]]);
            nNewAdd++;
        }
    }
    return nNewAdd;
}

}  // namespace se2lam
