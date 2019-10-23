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

namespace se2lam
{

using namespace cv;
using namespace std;
using namespace Eigen;
using namespace g2o;

typedef unique_lock<mutex> locker;

Map::Map() : mCurrentKF(nullptr), isEmpty(true), mpLocalMapper(nullptr)
{
    mCurrentFramePose = cv::Mat::eye(4, 4, CV_32FC1);
}

Map::~Map()
{}

void Map::insertKF(PtrKeyFrame& pKF)
{
    locker lock(mMutexGlobalGraph);
    printf("[ Map ][Info ] #%ld(KF#%ld) 当前新KF插入到地图里了!\n", pKF->id, pKF->mIdKF);
    pKF->setMap(this);
    mspKFs.insert(pKF);
    isEmpty = false;
    mCurrentKF = pKF;
}

void Map::insertMP(PtrMapPoint& pMP)
{
    locker lock(mMutexGlobalGraph);
    assert(pMP->countObservations() > 0);
    pMP->setMap(this);
    mspMPs.insert(pMP);
}

//! 未使用函数
void Map::eraseKF(const PtrKeyFrame& pKF)
{
    locker lock(mMutexGlobalGraph);
    mspKFs.erase(pKF);

    locker lock2(mMutexLocalGraph);
    auto iter1 = find(mvLocalGraphKFs.begin(), mvLocalGraphKFs.end(), pKF);
    if (iter1 != mvLocalGraphKFs.end())
        mvLocalGraphKFs.erase(iter1);
    auto iter2 = find(mvRefKFs.begin(), mvRefKFs.end(), pKF);
    if (iter2 != mvRefKFs.end())
        mvRefKFs.erase(iter2);
}

//! 未使用函数
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
    std::set<PtrKeyFrame> pKFs = toKeep->getObservations();
    for (auto it = pKFs.begin(), itend = pKFs.end(); it != itend; ++it) {
        PtrKeyFrame pKF = *it;
        //!@Vance: 有一KF能同时观测到这两个MP就返回，Why?
        if (pKF->hasObservation(toKeep) && pKF->hasObservation(toDelete)) {
            cerr << "[ Map ][Warni] Return for a kF(toKeep) has same observation." << endl;
            return;  //! TODO 应该是break？
        }
    }
    pKFs = toDelete->getObservations();
    for (auto it = pKFs.begin(), itend = pKFs.end(); it != itend; ++it) {
        PtrKeyFrame pKF = *it;
        if (pKF->hasObservation(toKeep) && pKF->hasObservation(toDelete)) {
            cerr << "[ Map ][Warni] Return for a kF(toDelete) has same observation." << endl;
            return;  //! TODO 应该是break？
        }
    }

    toDelete->mergedInto(toKeep);
    mspMPs.erase(toDelete);
    fprintf(stderr, "[ Map ][Info ] Have a merge between #%ld(keep) and #%ld(delete) MP.\n", toKeep->mId,
            toDelete->mId);

    // 还要更新局部的MP
    auto it = std::find(mvLocalGraphMPs.begin(), mvLocalGraphMPs.end(), toDelete);
    if (it != mvLocalGraphMPs.end())
        *it = toKeep;
}

vector<PtrKeyFrame> Map::getAllKF()
{
    locker lock(mMutexGlobalGraph);
    return vector<PtrKeyFrame>(mspKFs.begin(), mspKFs.end());
}

vector<PtrMapPoint> Map::getAllMP()
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
    return vector<PtrKeyFrame>(mvRefKFs.begin(), mvRefKFs.end());
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

void Map::clear()
{
    locker lock1(mMutexGlobalGraph);
    locker lock2(mMutexLocalGraph);
    mspKFs.clear();
    mspMPs.clear();
    mvLocalGraphKFs.clear();
    mvLocalGraphMPs.clear();
    mvRefKFs.clear();
    isEmpty = true;
    KeyFrame::mNextIdKF = 1;
    MapPoint::mNextId = 1;
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
 * LocalKFs和LocalMPs可以保证按id升序排序
 *
 * @return 返回是否有经过修剪的标志
 */
bool Map::pruneRedundantKF()
{
    printf("[Local][Info ] #%ld(KF#%ld) [Map]正在修剪冗余KF...\n",
           mCurrentKF->id, mCurrentKF->mIdKF);

    std::vector<PtrMapPoint> vLocalGraphMPs;
    std::vector<PtrKeyFrame> vLocalGraphKFs;

    {
        locker lock1(mMutexLocalGraph);
        vLocalGraphMPs = mvLocalGraphMPs;
        vLocalGraphKFs = mvLocalGraphKFs;
    }

    if (vLocalGraphKFs.size() <= 3)
        return false;

    bool pruned = false;
    int prunedIdxLocalKF = -1;

    vector<int> goodIdxLocalKF;
    vector<int> goodIdxLocalMP;
    size_t nKFsBefore = vLocalGraphKFs.size();
    size_t nMPsBefore = vLocalGraphMPs.size();

    //! FIXME 每次修剪都从0开始重新计算，LocalMapper重复调用时会有很多重复计算
    for (int i = 0, iend = vLocalGraphKFs.size(); i != iend; ++i) {
        bool prunedThis = false;
        if (!pruned) {  //! FIXME 这使得一次只会去除1帧
            PtrKeyFrame thisKF = vLocalGraphKFs[i];

            // 首帧和当前帧不能被修剪
            if (thisKF->isNull() || mCurrentKF->mIdKF == thisKF->mIdKF || thisKF->mIdKF == 1)
                continue;

            // Count MPs in thisKF observed by covKFs 2 times
            //! 共视KF中观测到本KF中MP两次以上的比例大于80%则视此KF为冗余的
            set<PtrMapPoint> spMPs;
            set<PtrKeyFrame> covKFs = thisKF->getAllCovisibleKFs();
            float ratio = compareViewMPs(thisKF, covKFs, spMPs, 2);
            bool bIsThisKFRedundant = (ratio >= 0.7f);  //! orig: 0.8

            // Do Prune if pass threashold test
            //! Local KF中的一帧thisKF要被修剪，则需要更新它们的联接关系
            if (bIsThisKFRedundant) {
                PtrKeyFrame lastKF = thisKF->mOdoMeasureTo.first;
                PtrKeyFrame nextKF = thisKF->mOdoMeasureFrom.first;

                // 这两句多余
                bool bIsInitKF = (thisKF->mIdKF == 1);
                bool bHasFeatEdge = (thisKF->mFtrMeasureFrom.size() != 0);

                // Prune this KF and link a new odometry constrait
                //! 非首尾帧
                if (lastKF && nextKF && !bIsInitKF && !bHasFeatEdge) {
                    Se2 dOdoLastThis = thisKF->odom - lastKF->odom;
                    Se2 dOdoThisNext = nextKF->odom - thisKF->odom;

                    double theshl = 10000;                 // 10m
                    double thesht = 45 * 3.1415926 / 180;  // 45 degree to rad

                    double dl1 = sqrt(dOdoLastThis.x * dOdoLastThis.x + dOdoLastThis.y * dOdoLastThis.y);
                    double dt1 = abs(dOdoLastThis.theta);
                    double dl2 = sqrt(dOdoThisNext.x * dOdoThisNext.x + dOdoThisNext.y * dOdoThisNext.y);
                    double dt2 = abs(dOdoThisNext.theta);

                    //! 被修剪帧的前后帧之间位移不能超过10m，角度不能超过45°，防止修剪掉在大旋转大平移之间的KF
                    if (dl1 < theshl && dl2 < theshl && dt1 < thesht && dt2 < thesht) {
//                        mspKFs.erase(thisKF);
                        thisKF->setNull(thisKF);
                        fprintf(stderr, "[ Map ][Info ] KF#%ld 此KF被修剪, 引用计数 = %ld\n",
                               thisKF->mIdKF, thisKF.use_count());

                        // 给前后帧之间添加共视关联和约束
                        Mat measure;
                        g2o::Matrix6d info;
                        Track::calcOdoConstraintCam(nextKF->odom - lastKF->odom, measure, info);
                        nextKF->setOdoMeasureTo(lastKF, measure, toCvMat6f(info));
                        lastKF->setOdoMeasureFrom(nextKF, measure, toCvMat6f(info));

                        nextKF->addCovisibleKF(lastKF);
                        lastKF->addCovisibleKF(nextKF);

                        pruned = true;
                        prunedThis = true;
                        prunedIdxLocalKF = i;
                    }
                }
            }
        }

        if (!prunedThis) {
            goodIdxLocalKF.push_back(i);
        }
    }

    //! Remove useless MP. 去除无用的MP(标记了null)
    //! NOTE 对KF的修剪基本不会影响到MP, 只是取消了对冗余KF的关联. 但MP观测数为0时会被置为null.
    if (pruned) {
        locker lock2(mMutexGlobalGraph);
        for (int i = 0, iend = vLocalGraphMPs.size(); i != iend; ++i) {
            PtrMapPoint pMP = vLocalGraphMPs[i];
            if (pMP->isNull()) {
                mspMPs.erase(pMP);
            } else {
                goodIdxLocalMP.push_back(i);
            }
        }
    }

    // 修剪后更新mLocalGraphMPs和mLocalGraphKFs
    if (pruned) {
        vector<PtrMapPoint> vpMPs;
        vector<PtrKeyFrame> vpKFs;
        vpMPs.reserve(goodIdxLocalMP.size());
        vpKFs.reserve(goodIdxLocalKF.size());

        for (int i = 0, iend = goodIdxLocalKF.size(); i != iend; ++i) {
            vpKFs.push_back(vLocalGraphKFs[goodIdxLocalKF[i]]);
        }

        for (int i = 0, iend = goodIdxLocalMP.size(); i != iend; ++i) {
            vpMPs.push_back(vLocalGraphMPs[goodIdxLocalMP[i]]);
        }

        locker lock3(mMutexLocalGraph);
        std::swap(vpMPs, mvLocalGraphMPs);
        std::swap(vpKFs, mvLocalGraphKFs);
    }

    printf("[Local][Info ] #%ld(KF#%ld) [Map]LocalMap 修剪量: KFs: %ld, Mps: %ld\n", mCurrentKF->id,
           mCurrentKF->mIdKF, nKFsBefore - mvLocalGraphKFs.size(),
           nMPsBefore - mvLocalGraphMPs.size());
    return pruned;
}

/**
 * @brief 更新局部地图的KFs与MPs, 以及参考KFs
 * 此函数在LocalMapper::run()函数中调用, mLocalGraphKFs, mRefKFs, mLocalGraphMPs在此更新
 */
void Map::updateLocalGraph(int maxLevel, int maxN, float searchRadius)
{
    locker lock(mMutexLocalGraph);

    WorkTimer timeCounter;

    mvLocalGraphKFs.clear();
    mvRefKFs.clear();
    mvLocalGraphMPs.clear();

    std::set<PtrKeyFrame> setLocalKFs;
    std::set<PtrKeyFrame> setRefKFs;
    std::set<PtrMapPoint> setLocalMPs;

    setLocalKFs.insert(mCurrentKF);

    //! kdtree查找, 利用几何关系找到当前KF附件的KF加入到localKFs中,
    //! 否则经过同一个地方不会考虑到之前添加的KF
    WorkTimer timer;
    addLocalGraphThroughKdtree(setLocalKFs, maxN * 0.5, searchRadius);
    size_t s1 = setLocalKFs.size();
    printf("[Local][Info ] #%ld(KF#%ld) [Map]使用FLANN搜索确定了%ld个Local KFs, 耗时%.2fms\n",
           mCurrentKF->id, mCurrentKF->mIdKF, s1 - 1, timer.count());

    //! 再根据共视关系, 获得当前KF附近的所有KF, 组成localKFs
    timer.start();
    int searchLevel = maxLevel;  // 3
    while (searchLevel > 0) {
        std::set<PtrKeyFrame> currentLocalKFs = setLocalKFs;
        for (auto i = currentLocalKFs.begin(), iend = currentLocalKFs.end(); i != iend; ++i) {
            PtrKeyFrame pKF = (*i);
            std::set<PtrKeyFrame> pKFs = pKF->getAllCovisibleKFs();
            setLocalKFs.insert(pKFs.begin(), pKFs.end());
        }
        searchLevel--;
    }
    size_t s2 = setLocalKFs.size();
    printf("[Local][Info ] #%ld(KF#%ld) [Map]根据共视关系递归%d层搜索确定了%ld个Local KFs, 耗时%.2fms\n",
           mCurrentKF->id, mCurrentKF->mIdKF, maxLevel, s2 - s1, timer.count());

    //! 获得localKFs的所有MPs, 不要求要有良好视差
    //! 如果要求要有良好视差, 则刚开始时视差都是差的, 会导致后面localMPs数量一直为0
    timer.start();
    for (auto i = setLocalKFs.begin(), iend = setLocalKFs.end(); i != iend; ++i) {
        PtrKeyFrame pKF = *i;
        bool checkPrl = false;
        set<PtrMapPoint> pMPs = pKF->getAllObsMPs(checkPrl);
        setLocalMPs.insert(pMPs.begin(), pMPs.end());
    }
    size_t s3 = setLocalMPs.size();
    printf("[Local][Info ] #%ld(KF#%ld) [Map]根据%ld个Local KFs添加了%ld个Local MPs, 耗时%.2fms\n",
           mCurrentKF->id, mCurrentKF->mIdKF, s2, s3, timer.count());

    //!@Vance: 获得refKFs
    timer.start();
    for (auto i = setLocalMPs.begin(), iend = setLocalMPs.end(); i != iend; ++i) {
        PtrMapPoint pMP = (*i);
        std::set<PtrKeyFrame> pKFs = pMP->getObservations();
        // 和Local KFs有共同的Local MPs观测的其他KF，将设置为RefKFs, 在优化时会被固定.
        for (auto j = pKFs.begin(), jend = pKFs.end(); j != jend; ++j) {
            if (setLocalKFs.find((*j)) != setLocalKFs.end() ||
                setRefKFs.find((*j)) != setRefKFs.end())
                continue;
            setRefKFs.insert((*j));
        }
    }
    size_t s4 = setRefKFs.size();
    printf("[Local][Info ] #%ld(KF#%ld) [Map]根据%ld个Local MPs添加了%ld个Reference KFs, 耗时%.2fms\n",
           mCurrentKF->id, mCurrentKF->mIdKF, s3, s4, timer.count());

    mvLocalGraphKFs = vector<PtrKeyFrame>(setLocalKFs.begin(), setLocalKFs.end());
    mvRefKFs = vector<PtrKeyFrame>(setRefKFs.begin(), setRefKFs.end());
    mvLocalGraphMPs = vector<PtrMapPoint>(setLocalMPs.begin(), setLocalMPs.end());

    printf("[Local][Info ] #%ld(KF#%ld) [Map]更新局部地图, 总共得到了%ld个LocalKFs, %ld个LocalMPs和 %ld个RefKFs, 共耗时%.2fms\n",
           mCurrentKF->id, mCurrentKF->mIdKF, mvLocalGraphKFs.size(), mvLocalGraphMPs.size(),
           mvRefKFs.size(), timeCounter.count());
}

/**
 * @brief   回环验证通过后, 将回环的两个关键帧的MP数据融合, 保留旧的MP
 *  此函数在GlobalMapper::VerifyLoopClose()中调用
 * @param mapMatchMP    回环两帧间匹配的MP点对，通过BoW匹配、RANSACN剔除误匹配得到
 * @param pKFCurr       当前KF
 * @param pKFLoop       与当前KF形成回环的KF
 */
void Map::mergeLoopClose(const std::map<int, int>& mapMatchMP, PtrKeyFrame& pKFCurr,
                         PtrKeyFrame& pKFLoop)
{
    {
        locker lock1(mMutexLocalGraph);
        locker lock2(mMutexGlobalGraph);

        pKFCurr->addCovisibleKF(pKFLoop);
        pKFLoop->addCovisibleKF(pKFCurr);
    }
    fprintf(stderr, "[ Map ][Info ] Merge loop close between KF#%ld and KF#%ld\n", pKFCurr->mIdKF,
            pKFLoop->mIdKF);

    for (auto iter = mapMatchMP.begin(), itend = mapMatchMP.end(); iter != itend; iter++) {
        size_t idKPCurr = iter->first;
        size_t idKPLoop = iter->second;

        if (pKFCurr->hasObservation(idKPCurr) && pKFLoop->hasObservation(idKPLoop)) {
            PtrMapPoint pMPCurr = pKFCurr->getObservation(idKPCurr);
            PtrMapPoint pMPLoop = pKFLoop->getObservation(idKPLoop);
            mergeMP(pMPLoop, pMPCurr); // setnull 会锁住Map的mutex
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
Point2f Map::compareViewMPs(const PtrKeyFrame& pKF1, const PtrKeyFrame& pKF2,
                            set<PtrMapPoint>& spMPs)
{
    spMPs.clear();
    set<PtrMapPoint> setMPs = pKF1->getAllObsMPs(false);

    int nSameMP = 0;
    for (auto i = setMPs.begin(), iend = setMPs.end(); i != iend; ++i) {
        PtrMapPoint pMP = *i;
        if (pKF2->hasObservation(pMP)) {
            spMPs.insert(pMP);
            nSameMP++;
        }
    }

    return Point2f(1.f*nSameMP/pKF1->countObservations(), 1.f*nSameMP/pKF2->countObservations());
}

/**
 * @brief 当前KF观测的MP可以被其它共视KFs同时观测次数>k次的比例, 主要用于冗余KF的判断.
 * @param pKFNow    当前帧KF
 * @param spKFsRef  参考KFs, 即LocalKFs(all covisible KFs)
 * @param spMPsRet  输出具有共同观测的MP集合[out]
 * @param k         共同观测次数, 默认为2
 * @return
 */
float Map::compareViewMPs(const PtrKeyFrame& pKFNow, const set<PtrKeyFrame>& spKFsRef,
                          set<PtrMapPoint>& spMPsRet, int k)
{
    spMPsRet.clear();
    set<PtrMapPoint> spMPsAll = pKFNow->getAllObsMPs(false);
    if (spMPsAll.size() == 0) {
        return -1.0;
    }

    for (auto iter = spMPsAll.begin(); iter != spMPsAll.end(); iter++) {
        PtrMapPoint pMP = *iter;
        int count = 0;

        for (auto iter2 = spKFsRef.begin(); iter2 != spKFsRef.end(); iter2++) {
            PtrKeyFrame pKFRef = *iter2;
            if (pKFRef->hasObservation(pMP)) {
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

int Map::getCovisibleWeight(const PtrKeyFrame& pKF1, const PtrKeyFrame& pKF2)
{
    set<PtrMapPoint> spMPs1 = pKF1->getAllObsMPs(false);
    if (spMPs1.size() == 0)
        return 0;

    int count = 0;
    for (auto iter = spMPs1.begin(); iter != spMPs1.end(); iter++) {
        if (pKF2->hasObservation(*iter))
            count++;
    }

    return count;
}

//! 此函数在LocalMap的removeOutlierChi2()函数中调用
void Map::loadLocalGraph(SlamOptimizer& optimizer, vector<vector<EdgeProjectXYZ2UV*>>& vpEdgesAll,
                         vector<vector<int>>& vnAllIdx)
{
    locker lock(mMutexLocalGraph);
    printf("[Local][Info ] #%ld(KF#%ld) [Map]正在加载局部BA以移除离群MP. removeOutlierChi2()...\n",
           mCurrentKF->id, mCurrentKF->mIdKF);

    int camParaId = 0;
    addCamPara(optimizer, Config::Kcam, camParaId);

    unsigned long maxKFid = 0;
    unsigned long minKFid = 0;
    // If no reference KF, the KF with minId should be fixed

    if (mvRefKFs.empty()) {
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
    const int nRefKFs = mvRefKFs.size();

    // Add local graph KeyFrames
    for (int i = 0; i != nLocalKFs; ++i) {
        PtrKeyFrame pKF = mvLocalGraphKFs[i];

        //! TODO 如果下面情况发生，vertexIdKF不就不连续了？出现空的Vertex？
        if (pKF->isNull()) {
            fprintf(stderr, "节点ID不连续，因为第%d个LocalKF是null!\n", i);
            continue;
        }

        int vertexIdKF = i;

        bool fixed = (pKF->mIdKF == minKFid) || (pKF->mIdKF == 1);
        // 添加位姿节点
        addVertexSE3Expmap(optimizer, toSE3Quat(pKF->getPose()), vertexIdKF, fixed);
        // 添加平面运动约束边
        addPlaneMotionSE3Expmap(optimizer, toSE3Quat(pKF->getPose()), vertexIdKF, Config::Tbc);
    }

    // Add odometry based constraints. 添加里程计约束边
    for (int i = 0; i != nLocalKFs; ++i) {
        PtrKeyFrame pKF = mvLocalGraphKFs[i];

        if (pKF->isNull())
            continue;

        PtrKeyFrame pKF1 = pKF->mOdoMeasureFrom.first;  //! 这里是From没错!
        auto it = std::find(mvLocalGraphKFs.begin(), mvLocalGraphKFs.end(), pKF1);
        if (it == mvLocalGraphKFs.end() || pKF1->isNull())
            continue;

        int id1 = it - mvLocalGraphKFs.begin();  // 这里id1倒是可以跟上面的i对应上

        g2o::Matrix6d info = toMatrix6d(pKF->mOdoMeasureFrom.second.info);
        // id1 - from, i - to
        addEdgeSE3Expmap(optimizer, toSE3Quat(pKF->mOdoMeasureFrom.second.measure), id1, i, info);
    }

    // Add Reference KeyFrames as fixed. 将RefKFs固定
    for (int i = 0; i != nRefKFs; ++i) {
        PtrKeyFrame pKF = mvRefKFs[i];

        if (pKF->isNull())
            continue;

        int vertexIdKF = i + nLocalKFs;

        addVertexSE3Expmap(optimizer, toSE3Quat(pKF->getPose()), vertexIdKF, true);
    }


    //! 接下来给MP添加节点和边
    maxKFid = nLocalKFs + nRefKFs + 1;

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

        int vertexIdMP = maxKFid + i;  // MP的节点编号

        addVertexSBAXYZ(optimizer, toVector3d(pMP->getPos()), vertexIdMP);

        std::set<PtrKeyFrame> pKFs = pMP->getObservations();  // 已可保证得到的都是有效的KF
        vector<EdgeProjectXYZ2UV*> vpEdges;
        vector<int> vnIdx;

        // 如果MP视差不好则只加节点而不加边
        if (!pMP->isGoodPrl()) {
            vpEdgesAll.push_back(vpEdges);
            vnAllIdx.push_back(vnIdx);
            continue;
        }

        for (auto j = pKFs.begin(), jend = pKFs.end(); j != jend; ++j) {
            PtrKeyFrame pKF = (*j);
            if (checkAssociationErr(pKF, pMP)) {  // 确认一下联接关系有没有错
                fprintf(stderr,
                        "[Local][Warni] removeOutlierChi2() Wrong Association! for KF#%ld and MP#%ld \n",
                        pKF->mIdKF, pMP->mId);
                continue;
            }

            int ftrIdx = pMP->getIndexInKF(pKF);
            int octave = pMP->getOctave(pKF);
            const float invSigma2 = pKF->mvInvLevelSigma2[octave];
            Eigen::Vector2d uv = toVector2d(pKF->mvKeyPoints[ftrIdx].pt);
            Eigen::Matrix2d info = Eigen::Matrix2d::Identity() * invSigma2;

            int vertexIdKF = -1;

            auto it1 = std::find(mvLocalGraphKFs.begin(), mvLocalGraphKFs.end(), pKF);
            if (it1 != mvLocalGraphKFs.end()) {
                vertexIdKF = it1 - mvLocalGraphKFs.begin();
            } else {
                auto it2 = std::find(mvRefKFs.begin(), mvRefKFs.end(), pKF);
                if (it2 != mvRefKFs.end())
                    vertexIdKF = it2 - mvRefKFs.begin() + nLocalKFs;
            }

            if (vertexIdKF == -1)
                continue;

            EdgeProjectXYZ2UV* ei =
                addEdgeXYZ2UV(optimizer, uv, vertexIdMP, vertexIdKF, camParaId, info, delta);
            ei->setLevel(0);

            optimizer.addEdge(ei);  //! FIXME addEdgeXYZ2UV()函数里不是添加过了吗？
            vpEdges.push_back(ei);

            vnIdx.push_back(vertexIdKF);
        }

        vpEdgesAll.push_back(vpEdges);
        vnAllIdx.push_back(vnIdx);
    }
    assert(vpEdgesAll.size() == mvLocalGraphMPs.size());
    assert(vnAllIdx.size() == mvLocalGraphMPs.size());
}

//! 此函数没有被调用!
void Map::loadLocalGraphOnlyBa(SlamOptimizer& optimizer,
                               vector<vector<EdgeProjectXYZ2UV*>>& vpEdgesAll,
                               vector<vector<int>>& vnAllIdx)
{
    locker lock(mMutexLocalGraph);

    int camParaId = 0;
    addCamPara(optimizer, Config::Kcam, camParaId);

    unsigned long maxKFid = 0;
    unsigned long minKFid = 0;
    // If no reference KF, the KF with minId should be fixed

    if (mvRefKFs.empty()) {
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
    const int nRefKFs = mvRefKFs.size();

    // Add local graph KeyFrames
    for (int i = 0; i != nLocalKFs; ++i) {
        PtrKeyFrame pKF = mvLocalGraphKFs[i];

        if (pKF->isNull())
            continue;

        int vertexIdKF = i;

        bool fixed = (pKF->mIdKF == minKFid) || pKF->mIdKF == 1;
        addVertexSE3Expmap(optimizer, toSE3Quat(pKF->getPose()), vertexIdKF, fixed);
    }

    // Add Reference KeyFrames as fixed
    for (int i = 0; i != nRefKFs; ++i) {
        PtrKeyFrame pKF = mvRefKFs[i];

        if (pKF->isNull())
            continue;

        int vertexIdKF = i + nLocalKFs;

        addVertexSE3Expmap(optimizer, toSE3Quat(pKF->getPose()), vertexIdKF, true);
    }

    maxKFid = nLocalKFs + nRefKFs + 1;

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

        std::set<PtrKeyFrame> pKFs = pMP->getObservations();
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
                fprintf(
                    stderr,
                    "[Local][Warni] loadLocalGraphOnlyBa() Wrong Association! for KF#%ld and MP#%ld \n",
                    pKF->mIdKF, pMP->mId);
                continue;
            }

            int ftrIdx = pMP->getIndexInKF(pKF);
            int octave = pMP->getOctave(pKF);
            const float invSigma2 = pKF->mvInvLevelSigma2[octave];
            Eigen::Vector2d uv = toVector2d(pKF->mvKeyPoints[ftrIdx].pt);
            Eigen::Matrix2d info = Eigen::Matrix2d::Identity() * invSigma2;

            int vertexIdKF = -1;

            auto it1 = std::find(mvLocalGraphKFs.begin(), mvLocalGraphKFs.end(), pKF);
            if (it1 != mvLocalGraphKFs.end()) {
                vertexIdKF = it1 - mvLocalGraphKFs.begin();
            } else {
                auto it2 = std::find(mvRefKFs.begin(), mvRefKFs.end(), pKF);
                if (it2 != mvRefKFs.end())
                    vertexIdKF = it2 - mvRefKFs.begin() + nLocalKFs;
            }

            if (vertexIdKF == -1)
                continue;

            EdgeProjectXYZ2UV* ei =
                addEdgeXYZ2UV(optimizer, uv, vertexIdMP, vertexIdKF, camParaId, info, delta);
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
 * @param vnOutlierIdxAll   所有离群MP的索引
 * @return  小于2帧观测数的离群MP数量
 * FIXME 最后对MP的setNull()并不会让MP析构, 需要查找原因
 */
int Map::removeLocalOutlierMP(const vector<vector<int>>& vnOutlierIdxAll)
{
    std::vector<PtrMapPoint> vLocalGraphMPs;
    std::vector<PtrKeyFrame> vLocalGraphKFs;
    std::vector<PtrKeyFrame> vRefKFs;

    {
        locker lock(mMutexLocalGraph);
        vLocalGraphMPs = mvLocalGraphMPs;
        vLocalGraphKFs = mvLocalGraphKFs;
        vRefKFs = mvRefKFs;
    }

    assert(vnOutlierIdxAll.size() == vLocalGraphMPs.size());

    const int nLocalKFs = vLocalGraphKFs.size();
    const int N = vLocalGraphMPs.size();
    int nBadMP = 0;

    for (int i = 0, iend = N; i != iend; ++i) {
        PtrMapPoint pMP = vLocalGraphMPs[i];

        for (int j = 0, jend = vnOutlierIdxAll[i].size(); j < jend; ++j) {
            PtrKeyFrame pKF = nullptr;
            int idxKF = vnOutlierIdxAll[i][j];

            if (idxKF < nLocalKFs) {
                pKF = vLocalGraphKFs[idxKF];
            } else {
                pKF = vRefKFs[idxKF - nLocalKFs];
            }

            if (pKF == nullptr) {
                printf("!! ERR MP: KF In Outlier Edge Not In LocalKF or RefKF! at line number %d "
                       "in file %s\n",
                       __LINE__, __FILE__);
                exit(-1);
                continue;
            }

            if (!pKF->hasObservation(pMP))
                continue;

            if (checkAssociationErr(pKF, pMP)) {
                printf("!! ERR MP: Wrong Association [Outlier removal] ! at line number %d in file "
                       "%s\n",
                       __LINE__, __FILE__);
                exit(-1);
                continue;
            }

            pMP->eraseObservation(pKF);
            pKF->eraseObservation(pMP);
        }

        if (pMP->countObservations() < 2) {
//            mspMPs.erase(pMP);
            pMP->setNull(pMP);  // 加了Map的锁
            fprintf(stderr, "[ Map ][Info ] MP#%ld 在移除外点中因观测数少于2而被修剪, 引用计数 = %ld\n",
                    pMP->mId, pMP.use_count());
            nBadMP++;
        }
    }
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

    const int nLocalKFs = mvLocalGraphKFs.size();
    const int nRefKFs = mvRefKFs.size();
    const int N = mvLocalGraphMPs.size();
    const int maxKFid = nLocalKFs + nRefKFs;

    for (int i = 0; i != nLocalKFs; ++i) {
        PtrKeyFrame pKF = mvLocalGraphKFs[i];
        if (pKF->isNull())
            continue;
        Eigen::Vector3d vp = estimateVertexSE2(optimizer, i).toVector();
        pKF->setPose(Se2(vp(0), vp(1), vp(2)));
    }

    for (int i = 0; i != N; ++i) {
        PtrMapPoint pMP = mvLocalGraphMPs[i];
        if (pMP->isNull() || !pMP->isGoodPrl())
            continue;

        Point3f pos = toCvPt3f(estimateVertexSBAXYZ(optimizer, i + maxKFid));
        pMP->setPos(pos);
        pMP->updateMeasureInKFs();
    }
}

//! 新添的KF在关联MP和生成新的MP之后，如果与Local KFs的共同MP观测数超过自身观测总数的30%，则为他们之间添加共视关系
void Map::updateCovisibility(PtrKeyFrame& pNewKF)
{
    locker lock1(mMutexLocalGraph);
    locker lock2(mMutexGlobalGraph);

    printf("[Local][Info ] #%ld(KF#%ld) [Map]正在更新共视关系... Local KFs数量为: %ld\n",
           pNewKF->id, pNewKF->mIdKF, mvLocalGraphKFs.size());
    for (auto i = mvLocalGraphKFs.begin(), iend = mvLocalGraphKFs.end(); i != iend; ++i) {
        set<PtrMapPoint> spMPs;
        PtrKeyFrame pKFi = *i;
        Point2f ratios = compareViewMPs(pNewKF, pKFi, spMPs);
        if (ratios.x > 0.3 || ratios.y > 0.3) {
            pNewKF->addCovisibleKF(pKFi);
            pKFi->addCovisibleKF(pNewKF);
            printf("[Local][Info ] #%ld(KF#%ld) [Map]和(KF#%ld)添加了共视关系, 共视比例为%.2f, %.2f.\n",
                   pNewKF->id, pNewKF->mIdKF, pKFi->mIdKF, ratios.x, ratios.y);
        }
    }
}


bool Map::checkAssociationErr(const PtrKeyFrame& pKF, const PtrMapPoint& pMP)
{
    int ftrIdx0 = pKF->getFeatureIndex(pMP);
    int ftrIdx1 = pMP->getIndexInKF(pKF);
    if (ftrIdx0 != ftrIdx1 || ftrIdx0 == -1 || ftrIdx1 == -1) {
        fprintf(stderr, "!! checkAssociationErr: pKF->pMP / pMP->pKF: %d / %d\n", ftrIdx0, ftrIdx1);
        return true;
    }
    return false;
}

// Select KF pairs to creat feature constraint between which
/**
 * @brief   提取特征图匹配对
 * @param _pKF  当前帧
 * @return      返回当前帧和即在共视图又在特征图里的KF的匹配对
 */
vector<pair<PtrKeyFrame, PtrKeyFrame>> Map::SelectKFPairFeat(const PtrKeyFrame& _pKF)
{
    vector<pair<PtrKeyFrame, PtrKeyFrame>> _vKFPairs;

    // Smallest distance between KFs in covis-graph to create a new feature edge
    int threshCovisGraphDist = 5;

    set<PtrKeyFrame> sKFSelected;
    set<PtrKeyFrame> sKFCovis = _pKF->getAllCovisibleKFs();
    set<PtrKeyFrame> sKFLocal = GlobalMapper::GetAllConnectedKFs_nLayers(_pKF, threshCovisGraphDist,
                                                                         sKFSelected);

    // 共视图里的KF如果在特征图里,就加入到sKFSelected里,然后会更新特征图
    for (auto iter = sKFCovis.begin(); iter != sKFCovis.end(); iter++) {
        PtrKeyFrame _pKFCand = *iter;
        if (sKFLocal.count(_pKFCand) == 0) {
            sKFSelected.insert(*iter);
            sKFLocal =
                GlobalMapper::GetAllConnectedKFs_nLayers(_pKF, threshCovisGraphDist, sKFSelected);
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
bool Map::UpdateFeatGraph(const PtrKeyFrame& _pKF)
{
    locker lock1(mMutexLocalGraph);
    locker lock2(mMutexGlobalGraph);

    vector<pair<PtrKeyFrame, PtrKeyFrame>> _vKFPairs = SelectKFPairFeat(_pKF);

    if (_vKFPairs.empty())
        return false;

    int numPairKFs = _vKFPairs.size();
    for (int i = 0; i != numPairKFs; ++i) {
        pair<PtrKeyFrame, PtrKeyFrame> pairKF = _vKFPairs[i];
        PtrKeyFrame ptKFFrom = pairKF.first;  //! 这里始终都是当前帧_pKF啊???
        PtrKeyFrame ptKFTo = pairKF.second;
        SE3Constraint ftrCnstr;

        //! NOTE 为KF对添加特征图约束,这里对应论文里的SE3XYZ约束??
        if (GlobalMapper::CreateFeatEdge(ptKFFrom, ptKFTo, ftrCnstr) == 0) {
            ptKFFrom->addFtrMeasureFrom(ptKFTo, ftrCnstr.measure, ftrCnstr.info);
            ptKFTo->addFtrMeasureTo(ptKFFrom, ftrCnstr.measure, ftrCnstr.info);
            if (Config::GlobalPrint) {
                fprintf(stderr, "\n\n[Globa][Info ] #%ld(KF#%ld) [Map]Add feature constraint from KF#%ld to KF#%ld!\n\n",
                        _pKF->id, _pKF->mIdKF, ptKFFrom->mIdKF, ptKFTo->mIdKF);
            }
        } else {
            if (Config::GlobalPrint)
                fprintf(stderr, "[Globa][Error] #%ld(KF#%ld) [Map]Add feature constraint failed!\n",
                        _pKF->id, _pKF->mIdKF);
        }
    }

    return true;
}

//! 此函数在LocalMap的localBA()函数中调用
void Map::loadLocalGraph(SlamOptimizer& optimizer)
{
    printf("[Local][Info ] #%ld(KF#%ld) [Map]正在加载LocalGraph以做局部图优化...\n", mCurrentKF->id,
           mCurrentKF->mIdKF);

    locker lock(mMutexLocalGraph);

    int camParaId = 0;
    CamPara* campr = addCamPara(optimizer, Config::Kcam, camParaId);

    int maxKFid = -1;
    unsigned long minKFid = 0;  // LocalKFs里最小的KFid

    // If no reference KF, the KF with minId should be fixed
    if (mvRefKFs.empty()) {
        minKFid = (*(mvLocalGraphKFs.begin()))->id;
        for (auto i = mvLocalGraphKFs.begin(), iend = mvLocalGraphKFs.end(); i != iend; ++i) {
            PtrKeyFrame pKF = *i;
            if (pKF->isNull())
                continue;
            if (pKF->id < minKFid)
                minKFid = pKF->id;
        }
    }

    const int nLocalKFs = mvLocalGraphKFs.size();
    const int nRefKFs = mvRefKFs.size();

    // Add local graph KeyFrames
    for (int i = 0; i < nLocalKFs; ++i) {
        PtrKeyFrame pKF = mvLocalGraphKFs[i];

        if (pKF->isNull())
            continue;

        int vertexIdKF = i;

        bool fixed = (pKF->id == minKFid) || (pKF->id == 1);
        if (fixed)
            printf("[Local][Info ] #%ld(KF#%ld) [Map]vertex of #KF%ld is fixed! minKFid = %ld\n",
                   mCurrentKF->id, mCurrentKF->mIdKF, pKF->id, minKFid);

        Se2 Twb = pKF->getTwb();
        g2o::SE2 pose(Twb.x, Twb.y, Twb.theta);
        addVertexSE2(optimizer, pose, vertexIdKF, fixed);
    }

    // Add odometry based constraints
    //! TODO preOdomFromSelf这个变量暂时都是0
    for (int i = 0; i < nLocalKFs; ++i) {
        PtrKeyFrame pKF = mvLocalGraphKFs[i];

        if (pKF->isNull())
            continue;

        PtrKeyFrame pKF1 = pKF->preOdomFromSelf.first;
        PreSE2 meas = pKF->preOdomFromSelf.second;
        auto it = std::find(mvLocalGraphKFs.begin(), mvLocalGraphKFs.end(), pKF1);
        if (it == mvLocalGraphKFs.end() || pKF1->isNull())
            continue;

        int id1 = it - mvLocalGraphKFs.begin();
        Eigen::Map<Eigen::Matrix3d, RowMajor> info(meas.cov);
        addEdgeSE2(optimizer, Vector3D(meas.meas), i, id1, info);
    }

    // Add Reference KeyFrames as fixed
    for (int i = 0; i < nRefKFs; ++i) {
        PtrKeyFrame pKF = mvRefKFs[i];

        if (pKF->isNull())
            continue;

        int vertexIdKF = i + nLocalKFs;

        Se2 Twb = pKF->getTwb();
        g2o::SE2 pose(Twb.x, Twb.y, Twb.theta);
        addVertexSE2(optimizer, pose, vertexIdKF, true);
    }

    maxKFid = nLocalKFs + nRefKFs;  //! 这里不应该+1

    // Store xyz2uv edges
    const int N = mvLocalGraphMPs.size();

    printf("[Local][Info ] #%ld(KF#%ld) [Map]LocalMap的顶点情况: nLocalKFs = %d, nRefKFs = %d, nMPs = %d\n",
           mCurrentKF->id, mCurrentKF->mIdKF, nLocalKFs, nRefKFs, N);

    const float delta = Config::ThHuber;

    // Add local graph MapPoints
    for (int i = 0; i < N; ++i) {
        PtrMapPoint pMP = mvLocalGraphMPs[i];
        assert(!pMP->isNull());  // TODO to delete

        int vertexIdMP = maxKFid + i;
        Vector3d lw = toVector3d(pMP->getPos());

        addVertexSBAXYZ(optimizer, lw, vertexIdMP);

        std::set<PtrKeyFrame> pKFs = pMP->getObservations();
        for (auto j = pKFs.begin(), jend = pKFs.end(); j != jend; ++j) {
            PtrKeyFrame pKF = (*j);
            if (checkAssociationErr(pKF, pMP)) {
                fprintf(stderr, "[ Map ][Warni] localBA() Wrong Association! for KF#%ld-%d and MP#%ld-%d\n",
                pKF->mIdKF, pKF->getFeatureIndex(pMP), pMP->mId, pMP->getIndexInKF(pKF));
                continue;
            }

            int octave = pMP->getOctave(pKF);
            size_t ftrIdx = pMP->getIndexInKF(pKF);
            const float Sigma2 = pKF->mvLevelSigma2[octave];
            Eigen::Vector2d uv = toVector2d(pKF->mvKeyPoints[ftrIdx].pt);

            int vertexIdKF = -1;

            auto it1 = std::find(mvLocalGraphKFs.begin(), mvLocalGraphKFs.end(), pKF);
            if (it1 != mvLocalGraphKFs.end()) {
                vertexIdKF = it1 - mvLocalGraphKFs.begin();
            } else {
                auto it2 = std::find(mvRefKFs.begin(), mvRefKFs.end(), pKF);
                if (it2 != mvRefKFs.end())
                    vertexIdKF = it2 - mvRefKFs.begin() + nLocalKFs;
            }

            if (vertexIdKF == -1)
                continue;

            // compute covariance/information
            Matrix2d Sigma_u = Eigen::Matrix2d::Identity() * Sigma2;
            Vector3d lc = toVector3d(pKF->mvViewMPs[ftrIdx]);

            double zc = lc(2);
            double zc_inv = 1. / zc;
            double zc_inv2 = zc_inv * zc_inv;
            const float& fx = Config::fx;
            Matrix23d J_pi;
            J_pi << fx * zc_inv, 0, -fx * lc(0) * zc_inv2, 0, fx * zc_inv, -fx * lc(1) * zc_inv2;
            Matrix3d Rcw = toMatrix3d(pKF->getPose().rowRange(0, 3).colRange(0, 3));
            Se2 Twb = pKF->getTwb();
            Vector3d pi(Twb.x, Twb.y, 0);

            Matrix23d J_pi_Rcw = J_pi * Rcw;

            Matrix2d J_rotxy = (J_pi_Rcw * skew(lw - pi)).block<2, 2>(0, 0);
            Matrix<double, 2, 1> J_z = -J_pi_Rcw.block<2, 1>(0, 2);
            float Sigma_rotxy = 1. / Config::PlaneMotionInfoXrot;
            float Sigma_z = 1. / Config::PlaneMotionInfoZ;
            Matrix2d Sigma_all = Sigma_rotxy * J_rotxy * J_rotxy.transpose() +
                                 Sigma_z * J_z * J_z.transpose() + Sigma_u;

            addEdgeSE2XYZ(optimizer, uv, vertexIdKF, vertexIdMP, campr, toSE3Quat(Config::Tbc),
                          Sigma_all.inverse(), delta);
        }
    }

    size_t nVertices = optimizer.vertices().size();
    size_t nEdges = optimizer.edges().size();
    printf("[Local][Info ] #%ld(KF#%ld) [Map]loadLocalGraph() optimizer nVertices = %ld, nEdges = %ld\n",
           mCurrentKF->id, mCurrentKF->mIdKF, nVertices, nEdges);
    // check optimizer

    //    auto edges = optimizer.edges();
    //    for (size_t i = 0; i < nEdges; ++i) {
    //        auto v1 = edges[i]->vertices()[0];  // KF
    //        auto v2 = edges[i]->vertices()[1];  // MP
    //        fprintf(stderr, "[TEST] edge %ld vertex1.KFid = %ld, vertex2.MPid = %ld, oberservation
    //        = %d\n",
    //                i, v1)
    //    }
}

void Map::addLocalGraphThroughKdtree(std::set<PtrKeyFrame>& setLocalKFs,
                                     int maxN, float searchRadius)
{
    vector<PtrKeyFrame> vKFsAll = getAllKF();
    vector<Point3f> vKFPoses;
    for (size_t i = 0, iend = vKFsAll.size(); i != iend; ++i) {
        Mat Twc = cvu::inv(vKFsAll[i]->getPose());
        Point3f pose(Twc.at<float>(0, 3) * 0.001f, Twc.at<float>(1, 3) * 0.001f,
                     Twc.at<float>(2, 3) * 0.001f);
        vKFPoses.push_back(pose);
    }

    cv::flann::KDTreeIndexParams kdtreeParams;
    cv::flann::Index kdtree(Mat(vKFPoses).reshape(1), kdtreeParams);

    Mat pose = cvu::inv(getCurrentKF()->getPose());
    std::vector<float> query = {pose.at<float>(0, 3) / 1000.f, pose.at<float>(1, 3) / 1000.f,
                                pose.at<float>(2, 3) / 1000.f};
//    int size = std::min(static_cast<int>(vKFsAll.size()), 8);  // 最近的5个KF
    std::vector<int> indices;
    std::vector<float> dists;
//    kdtree.knnSearch(query, indices, dists, size, cv::flann::SearchParams());
    kdtree.radiusSearch(query, indices, dists, searchRadius, maxN, cv::flann::SearchParams());
    for (size_t i = 0, iend = indices.size(); i != iend; ++i) {
        if (indices[i] > 0 && dists[i] < searchRadius && vKFsAll[indices[i]])  // 距离在0.3m以内
            setLocalKFs.insert(vKFsAll[indices[i]]);
    }
}


}  // namespace se2lam
