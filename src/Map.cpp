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
typedef lock_guard<mutex> locker;

Map::Map() : mCurrentKF(nullptr), isEmpty(true), mpLocalMapper(nullptr)
{
    mCurrentFramePose = Mat::eye(4, 4, CV_32FC1);
    mbKFUpdated = false;
    mbMPUpdated = false;
}
Map::~Map() {}

bool Map::empty()
{
    return isEmpty;
}

PtrKeyFrame Map::getCurrentKF()
{
    locker lock(mMutexCurrentKF);
    return mCurrentKF;
}

void Map::setCurrentKF(const PtrKeyFrame& pKF)
{
    locker lock(mMutexCurrentKF);
    mCurrentKF = pKF;
}


vector<PtrKeyFrame> Map::getAllKFs()
{
    locker lock(mMutexGlobalGraph);
    return vector<PtrKeyFrame>(mKFs.begin(), mKFs.end());
}


vector<PtrMapPoint> Map::getAllMPs()
{
    locker lock(mMutexGlobalGraph);
    return vector<PtrMapPoint>(mMPs.begin(), mMPs.end());
}

void Map::clear()
{
    locker lock(mMutexGlobalGraph);
    mKFs.clear();
    mMPs.clear();
    mLocalGraphKFs.clear();
    mLocalGraphMPs.clear();
    mLocalRefKFs.clear();
    isEmpty = true;
    KeyFrame::mNextIdKF = 0;
    MapPoint::mNextId = 0;
}

void Map::insertKF(const PtrKeyFrame& pkf)
{
    locker lock(mMutexGlobalGraph);
    // pkf->setMap(this);
    mKFs.insert(pkf);
    mCurrentKF = pkf;
    isEmpty = false;
    mbKFUpdated = true;
}

void Map::insertMP(const PtrMapPoint& pmp)
{
    locker lock(mMutexGlobalGraph);
    // pMP->setMap(this);
    mMPs.insert(pmp);
    mbMPUpdated = true;
}

void Map::eraseKF(const PtrKeyFrame& pKF)
{
    locker lock(mMutexGlobalGraph);
    if (mKFs.erase(pKF))
        mbKFUpdated = true;

    locker lock2(mMutexLocalGraph);
    auto iter1 = find(mLocalGraphKFs.begin(), mLocalGraphKFs.end(), pKF);
    if (iter1 != mLocalGraphKFs.end())
        mLocalGraphKFs.erase(iter1);
    auto iter2 = find(mLocalRefKFs.begin(), mLocalRefKFs.end(), pKF);
    if (iter2 != mLocalRefKFs.end())
        mLocalRefKFs.erase(iter2);
}

void Map::eraseMP(const PtrMapPoint& pMP)
{
    locker lock(mMutexGlobalGraph);
    if (mMPs.erase(pMP))
        mbMPUpdated = true;

    locker lock2(mMutexLocalGraph);
    auto iter = find(mLocalGraphMPs.begin(), mLocalGraphMPs.end(), pMP);
    if (iter != mLocalGraphMPs.end())
        mLocalGraphMPs.erase(iter);
}

Mat Map::getCurrentFramePose()
{
    locker lock(mMutexCurrentFrame);
    return mCurrentFramePose.clone();
}


void Map::setCurrentFramePose(const Mat& pose)
{
    locker lock(mMutexCurrentFrame);
    pose.copyTo(mCurrentFramePose);
}

size_t Map::countKFs()
{
    locker lock(mMutexGlobalGraph);
    return mKFs.size();
}

size_t Map::countMPs()
{
    locker lock(mMutexGlobalGraph);
    return mMPs.size();
}

size_t Map::countLocalKFs()
{
    locker lock(mMutexLocalGraph);
    return mLocalGraphKFs.size();
}

size_t Map::countLocalMPs()
{
    locker lock(mMutexLocalGraph);
    return mLocalGraphMPs.size();
}

size_t Map::countLocalRefKFs()
{
    locker lock(mMutexLocalGraph);
    return mLocalRefKFs.size();
}


void Map::mergeMP(PtrMapPoint& toKeep, PtrMapPoint& toDelete)
{

    std::set<PtrKeyFrame> pKFs = toKeep->getObservations();
    for (auto it = pKFs.begin(), itend = pKFs.end(); it != itend; it++) {
        PtrKeyFrame pKF = *it;
        if (pKF->hasObservation(toKeep) && pKF->hasObservation(toDelete)) {
            return;
        }
    }
    pKFs = toDelete->getObservations();
    for (auto it = pKFs.begin(), itend = pKFs.end(); it != itend; it++) {
        PtrKeyFrame pKF = *it;
        if (pKF->hasObservation(toKeep) && pKF->hasObservation(toDelete)) {
            return;
        }
    }

    toDelete->mergedInto(toKeep);
    if (mMPs.erase(toDelete))
        mbMPUpdated = true;

    auto it = std::find(mLocalGraphMPs.begin(), mLocalGraphMPs.end(), toDelete);
    if (it != mLocalGraphMPs.end())
        *it = toKeep;
}

void Map::setLocalMapper(LocalMapper* pLocalMapper)
{
    mpLocalMapper = pLocalMapper;
}

bool Map::pruneRedundantKF()
{

    locker lock1(mMutexLocalGraph);
    locker lock2(mMutexGlobalGraph);

    bool pruned = false;
    int prunedIdxLocalKF = -1;
    // int count;

    if (mLocalGraphKFs.size() <= 3) {
        return pruned;
    }

    vector<int> goodIdxLocalKF;
    vector<int> goodIdxLocalMP;

    for (int i = 0, iend = mLocalGraphKFs.size(); i < iend; i++) {

        bool prunedThis = false;

        if (!pruned) {

            PtrKeyFrame thisKF = mLocalGraphKFs[i];

            // The first and the current KF should not be pruned
            if (thisKF->isNull() || mCurrentKF->mIdKF == thisKF->mIdKF || thisKF->mIdKF <= 1)
                continue;

            // Count how many KF share same MPs more than 80% of each's observation
            //            count = 0;
            //            set<PtrKeyFrame> covKFs = thisKF->getAllCovisibleKFs();
            //            for(auto j = covKFs.begin(), jend = covKFs.end(); j!=jend; j++ ) {
            //                PtrKeyFrame pKFj = *j;
            //                if(pKFj->isNull())
            //                    continue;
            //                set<PtrMapPoint> spMPs;
            //                Point2f ratio = compareViewMPs(thisKF, pKFj, spMPs);
            //                //            if(ratio.x > 0.8f && ratio.y > 0.8f) {
            //                //                count++;
            //                //            }
            //                if (ratio.x > 0.6f) {
            //                    count++;
            //                }
            //            }
            //            bool bIsThisKFRedundant = (count >= 2);

            // Count MPs in thisKF observed by covKFs 2 times
            set<PtrMapPoint> spMPs;
            set<PtrKeyFrame> covKFs = thisKF->getAllCovisibleKFs();
            double ratio = compareViewMPs(thisKF, covKFs, spMPs, 2);
            bool bIsThisKFRedundant = (ratio >= 0.8);


            // Do Prune if pass threashold test
            if (bIsThisKFRedundant) {
                PtrKeyFrame lastKF = thisKF->mOdoMeasureTo.first;
                PtrKeyFrame nextKF = thisKF->mOdoMeasureFrom.first;

                bool bIsInitKF = (thisKF->mIdKF == 0);
                bool bHasFeatEdge = (thisKF->mFtrMeasureFrom.size() != 0);

                // Prune this KF and link a new odometry constrait
                if (lastKF && nextKF && !bIsInitKF && !bHasFeatEdge) {

                    Se2 dOdoLastThis = thisKF->odom - lastKF->odom;
                    Se2 dOdoThisNext = nextKF->odom - thisKF->odom;

                    double theshl = 10000;
                    double thesht = 45 * 3.1415926 / 180;

                    double dl1 =
                        sqrt(dOdoLastThis.x * dOdoLastThis.x + dOdoLastThis.y * dOdoLastThis.y);
                    double dt1 = abs(dOdoLastThis.theta);
                    double dl2 =
                        sqrt(dOdoThisNext.x * dOdoThisNext.x + dOdoThisNext.y * dOdoThisNext.y);
                    double dt2 = abs(dOdoThisNext.theta);

                    if (dl1 < theshl && dl2 < theshl && dt1 < thesht && dt2 < thesht) {
                        mKFs.erase(thisKF);
                        thisKF->setNull(thisKF);

                        Mat measure;
                        g2o::Matrix6d info;
                        Track::calcOdoConstraintCam(nextKF->odom - lastKF->odom, measure, info);
                        nextKF->setOdoMeasureTo(lastKF, measure, toCvMat6f(info));
                        lastKF->setOdoMeasureFrom(nextKF, measure, toCvMat6f(info));

                        nextKF->addCovisibleKF(lastKF);
                        lastKF->addCovisibleKF(nextKF);

                        printf("!! INFO MP: Prune KF %d\n", thisKF->mIdKF);

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

    // Remove useless MP
    if (pruned) {
        for (int i = 0, iend = mLocalGraphMPs.size(); i < iend; i++) {
            PtrMapPoint pMP = mLocalGraphMPs[i];
            if (pMP->isNull()) {
                mMPs.erase(pMP);
            } else {
                goodIdxLocalMP.push_back(i);
            }
        }
    }

    if (pruned) {
        vector<PtrMapPoint> vpMPs;
        vector<PtrKeyFrame> vpKFs;
        vpMPs.reserve(goodIdxLocalMP.size());
        vpKFs.reserve(goodIdxLocalKF.size());

        for (int i = 0, iend = goodIdxLocalKF.size(); i < iend; i++) {
            vpKFs.push_back(mLocalGraphKFs[goodIdxLocalKF[i]]);
        }

        for (int i = 0, iend = goodIdxLocalMP.size(); i < iend; i++) {
            vpMPs.push_back(mLocalGraphMPs[goodIdxLocalMP[i]]);
        }

        std::swap(vpMPs, mLocalGraphMPs);
        std::swap(vpKFs, mLocalGraphKFs);
    }

    return pruned;
}

void Map::updateLocalGraph()
{
    WorkTimer timer;

    locker lock(mMutexLocalGraph);

    mLocalGraphKFs.clear();
    mLocalRefKFs.clear();
    mLocalGraphMPs.clear();

    std::set<PtrKeyFrame, KeyFrame::IdLessThan> setLocalKFs;
    std::set<PtrKeyFrame, KeyFrame::IdLessThan> setLocalRefKFs;
    std::set<PtrMapPoint, MapPoint::IdLessThan> setLocalMPs;

    setLocalKFs.insert(mCurrentKF);

    if (static_cast<int>(countKFs()) <= Config::MaxLocalFrameNum) {
        locker lock(mMutexGlobalGraph);
        setLocalKFs.insert(mKFs.begin(), mKFs.end());
    } else {
        setLocalKFs.insert(mCurrentKF);

        updateLocalGraphKdtree(setLocalKFs, Config::MaxLocalFrameNum, Config::LocalFrameSearchRadius);  // lock global

        int toAdd = Config::MaxLocalFrameNum - setLocalKFs.size();
        int searchLevel = Config::LocalFrameSearchLevel;
        while (searchLevel > 0 && toAdd > 0) {
            std::set<PtrKeyFrame, KeyFrame::IdLessThan> currentLocalKFs = setLocalKFs;
            for (auto i = currentLocalKFs.begin(), iend = currentLocalKFs.end(); i != iend; i++) {
                PtrKeyFrame pKF = (*i);
                std::set<PtrKeyFrame> pKFs = pKF->getAllCovisibleKFs();
                setLocalKFs.insert(pKFs.begin(), pKFs.end());
                if (--toAdd <= 0)
                    break;
            }
            searchLevel--;
        }
    }

    for (auto i = setLocalKFs.begin(), iend = setLocalKFs.end(); i != iend; i++) {
        PtrKeyFrame pKF = *i;
        bool checkPrl = false;
        set<PtrMapPoint> pMPs = pKF->getAllObsMPs(checkPrl);
        setLocalMPs.insert(pMPs.begin(), pMPs.end());
    }

    setLocalRefKFs.clear();
    for (auto i = setLocalMPs.begin(), iend = setLocalMPs.end(); i != iend; i++) {
        PtrMapPoint pMP = (*i);
        std::set<PtrKeyFrame> pKFs = pMP->getObservations();
        for (auto j = pKFs.begin(), jend = pKFs.end(); j != jend; j++) {
            if (setLocalKFs.find((*j)) != setLocalKFs.end() ||
                setLocalRefKFs.find((*j)) != setLocalRefKFs.end())
                continue;
            setLocalRefKFs.insert((*j));
        }
    }

    cout << "[Local][ Map ] #" << mCurrentKF->id << "(KF#" << mCurrentKF->mIdKF << ") "
         << "更新局部地图, 局部地图成员个数分别为: LocalKFs/RefKFs/LocalMPs = " << setLocalKFs.size()
         << "/" << setLocalRefKFs.size() << "/" << setLocalMPs.size() << ", 共耗时" << timer.count()
         << "ms." << endl;

    mLocalGraphKFs = vector<PtrKeyFrame>(setLocalKFs.begin(), setLocalKFs.end());
    mLocalRefKFs = vector<PtrKeyFrame>(setLocalRefKFs.begin(), setLocalRefKFs.end());
    mLocalGraphMPs = vector<PtrMapPoint>(setLocalMPs.begin(), setLocalMPs.end());
}

void Map::updateLocalGraphKdtree(std::set<PtrKeyFrame, KeyFrame::IdLessThan>& setLocalKFs, int maxN, float searchRadius)
{
    const vector<PtrKeyFrame> vKFsAll = getAllKFs();  // lock global
    vector<Point3f> vKFPoses(vKFsAll.size());
    for (size_t i = 0, iend = vKFsAll.size(); i != iend; ++i) {
        Mat Twc = cvu::inv(vKFsAll[i]->getPose());
        Point3f pose(Twc.at<float>(0, 3) * 0.001f, Twc.at<float>(1, 3) * 0.001f, Twc.at<float>(2, 3) * 0.001f);
        vKFPoses[i] = pose;
    }

    cv::flann::KDTreeIndexParams kdtreeParams;
    cv::flann::Index kdtree(Mat(vKFPoses).reshape(1), kdtreeParams);

    Mat pose = cvu::inv(mCurrentKF->getPose());
    std::vector<float> query = {pose.at<float>(0, 3) * 0.001f, pose.at<float>(1, 3) * 0.001f,
                                pose.at<float>(2, 3) * 0.001f};
    std::vector<int> indices;
    std::vector<float> dists;
    kdtree.radiusSearch(query, indices, dists, searchRadius, maxN, cv::flann::SearchParams());
    for (size_t i = 0, iend = indices.size(); i != iend; ++i) {
        if (indices[i] > 0 && dists[i] < searchRadius)  // 距离在0.3m以内
            setLocalKFs.insert(vKFsAll[indices[i]]);
    }
}

void Map::mergeLoopClose(const std::map<int, int>& mapMatchMP, PtrKeyFrame& pKFCurr, PtrKeyFrame& pKFLoop)
{
    locker lock1(mMutexLocalGraph);
    locker lock2(mMutexGlobalGraph);

    pKFCurr->addCovisibleKF(pKFLoop);
    pKFLoop->addCovisibleKF(pKFCurr);

    for (auto iter = mapMatchMP.begin(); iter != mapMatchMP.end(); iter++) {
        int idKPCurr = iter->first;
        int idKPLoop = iter->second;

        if (pKFCurr->hasObservation(idKPCurr) && pKFLoop->hasObservation(idKPLoop)) {
            PtrMapPoint pMPCurr = pKFCurr->getObservation(idKPCurr);
            if (!pMPCurr)
                cerr << "This is NULL /in M::mergeLoopClose \n";
            PtrMapPoint pMPLoop = pKFLoop->getObservation(idKPLoop);
            if (!pMPLoop)
                cerr << "This is NULL /in M::mergeLoopClose \n";
            mergeMP(pMPLoop, pMPCurr);
        }
    }
}

Point2f Map::compareViewMPs(const PtrKeyFrame& pKF1, const PtrKeyFrame& pKF2, set<PtrMapPoint>& spMPs)
{
    int nSameMP = 0;
    spMPs.clear();
    bool checkPrl = false;
    set<PtrMapPoint> setMPs = pKF1->getAllObsMPs(checkPrl);
    for (auto i = setMPs.begin(), iend = setMPs.end(); i != iend; i++) {
        PtrMapPoint pMP = *i;
        if (pKF2->hasObservation(pMP)) {
            spMPs.insert(pMP);
            nSameMP++;
        }
    }

    return Point2f((float)nSameMP / (float)(pKF1->countObservations()),
                   (float)nSameMP / (float)(pKF2->countObservations()));
}

// Find MPs in pKF which are observed by vpKFs for k times
// MPs are returned by vpMPs
double Map::compareViewMPs(const PtrKeyFrame& pKFNow, const set<PtrKeyFrame>& spKFsRef,
                           set<PtrMapPoint>& spMPsRet, int k)
{

    spMPsRet.clear();
    set<PtrMapPoint> spMPsAll = pKFNow->getAllObsMPs();
    if (spMPsAll.size() == 0) {
        return -1;
    }

    for (auto iter = spMPsAll.begin(); iter != spMPsAll.end(); iter++) {

        PtrMapPoint pMP = *iter;
        int count = 0;

        for (auto iter2 = spKFsRef.begin(); iter2 != spKFsRef.end(); iter2++) {
            PtrKeyFrame pKFRef = *iter2;
            if (pKFRef->hasObservation(pMP)) {
                count++;
            }
        }

        if (count >= k) {
            spMPsRet.insert(pMP);
        }
    }

    double ratio = spMPsRet.size() * 1.0 / spMPsAll.size();
    return ratio;
}

void Map::loadLocalGraph(SlamOptimizer& optimizer, vector<vector<EdgeProjectXYZ2UV*>>& vpEdgesAll,
                         vector<vector<int>>& vnAllIdx)
{

    locker lock(mMutexLocalGraph);

    int camParaId = 0;
    addCamPara(optimizer, Config::Kcam, camParaId);

    int maxKFid = -1;
    int minKFid = -1;
    // If no reference KF, the KF with minId should be fixed

    if (mLocalRefKFs.empty()) {
        minKFid = (*(mLocalGraphKFs.begin()))->mIdKF;
        for (auto i = mLocalGraphKFs.begin(), iend = mLocalGraphKFs.end(); i != iend; i++) {
            PtrKeyFrame pKF = *i;
            if (pKF->isNull())
                continue;
            if (pKF->mIdKF < minKFid)
                minKFid = pKF->mIdKF;
        }
    }

    const int nLocalKFs = mLocalGraphKFs.size();
    const int nRefKFs = mLocalRefKFs.size();

    // Add local graph KeyFrames
    for (int i = 0; i < nLocalKFs; i++) {

        PtrKeyFrame pKF = mLocalGraphKFs[i];

        if (pKF->isNull())
            continue;

        int vertexIdKF = i;

        bool fixed = (pKF->mIdKF == minKFid) || pKF->mIdKF == 1;
        addVertexSE3Expmap(optimizer, toSE3Quat(pKF->getPose()), vertexIdKF, fixed);
        addPlaneMotionSE3Expmap(optimizer, toSE3Quat(pKF->getPose()), vertexIdKF, Config::Tbc);
    }

    // Add odometry based constraints
    for (int i = 0; i < nLocalKFs; i++) {

        PtrKeyFrame pKF = mLocalGraphKFs[i];

        if (pKF->isNull())
            continue;

        PtrKeyFrame pKF1 = pKF->mOdoMeasureFrom.first;
        auto it = std::find(mLocalGraphKFs.begin(), mLocalGraphKFs.end(), pKF1);
        if (it == mLocalGraphKFs.end() || pKF1->isNull())
            continue;

        int id1 = distance(mLocalGraphKFs.begin(), it);

        g2o::Matrix6d info = toMatrix6d(pKF->mOdoMeasureFrom.second.info);
        addEdgeSE3Expmap(optimizer, toSE3Quat(pKF->mOdoMeasureFrom.second.measure), id1, i, info);
    }

    // Add Reference KeyFrames as fixed
    for (int i = 0; i < nRefKFs; i++) {

        PtrKeyFrame pKF = mLocalRefKFs[i];

        if (pKF->isNull())
            continue;

        int vertexIdKF = i + nLocalKFs;

        addVertexSE3Expmap(optimizer, toSE3Quat(pKF->getPose()), vertexIdKF, true);
    }

    maxKFid = nLocalKFs + nRefKFs + 1;

    // Store xyz2uv edges
    const int N = mLocalGraphMPs.size();
    vpEdgesAll.clear();
    vpEdgesAll.reserve(N);
    vnAllIdx.clear();
    vnAllIdx.reserve(N);

    const float delta = Config::ThHuber;

    // Add local graph MapPoints
    for (int i = 0; i < N; i++) {

        PtrMapPoint pMP = mLocalGraphMPs[i];
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

        for (auto j = pKFs.begin(), jend = pKFs.end(); j != jend; j++) {

            PtrKeyFrame pKF = (*j);
            if (pKF->isNull())
                continue;

            if (checkAssociationErr(pKF, pMP)) {
                printf("!! ERR MP: Wrong Association! [LocalGraph loading] \n");
                continue;
            }

            int ftrIdx = pMP->getFtrIdx(pKF);
            int octave = pMP->getOctave(pKF);
            const float invSigma2 = pKF->mvInvLevelSigma2[octave];
            Eigen::Vector2d uv = toVector2d(pKF->keyPointsUn[ftrIdx].pt);
            Eigen::Matrix2d info = Eigen::Matrix2d::Identity() * invSigma2;

            int vertexIdKF = -1;

            auto it1 = std::find(mLocalGraphKFs.begin(), mLocalGraphKFs.end(), pKF);
            if (it1 != mLocalGraphKFs.end()) {
                vertexIdKF = it1 - mLocalGraphKFs.begin();
            } else {
                auto it2 = std::find(mLocalRefKFs.begin(), mLocalRefKFs.end(), pKF);
                if (it2 != mLocalRefKFs.end())
                    vertexIdKF = it2 - mLocalRefKFs.begin() + nLocalKFs;
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
    assert(vpEdgesAll.size() == mLocalGraphMPs.size());
    assert(vnAllIdx.size() == mLocalGraphMPs.size());
}

void Map::loadLocalGraphOnlyBa(SlamOptimizer& optimizer, vector<vector<EdgeProjectXYZ2UV*>>& vpEdgesAll,
                               vector<vector<int>>& vnAllIdx)
{

    locker lock(mMutexLocalGraph);

    int camParaId = 0;
    addCamPara(optimizer, Config::Kcam, camParaId);

    int maxKFid = -1;
    int minKFid = -1;
    // If no reference KF, the KF with minId should be fixed

    if (mLocalRefKFs.empty()) {
        minKFid = (*(mLocalGraphKFs.begin()))->mIdKF;
        for (auto i = mLocalGraphKFs.begin(), iend = mLocalGraphKFs.end(); i != iend; i++) {
            PtrKeyFrame pKF = *i;
            if (pKF->isNull())
                continue;
            if (pKF->mIdKF < minKFid)
                minKFid = pKF->mIdKF;
        }
    }

    const int nLocalKFs = mLocalGraphKFs.size();
    const int nRefKFs = mLocalRefKFs.size();

    // Add local graph KeyFrames
    for (int i = 0; i < nLocalKFs; i++) {

        PtrKeyFrame pKF = mLocalGraphKFs[i];

        if (pKF->isNull())
            continue;

        int vertexIdKF = i;

        bool fixed = (pKF->mIdKF == minKFid) || pKF->mIdKF == 1;
        addVertexSE3Expmap(optimizer, toSE3Quat(pKF->getPose()), vertexIdKF, fixed);
    }

    // Add Reference KeyFrames as fixed
    for (int i = 0; i < nRefKFs; i++) {

        PtrKeyFrame pKF = mLocalRefKFs[i];

        if (pKF->isNull())
            continue;

        int vertexIdKF = i + nLocalKFs;

        addVertexSE3Expmap(optimizer, toSE3Quat(pKF->getPose()), vertexIdKF, true);
    }

    maxKFid = nLocalKFs + nRefKFs + 1;

    // Store xyz2uv edges
    const int N = mLocalGraphMPs.size();
    vpEdgesAll.clear();
    vpEdgesAll.reserve(N);
    vnAllIdx.clear();
    vnAllIdx.reserve(N);

    const float delta = Config::ThHuber;

    // Add local graph MapPoints
    for (int i = 0; i < N; i++) {

        PtrMapPoint pMP = mLocalGraphMPs[i];
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

        for (auto j = pKFs.begin(), jend = pKFs.end(); j != jend; j++) {

            PtrKeyFrame pKF = (*j);
            if (pKF->isNull())
                continue;

            if (checkAssociationErr(pKF, pMP)) {
                printf("!! ERR MP: Wrong Association! [LocalGraph loading] \n");
                continue;
            }

            int ftrIdx = pMP->getFtrIdx(pKF);
            int octave = pMP->getOctave(pKF);
            const float invSigma2 = pKF->mvInvLevelSigma2[octave];
            Eigen::Vector2d uv = toVector2d(pKF->keyPointsUn[ftrIdx].pt);
            Eigen::Matrix2d info = Eigen::Matrix2d::Identity() * invSigma2;

            int vertexIdKF = -1;

            auto it1 = std::find(mLocalGraphKFs.begin(), mLocalGraphKFs.end(), pKF);
            if (it1 != mLocalGraphKFs.end()) {
                vertexIdKF = it1 - mLocalGraphKFs.begin();
            } else {
                auto it2 = std::find(mLocalRefKFs.begin(), mLocalRefKFs.end(), pKF);
                if (it2 != mLocalRefKFs.end())
                    vertexIdKF = it2 - mLocalRefKFs.begin() + nLocalKFs;
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
    assert(vpEdgesAll.size() == mLocalGraphMPs.size());
    assert(vnAllIdx.size() == mLocalGraphMPs.size());
}

int Map::removeLocalOutlierMP(const vector<vector<int>>& vnOutlierIdxAll)
{

    locker lock1(mMutexLocalGraph);
    locker lock2(mMutexGlobalGraph);

    assert(vnOutlierIdxAll.size() == mLocalGraphMPs.size());

    const int nLocalKFs = mLocalGraphKFs.size();
    const int N = mLocalGraphMPs.size();
    int nBadMP = 0;

    for (int i = 0, iend = N; i < iend; i++) {

        PtrMapPoint pMP = mLocalGraphMPs[i];

        for (int j = 0, jend = vnOutlierIdxAll[i].size(); j < jend; j++) {
            PtrKeyFrame pKF = NULL;
            int idxKF = vnOutlierIdxAll[i][j];

            if (idxKF < nLocalKFs) {
                pKF = mLocalGraphKFs[idxKF];
            } else {
                pKF = mLocalRefKFs[idxKF - nLocalKFs];
            }

            if (pKF == NULL) {
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

        if (pMP->countObservation() < 2) {
            mMPs.erase(pMP);
            pMP->setNull(pMP);
            nBadMP++;
        }
    }
    return nBadMP;
}

void Map::optimizeLocalGraph(SlamOptimizer& optimizer)
{

    locker lock1(mMutexLocalGraph);
    locker lock2(mMutexGlobalGraph);

    const int nLocalKFs = mLocalGraphKFs.size();
    const int nRefKFs = mLocalRefKFs.size();
    const int N = mLocalGraphMPs.size();
    const int maxKFid = nLocalKFs + nRefKFs + 1;

    for (int i = 0; i < nLocalKFs; i++) {
        PtrKeyFrame pKF = mLocalGraphKFs[i];
        if (pKF->isNull())
            continue;
        Eigen::Vector3d vp = estimateVertexSE2(optimizer, i).toVector();
        pKF->setPose(Se2(vp(0), vp(1), vp(2)));
    }

    for (int i = 0; i < N; i++) {
        PtrMapPoint pMP = mLocalGraphMPs[i];
        if (pMP->isNull() || !pMP->isGoodPrl())
            continue;

        Point3f pos = toCvPt3f(estimateVertexSBAXYZ(optimizer, i + maxKFid));
        pMP->setPos(pos);
        pMP->updateMeasureInKFs();
    }
}

void Map::updateCovisibility(PtrKeyFrame& pNewKF)
{

    locker lock1(mMutexLocalGraph);
    locker lock2(mMutexGlobalGraph);

    for (auto i = mLocalGraphKFs.begin(), iend = mLocalGraphKFs.end(); i != iend; i++) {
        set<PtrMapPoint> spMPs;
        PtrKeyFrame pKFi = *i;
        compareViewMPs(pNewKF, pKFi, spMPs);
        if (spMPs.size() > 0.3f * pNewKF->countObservations()) {
            pNewKF->addCovisibleKF(pKFi);
            pKFi->addCovisibleKF(pNewKF);
        }
    }
}

vector<PtrKeyFrame> Map::getLocalKFs()
{
    locker lock(mMutexLocalGraph);
    return vector<PtrKeyFrame>(mLocalGraphKFs.begin(), mLocalGraphKFs.end());
}

vector<PtrMapPoint> Map::getLocalMPs()
{
    locker lock(mMutexLocalGraph);
    return vector<PtrMapPoint>(mLocalGraphMPs.begin(), mLocalGraphMPs.end());
}

vector<PtrKeyFrame> Map::getRefKFs()
{
    locker lock(mMutexLocalGraph);
    return vector<PtrKeyFrame>(mLocalRefKFs.begin(), mLocalRefKFs.end());
}

bool Map::checkAssociationErr(const PtrKeyFrame& pKF, const PtrMapPoint& pMP)
{
    int ftrIdx0 = pKF->getFtrIdx(pMP);
    int ftrIdx1 = pMP->getFtrIdx(pKF);
    if (ftrIdx0 != ftrIdx1) {
        printf("!! ERR AS: pKF->pMP / pMP->pKF: %d / %d\n", ftrIdx0, ftrIdx1);
    }
    return (ftrIdx0 != ftrIdx1);
}

// Select KF pairs to creat feature constraint between which
vector<pair<PtrKeyFrame, PtrKeyFrame>> Map::SelectKFPairFeat(const PtrKeyFrame& _pKF)
{

    vector<pair<PtrKeyFrame, PtrKeyFrame>> _vKFPairs;

    // Smallest distance between KFs in covis-graph to create a new feature edge
    int threshCovisGraphDist = 5;

    set<PtrKeyFrame> sKFSelected;
    set<PtrKeyFrame> sKFCovis = _pKF->getAllCovisibleKFs();
    set<PtrKeyFrame> sKFLocal =
        GlobalMapper::GetAllConnectedKFs_nLayers(_pKF, threshCovisGraphDist, sKFSelected);

    for (auto iter = sKFCovis.begin(); iter != sKFCovis.end(); iter++) {

        PtrKeyFrame _pKFCand = *iter;
        if (sKFLocal.count(_pKFCand) == 0) {
            sKFSelected.insert(*iter);
            sKFLocal = GlobalMapper::GetAllConnectedKFs_nLayers(_pKF, threshCovisGraphDist, sKFSelected);
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

bool Map::UpdateFeatGraph(const PtrKeyFrame& _pKF)
{

    locker lock1(mMutexLocalGraph);
    locker lock2(mMutexGlobalGraph);

    vector<pair<PtrKeyFrame, PtrKeyFrame>> _vKFPairs = SelectKFPairFeat(_pKF);

    if (_vKFPairs.empty())
        return false;

    int numPairKFs = _vKFPairs.size();
    for (int i = 0; i < numPairKFs; i++) {
        pair<PtrKeyFrame, PtrKeyFrame> pairKF = _vKFPairs[i];
        PtrKeyFrame ptKFFrom = pairKF.first;
        PtrKeyFrame ptKFTo = pairKF.second;
        SE3Constraint ftrCnstr;

        if (GlobalMapper::CreateFeatEdge(ptKFFrom, ptKFTo, ftrCnstr) == 0) {
            ptKFFrom->addFtrMeasureFrom(ptKFTo, ftrCnstr.measure, ftrCnstr.info);
            ptKFTo->addFtrMeasureTo(ptKFFrom, ftrCnstr.measure, ftrCnstr.info);
            if (Config::GlobalPrint) {
                cerr << "## DEBUG GM: add feature constraint from " << ptKFFrom->id << " to "
                     << ptKFTo->id << endl;
            }
        } else {
            if (Config::GlobalPrint)
                cerr << "## DEBUG GM: add feature constraint failed" << endl;
        }
    }

    return true;
}

void Map::loadLocalGraph(SlamOptimizer& optimizer)
{
    WorkTimer timer;
    locker lock(mMutexLocalGraph);

    int camParaId = 0;
    CamPara* campr = addCamPara(optimizer, (Config::Kcam), camParaId);

    int maxKFid = -1;
    int minKFid = -1;

    // If no reference KF, the KF with minId should be fixed
    if (mLocalRefKFs.empty()) {
        minKFid = (*(mLocalGraphKFs.begin()))->id;
        for (auto i = mLocalGraphKFs.begin(), iend = mLocalGraphKFs.end(); i != iend; i++) {
            const PtrKeyFrame pKF = *i;
            if (pKF->isNull())
                continue;
            if (pKF->id < minKFid)
                minKFid = pKF->id;
        }
    }

    const int nLocalKFs = mLocalGraphKFs.size();
    const int nRefKFs = mLocalRefKFs.size();

    // Add local graph KeyFrames
    for (int i = 0; i < nLocalKFs; i++) {
        const PtrKeyFrame& pKF = mLocalGraphKFs[i];
        if (pKF->isNull())
            continue;

        const int vertexIdKF = i;
        const bool fixed = (pKF->id == minKFid) || pKF->id == 1;

        const g2o::SE2 pose(pKF->Twb.x, pKF->Twb.y, pKF->Twb.theta);
        addVertexSE2(optimizer, pose, vertexIdKF, fixed);
    }

    // Add odometry based constraints
    for (int i = 0; i < nLocalKFs; i++) {
        const PtrKeyFrame& pKF = mLocalGraphKFs[i];
        if (pKF->isNull())
            continue;

        const PtrKeyFrame pKF1 = pKF->preOdomFromSelf.first;
        PreSE2 meas = pKF->preOdomFromSelf.second;
        auto it = std::find(mLocalGraphKFs.begin(), mLocalGraphKFs.end(), pKF1);
        if (it == mLocalGraphKFs.end() || pKF1->isNull())
            continue;

        const int id1 = it - mLocalGraphKFs.begin();
        {
            Eigen::Map<Eigen::Matrix3d, RowMajor> info(meas.cov);
            addEdgeSE2(optimizer, Vector3D(meas.meas), i, id1, info);
            //addEdgeSE2_g2o(optimizer, Vector3D(meas.meas), i, id1, info);
        }
    }

    // Add Reference KeyFrames as fixed
    for (int i = 0; i < nRefKFs; i++) {
        const PtrKeyFrame& pKF = mLocalRefKFs[i];
        if (pKF->isNull())
            continue;

        const int vertexIdKF = i + nLocalKFs;

        const g2o::SE2 pose(pKF->Twb.x, pKF->Twb.y, pKF->Twb.theta);
        addVertexSE2(optimizer, pose, vertexIdKF, true);
    }

    maxKFid = nLocalKFs + nRefKFs + 1;

    // Store xyz2uv edges
    const int N = mLocalGraphMPs.size();

    const float delta = Config::ThHuber;

    // Add local graph MapPoints
    for (int i = 0; i < N; i++) {
        const PtrMapPoint& pMP = mLocalGraphMPs[i];
        assert(!pMP->isNull());

        const int vertexIdMP = maxKFid + i;
        const Vector3d lw = toVector3d(pMP->getPos());
        addVertexSBAXYZ(optimizer, lw, vertexIdMP);

        const std::set<PtrKeyFrame> pKFs = pMP->getObservations();
        for (auto j = pKFs.begin(), jend = pKFs.end(); j != jend; j++) {
            const PtrKeyFrame pKFj = (*j);
            if (pKFj->isNull())
                continue;

            if (checkAssociationErr(pKFj, pMP)) {
                printf("!! ERR MP: Wrong Association! [LocalGraph loading] \n");
                continue;
            }

            const int ftrIdx = pMP->getFtrIdx(pKFj);
            const int octave = pMP->getOctave(pKFj);
            const float Sigma2 = pKFj->mvLevelSigma2[octave];
            const Eigen::Vector2d uv = toVector2d(pKFj->keyPointsUn[ftrIdx].pt);

            int vertexIdKF = -1;
            auto it1 = std::find(mLocalGraphKFs.begin(), mLocalGraphKFs.end(), pKFj);
            if (it1 != mLocalGraphKFs.end()) {
                vertexIdKF = it1 - mLocalGraphKFs.begin();
            } else {
                auto it2 = std::find(mLocalRefKFs.begin(), mLocalRefKFs.end(), pKFj);
                if (it2 != mLocalRefKFs.end())
                    vertexIdKF = it2 - mLocalRefKFs.begin() + nLocalKFs;
            }
            if (vertexIdKF == -1)
                continue;

            // compute covariance/information
            const Matrix2d Sigma_u = Eigen::Matrix2d::Identity() * Sigma2;
            const Vector3d lc = toVector3d(pKFj->mViewMPs[ftrIdx]);

            const double zc = lc(2);
            const double zc_inv = 1. / zc;
            const double zc_inv2 = zc_inv * zc_inv;
            const float& fx = Config::fx;
            Matrix23d J_pi;
            J_pi << fx * zc_inv, 0, -fx * lc(0) * zc_inv2, 0, fx * zc_inv, -fx * lc(1) * zc_inv2;
            const Matrix3d Rcw = toMatrix3d(pKFj->Tcw.rowRange(0, 3).colRange(0, 3));
            const Vector3d pi(pKFj->Twb.x, pKFj->Twb.y, 0);

            const Matrix23d J_pi_Rcw = J_pi * Rcw;

            const Matrix2d J_rotxy = (J_pi_Rcw * skew(lw - pi)).block<2, 2>(0, 0);
            const Matrix<double, 2, 1> J_z = -J_pi_Rcw.block<2, 1>(0, 2);
            const float Sigma_rotxy = 1. / Config::PlaneMotionInfoXrot;
            const float Sigma_z = 1. / Config::PlaneMotionInfoZ;
            const Matrix2d Sigma_all =
                Sigma_rotxy * J_rotxy * J_rotxy.transpose() + Sigma_z * J_z * J_z.transpose() + Sigma_u;

            addEdgeSE2XYZ(optimizer, uv, vertexIdKF, vertexIdMP, campr, toSE3Quat(Config::Tbc),
                          Sigma_all.inverse(), delta);
        }
    }

    size_t nVertices = optimizer.vertices().size();
    size_t nEdges = optimizer.edges().size();
    cout << "[ Map ][Info ] #" << mCurrentKF->id << "(KF#" << mCurrentKF->mIdKF
         << ") 加载LocalGraph: 边数为" << nEdges << ", 节点数为" << nVertices
         << ": LocalKFs/nRefKFs/nMPs = " << nLocalKFs << "/" << nRefKFs << "/" << N << ", 耗时"
         << setiosflags(ios::fixed) << setprecision(2) << timer.count() << "ms" << endl;
}

}  // namespace se2lam
