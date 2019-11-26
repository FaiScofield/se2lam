/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/


#ifndef MAP_H
#define MAP_H
#pragma once

#include "Config.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "optimizer.h"

#include <set>
#include <unordered_map>

namespace se2lam
{

class MapPoint;
class KeyFrame;
class LocalMapper;
typedef std::shared_ptr<MapPoint> PtrMapPoint;
typedef std::shared_ptr<KeyFrame> PtrKeyFrame;

class Map
{

public:
    Map();
    ~Map();

    void insertKF(const PtrKeyFrame& pkf);
    void insertMP(const PtrMapPoint& pmp);
    void eraseKF(const PtrKeyFrame& pKF);
    void eraseMP(const PtrMapPoint& pMP);
    void mergeMP(PtrMapPoint& toKeep, PtrMapPoint& toDelete);
    size_t countKFs();
    size_t countMPs();
    size_t countLocalKFs();
    size_t countLocalMPs();
    size_t countRefKFs();
    std::vector<PtrKeyFrame> getAllKFs();
    std::vector<PtrMapPoint> getAllMPs();
    std::vector<PtrKeyFrame> getLocalKFs();
    std::vector<PtrMapPoint> getLocalMPs();
    std::vector<PtrKeyFrame> getRefKFs();

    PtrKeyFrame getCurrentKF();
    void setCurrentKF(const PtrKeyFrame& pKF);
    void setCurrentFramePose(const cv::Mat& pose);
    cv::Mat getCurrentFramePose();

    void clear();
    bool empty() { return isEmpty; }

    static cv::Point2f compareViewMPs(const PtrKeyFrame& pKF1, const PtrKeyFrame& pKF2,
                                      std::set<PtrMapPoint>& spMPs);
    static float compareViewMPs(const PtrKeyFrame& pKF, const std::vector<PtrKeyFrame>& vpKFs,
                                std::set<PtrMapPoint>& spMPs, int k = 2);
    static bool checkAssociationErr(const PtrKeyFrame& pKF, const PtrMapPoint& pMP);

    //! For LocalMapper
    void setLocalMapper(LocalMapper* pLocalMapper) { mpLocalMapper = pLocalMapper; }
    void updateLocalGraph(int maxLevel = 3, int maxN = 20, float searchRadius = 5.f);
    void updateLocalGraph_new();
    void updateCovisibility(const PtrKeyFrame& pNewKF);
    void addLocalGraphThroughKdtree(std::set<PtrKeyFrame>& setLocalKFs, int maxN = 10,
                                    float searchRadius = 5.f);
    bool pruneRedundantKF();
    int removeLocalOutlierMP(const std::vector<std::vector<int>>& vnOutlierIdxAll);

    void loadLocalGraph(SlamOptimizer& optimizer);
    void loadLocalGraph(SlamOptimizer& optimizer,
                        std::vector<std::vector<g2o::EdgeProjectXYZ2UV*>>& vpEdgesAll,
                        std::vector<std::vector<int>>& vnAllIdx);

    void loadLocalGraphOnlyBa(SlamOptimizer& optimizer,
                              std::vector<std::vector<g2o::EdgeProjectXYZ2UV*>>& vpEdgesAll,
                              std::vector<std::vector<int>>& vnAllIdx);
    void optimizeLocalGraph(SlamOptimizer& optimizer);

    //! For GlobalMapper
    void mergeLoopClose(const std::map<int, int>& mapMatchMP, PtrKeyFrame& pKFCurr,
                        PtrKeyFrame& pKFLoop);

    //! Set KF pair waiting for feature constraint generation, called by localmapper
    std::vector<std::pair<PtrKeyFrame, PtrKeyFrame>> SelectKFPairFeat(const PtrKeyFrame& _pKF);

    //! Update feature constraint graph, on KFs pairs given by LocalMapper
    bool updateFeatGraph(const PtrKeyFrame& _pKF);

    cv::SparseMat mFtrBasedGraph;
    cv::SparseMat mOdoBasedGraph;
    std::unordered_map<int, SE3Constraint> mFtrBasedEdges;
    std::unordered_map<int, SE3Constraint> mOdoBasedEdges;
    std::vector<int> mIdxFtrBased;
    std::vector<int> mIdxOdoBased;

    bool mbNewKFinserted; // for visualizaiton.
protected:
    PtrKeyFrame mCurrentKF;
    cv::Mat mCurrentFramePose;  // Tcw

    bool isEmpty;

    //! Global Map
    std::set<PtrMapPoint, MapPoint::IdLessThan> mspMPs;  // 全局地图点集合，以id升序排序
    std::set<PtrKeyFrame, KeyFrame::IdLessThan> mspKFs;  // 全局关键帧集合，以id升序排序

    //! Local Map
    //! updateLocalGraph()和pruneRedundantKF()会更新此变量, 都是LocalMapper在调用. 根据id升序排序
    std::vector<PtrMapPoint> mvLocalGraphMPs;
    std::vector<PtrKeyFrame> mvLocalGraphKFs;
    std::vector<PtrKeyFrame> mvRefKFs;
    LocalMapper* mpLocalMapper;

    std::mutex mMutexGlobalGraph;
    std::mutex mMutexLocalGraph;
    std::mutex mMutexCurrentKF;
    std::mutex mMutexCurrentFrame;
};  // class Map

}  // namespace se2lam

#endif
