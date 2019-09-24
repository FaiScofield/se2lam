/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/


#ifndef MAP_H
#define MAP_H

#include "Config.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "optimizer.h"
#include <opencv2/flann.hpp>
#include <set>
#include <unordered_map>

namespace se2lam
{

class LocalMapper;

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

    std::vector<PtrKeyFrame> getAllKF();
    std::vector<PtrMapPoint> getAllMP();
    std::vector<PtrKeyFrame> getLocalKFs();
    std::vector<PtrMapPoint> getLocalMPs();
    std::vector<PtrKeyFrame> getRefKFs();
    size_t countKFs();
    size_t countMPs();
    size_t countLocalKFs();
    size_t countLocalMPs();

    void clear();
    bool empty();

    void setCurrentKF(const PtrKeyFrame& pKF);
    PtrKeyFrame getCurrentKF();

    void setCurrentFramePose(const cv::Mat& pose);
    cv::Mat getCurrentFramePose();

    static cv::Point2f compareViewMPs(const PtrKeyFrame& pKF1, const PtrKeyFrame& pKF2,
                                      std::set<PtrMapPoint>& spMPs);
    static float compareViewMPs(const PtrKeyFrame& pKF, const set<PtrKeyFrame>& spKFs,
                                 std::set<PtrMapPoint>& spMPs, int k = 2);
    static bool checkAssociationErr(const PtrKeyFrame& pKF, const PtrMapPoint& pMP);


    //! For LocalMapper
    void setLocalMapper(LocalMapper* pLocalMapper);
    void updateLocalGraph();
    void updateCovisibility(PtrKeyFrame& pNewKF);
    void addLocalGraphThroughKdtree(std::set<PtrKeyFrame, KeyFrame::IdLessThan>& setLocalKFs);

    bool pruneRedundantKF();
    int removeLocalOutlierMP(const vector<vector<int>>& vnOutlierIdxAll);

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
    std::vector<pair<PtrKeyFrame, PtrKeyFrame>> SelectKFPairFeat(const PtrKeyFrame& _pKF);

    //! Update feature constraint graph, on KFs pairs given by LocalMapper
    bool UpdateFeatGraph(const PtrKeyFrame& _pKF);

public:
    cv::SparseMat mFtrBasedGraph;
    cv::SparseMat mOdoBasedGraph;
    std::unordered_map<int, SE3Constraint> mFtrBasedEdges;
    std::unordered_map<int, SE3Constraint> mOdoBasedEdges;
    std::vector<int> mIdxFtrBased;
    std::vector<int> mIdxOdoBased;

protected:
    PtrKeyFrame mCurrentKF;
    cv::Mat mCurrentFramePose;  // Tcw

    bool isEmpty;

    //! Global Map
    std::set<PtrMapPoint, MapPoint::IdLessThan> mMPs;  // 全局地图点集合，以id升序排序
    std::set<PtrKeyFrame, KeyFrame::IdLessThan> mKFs;  // 全局关键帧集合，以id升序排序

    //! Local Map
    //! updateLocalGraph()和pruneRedundantKF()会更新此变量, 都是LocalMapper在调用
    std::vector<PtrMapPoint> mLocalGraphMPs;
    std::vector<PtrKeyFrame> mLocalGraphKFs;
    std::vector<PtrKeyFrame> mRefKFs;
    LocalMapper* mpLocalMapper;

    std::mutex mMutexGlobalGraph;
    std::mutex mMutexLocalGraph;
    std::mutex mMutexCurrentKF;
    std::mutex mMutexCurrentFrame;

};  // class Map

}  // namespace se2lam

#endif
