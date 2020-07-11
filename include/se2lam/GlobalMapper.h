/**
 * This file is part of se2lam
 *
 * Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
 */

#ifndef GLOBALMAPPER_H
#define GLOBALMAPPER_H

#include "ORBVocabulary.h"
#include "Sparsifier.h"
#include "optimizer.h"
#include <condition_variable>

namespace se2lam
{

class Map;
class LocalMapper;
class MapPublish;

class GlobalMapper
{
public:
    GlobalMapper();

    void setMap(Map* pMap) { mpMap = pMap; }
    void setLocalMapper(LocalMapper* pLocalMapper) { mpLocalMapper = pLocalMapper; }
    void setMapPublisher(MapPublish* pMapPublisher) { mpMapPublisher = pMapPublisher; }
    void setORBVoc(ORBVocabulary* pORBVoc) { mpORBVoc = pORBVoc; }
    void setUpdated(bool val);

    bool CheckGMReady();

    void run();

    // Feature Constraint Graph Functions ...

    // Update feature constraint graph, on KFs pairs given by LocalMapper
    //    void UpdateFeatGraph();
    void UpdataFeatGraph(vector<pair<PtrKeyFrame, PtrKeyFrame>>& _vKFPairs);

    // Set KF pair waiting for feature constraint generation, called by localmapper
    void SetKFPairFeatEdge(PtrKeyFrame _pKFFrom, PtrKeyFrame _pKFTo);

    // Create single feature constraint between 2 KFs
    static int CreateFeatEdge(PtrKeyFrame _pKFFrom, PtrKeyFrame _pKFTo, SE3Constraint& SE3CnstrOutput);
    static int CreateFeatEdge(PtrKeyFrame _pKFFrom, PtrKeyFrame _pKFTo, map<int, int>& mapMatch,
                              SE3Constraint& SE3CnstrOutput);

    // Do local optimization with chosen KFs and MPs
    static void OptKFPair(const vector<PtrKeyFrame>& _vPtrKFs, const vector<PtrMapPoint>& _vPtrMPs,
                          vector<g2o::SE3Quat, Eigen::aligned_allocator<g2o::SE3Quat>>& _vSe3KFs,
                          vector<g2o::Vector3D, Eigen::aligned_allocator<g2o::Vector3D>>& _vPt3MPs);
    static void OptKFPairMatch(PtrKeyFrame _pKF1, PtrKeyFrame _pKF2, map<int, int>& mapMatch,
                               vector<g2o::SE3Quat, Eigen::aligned_allocator<g2o::SE3Quat>>& _vSe3KFs,
                               vector<g2o::Vector3D, Eigen::aligned_allocator<g2o::Vector3D>>& _vPt3MPs,
                               set<int>& sIdMPin1Outlier);

    // generate MeasSE3XYZ measurement vector
    static void CreateVecMeasSE3XYZ(const vector<PtrKeyFrame> _vpKFs, const vector<PtrMapPoint> _vpMPs,
                                    vector<MeasSE3XYZ, Eigen::aligned_allocator<MeasSE3XYZ>>& vMeas);

    vector<pair<PtrKeyFrame, PtrKeyFrame>> SelectKFPairFeat(const PtrKeyFrame _pKF);

    static set<PtrKeyFrame> GetAllConnectedKFs(const PtrKeyFrame _pKF,
                                               set<PtrKeyFrame> _sKFSelected = set<PtrKeyFrame>());
    static set<PtrKeyFrame> GetAllConnectedKFs_nLayers(const PtrKeyFrame _pKF, int numLayers = 10,
                                                       set<PtrKeyFrame> _sKFSelected = set<PtrKeyFrame>());


    // Loop Closing ...
    void ComputeBowVecAll();
    bool DetectLoopClose();
    bool VerifyLoopClose(map<int, int>& _mapMatchMP, map<int, int>& _mapMatchAll, map<int, int>& _mapMatchRaw);
    void GlobalBA();

    void RemoveMatchOutlierRansac(PtrKeyFrame _pKFCurrent, PtrKeyFrame _pKFLoop, map<int, int>& mapiMatch);
    void RemoveKPMatch(PtrKeyFrame _pKFCurrent, PtrKeyFrame _pKFLoop, map<int, int>& mapiMatch);

    // DEBUG Functions ...
    // Print SE3Quat
    void PrintSE3(const g2o::SE3Quat se3);
    void PrintOptInfo(const SlamOptimizer& _optimizer);
    void PrintOptInfo(const vector<g2o::EdgeSE3*>& vpEdgeOdo, const vector<g2o::EdgeSE3*>& vpEdgeFeat,
                      const vector<g2o::EdgeSE3Prior*>& vpEdgePlane, double threshChi2 = 30.0,
                      bool bPrintMatInfo = false);

    void setBusy(bool v);
    void waitIfBusy();

    void requestFinish();
    bool isFinished();

    PtrKeyFrame mpLastKFLoopDetect;
    bool mbExit;

private:
    // visulization
    void DrawMatch(const map<int, int>& mapiMatch);
    void DrawMatch(const vector<int>& viMatch);
    void copyForPub(const map<int, int>& mapiMatch, bool closed);
    cv::Mat mImgLoop;
    cv::Mat mImgCurr;
    cv::Mat mImgMatch;

    Map* mpMap;
    LocalMapper* mpLocalMapper;
    MapPublish* mpMapPublisher;

    ORBVocabulary* mpORBVoc;


    PtrKeyFrame mpKFCurr;
    PtrKeyFrame mpKFLoop;
    std::deque<std::pair<PtrKeyFrame, PtrKeyFrame>> mdeqPairKFs;

    bool mbUpdated;
    bool mbNewKF;

    bool mbGlobalBALastLoop;

    bool mbIsBusy;
    std::condition_variable mcIsBusy;
    std::mutex mMutexBusy;

    bool checkFinish();
    void setFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;
};

}  // namespace se2lam

#endif  // GLOBALMAPPER_H
