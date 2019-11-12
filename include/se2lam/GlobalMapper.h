/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#ifndef GLOBALMAPPER_H
#define GLOBALMAPPER_H

#include "KeyFrame.h"
#include "LocalMapper.h"
#include "Map.h"
#include "Track.h"
#include <condition_variable>

#include "converter.h"
#include "optimizer.h"
#include "sparsifier.h"

#include "Config.h"

#include "ORBVocabulary.h"
#include "ORBmatcher.h"

namespace se2lam
{

class Map;
class LocalMapper;

class GlobalMapper
{
public:
    GlobalMapper();

    void run();

    void setMap(Map* pMap) { mpMap = pMap; }
    void setLocalMapper(LocalMapper* pLocalMapper) { mpLocalMapper = pLocalMapper; }
    void setUpdated(bool val);
    bool checkGMReady();

    // Feature Constraint Graph Functions ...
    // Update feature constraint graph, on KFs pairs given by LocalMapper
    void updataFeatGraph(vector<pair<PtrKeyFrame, PtrKeyFrame>>& _vKFPairs);

    // Set KF pair waiting for feature constraint generation
    //! 并没有用上
    void setKFPairFeatEdge(PtrKeyFrame _pKFFrom, PtrKeyFrame _pKFTo);

    // Create single feature constraint between 2 KFs
    static int createFeatEdge(PtrKeyFrame _pKFFrom, PtrKeyFrame _pKFTo,
                              SE3Constraint& SE3CnstrOutput);
    static int createFeatEdge(PtrKeyFrame _pKFFrom, PtrKeyFrame _pKFTo, map<int, int>& mapMatch,
                              SE3Constraint& SE3CnstrOutput);

    // Do local optimization with chosen KFs and MPs
    static void optKFPair(const vector<PtrKeyFrame>& _vPtrKFs, const vector<PtrMapPoint>& _vPtrMPs,
                          vector<g2o::SE3Quat, Eigen::aligned_allocator<g2o::SE3Quat>>& _vSe3KFs,
                          vector<g2o::Vector3D, Eigen::aligned_allocator<g2o::Vector3D>>& _vPt3MPs);
    static void optKFPairMatch(PtrKeyFrame _pKF1, PtrKeyFrame _pKF2, map<int, int>& mapMatch,
                   vector<g2o::SE3Quat, Eigen::aligned_allocator<g2o::SE3Quat>>& _vSe3KFs,
                   vector<g2o::Vector3D, Eigen::aligned_allocator<g2o::Vector3D>>& _vPt3MPs);

    // generate MeasSE3XYZ measurement vector
    static void createVecMeasSE3XYZ(const vector<PtrKeyFrame> _vpKFs, const vector<PtrMapPoint> _vpMPs,
                        vector<MeasSE3XYZ, Eigen::aligned_allocator<MeasSE3XYZ>>& vMeas);

    vector<pair<PtrKeyFrame, PtrKeyFrame>> selectKFPairFeat(const PtrKeyFrame _pKF);

    static set<PtrKeyFrame> getAllConnectedKFs(const PtrKeyFrame _pKF,
                                               set<PtrKeyFrame> _sKFSelected = set<PtrKeyFrame>());
    static set<PtrKeyFrame> getAllConnectedKFs_nLayers(const PtrKeyFrame _pKF, int numLayers = 10,
                               set<PtrKeyFrame> _sKFSelected = set<PtrKeyFrame>());


    // Loop Closing ...
    void setORBVoc(ORBVocabulary* pORBVoc) { mpORBVoc = pORBVoc; }
    void computeBowVecAll();
    bool detectLoopClose();
    bool verifyLoopClose(map<int, int>& _mapMatchMP, map<int, int>& _mapMatchAll,
                         map<int, int>& _mapMatchRaw);

    void globalBA();

    void removeMatchOutlierRansac(PtrKeyFrame _pKFCurrent, PtrKeyFrame _pKFLoop,
                                  map<int, int>& mapiMatch);
    void removeKPMatch(PtrKeyFrame _pKFCurrent, PtrKeyFrame _pKFLoop, map<int, int>& mapiMatch);

    void setBusy(bool v);
    void waitIfBusy();

    void requestFinish();
    bool isFinished();

    void drawMatch(const map<int, int>& mapiMatch);
    void drawMatch(const vector<int>& viMatch);

    // DEBUG Functions ... Print SE3Quat
    void printSE3(const g2o::SE3Quat se3);
    void printOptInfo(const SlamOptimizer& _optimizer);
    void printOptInfo(const vector<g2o::EdgeSE3*>& vpEdgeOdo,
                      const vector<g2o::EdgeSE3*>& vpEdgeFeat,
                      const vector<g2o::EdgeSE3Prior*>& vpEdgePlane, double threshChi2 = 30.0,
                      bool bPrintMatInfo = false);

public:
    ORBVocabulary* mpORBVoc;
    PtrKeyFrame mpLastKFLoopDetect;

    cv::Mat mImgLoop;
    cv::Mat mImgCurr;
    cv::Mat mImgMatch;

protected:
    bool checkFinish();
    void setFinish();

    Map* mpMap;
    LocalMapper* mpLocalMapper;

    PtrKeyFrame mpKFCurr;
    PtrKeyFrame mpKFLoop;
    std::deque<std::pair<PtrKeyFrame, PtrKeyFrame>> mdeqPairKFs;

    bool mbUpdated;  //! 没用
    bool mbNewKF;

    bool mbGlobalBALastLoop;

    std::condition_variable mcIsBusy;
    bool mbIsBusy;
    bool mbFinished;
    bool mbFinishRequested;

    std::mutex mMutexBusy;
    std::mutex mMutexFinish;
};

}  // namespace se2lam

#endif  // GLOBALMAPPER_H
