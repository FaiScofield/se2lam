/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/


#ifndef LOCALIZE_H
#define LOCALIZE_H

#include "Config.h"
#include "Frame.h"
#include "KeyFrame.h"
#include "Map.h"
#include "Sensors.h"
#include "cvutil.h"
#include "ORBVocabulary.h"

namespace se2lam
{

class Localizer
{
public:
    //! Functions
    Localizer();
    ~Localizer();

    // Main
    void run();
    bool relocalization();

    // Initialization
    void setMap(Map* pMap);
    void setORBVoc(ORBVocabulary* pORBVoc);
    void setSensors(Sensors* pSensors);

    void ReadFrameInfo(const cv::Mat& img, const float& imgTime, const Se2& odo);

    void MatchLocalMap();

    // Loop closing
    bool DetectLoopClose();
    bool VerifyLoopClose(std::map<int, int>& mapMatchMP, std::map<int, int>& mapMatchAll,
                         std::map<int, int>& mapMatchRaw);
    void MatchLoopClose(std::map<int, int> mapMatchGood);

    void DoLocalBA();
    void DetectIfLost();
    cv::Mat DoPoseGraphOptimization(int iterNum);
    bool TrackLocalMap();

    // Local map
    void UpdateLocalMap(int searchLevel = 3);
    void UpdateLocalMapTrack();
    void ResetLocalMap();

    // Subfunctions
    void RemoveMatchOutlierRansac(PtrKeyFrame pKFCurr, PtrKeyFrame pKFLoop,
                                  std::map<int, int>& mapMatch);
    void ComputeBowVecAll();
    std::vector<PtrKeyFrame> GetLocalKFs();
    std::vector<PtrMapPoint> GetLocalMPs();

    // IO
    void UpdatePoseCurr();
    void DrawImgMatch(const std::map<int, int>& mapMatch);
    void DrawImgCurr();

    void UpdateCovisKFCurr();
    int FindCommonMPs(const PtrKeyFrame pKF1, const PtrKeyFrame pKF2, std::set<PtrMapPoint>& spMPs);

    // DEBUG
    void WriteTrajFile(std::ofstream& file);

    void requestFinish();
    bool isFinished();

    Se2 getCurrentFrameOdom();
    Se2 getCurrKFPose();
    Se2 getRefKFPose();
    PtrKeyFrame getKFCurr();

    void setTrackingState(const cvu::eTrackingState& s);
    void setLastTrackingState(const cvu::eTrackingState& s);
    cvu::eTrackingState getTrackingState();
    cvu::eTrackingState getLastTrackingState();
    void addLocalGraphThroughKdtree(std::set<PtrKeyFrame>& setLocalKFs);

public:
    //! Variables
    Map* mpMap;
    ORBextractor* mpORBextractor;
    ORBVocabulary* mpORBVoc;

    Frame mFrameCurr;
    PtrKeyFrame mpKFCurr;
    PtrKeyFrame mpKFCurrRefined;

    Frame mFrameRef;
    PtrKeyFrame mpKFRef;
    PtrKeyFrame mpKFLoop;

    std::set<PtrMapPoint> mspMPLocal;
    std::set<PtrKeyFrame> mspKFLocal;

    cv::Mat mImgCurr;
    cv::Mat mImgLoop;
    cv::Mat mImgMatch;

    std::mutex mMutexImg;
    std::mutex mMutexMPLocal;
    std::mutex mMutexKFLocal;
    std::mutex mMutexLocalMap;

    std::vector<double> mvScores;
    std::vector<double> mvLocalScores;

protected:
    bool checkFinish();
    void setFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    Sensors* mpSensors;

    int nLostFrames;
    std::condition_variable cndvFirstKFUpdate;

    float thMaxDistance;
    float thMaxAngular;

    std::mutex mMutexState;
    cvu::eTrackingState mState;
    cvu::eTrackingState mLastState;
};
}


#endif  // LOCALIZE_H
