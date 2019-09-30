/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/


#ifndef LOCALMAPPER_H
#define LOCALMAPPER_H

#include "Map.h"
#include "Track.h"
#include "optimizer.h"

namespace se2lam
{

class Track;
class GlobalMapper;

class LocalMapper
{
public:
    LocalMapper();

    void run();

    void setMap(Map* pMap);
    void setGlobalMapper(GlobalMapper* pGlobalMapper);

    void addNewKF(PtrKeyFrame& pKF, const std::vector<cv::Point3f>& localMPs,
                  const std::vector<int>& vMatched12, const std::vector<bool>& vbGoodPrl);

    void findCorrespd(const std::vector<int>& vMatched12, const std::vector<cv::Point3f>& localMPs,
                      const std::vector<bool>& vbGoodPrl);

    void removeOutlierChi2();

    void localBA();
    void setAbortBA();
    bool acceptNewKF();
    void setAcceptNewKF(bool value);
    void setGlobalBABegin(bool value);

    // For debugging by hbtang
    void printOptInfo(const SlamOptimizer& _optimizer);

    void requestFinish();
    bool isFinished();

    void updateLocalGraphInMap();
    void pruneRedundantKFinMap();

    //    int getNumFKsInQueue();

    bool mbPrintDebugInfo;

    std::mutex mutexMapper;

protected:
    Map* mpMap;
    GlobalMapper* mpGlobalMapper;
    ORBVocabulary* mpORBVoc;

    PtrKeyFrame mpNewKF;
    std::mutex mMutexNewKFs;

    bool mbAcceptNewKF;
    bool mbUpdated;
    std::mutex mMutexAccept;

    bool mbAbortBA;
    bool mbGlobalBABegin;
    std::mutex mMutexLocalGraph;

    bool checkFinish();
    void setFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    //    bool mbStopped;
    //    bool mbStopRequested;
    //    bool mbNotStop;
    //    std::mutex mMutexStop;
};
}


#endif  // LOCALMAPPER_H
