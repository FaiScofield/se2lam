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

    void setMap(Map* pMap) { mpMap = pMap; mpMap->setLocalMapper(this); }
    void setGlobalMapper(GlobalMapper* pGlobalMapper) { mpGlobalMapper = pGlobalMapper; }

    void addNewKF(PtrKeyFrame& pKF, const std::vector<cv::Point3f>& localMPs,
                  const std::vector<int>& vMatched12, const std::vector<bool>& vbGoodPrl);

    void findCorrespd(const std::vector<int>& vMatched12, const std::vector<cv::Point3f>& localMPs,
                      const std::vector<bool>& vbGoodPrl);

    void updateLocalGraphInMap();
    void pruneRedundantKFinMap();
    void removeOutlierChi2();
    void localBA();

    void setAbortBA() { mbAbortBA = true; }
    bool acceptNewKF();
    void setAcceptNewKF(bool value);
    void setGlobalBABegin(bool value);

    void requestFinish();
    bool isFinished();

    // For debugging by hbtang
    void printOptInfo(const SlamOptimizer& _optimizer);

    bool mbPrintDebugInfo;
    std::mutex mutexMapper;

protected:
    bool checkFinish();
    void setFinish();

    Map* mpMap;
    GlobalMapper* mpGlobalMapper;

    PtrKeyFrame mpNewKF;
    std::mutex mMutexNewKFs;

    bool mbAcceptNewKF;
    bool mbUpdated;
    std::mutex mMutexAccept;

    bool mbAbortBA;
    bool mbGlobalBABegin;
    std::mutex mMutexLocalGraph;

    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    int mnMaxLocalFrames;  // 0表示无上限
    int mnSearchLevel;
    float mfSearchRadius;
    //    bool mbStopped;
    //    bool mbStopRequested;
    //    bool mbNotStop;
    //    std::mutex mMutexStop;
};
}


#endif  // LOCALMAPPER_H
