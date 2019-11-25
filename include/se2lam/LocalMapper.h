/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/


#ifndef LOCALMAPPER_H
#define LOCALMAPPER_H

#include "Map.h"
#include "optimizer.h"

namespace se2lam
{

class GlobalMapper;

class LocalMapper
{
public:
    LocalMapper();

    void setMap(Map* pMap) { mpMap = pMap; mpMap->setLocalMapper(this); }
    void setGlobalMapper(GlobalMapper* pGlobalMapper) { mpGlobalMapper = pGlobalMapper; }

    void run();

    void addNewKF(const PtrKeyFrame& pKF, const std::map<size_t, MPCandidate>& MPCandidates);
    bool checkNewKF();
    void processNewKF();
    void findCorresponds(const std::map<size_t, MPCandidate>& MPCandidates);

    void updateLocalGraphInMap();
    void pruneRedundantKFinMap();
    void removeOutlierChi2();
    void localBA();

    bool checkIfAcceptNewKF();
    void setAbortBA() { mbAbortBA = true; }
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
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    Map* mpMap;
    GlobalMapper* mpGlobalMapper;

    int mnMaxLocalFrames;  // 局部KF的最大数量, 0表示无上限
    int mnSearchLevel;     // 局部KF的搜索层数
    float mfSearchRadius;  // 局部KF的FLANN近邻搜索半v径

    std::map<size_t, MPCandidate> mMPCandidates;  // 等待处理的MP候选点
    std::list<PtrKeyFrame> mlNewKFs; // 等待处理的关键帧列表
    PtrKeyFrame mpNewKF;
    std::mutex mMutexNewKFs;

    bool mbAcceptNewKF;
    std::mutex mMutexAccept;

    bool mbAbortBA;
    bool mbGlobalBABegin;
    std::mutex mMutexLocalGraph;
};

}  // namespace se2lam

#endif  // LOCALMAPPER_H
