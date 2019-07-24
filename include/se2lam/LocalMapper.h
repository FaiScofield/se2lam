/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/


#ifndef LOCALMAPPER_H
#define LOCALMAPPER_H

#include "Map.h"
#include "optimizer.h"

namespace se2lam{

//#define TIME_TO_LOG_LOCAL_BA

class GlobalMapper;

class LocalMapper{
public:
    LocalMapper();

    void run();

    void setMap(Map *pMap);
    ///
    /// \brief setGlobalMapper
    /// \param pGlobalMapper
    ///
    void setGlobalMapper(GlobalMapper* pGlobalMapper);

    /**
     * @brief addNewKF
     * @param pKF - key frame pointer
     * @param localMPs - local map points
     * @param vMatched12 - matches
     * @param vbGoodPrl - vector of flags for map points with good parallax
     */
    void addNewKF(PtrKeyFrame &pKF, const std::vector<cv::Point3f>& localMPs, const std::vector<int> &vMatched12, const std::vector<bool>& vbGoodPrl);

    /**
     * @brief findCorrespd
     * @param vMatched12
     * @param localMPs
     * @param vbGoodPrl
     */
    void findCorrespd(const std::vector<int> &vMatched12, const std::vector<cv::Point3f> &localMPs, const std::vector<bool>& vbGoodPrl);

    void removeOutlierChi2();

    void localBA();

    void setAbortBA();

    bool acceptNewKF();

    void setGlobalBABegin(bool value);

    void printOptInfo(const SlamOptimizer & _optimizer);    // For debugging by hbtang

    void requestFinish();
    bool isFinished();

    void updateLocalGraphInMap();

    void pruneRedundantKFinMap();

    bool mbPrintDebugInfo;
    std::mutex mutexMapper;

protected:
    Map* mpMap;
    GlobalMapper* mpGlobalMapper;
    PtrKeyFrame mNewKF;

    bool mbUpdated;
    bool mbAbortBA;
    bool mbAcceptNewKF;
    bool mbGlobalBABegin;

    std::mutex mMutexLocalGraph;

    bool checkFinish();
    void setFinish();
    bool mbFinishRequested;
    bool mbFinished;

    std::mutex mMutexFinish;
};

}


#endif // LOCALMAPPER_H
