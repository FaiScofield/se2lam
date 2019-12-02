/**
 * This file is part of se2lam
 *
 * Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
 */

#ifndef TRACK_H
#define TRACK_H

#include "Config.h"
#include "Frame.h"
#include "ORBmatcher.h"
#include "Sensors.h"
#include "Thirdparty/g2o/g2o/types/sba/types_six_dof_expmap.h"
#include "cvutil.h"

namespace se2lam
{

class KeyFrame;
class Map;
class LocalMapper;
class GlobalMapper;
class MapPublish;

typedef std::shared_ptr<KeyFrame> PtrKeyFrame;


class Track
{
public:
    Track();
    ~Track();

    void run();

    void setMap(Map* pMap) { mpMap = pMap; }
    void setLocalMapper(LocalMapper* pLocalMapper) { mpLocalMapper = pLocalMapper; }
    void setGlobalMapper(GlobalMapper* pGlobalMapper) { mpGlobalMapper = pGlobalMapper; }
    void setSensors(Sensors* pSensors) { mpSensors = pSensors; }
    void setMapPublisher(MapPublish* pMapPublisher) { mpMapPublisher = pMapPublisher; }

    cv::Mat getAffineMatrix(const Se2& dOdo);
    int removeOutliers(const PtrKeyFrame& pKFRef, const Frame* pFCur, std::vector<int>& vKPMatchIdx, cv::Mat& A12);
    int removeOutliers(const PtrKeyFrame& pKFRef, const Frame* pFCur, std::map<int, int>& mapKPMatchIdx, cv::Mat& A12);

    // for visulization
    unsigned long getCurrentFrameID() { return mCurrentFrame.id; }
    Se2 getCurrentFrameOdo() { return mCurrentFrame.odom; }
    cv::Mat getCurrentFramePose() { return mCurrentFrame.getPose(); }
    void copyForPub();

    bool isFinished();
    void requestFinish();

    cv::Mat computeA12(const PtrKeyFrame& pKFRef, const Frame* pFCur, std::vector<int>& vKPMatchIdx);

public:
    // Tracking states
    cvu::eTrackingState mState;
    cvu::eTrackingState mLastState;

    int N1 = 0, N2 = 0, N3 = 0;  // for debug print
    double trackTimeTatal = 0.;

private:
    void processFirstFrame();
    bool trackLastFrame();
    bool trackReferenceKF();
    bool trackLocalMap();

    void updateFramePoseFromLast();
    void updateFramePoseFromRef();

    void doTriangulate(PtrKeyFrame& pKF);
    void resetLocalTrack();
    bool needNewKF();
    void addNewKF();

    // Relocalization
    bool detectIfLost();
    bool detectIfLost(int cros, double projError);

    bool doRelocalization();
    bool detectLoopClose();
    bool verifyLoopClose();
    void doLocalBA(Frame& pKF);
    void startNewTrack();

    void checkReady();
    bool checkFinish();
    void setFinish();

private:
    static bool mbUseOdometry;  //! TODO 冗余变量
    bool mbPrint;
    bool mbNeedVisualization;
    bool mbRelocalized;  // 成功重定位标志
    std::string mImageText;

    // set in OdoSLAM class
    Map* mpMap;
    LocalMapper* mpLocalMapper;
    GlobalMapper* mpGlobalMapper;
    Sensors* mpSensors;
    MapPublish* mpMapPublisher;

    ORBextractor* mpORBextractor;  // 这里有new
    ORBmatcher* mpORBmatcher;  // 这里有new

    // local map
    Frame mCurrentFrame, mLastFrame;
    PtrKeyFrame mpReferenceKF;
    PtrKeyFrame mpLoopKF;
    std::map<size_t, MPCandidate> mMPCandidates;  // 参考帧的MP候选, 在LocalMap线程中会生成真正的MP
    std::map<int, int> mKPMatchesLoop;  // 重定位回环中通过BoW匹配的KP匹配对
    std::vector<int> mvKPMatchIdx, mvKPMatchIdxGood;  // Matches12, 参考帧到当前帧的KP匹配索引. Good指有对应的MP
    int mnMPsNewAdded, mnMPsCandidate, mnKPMatchesBad;  // 新增/潜在的MP数及不好的匹配点数
    int mnKPMatches, mnKPsInline, mnMPsInline, mnMPsTracked;  // 匹配内点数/三角化丢弃后的内点数/关联上参考帧MP数
    int mnLostFrames;  // 连续追踪失败的帧数
    double mLoopScore;

    // New KeyFrame rules (according to fps)
    int nMinFrames, nMaxFrames, nMinMatches;
    float mMaxAngle, mMaxDistance;
    float mCurrRatioGoodDepth, mCurrRatioGoodParl;
    float mLastRatioGoodDepth, mLastRatioGoodParl;

    cv::Mat mAffineMatrix;  // A12

    // preintegration on SE2
    PreSE2 preSE2;

    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexForPub;
    std::mutex mMutexFinish;
};

}  // namespace se2lam

#endif  // TRACK_H
