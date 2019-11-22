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
#include "cvutil.h"
#include <g2o/types/sba/types_six_dof_expmap.h>

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

    Se2 getCurrentFrameOdo() { return mCurrentFrame.odom; }

    static void calcOdoConstraintCam(const Se2& dOdo, cv::Mat& cTc, g2o::Matrix6d& Info_se3);
    static void calcSE3toXYZInfo(const cv::Point3f& Pc1, const cv::Mat& Tc1w, const cv::Mat& Tc2w,
                                 Eigen::Matrix3d& info1, Eigen::Matrix3d& info2);

    // for visulization message publisher
    size_t copyForPub(cv::Mat& img1, cv::Mat& img2, std::vector<cv::KeyPoint>& kp1,
                      std::vector<cv::KeyPoint>& kp2, std::vector<int>& vMatches12);


    bool isFinished();
    void requestFinish();

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
    void removeOutliers();
    void doTriangulate();
    void resetLocalTrack();
    bool needNewKF();

    // Relocalization
    bool detectIfLost();
    bool detectIfLost(int cros, double projError);

    bool relocalization();
    bool detectLoopClose();
    bool verifyLoopClose();
    void removeMatchOutlierRansac(const Frame* _pKFCurrent, const PtrKeyFrame& _pKFLoop,
                                  std::map<int, int>& mapiMatch);
    void doLocalBA(Frame& pKF);
    void resartTracking();

    bool checkFinish();
    void setFinish();

private:
    static bool mbUseOdometry;  //! TODO 冗余变量
    bool mbPrint;
    bool mbNeedVisualization;
    std::string mImageText;

    // set in OdoSLAM class
    Map* mpMap;
    LocalMapper* mpLocalMapper;
    GlobalMapper* mpGlobalMapper;
    Sensors* mpSensors;
    ORBextractor* mpORBextractor;  // 这里有new
    ORBmatcher* mpORBmatcher;
    MapPublish* mpMapPublisher;

    // local map
    Frame mCurrentFrame, mLastFrame;
    PtrKeyFrame mpReferenceKF;
    PtrKeyFrame mpLoopKF;
    std::map<size_t, cv::Point3f> mMPCandidates;  // 参考帧的MP候选, 在LocalMap线程中会生成真正的MP
    std::vector<int> mvMatchIdx, mvGoodMatchIdx;  // Matches12, 参考帧到当前帧的KP匹配索引
    int mnNewAddedMPs, mnCandidateMPs, mnBadMatches;  // 新增/潜在的MP数及不好的匹配点数
    int mnInliers, mnGoodInliers, mnTrackedOld;  // 匹配内点数/三角化丢弃后的内点数/关联上参考帧MP数
    int mnLostFrames;                            // 连续追踪失败的帧数

    // New KeyFrame rules (according to fps)
    int nMinFrames, nMaxFrames, nMinMatches;
    float mMaxAngle, mMaxDistance;
    float mCurrRatioGoodDepth, mCurrRatioGoodParl;
    float mLastRatioGoodDepth, mLastRatioGoodParl;

    cv::Mat mK, mD;
    cv::Mat mAffineMatrix;

    // preintegration on SE2
    PreSE2 preSE2;

    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexForPub;
    std::mutex mMutexFinish;
};

}  // namespace se2lam

#endif  // TRACK_H
