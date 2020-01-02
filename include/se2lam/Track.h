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

    void Run();

    void setMap(Map* pMap) { mpMap = pMap; }
    void setLocalMapper(LocalMapper* pLocalMapper) { mpLocalMapper = pLocalMapper; }
    void setGlobalMapper(GlobalMapper* pGlobalMapper) { mpGlobalMapper = pGlobalMapper; }
    void setSensors(Sensors* pSensors) { mpSensors = pSensors; }
    void setMapPublisher(MapPublish* pMapPublisher) { mpMapPublisher = pMapPublisher; }

    void RequestFinish();
    bool IsFinished();

    static void calcOdoConstraintCam(const Se2& dOdo, cv::Mat& cTc, g2o::Matrix6d& Info_se3);

    static void calcSE3toXYZInfo(cv::Point3f xyz1, const cv::Mat& Tcw1, const cv::Mat& Tcw2,
                                 Eigen::Matrix3d& info1, Eigen::Matrix3d& info2);

    static bool mbUseOdometry;
    bool mbNeedVisualization;
    int nMinFrames;
    int nMaxFrames;

    double trackTimeTatal = 0.;

private:
    void CheckReady();
    void ProcessFirstFrame(const cv::Mat& img, const Se2& odo);
    void TrackReferenceKF(const cv::Mat& img, const Se2& odo);
    void UpdateFramePose();

    int RemoveOutliers(const std::vector<cv::KeyPoint>& kp1, const std::vector<cv::KeyPoint>& kp2,
                       std::vector<int>& matches);
    int DoTriangulate();
    bool NeedNewKF();

    void ResetLocalTrack();

    void CopyForPub();
    std::mutex mMutexForPub;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

private:
    Map* mpMap;
    Sensors* mpSensors;
    LocalMapper* mpLocalMapper;
    GlobalMapper* mpGlobalMapper;
    MapPublish* mpMapPublisher;
    ORBextractor* mpORBextractor;
    ORBmatcher* mpORBmatcher;

    std::vector<int> mvKPMatchIdx, mvKPMatchIdxGood;
    std::vector<cv::Point3f> mLocalMPs;
    int mnGoodPrl;  // count number of mLocalMPs with good parallax
    std::vector<bool> mvbGoodPrl;

    Frame mCurrentFrame, mLastFrame;
    PtrKeyFrame mpReferenceKF, mpLoopKF;
    std::vector<cv::Point2f> mPrevMatched;

    int mnKPMatches, mnKPMatchesGood, mnKPsInline;
    int mnMPsTracked, mnMPsNewAdded, mnMPsInline;
    int mnLessMatchFrames, mnLostFrames;

    // New KeyFrame rules (according to fps)
    int nMinFrames, nMaxFrames, nMinMatches;
    float mMaxAngle, mMinDistance;

    cv::Mat mAffineMatrix;
    PreSE2 preSE2;
    Se2 lastOdom;
};


}  // namespace se2lam

#endif  // TRACK_H
