/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#ifndef TRACK_H
#define TRACK_H
#include "Config.h"
#include "Frame.h"
#include "GlobalMapper.h"
#include "Sensors.h"
#include "cvutil.h"
#include <g2o/types/sba/types_six_dof_expmap.h>

namespace se2lam
{

class KeyFrame;
class Map;
class LocalMapper;
class GlobalMapper;

typedef std::shared_ptr<KeyFrame> PtrKeyFrame;

class Track
{
public:
    Track();
    ~Track();

    void run();

    void setMap(Map *pMap);
    void setLocalMapper(LocalMapper *pLocalMapper);
    void setGlobalMapper(GlobalMapper *pGlobalMapper);
    void setSensors(Sensors *pSensors);

    static void calcOdoConstraintCam(const Se2 &dOdo, cv::Mat &cTc, g2o::Matrix6d &Info_se3);
    static void calcSE3toXYZInfo(cv::Point3f xyz1, const cv::Mat &Tcw1, const cv::Mat &Tcw2,
                                 Eigen::Matrix3d &info1, Eigen::Matrix3d &info2);

    // for frame publisher
    std::vector<int> mMatchIdx;
    int copyForPub(std::vector<cv::KeyPoint> &kp1, std::vector<cv::KeyPoint> &kp2, cv::Mat &img1,
                   cv::Mat &img2, std::vector<int> &vMatches12);

    void requestFinish();
    bool isFinished();
    bool checkFinish();
    void setFinish();

    void relocalization(const cv::Mat &img, const Se2 &odo);
    void DrawMachesPoints(const cv::Mat fmg, const cv::Mat img, std::vector<int> vMatchesDistance);


    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

private:
    static bool mbUseOdometry;

    // only useful when odo time not sync with img time
    unsigned long int mTimeOdo;
    unsigned long int mTimeImg;

    // set in OdoSLAM class
    Map *mpMap;
    LocalMapper *mpLocalMapper;
    GlobalMapper *mpGlobalMapper;
    Sensors *mpSensors;

    ORBextractor *mpORBextractor;

    std::vector<cv::Point3f> mLocalMPs; // 当前帧KP的候选MP观测，相机坐标系
    int mnGoodPrl;  // count number of mLocalMPs with good parallax
    std::vector<bool> mvbGoodPrl;

    // New KeyFrame rules (according to fps)
    int nMinFrames;
    int nMaxFrames;

    Frame mFrame;
    Frame mRefFrame;
    PtrKeyFrame mpKF;
    std::vector<cv::Point2f> mPrevMatched;  // 上一参考帧的特征点

    void mCreateFrame(const cv::Mat &img, const Se2 &odo);
    void mTrack(const cv::Mat &img, const Se2 &odo);
    void resetLocalTrack();
    void updateFramePose();
    int removeOutliers(const std::vector<cv::KeyPoint> &kp1, const std::vector<cv::KeyPoint> &kp2,
                       std::vector<int> &matches);
    bool needNewKF(int nTrackedOldMP, int nMatched);
    int doTriangulate();

    std::mutex mMutexForPub;

    // preintegration on SE2
    PreSE2 preSE2;
    Se2 mLastOdom;

    // Tracking states
    cvu::eTrackingState mState;
    cvu::eTrackingState mLastState;

    int nLostFrames;
};


}  // namespace se2lam

#endif  // TRACK_H
