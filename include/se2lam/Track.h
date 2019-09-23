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
    Se2 dataAlignment(std::vector<Se2>& dataOdoSeq, cv::Mat& dataImg, float& timeImg);

    void setMap(Map* pMap);
    void setLocalMapper(LocalMapper* pLocalMapper);
    void setGlobalMapper(GlobalMapper* pGlobalMapper);
    void setSensors(Sensors* pSensors);

    static void calcOdoConstraintCam(const Se2& dOdo, cv::Mat& cTc, g2o::Matrix6d& Info_se3);
    static void calcSE3toXYZInfo(cv::Point3f xyz1, const cv::Mat& Tcw1, const cv::Mat& Tcw2,
                                 Eigen::Matrix3d& info1, Eigen::Matrix3d& info2);

    // for frame publisher
    int copyForPub(std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2, cv::Mat& img1,
                   cv::Mat& img2, std::vector<int>& vMatches12);

    // for finishing
    void requestFinish();
    bool isFinished();
    bool checkFinish();
    void setFinish();

public:
    // Tracking states
    cvu::eTrackingState mState;
    cvu::eTrackingState mLastState;

    std::vector<int> mMatchIdx;  // Matches12, 参考帧到当前帧的KP匹配索引

    //! for debug print
    int N1 = 0, N2 = 0, N3 = 0;

private:
    void mCreateFirstFrame(const cv::Mat& img, const float& imgTime, const Se2& odo);
    void mTrack(const cv::Mat& img, const float& imgTime, const Se2& odo);
    void relocalization(const cv::Mat& img, const float& imgTime, const Se2& odo);
    void resetLocalTrack();

    void updateFramePose();
    int removeOutliers();

    bool needNewKF(int nTrackedOldMP, int nMatched);
    int doTriangulate();

private:
    static bool mbUseOdometry;  //! TODO 冗余变量

    // only useful when odo time not sync with img time
    float mTimeOdo;
    float mTimeImg;

    // set in OdoSLAM class
    Map* mpMap;
    LocalMapper* mpLocalMapper;
    GlobalMapper* mpGlobalMapper;
    Sensors* mpSensors;
    ORBextractor* mpORBextractor;  // 这里有new

    // local map
    Frame mCurrentFrame;
    PtrKeyFrame mpReferenceKF;
    std::vector<cv::Point2f> mPrevMatched;  // 其实就是参考帧的特征点, 匹配过程中会更新
    std::vector<cv::Point3f> mLocalMPs;  // 当前帧KP的候选MP观测在相机坐标系下的坐标即Pc
    std::set<PtrKeyFrame> mspKFLocal;
    std::set<PtrMapPoint> mspMPLocal;
    std::vector<bool> mvbGoodPrl;
    int mnGoodPrl;  // count number of mLocalMPs with good parallax

    // New KeyFrame rules (according to fps)
    int nMinFrames;
    int nMaxFrames;

    // preintegration on SE2
    PreSE2 preSE2;
    Se2 mLastOdom;

    int nLostFrames;

    cv::Mat Homography;
    cv::Mat mVelocity;

    bool mbFinishRequested;
    bool mbFinished;

    std::mutex mMutexForPub;
    std::mutex mMutexFinish;
};


}  // namespace se2lam

#endif  // TRACK_H
