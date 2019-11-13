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

    Se2 getCurrentFrameOdo() { return mCurrentFrame.odom; }

    static void calcOdoConstraintCam(const Se2& dOdo, cv::Mat& cTc, g2o::Matrix6d& Info_se3);
    static void calcSE3toXYZInfo(const cv::Point3f& Pc1, const cv::Mat& Tc1w, const cv::Mat& Tc2w,
                                 Eigen::Matrix3d& info1, Eigen::Matrix3d& info2);

    // for visulization message publisher
    size_t copyForPub(cv::Mat& img1, cv::Mat& img2, std::vector<cv::KeyPoint>& kp1,
                      std::vector<cv::KeyPoint>& kp2, std::vector<int>& vMatches12);
    void drawFrameForPub(cv::Mat& imgLeft);
    cv::Mat getImageMatches();

    bool isFinished();
    void requestFinish();

public:
    // Tracking states
    cvu::eTrackingState mState;
    cvu::eTrackingState mLastState;

    int N1 = 0, N2 = 0, N3 = 0;  // for debug print
    double trackTimeTatal = 0.;

private:
    void createFirstFrame(const cv::Mat& img, const double& imgTime, const Se2& odo);
    void trackReferenceKF(const cv::Mat& img, const double& imgTime, const Se2& odo);
    void relocalization(const cv::Mat& img, const double& imgTime, const Se2& odo);
    void resetLocalTrack();

    void updateFramePose();
    int removeOutliers();

    bool needNewKF();
    int doTriangulate();

    void drawMatchesForPub(bool warp);

    bool checkFinish();
    void setFinish();

private:
    static bool mbUseOdometry;  //! TODO 冗余变量
    bool mbPrint;
    bool mbNeedVisualization;

    // set in OdoSLAM class
    Map* mpMap;
    LocalMapper* mpLocalMapper;
    GlobalMapper* mpGlobalMapper;
    Sensors* mpSensors;
    ORBextractor* mpORBextractor;  // 这里有new
    ORBmatcher* mpORBmatcher;

    // local map
    Frame mCurrentFrame;
    PtrKeyFrame mpReferenceKF;
    std::vector<cv::Point2f> mPrevMatched;  // 其实就是参考帧的特征点, 匹配过程中会更新
    std::vector<cv::Point3f> mLocalMPs;  // 参考帧的MP观测(Pc非Pw), 每帧处理会更新此变量
    std::vector<int> mvMatchIdx;         // Matches12, 参考帧到当前帧的KP匹配索引
    std::vector<bool> mvbGoodPrl;
    int mnGoodPrl;  // count number of mLocalMPs with good parallax
    int mnInliers, mnMatchSum, mnTrackedOld;
    int mnLostFrames;

    // New KeyFrame rules (according to fps)
    int nMinFrames, nMaxFrames;
    double mMaxAngle, mMaxDistance;

    // preintegration on SE2
    PreSE2 preSE2;
    Se2 mLastOdom;

    cv::Mat mK, mD;
    cv::Mat mAffineMatrix;
    cv::Mat mImgOutMatch;

    bool mbFinishRequested;
    bool mbFinished;

    std::mutex mMutexForPub;
    std::mutex mMutexFinish;
};

}  // namespace se2lam

#endif  // TRACK_H
