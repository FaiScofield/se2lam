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

    void run();

    void setMap(Map* pMap) { mpMap = pMap; }
    void setLocalMapper(LocalMapper* pLocalMapper) { mpLocalMapper = pLocalMapper; }
    void setGlobalMapper(GlobalMapper* pGlobalMapper) { mpGlobalMapper = pGlobalMapper; }
    void setSensors(Sensors* pSensors) { mpSensors = pSensors; }
    void setMapPublisher(MapPublish* pMapPublisher) { mpMapPublisher = pMapPublisher; }

    static void calcOdoConstraintCam(const Se2& dOdo, cv::Mat& cTc, g2o::Matrix6d& Info_se3);

    static void calcSE3toXYZInfo(cv::Point3f xyz1, const cv::Mat& Tcw1, const cv::Mat& Tcw2,
                                 Eigen::Matrix3d& info1, Eigen::Matrix3d& info2);

    // for frame publisher
    std::vector<int> mMatchIdx;
    int copyForPub(std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2, cv::Mat& img1,
                   cv::Mat& img2, std::vector<int>& vMatches12);


    void requestFinish();
    bool isFinished();
    bool checkFinish();
    void setFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    double trackTimeTatal = 0.;

private:
    void mCreateFrame(const cv::Mat& img, const Se2& odo);
    void mTrack(const cv::Mat& img, const Se2& odo);
    void resetLocalTrack();
    void updateFramePose();
    int removeOutliers(const std::vector<cv::KeyPoint>& kp1, const std::vector<cv::KeyPoint>& kp2,
                       std::vector<int>& matches);
    bool needNewKF(int nTrackedOldMP, int nMatched);
    int doTriangulate();
    void checkReady();


    static bool mbUseOdometry;
    bool mbNeedVisualization;

    Map* mpMap;
    Sensors* mpSensors;
    LocalMapper* mpLocalMapper;
    GlobalMapper* mpGlobalMapper;
    MapPublish* mpMapPublisher;

    ORBextractor* mpORBextractor;  // 这里有new
    ORBmatcher* mpORBmatcher;

    std::vector<cv::Point3f> mLocalMPs;
    int mnGoodPrl;  // count number of mLocalMPs with good parallax
    std::vector<bool> mvbGoodPrl;

    int nMinFrames;
    int nMaxFrames;

    Frame mFrame;
    Frame mRefFrame;
    PtrKeyFrame mpKF;
    std::vector<cv::Point2f> mPrevMatched;

    std::mutex mMutexForPub;

    PreSE2 preSE2;
    Se2 lastOdom;
};


}  // namespace se2lam

#endif  // TRACK_H
