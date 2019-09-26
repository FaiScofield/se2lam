/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#ifndef MAPPUBLISH_H
#define MAPPUBLISH_H

#include "FramePublish.h"
#include "LocalMapper.h"
#include "Localizer.h"
#include <opencv2/core/core.hpp>
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>

namespace se2lam
{

class Map;

class MapPublish
{
public:
    MapPublish();
    MapPublish(Map* pMap);
    ~MapPublish();

    void run();

    void setFramePub(FramePublish* pFP) { mpFramePub = pFP; }
    void setMap(Map* pMap) { mpMap = pMap; }
    void setTracker(Track* pTrack) { mpTracker = pTrack; }
    void setLocalMapper(LocalMapper* pLocal) { mpLocalMapper = pLocal; }
    void setLocalizer(Localizer* pLocalize) { mpLocalizer = pLocalize; }

    void PublishMapPoints();
    void PublishKeyFrames();
    void PublishCameraCurr(const cv::Mat& Twc);

    void PublishOdomInformation();
    cv::Mat drawMatchesInOneImg();

    bool isFinished();
    void RequestFinish();

public:
    bool mbIsLocalize;

    Map* mpMap;
    LocalMapper* mpLocalMapper;
    Localizer* mpLocalizer;
    FramePublish* mpFramePub;
    Track* mpTracker;

private:
    bool CheckFinish();
    void SetFinish();

    ros::NodeHandle nh;
    ros::Publisher publisher;
    tf::TransformBroadcaster tfb;

    visualization_msgs::Marker mMPsNeg;
    visualization_msgs::Marker mMPsAct;
    visualization_msgs::Marker mMPsNow;
    visualization_msgs::Marker mMPsNoGoodPrl;

    visualization_msgs::Marker mKFsNeg;
    visualization_msgs::Marker mKFsAct;
    visualization_msgs::Marker mKFNow;

    visualization_msgs::Marker mCovisGraph;
    visualization_msgs::Marker mFeatGraph;
    visualization_msgs::Marker mVIGraph;
    visualization_msgs::Marker mMST;

    visualization_msgs::Marker mOdomRawGraph;

    float mPointSize;
    float mCameraSize;
    float mScaleRatio;


    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    double mErrorSum = 0.;
};  // class MapPublisher

}  // namespace se2lam

#endif  // MAPPUBLISH_H
