/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#ifndef MAPPUBLISH_H
#define MAPPUBLISH_H

#include "LocalMapper.h"
#include "Localizer.h"
#include "Track.h"
#include "TrackKlt.h"
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

    void setMap(Map* pMap) { mpMap = pMap; }
#ifdef USEKLT
    void setTracker(TrackKlt* pTrack) { mpTracker = pTrack; }
#else
    void setTracker(Track* pTrack) { mpTracker = pTrack; }
#endif
    void setLocalMapper(LocalMapper* pLocal) { mpLocalMapper = pLocal; }
    void setLocalizer(Localizer* pLocalize) { mpLocalizer = pLocalize; }

    void publishMapPoints();
    void publishKeyFrames();
    void publishCameraCurr(const cv::Mat& Twc);

    void publishOdomInformation();
    cv::Mat drawMatchesInOneImg();

    bool isFinished();
    void requestFinish();

public:
    bool mbIsLocalize;

    Map* mpMap;
    LocalMapper* mpLocalMapper;
    Localizer* mpLocalizer;
#ifdef USEKLT
    TrackKlt* mpTracker;
#else
    Track* mpTracker;
#endif

private:
    bool checkFinish();
    void setFinish();

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
    visualization_msgs::Marker mOdomRawGraph;

    visualization_msgs::Marker mMST;

    float mPointSize;
    float mCameraSize;
    float mScaleRatio;

    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;
};  // class MapPublisher

}  // namespace se2lam

#endif  // MAPPUBLISH_H
