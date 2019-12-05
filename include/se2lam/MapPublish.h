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
#include "Track.h"
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>


namespace se2lam
{

class Map;

class MapPublish
{
public:
    MapPublish(Map* pMap);
    ~MapPublish();

    void run();

    void setFramePub(FramePublish* pFP) { mpFramePub = pFP; }
    void setMap(Map* pMap) { mpMap = pMap; }
    void setTracker(Track* pTrack) { mpTracker = pTrack; }
    void setLocalMapper(LocalMapper* pLocal) { mpLocalMapper = pLocal; }
    void setLocalizer(Localizer* pLocalizer) { mpLocalizer = pLocalizer; }

    void publishMapPoints();
    void publishKeyFrames();
    void publishCameraCurr(const cv::Mat& Twc);
    void publishOdomInformation();
    void publishGroundTruth();

    cv::Mat drawCurrentFrameMatches();
    cv::Mat drawMatchesInOneImg();    

    void requestFinish();
    bool isFinished();

public:
    bool mbIsLocalize;
    std::vector<Se2> mvGroundTruth;

    Map* mpMap;
    Track* mpTracker;
    LocalMapper* mpLocalMapper;
    Localizer* mpLocalizer;
    FramePublish* mpFramePub;

    // for visulization
    bool mbDataUpdated;
    unsigned long mnCurrentFrameID;
    cv::Mat mCurrentFramePose;
    cv::Mat mCurrentImage, mReferenceImage;
    cv::Mat mAffineMatrix, mHomography;
    std::vector<cv::KeyPoint> mvCurrentKPs, mvReferenceKPs;
    std::vector<int> mvMatchIdx, mvMatchIdxGood;
    std::string mImageText;
    std::mutex mMutexUpdate;

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
    visualization_msgs::Marker mGroundTruthGraph;

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
