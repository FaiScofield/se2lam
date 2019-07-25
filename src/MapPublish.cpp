/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include "MapPublish.h"
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include "Map.h"
#include "converter.h"
#include "FramePublish.h"
#include <opencv2/highgui/highgui.hpp>

namespace se2lam {
using namespace cv;
using namespace std;
typedef lock_guard<mutex> locker;

MapPublish::MapPublish(Map* pMap){

    mbFinished = false;
    mbFinishRequested = false;

    mpMap = pMap;

    const char* MAP_FRAME_ID = "/se2lam/World";

    //const char* KEYFRAMES_NAMESPACE = "KeyFrames";
    const char* CAMERA_NAMESPACE = "Camera";
    const char* KEYFRAMESACTIVE_NAMESPACE = "KeyFramesActive";
    const char* KEYFRAMESNEGATIVE_NAMESPACE = "KeyFramesNegative";

    //const char* POINTS_NAMESPACE = "MapPoints";
    const char* POINTSNOW_NAMESPACE = "MapPointsNow";
    const char* POINTSACTIVE_NAMESPACE = "MapPointsActive";
    const char* POINTSNEGATIVE_NAMESPACE = "MapPointsNegative";

    const char* GRAPH_NAMESPACE = "Graph";
    const char* FEATGRAPH_NAMESPACE = "FeatGraph";
    const char* ODOGRAPH_NAMESPACE = "OdoGraph";
    const char* COVISGRAPH_NAMESPACE = "CovisGraph";

    // Set Scale Ratio
    mScaleRatio = Config::MAPPUB_SCALE_RATIO;


    //Configure KF not in local map
    mCameraSize = 0.4;
    mKFsNeg.header.frame_id = MAP_FRAME_ID;
    mKFsNeg.ns = KEYFRAMESNEGATIVE_NAMESPACE;
    mKFsNeg.id = 0;
    mKFsNeg.type = visualization_msgs::Marker::LINE_LIST;
    mKFsNeg.scale.x = 0.05;
    mKFsNeg.scale.y = 0.05;
    mKFsNeg.pose.orientation.w = 1.0;
    mKFsNeg.action = visualization_msgs::Marker::ADD;
    mKFsNeg.color.r = 0.0;
    mKFsNeg.color.g = 0.0;
    mKFsNeg.color.b = 0.0;
    mKFsNeg.color.a = 1.0;

    //Configure KFs in local map
    mKFsAct.header.frame_id = MAP_FRAME_ID;
    mKFsAct.ns = KEYFRAMESACTIVE_NAMESPACE;
    mKFsAct.id = 8;
    mKFsAct.type = visualization_msgs::Marker::LINE_LIST;
    mKFsAct.scale.x = 0.05;
    mKFsAct.scale.y = 0.05;
    mKFsAct.pose.orientation.w = 1.0;
    mKFsAct.action = visualization_msgs::Marker::ADD;
    mKFsAct.color.b = 1.0;
    mKFsAct.color.a = 1.0;

    //Configure Current Camera
    mKFNow.header.frame_id = MAP_FRAME_ID;
    mKFNow.ns = CAMERA_NAMESPACE;
    mKFNow.id = 1;
    mKFNow.type = visualization_msgs::Marker::LINE_LIST;
    mKFNow.scale.x = 0.05;//0.1;0.2; 0.03
    mKFNow.scale.y = 0.05;//0.1;0.2; 0.03
    mKFNow.pose.orientation.w = 1.0;
    mKFNow.action = visualization_msgs::Marker::ADD;
    mKFNow.color.r = 1.0;
    mKFNow.color.a = 1.0;

    //Configure MPs not in local map
    mPointSize = 0.1;
    mMPsNeg.header.frame_id = MAP_FRAME_ID;
    mMPsNeg.ns = POINTSNEGATIVE_NAMESPACE;
    mMPsNeg.id = 2;
    mMPsNeg.type = visualization_msgs::Marker::POINTS;
    mMPsNeg.scale.x = mPointSize;
    mMPsNeg.scale.y = mPointSize;
    mMPsNeg.pose.orientation.w = 1.0;
    mMPsNeg.action = visualization_msgs::Marker::ADD;
    mMPsNeg.color.r = 0.0;
    mMPsNeg.color.g = 0.0;
    mMPsNeg.color.b = 0.0;
    mMPsNeg.color.a = 1.0;

    //Configure MPs in local map
    mMPsAct.header.frame_id = MAP_FRAME_ID;
    mMPsAct.ns = POINTSACTIVE_NAMESPACE;
    mMPsAct.id = 3;
    mMPsAct.type = visualization_msgs::Marker::POINTS;
    mMPsAct.scale.x = mPointSize;
    mMPsAct.scale.y = mPointSize;
    mMPsAct.pose.orientation.w = 1.0;
    mMPsAct.action = visualization_msgs::Marker::ADD;
    mMPsAct.color.r = 0.0;
    mMPsAct.color.g = 0.0;
    mMPsAct.color.b = 1.0;
    mMPsAct.color.a = 1.0;

    // Configure MPs currently observed
    mMPsNow.header.frame_id = MAP_FRAME_ID;
    mMPsNow.ns = POINTSNOW_NAMESPACE;
    mMPsNow.id = 3;
    mMPsNow.type = visualization_msgs::Marker::POINTS;
    mMPsNow.scale.x = mPointSize;
    mMPsNow.scale.y = mPointSize;
    mMPsNow.pose.orientation.w = 1.0;
    mMPsNow.action = visualization_msgs::Marker::ADD;
    mMPsNow.color.r = 1.0;
    mMPsNow.color.g = 0.0;
    mMPsNow.color.b = 0.0;
    mMPsNow.color.a = 1.0;

    //Configure Covisibility Graph
    mCovisGraph.header.frame_id = MAP_FRAME_ID;
    mCovisGraph.ns = COVISGRAPH_NAMESPACE;
    mCovisGraph.id = 4;
    mCovisGraph.type = visualization_msgs::Marker::LINE_LIST;
    mCovisGraph.scale.x = 0.05;
    mCovisGraph.scale.y = 0.05;
    mCovisGraph.pose.orientation.w = 1.0;
    mCovisGraph.action = visualization_msgs::Marker::ADD;
    mCovisGraph.color.r = 0.0;
    mCovisGraph.color.g = 1.0;
    mCovisGraph.color.b = 0.0;
    mCovisGraph.color.a = 1.0;

    // Configure Feature Constraint Graph
    mFeatGraph.header.frame_id = MAP_FRAME_ID;
    mFeatGraph.ns = FEATGRAPH_NAMESPACE;
    mFeatGraph.id = 5;
    mFeatGraph.type = visualization_msgs::Marker::LINE_LIST;
    mFeatGraph.scale.x = 0.05;
    mFeatGraph.scale.y = 0.05;
    mFeatGraph.pose.orientation.w=1.0;
    mFeatGraph.action=visualization_msgs::Marker::ADD;
    mFeatGraph.color.r = 0.0;
    mFeatGraph.color.g = 0.0;
    mFeatGraph.color.b = 1.0;
    mFeatGraph.color.a = 1.0;

    // Configure Odometry Constraint Graph
    mOdoGraph.header.frame_id = MAP_FRAME_ID;
    mOdoGraph.ns = ODOGRAPH_NAMESPACE;
    mOdoGraph.id = 6;
    mOdoGraph.type = visualization_msgs::Marker::LINE_LIST;
    mOdoGraph.scale.x = 0.05;
    mOdoGraph.scale.y = 0.05;
    mOdoGraph.pose.orientation.w=1.0;
    mOdoGraph.action=visualization_msgs::Marker::ADD;
    mOdoGraph.color.r = 0.0;
    mOdoGraph.color.g = 0.0;
    mOdoGraph.color.b = 0.0;
    mOdoGraph.color.a = 1.0;

    //Configure KeyFrames Spanning Tree
    mMST.header.frame_id = MAP_FRAME_ID;
    mMST.ns = GRAPH_NAMESPACE;
    mMST.id = 7;
    mMST.type = visualization_msgs::Marker::LINE_LIST;
    mMST.scale.x=0.005;
    mMST.pose.orientation.w=1.0;
    mMST.action=visualization_msgs::Marker::ADD;
    mMST.color.b = 1.0f;
    mMST.color.g = 1.0f;
    mMST.color.a = 1.0;

    // Configure Odometry Raw Constraint Graph
    mOdomRawGraph.header.frame_id = MAP_FRAME_ID;
    mOdomRawGraph.ns = "OdomRawGraph";
    mOdomRawGraph.id = 9;
    mOdomRawGraph.type = visualization_msgs::Marker::LINE_LIST;
    mOdomRawGraph.action=visualization_msgs::Marker::ADD;
    mOdomRawGraph.scale.x = 0.03;
    mOdomRawGraph.scale.y = 0.03;
    mOdomRawGraph.pose.orientation.w = 1.0;
    mOdomRawGraph.color.r = 0.95;
    mOdomRawGraph.color.g = 0.0;
    mOdomRawGraph.color.b = 0.0;
    mOdomRawGraph.color.a = 0.7;

    tf::Transform tfT;
    tfT.setIdentity();
    tfb.sendTransform(tf::StampedTransform(tfT,ros::Time::now(),"/se2lam/World","/se2lam/Camera"));

    publisher = nh.advertise<visualization_msgs::Marker>("se2lam/Map", 10);

    publisher.publish(mKFsNeg);
    publisher.publish(mKFsAct);
    publisher.publish(mKFNow);

    publisher.publish(mMPsNeg);
    publisher.publish(mMPsAct);
    publisher.publish(mMPsNow);

    publisher.publish(mCovisGraph);
    publisher.publish(mFeatGraph);
    publisher.publish(mOdoGraph);

    publisher.publish(mOdomRawGraph);
}

MapPublish::~MapPublish(){}

void MapPublish::PublishKeyFrames(){
    mKFsNeg.points.clear();
    mKFsAct.points.clear();

    mCovisGraph.points.clear();
    mFeatGraph.points.clear();

    if (!mbIsLocalize)
        mOdoGraph.points.clear();

    //Camera is a pyramid. Define in camera coordinate system
    float d = mCameraSize;
    cv::Mat o = (cv::Mat_<float>(4,1) << 0, 0, 0, 1);
    cv::Mat p1 = (cv::Mat_<float>(4,1) << d, d*0.8, d*0.5, 1);
    cv::Mat p2 = (cv::Mat_<float>(4,1) << d, -d*0.8, d*0.5, 1);
    cv::Mat p3 = (cv::Mat_<float>(4,1) << -d, -d*0.8, d*0.5, 1);
    cv::Mat p4 = (cv::Mat_<float>(4,1) << -d, d*0.8, d*0.5, 1);

    vector<PtrKeyFrame> vKFsAll = mpMap->getAllKF();
    if(vKFsAll.empty()){
        return;
    }

    vector<PtrKeyFrame> vKFsAct;
    if (mbIsLocalize) {
        vKFsAct = mpLocalize->GetLocalKFs();
    } else {
        vKFsAct = mpMap->getLocalKFs();
    }

    //! vKFsAll是从Map里获取的，在LocalizeOnly模式下是固定的
    for (int i=0, iend=vKFsAll.size(); i<iend; i++) {
        if(vKFsAll[i]->isNull())
            continue;

        cv::Mat Twc = vKFsAll[i]->getPose().inv();
        Twc *= Config::cTb; //! 这里实际把它变成Twb，转到body坐标系下

        Twc.at<float>(0,3) = Twc.at<float>(0,3)/mScaleRatio;
        Twc.at<float>(1,3) = Twc.at<float>(1,3)/mScaleRatio;
        Twc.at<float>(2,3) = Twc.at<float>(2,3)/mScaleRatio;

        cv::Mat ow = Twc*o;
        cv::Mat p1w = Twc*p1;
        cv::Mat p2w = Twc*p2;
        cv::Mat p3w = Twc*p3;
        cv::Mat p4w = Twc*p4;

        geometry_msgs::Point msgs_o,msgs_p1, msgs_p2, msgs_p3, msgs_p4;
        msgs_o.x=ow.at<float>(0);
        msgs_o.y=ow.at<float>(1);
        msgs_o.z=ow.at<float>(2);
        msgs_p1.x=p1w.at<float>(0);
        msgs_p1.y=p1w.at<float>(1);
        msgs_p1.z=p1w.at<float>(2);
        msgs_p2.x=p2w.at<float>(0);
        msgs_p2.y=p2w.at<float>(1);
        msgs_p2.z=p2w.at<float>(2);
        msgs_p3.x=p3w.at<float>(0);
        msgs_p3.y=p3w.at<float>(1);
        msgs_p3.z=p3w.at<float>(2);
        msgs_p4.x=p4w.at<float>(0);
        msgs_p4.y=p4w.at<float>(1);
        msgs_p4.z=p4w.at<float>(2);


        PtrKeyFrame pKFtmp = vKFsAll[i];
        int count = std::count(vKFsAct.begin(), vKFsAct.end(), pKFtmp);
        if (count == 0) {
            mKFsNeg.points.push_back(msgs_o);
            mKFsNeg.points.push_back(msgs_p1);
            mKFsNeg.points.push_back(msgs_o);
            mKFsNeg.points.push_back(msgs_p2);
            mKFsNeg.points.push_back(msgs_o);
            mKFsNeg.points.push_back(msgs_p3);
            mKFsNeg.points.push_back(msgs_o);
            mKFsNeg.points.push_back(msgs_p4);
            mKFsNeg.points.push_back(msgs_p1);
            mKFsNeg.points.push_back(msgs_p2);
            mKFsNeg.points.push_back(msgs_p2);
            mKFsNeg.points.push_back(msgs_p3);
            mKFsNeg.points.push_back(msgs_p3);
            mKFsNeg.points.push_back(msgs_p4);
            mKFsNeg.points.push_back(msgs_p4);
            mKFsNeg.points.push_back(msgs_p1);
        } else {
            mKFsAct.points.push_back(msgs_o);
            mKFsAct.points.push_back(msgs_p1);
            mKFsAct.points.push_back(msgs_o);
            mKFsAct.points.push_back(msgs_p2);
            mKFsAct.points.push_back(msgs_o);
            mKFsAct.points.push_back(msgs_p3);
            mKFsAct.points.push_back(msgs_o);
            mKFsAct.points.push_back(msgs_p4);
            mKFsAct.points.push_back(msgs_p1);
            mKFsAct.points.push_back(msgs_p2);
            mKFsAct.points.push_back(msgs_p2);
            mKFsAct.points.push_back(msgs_p3);
            mKFsAct.points.push_back(msgs_p3);
            mKFsAct.points.push_back(msgs_p4);
            mKFsAct.points.push_back(msgs_p4);
            mKFsAct.points.push_back(msgs_p1);
        }

        // Covisibility Graph
        std::set<PtrKeyFrame> covKFs = vKFsAll[i]->getAllCovisibleKFs();

        if(!covKFs.empty()){
            for(auto it = covKFs.begin(), iend = covKFs.end(); it != iend; it++){
                if( (*it)->mIdKF > vKFsAll[i]->mIdKF )
                    continue;
                Mat Twb = (*it)->getPose().inv() * Config::cTb;
                geometry_msgs::Point msgs_o2;
                msgs_o2.x = Twb.at<float>(0,3)/mScaleRatio;
                msgs_o2.y = Twb.at<float>(1,3)/mScaleRatio;
                msgs_o2.z = Twb.at<float>(2,3)/mScaleRatio;
                mCovisGraph.points.push_back(msgs_o);
                mCovisGraph.points.push_back(msgs_o2);
            }
        }

        // Feature Graph
        PtrKeyFrame pKF = vKFsAll[i];
        for (auto iter = pKF->mFtrMeasureFrom.begin();
             iter != pKF->mFtrMeasureFrom.end(); iter++) {
            PtrKeyFrame pKF2 = iter->first;
            Mat Twb = pKF2->getPose().inv() * Config::cTb;
            geometry_msgs::Point msgs_o2;
            msgs_o2.x = Twb.at<float>(0,3)/mScaleRatio;
            msgs_o2.y = Twb.at<float>(1,3)/mScaleRatio;
            msgs_o2.z = Twb.at<float>(2,3)/mScaleRatio;
            mFeatGraph.points.push_back(msgs_o);
            mFeatGraph.points.push_back(msgs_o2);
        }

        // Odometry Graph
        if (mbIsLocalize)
            continue;
        PtrKeyFrame pKFOdoChild = pKF->mOdoMeasureFrom.first;
        if (pKFOdoChild != NULL && !mbIsLocalize) {
            Mat Twb = pKFOdoChild->getPose().inv() * Config::cTb;
            geometry_msgs::Point msgs_o2;
            msgs_o2.x = Twb.at<float>(0,3)/mScaleRatio;
            msgs_o2.y = Twb.at<float>(1,3)/mScaleRatio;
            msgs_o2.z = Twb.at<float>(2,3)/mScaleRatio;
            mOdoGraph.points.push_back(msgs_o);
            mOdoGraph.points.push_back(msgs_o2);
        }
    }
    // Odometry Graph for Localize only case
    static geometry_msgs::Point msgsLast;
    if (mbIsLocalize) {
        geometry_msgs::Point msgs;
        msgs.x = mpLocalize->getCurrKFPose().x/mScaleRatio;
        msgs.y = mpLocalize->getCurrKFPose().y/mScaleRatio;
//        msgs.x = mpLocalize->mpKFCurr->Twb.x/mScaleRatio;
//        msgs.y = mpLocalize->mpKFCurr->Twb.y/mScaleRatio;
        mOdoGraph.points.push_back(msgsLast);
        mOdoGraph.points.push_back(msgs);
        printf("[MapPublish] mOdoGraph msgsLast = [%f, %f]\n", msgsLast.x, msgsLast.y);
        printf("[MapPublish] mOdoGraph     msgs = [%f, %f]\n", msgs.x, msgs.y);
        msgsLast = msgs;

//        msgsLast.x = mpLocalize->mpKFRef->Twb.x/mScaleRatio;
//        msgsLast.y = mpLocalize->mpKFRef->Twb.y/mScaleRatio;
    }
//    printf("[MapPublish] Size of mOdoGraph = %ld\n", mOdoGraph.points.size());

    mKFsNeg.header.stamp = ros::Time::now();
    mCovisGraph.header.stamp = ros::Time::now();
    mFeatGraph.header.stamp = ros::Time::now();
    mOdoGraph.header.stamp = ros::Time::now();

    publisher.publish(mKFsNeg);
    publisher.publish(mKFsAct);

    publisher.publish(mCovisGraph);
    publisher.publish(mFeatGraph);
    publisher.publish(mOdoGraph);
}

void MapPublish::PublishMapPoints() {

    mMPsNeg.points.clear();
    mMPsAct.points.clear();
    mMPsNow.points.clear();

    set<PtrMapPoint> spMPNow;
    vector<PtrMapPoint> vpMPAct;
    if (mbIsLocalize) {
        locker lock(mpLocalize->mMutexLocalMap);
        vpMPAct = mpLocalize->GetLocalMPs();
        spMPNow = mpLocalize->mpKFCurr->getAllObsMPs();
    } else {
        vpMPAct = mpMap->getLocalMPs();
        spMPNow = mpMap->getCurrentKF()->getAllObsMPs();
    }

    vector<PtrMapPoint> vpMPNeg = mpMap->getAllMP();

    vector<PtrMapPoint> vpMPNegGood;
    for (auto iter = vpMPNeg.begin(); iter != vpMPNeg.end(); iter ++) {
        PtrMapPoint pMPtemp = *iter;
        int count = std::count(vpMPAct.begin(), vpMPAct.end(), pMPtemp);
        if (count == 0) {
            vpMPNegGood.push_back(pMPtemp);
        }
    }
    vpMPNeg.swap(vpMPNegGood);

    vector<PtrMapPoint> vpMPActGood;
    for (auto iter = vpMPAct.begin(); iter != vpMPAct.end(); iter ++) {
        PtrMapPoint pMPtemp = *iter;
        int count = spMPNow.count(pMPtemp);
        if (count == 0) {
            vpMPActGood.push_back(pMPtemp);
        }
    }
    vpMPAct.swap(vpMPActGood);


    mMPsNeg.points.reserve(vpMPNeg.size());
    for(int i = 0, iend = vpMPNeg.size(); i < iend; i++){
        if(vpMPNeg[i]->isNull() || !vpMPNeg[i]->isGoodPrl()){
            continue;
        }
        geometry_msgs::Point msg_p;
        msg_p.x = vpMPNeg[i]->getPos().x/mScaleRatio;
        msg_p.y = vpMPNeg[i]->getPos().y/mScaleRatio;
        msg_p.z = vpMPNeg[i]->getPos().z/mScaleRatio;
        mMPsNeg.points.push_back(msg_p);
    }


    mMPsAct.points.reserve(vpMPAct.size());
    for(int i = 0, iend = vpMPAct.size(); i < iend; i++){
        if(vpMPAct[i]->isNull() || !vpMPAct[i]->isGoodPrl()){
            continue;
        }
        geometry_msgs::Point msg_p;
        msg_p.x = vpMPAct[i]->getPos().x/mScaleRatio;
        msg_p.y = vpMPAct[i]->getPos().y/mScaleRatio;
        msg_p.z = vpMPAct[i]->getPos().z/mScaleRatio;
        mMPsAct.points.push_back(msg_p);
    }


    mMPsNow.points.reserve(spMPNow.size());
    for (auto iter = spMPNow.begin(); iter != spMPNow.end(); iter ++) {
        PtrMapPoint pMPtmp = *iter;
        if(pMPtmp->isNull() || !pMPtmp->isGoodPrl()) {
            continue;
        }
        geometry_msgs::Point msg_p;
        msg_p.x = pMPtmp->getPos().x/mScaleRatio;
        msg_p.y = pMPtmp->getPos().y/mScaleRatio;
        msg_p.z = pMPtmp->getPos().z/mScaleRatio;
        mMPsNow.points.push_back(msg_p);
    }

    mMPsNeg.header.stamp = ros::Time::now();
    mMPsAct.header.stamp = ros::Time::now();
    mMPsNow.header.stamp = ros::Time::now();

    publisher.publish(mMPsNeg);
    publisher.publish(mMPsAct);
    publisher.publish(mMPsNow);
}

void MapPublish::PublishCameraCurr(const cv::Mat &Twc)
{
    mKFNow.points.clear();

    tf::Matrix3x3 R(Twc.at<float>(0,0),Twc.at<float>(0,1),Twc.at<float>(0,2),
                    Twc.at<float>(1,0),Twc.at<float>(1,1),Twc.at<float>(1,2),
                    Twc.at<float>(2,0),Twc.at<float>(2,1),Twc.at<float>(2,2));
    tf::Vector3 t(Twc.at<float>(0,3)/mScaleRatio,Twc.at<float>(1,3)/mScaleRatio,Twc.at<float>(2,3)/mScaleRatio);
    tf::Transform tfwTc(R,t);
    tfb.sendTransform(tf::StampedTransform(tfwTc,ros::Time::now(),"se2lam/World","se2lam/Camera"));

    float d = mCameraSize;

    //Camera is a pyramid. Define in camera coordinate system
    cv::Mat o = (cv::Mat_<float>(4,1) << 0, 0, 0, 1);
    cv::Mat p1 = (cv::Mat_<float>(4,1) << d, d*0.8, d*0.5, 1);
    cv::Mat p2 = (cv::Mat_<float>(4,1) << d, -d*0.8, d*0.5, 1);
    cv::Mat p3 = (cv::Mat_<float>(4,1) << -d, -d*0.8, d*0.5, 1);
    cv::Mat p4 = (cv::Mat_<float>(4,1) << -d, d*0.8, d*0.5, 1);

    cv::Mat T = Twc.clone();
    T.at<float>(0,3) = T.at<float>(0,3)/mScaleRatio;
    T.at<float>(1,3) = T.at<float>(1,3)/mScaleRatio;
    T.at<float>(2,3) = T.at<float>(2,3)/mScaleRatio;
    cv::Mat ow = T*o;
    cv::Mat p1w = T*p1;
    cv::Mat p2w = T*p2;
    cv::Mat p3w = T*p3;
    cv::Mat p4w = T*p4;

    geometry_msgs::Point msgs_o,msgs_p1, msgs_p2, msgs_p3, msgs_p4;
    msgs_o.x  = ow. at<float>(0);
    msgs_o.y  = ow. at<float>(1);
    msgs_o.z  = ow. at<float>(2);
    msgs_p1.x = p1w.at<float>(0);
    msgs_p1.y = p1w.at<float>(1);
    msgs_p1.z = p1w.at<float>(2);
    msgs_p2.x = p2w.at<float>(0);
    msgs_p2.y = p2w.at<float>(1);
    msgs_p2.z = p2w.at<float>(2);
    msgs_p3.x = p3w.at<float>(0);
    msgs_p3.y = p3w.at<float>(1);
    msgs_p3.z = p3w.at<float>(2);
    msgs_p4.x = p4w.at<float>(0);
    msgs_p4.y = p4w.at<float>(1);
    msgs_p4.z = p4w.at<float>(2);

    mKFNow.points.push_back(msgs_o);
    mKFNow.points.push_back(msgs_p1);
    mKFNow.points.push_back(msgs_o);
    mKFNow.points.push_back(msgs_p2);
    mKFNow.points.push_back(msgs_o);
    mKFNow.points.push_back(msgs_p3);
    mKFNow.points.push_back(msgs_o);
    mKFNow.points.push_back(msgs_p4);
    mKFNow.points.push_back(msgs_p1);
    mKFNow.points.push_back(msgs_p2);
    mKFNow.points.push_back(msgs_p2);
    mKFNow.points.push_back(msgs_p3);
    mKFNow.points.push_back(msgs_p3);
    mKFNow.points.push_back(msgs_p4);
    mKFNow.points.push_back(msgs_p4);
    mKFNow.points.push_back(msgs_p1);

    mKFNow.header.stamp = ros::Time::now();

    publisher.publish(mKFNow);
}

void MapPublish::run(){

    mbIsLocalize = Config::LOCALIZATION_ONLY;

    image_transport::ImageTransport it(nh);
    image_transport::Publisher pub = it.advertise("/camera/framepub", 1);
    image_transport::Publisher pubImgMatches = it.advertise("/camera/imageMatches", 1);

    ros::Rate rate(Config::FPS/* / 3*/);
    while (nh.ok()) {

        if (CheckFinish()) {
            break;
        }

        cv::Mat img = mpFramePub->drawFrame();
        if (img.empty())
            continue;

        sensor_msgs::ImagePtr msg =
            cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg();
        pub.publish(msg);

        cv::Mat imgMatch = mpFramePub->drawMatch();
        float lastThetaKF = mpMap->getCurrentKF()->odom.theta;
        float currTheta = mpLocalize->getCurrentFrameOdom().z;
        float dt = (currTheta - lastThetaKF) * 180 / M_PI;
        if (dt > 180.)       dt -= 360;
        else if (dt < -180.) dt += 360;
        string str = "d_theta: " + to_string(dt);
        cv::putText(imgMatch, str, Point(30, 15), 1, 1.1, Scalar(0,0,255), 2);
        sensor_msgs::ImagePtr msgMatch =
            cv_bridge::CvImage(std_msgs::Header(), "bgr8", imgMatch).toImageMsg();
        pubImgMatches.publish(msgMatch);
        // debug 存下match图片
        if (Config::SAVE_MATCH_IMAGE) {
            string fileName = Config::SAVE_MATCH_IMAGE_PATH +
                              to_string(mpMap->getCurrentKF()->mIdKF) + ".jpg";
            cv::imwrite(fileName, imgMatch);
        }

        if (mpMap->empty())
            continue;

        cv::Mat cTw;
        if (mbIsLocalize) {
            if (mpLocalize->mpKFCurr == nullptr || mpLocalize->mpKFCurr->isNull()) {
                continue;
            }
            cTw = mpLocalize->mpKFCurr->getPose();
        } else {
            cTw = mpMap->getCurrentFramePose();
        }

//                cv::Mat cTw;
//                cTw = mpMap->getCurrentFramePose();

        PublishCameraCurr(cTw.inv());
        PublishKeyFrames();
        PublishMapPoints();
        PublishOdomInformation();

        rate.sleep();
        ros::spinOnce();
    }
    cout << "[MapPublish] Exiting mappublish .." << endl;

    nh.shutdown();

    SetFinish();
}

void MapPublish::setFramePub(FramePublish *pFP){
    mpFramePub = pFP;
}

void MapPublish::RequestFinish() {
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool MapPublish::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void MapPublish::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool MapPublish::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}


void MapPublish::PublishOdomInformation()
{
//    mOdomRawGraph.points.clear();
//    static bool firstFrame = true;
    static geometry_msgs::Point msgsLast;
//    static Se2 odomDelta;

//    if (firstFrame) {
//        odomDelta = mpMap->getCurrentKF()->odom;
//        firstFrame = false;
//    }

//    geometry_msgs::Point msgs, msgsCurr;
//    msgs.x = mpMap->getCurrentKF()->odom.x/mScaleRatio;
//    msgs.y = mpMap->getCurrentKF()->odom.y/mScaleRatio;
//    msgsCurr.x = (msgs.x - odomDelta.x) * cos(odomDelta.theta) - (msgs.y - odomDelta.y) * sin(odomDelta.theta);
//    msgsCurr.y = (msgs.x - odomDelta.x) * sin(odomDelta.theta) + (msgs.y - odomDelta.y) * cos(odomDelta.theta);
//    mOdomRawGraph.points.push_back(msgsLast);
//    mOdomRawGraph.points.push_back(msgsCurr);
//    msgsLast = msgsCurr;

    //! odom在body系下，转到camera系下和计算值进行对比
    geometry_msgs::Point msgs;

    if (!mbIsLocalize) {
        msgs.x = mpMap->getCurrentKF()->odom.x/mScaleRatio;
        msgs.y = mpMap->getCurrentKF()->odom.y/mScaleRatio;
        mOdomRawGraph.points.push_back(msgsLast);
        mOdomRawGraph.points.push_back(msgs);
        msgsLast = msgs;
    } else {
        msgs.x = mpLocalize->mpKFCurr->odom.x/mScaleRatio;
        msgs.y = mpLocalize->mpKFCurr->odom.y/mScaleRatio;
        mOdomRawGraph.points.push_back(msgsLast);
        mOdomRawGraph.points.push_back(msgs);
        msgsLast = msgs;
//        msgsLast.x = mpLocalize->mpKFRef->Twb.x/mScaleRatio;
//        msgsLast.y = mpLocalize->mpKFRef->Twb.y/mScaleRatio;
    }


//    fprintf(stderr, "[MapPublish] A OdomRawGraph marker (%f, %f) published.\n",
//            msgs.x, msgs.y);
//    fprintf(stderr, "[MapPublish] mOdomRawGraph size: %ld.\n", mOdomRawGraph.points.size());
    mOdomRawGraph.header.stamp = ros::Time::now();
    publisher.publish(mOdomRawGraph);
}

}
