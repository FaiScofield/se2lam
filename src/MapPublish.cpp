/**
 * This file is part of se2lam
 *
 * Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG
 * (github.com/hbtang)
 */

#include "MapPublish.h"
#include "Map.h"
#include "converter.h"
#include <cstdio>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <sensor_msgs/Image.h>

namespace se2lam
{

using namespace cv;
using namespace std;

typedef unique_lock<mutex> locker;

MapPublish::MapPublish(Map* pMap)
    : mbIsLocalize(Config::LocalizationOnly), mpMap(pMap), mPointSize(0.1f), mCameraSize(0.3f),
      mScaleRatio(Config::MappubScaleRatio), mbFinishRequested(false), mbFinished(false)
{
    const char* MAP_FRAME_ID = "/se2lam/World";

    // Global (Negtive) KFs, black
    mKFsNeg.header.frame_id = MAP_FRAME_ID;
    mKFsNeg.ns = "KeyFramesNegative";
    mKFsNeg.id = 0;
    mKFsNeg.type = visualization_msgs::Marker::LINE_LIST;
    mKFsNeg.action = visualization_msgs::Marker::ADD;
    mKFsNeg.scale.x = 0.1;
    mKFsNeg.scale.y = 0.1;
    mKFsNeg.pose.orientation.w = 1.0;
    mKFsNeg.color.r = 0.0;
    mKFsNeg.color.g = 0.0;
    mKFsNeg.color.b = 0.0;
    mKFsNeg.color.a = 1.0;

    // Local (Active) KFs, blue
    mKFsAct.header.frame_id = MAP_FRAME_ID;
    mKFsAct.ns = "KeyFramesActive";
    mKFsAct.id = 1;
    mKFsAct.type = visualization_msgs::Marker::LINE_LIST;
    mKFsAct.action = visualization_msgs::Marker::ADD;
    mKFsAct.scale.x = 0.1;
    mKFsAct.scale.y = 0.1;
    mKFsAct.pose.orientation.w = 1.0;
    mKFsAct.color.b = 1.0;
    mKFsAct.color.a = 1.0;

    // Configure Current Camera, red
    mKFNow.header.frame_id = MAP_FRAME_ID;
    mKFNow.ns = "Camera";
    mKFNow.id = 2;
    mKFNow.type = visualization_msgs::Marker::LINE_LIST;
    mKFNow.action = visualization_msgs::Marker::ADD;
    mKFNow.scale.x = 0.1;
    mKFNow.scale.y = 0.1;
    mKFNow.pose.orientation.w = 1.0;
    mKFNow.color.r = 1.0;
    mKFNow.color.a = 1.0;

    // Configure MPs not in local map, black
    mMPsNeg.header.frame_id = MAP_FRAME_ID;
    mMPsNeg.ns = "MapPointsNegative";
    mMPsNeg.id = 3;
    mMPsNeg.type = visualization_msgs::Marker::POINTS;
    mMPsNeg.action = visualization_msgs::Marker::ADD;
    mMPsNeg.scale.x = mPointSize;
    mMPsNeg.scale.y = mPointSize;
    mMPsNeg.pose.orientation.w = 1.0;
    mMPsNeg.color.r = 0.0;
    mMPsNeg.color.g = 0.0;
    mMPsNeg.color.b = 0.0;
    mMPsNeg.color.a = 1.0;

    // Configure MPs in local map, blue
    mMPsAct.header.frame_id = MAP_FRAME_ID;
    mMPsAct.ns = "MapPointsActive";
    mMPsAct.id = 4;
    mMPsAct.type = visualization_msgs::Marker::POINTS;
    mMPsAct.action = visualization_msgs::Marker::ADD;
    mMPsAct.scale.x = mPointSize;
    mMPsAct.scale.y = mPointSize;
    mMPsAct.pose.orientation.w = 1.0;
    mMPsAct.color.r = 0.0;
    mMPsAct.color.g = 0.0;
    mMPsAct.color.b = 1.0;
    mMPsAct.color.a = 1.0;

    // Configure MPs currently observed, red
    mMPsNow.header.frame_id = MAP_FRAME_ID;
    mMPsNow.ns = "MapPointsNow";
    mMPsNow.id = 5;
    mMPsNow.type = visualization_msgs::Marker::POINTS;
    mMPsNow.action = visualization_msgs::Marker::ADD;
    mMPsNow.scale.x = mPointSize * 1.5f;
    mMPsNow.scale.y = mPointSize * 1.5f;
    mMPsNow.pose.orientation.w = 1.0;
    mMPsNow.color.r = 1.0;
    mMPsNow.color.g = 0.0;
    mMPsNow.color.b = 0.0;
    mMPsNow.color.a = 1.0;

    // Configure Covisibility Graph
    mCovisGraph.header.frame_id = MAP_FRAME_ID;
    mCovisGraph.ns = "CovisGraph";
    mCovisGraph.id = 6;
    mCovisGraph.type = visualization_msgs::Marker::LINE_LIST;
    mCovisGraph.action = visualization_msgs::Marker::ADD;
    mCovisGraph.scale.x = 0.03;
    mCovisGraph.scale.y = 0.03;
    mCovisGraph.pose.orientation.w = 1.0;
    mCovisGraph.color.r = 0.0;
    mCovisGraph.color.g = 1.0;
    mCovisGraph.color.b = 0.0;
    mCovisGraph.color.a = 0.5;

    // Configure Feature Constraint Graph
    mFeatGraph.header.frame_id = MAP_FRAME_ID;
    mFeatGraph.ns = "FeatGraph";
    mFeatGraph.id = 7;
    mFeatGraph.type = visualization_msgs::Marker::LINE_LIST;
    mFeatGraph.action = visualization_msgs::Marker::ADD;
    mFeatGraph.scale.x = 0.03;
    mFeatGraph.scale.y = 0.03;
    mFeatGraph.pose.orientation.w = 1.0;
    mFeatGraph.color.r = 0.0;
    mFeatGraph.color.g = 0.0;
    mFeatGraph.color.b = 1.0;
    mFeatGraph.color.a = 0.5;

    // Configure Odometry Constraint Graph, black
    mVIGraph.header.frame_id = MAP_FRAME_ID;
    mVIGraph.ns = "VIGraph";
    mVIGraph.id = 8;
    mVIGraph.type = visualization_msgs::Marker::LINE_LIST;
    mVIGraph.action = visualization_msgs::Marker::ADD;
    mVIGraph.scale.x = 0.06;
    mVIGraph.scale.y = 0.06;
    mVIGraph.pose.orientation.w = 1.0;

    mVIGraph.color.r = 0.0;
    mVIGraph.color.g = 0.0;
    mVIGraph.color.b = 0.0;
    mVIGraph.color.a = 1.0;

    // Configure Odometry Raw Graph, light red
    mOdomRawGraph.header.frame_id = MAP_FRAME_ID;
    mOdomRawGraph.ns = "OdomRawGraph";
    mOdomRawGraph.id = 9;
    mOdomRawGraph.type = visualization_msgs::Marker::LINE_LIST;
    mOdomRawGraph.action = visualization_msgs::Marker::ADD;
    mOdomRawGraph.scale.x = 0.05;
    mOdomRawGraph.scale.y = 0.05;
    mOdomRawGraph.pose.orientation.w = 1.0;
    mOdomRawGraph.color.r = 0.8;
    mOdomRawGraph.color.g = 0.0;
    mOdomRawGraph.color.b = 0.0;
    mOdomRawGraph.color.a = 0.75;

    // Cofigure MPs with no good Parallax, purple
    mMPsNoGoodPrl.header.frame_id = MAP_FRAME_ID;
    mMPsNoGoodPrl.ns = "MapPointsNoGoodParallax";
    mMPsNoGoodPrl.id = 10;
    mMPsNoGoodPrl.type = visualization_msgs::Marker::POINTS;
    mMPsNoGoodPrl.action = visualization_msgs::Marker::ADD;
    mMPsNoGoodPrl.scale.x = 0.08;
    mMPsNoGoodPrl.scale.y = 0.08;
    mMPsNoGoodPrl.pose.orientation.w = 1.0;
    mMPsNoGoodPrl.color.r = 1.0;
    mMPsNoGoodPrl.color.g = 0.2;
    mMPsNoGoodPrl.color.b = 1.0;
    mMPsNoGoodPrl.color.a = 0.8;

    tf::Transform tfT;
    tfT.setIdentity();
    tfb.sendTransform(tf::StampedTransform(tfT, ros::Time::now(), "/se2lam/World", "/se2lam/Camera"));

    publisher = nh.advertise<visualization_msgs::Marker>("se2lam/Map", 10);

    publisher.publish(mKFsNeg);
    publisher.publish(mKFsAct);
    publisher.publish(mKFNow);

    publisher.publish(mMPsNeg);
    publisher.publish(mMPsAct);
    publisher.publish(mMPsNow);
    publisher.publish(mMPsNoGoodPrl);

    publisher.publish(mCovisGraph);
    publisher.publish(mFeatGraph);

    publisher.publish(mVIGraph);
    publisher.publish(mOdomRawGraph);
}

MapPublish::~MapPublish() {}

void MapPublish::run()
{
    assert(Config::NeedVisualization == 1);

    image_transport::ImageTransport it(nh);
    image_transport::Publisher pub = it.advertise("/camera/framepub", 1);
    image_transport::Publisher pubImgMatches = it.advertise("/camera/imageMatches", 1);

    ros::Rate rate(Config::FPS * 3);
    while (nh.ok()) {
        if (checkFinish())
            break;

        if (mpMap->empty()) {
            rate.sleep();
            continue;
        }

        if (!mbUpdated) {  // Tracker/Localizer 更新此标志
            rate.sleep();
            ros::spinOnce();
            continue;
        }
        mbUpdated = false;

        Mat imgMatch = drawCurrentFrameMatches();
        sensor_msgs::ImagePtr msgMatch =
                cv_bridge::CvImage(std_msgs::Header(), "bgr8", imgMatch).toImageMsg();
        pubImgMatches.publish(msgMatch);

        // debug 存下match图片
        if (Config::SaveMatchImage) {
            string fileName = Config::MatchImageStorePath + to_string(mnCurrentFrameID) + ".jpg";
            imwrite(fileName, imgMatch);
        }

        publishCameraCurr(cvu::inv(mCurrentFramePose));
        publishOdomInformation();
        if (mpMap->mbNewKFInserted) {  // Map不变时没必要重复显示
            mpMap->mbNewKFInserted = false;
            publishKeyFrames();
            publishMapPoints();
        }

        rate.sleep();
        ros::spinOnce();
    }
    cerr << "[MapPu][Info ] Exiting Mappublish..." << endl;

    nh.shutdown();

    setFinish();
}

void MapPublish::publishKeyFrames()
{
    mKFsNeg.points.clear();
    mKFsAct.points.clear();
    mCovisGraph.points.clear();
    mFeatGraph.points.clear();
    if (!mbIsLocalize)
        mVIGraph.points.clear();

    // Camera is a pyramid. Define in camera coordinate system
    const float d = mCameraSize;
    const Mat o = (Mat_<float>(4, 1) << 0, 0, 0, 1);
    const Mat p1 = (Mat_<float>(4, 1) << d, d * 0.8, d * 0.5, 1);
    const Mat p2 = (Mat_<float>(4, 1) << d, -d * 0.8, d * 0.5, 1);
    const Mat p3 = (Mat_<float>(4, 1) << -d, -d * 0.8, d * 0.5, 1);
    const Mat p4 = (Mat_<float>(4, 1) << -d, d * 0.8, d * 0.5, 1);

    const vector<PtrKeyFrame> vKFsAll = mpMap->getAllKFs();
    if (vKFsAll.empty())
        return;

    vector<PtrKeyFrame> vKFsAct, vKFsNeg;
    if (mbIsLocalize)
        vKFsAct = mpLocalizer->getLocalKFs();
    else
        vKFsAct = mpMap->getLocalKFs();

    //! vKFsAll是从Map里获取的，在LocalizeOnly模式下是固定的
    for (int i = 0, iend = vKFsAll.size(); i != iend; ++i) {
        const PtrKeyFrame& pKFi = vKFsAll[i];
        if (!pKFi || pKFi->isNull())
            continue;

        // 按比例缩放地图, mScaleRatio = 300 时显示比例为 1:3.333，数据源单位是[mm]
        Mat Twc = cvu::inv(pKFi->getPose());
        Mat Twb = Twc * Config::Tcb;
        Twc.at<float>(0, 3) = Twc.at<float>(0, 3) / mScaleRatio;
        Twc.at<float>(1, 3) = Twc.at<float>(1, 3) / mScaleRatio;
        Twc.at<float>(2, 3) = Twc.at<float>(2, 3) / mScaleRatio;

        Mat ow = Twc * o;  // 第i帧KF的相机中心位姿
        Mat p1w = Twc * p1;
        Mat p2w = Twc * p2;
        Mat p3w = Twc * p3;
        Mat p4w = Twc * p4;

        Mat ob = Twb * o;  // 第i帧KF的Body中心位姿
        geometry_msgs::Point msgs_b;
        msgs_b.x = ob.at<float>(0) / mScaleRatio;
        msgs_b.y = ob.at<float>(1) / mScaleRatio;
        msgs_b.z = ob.at<float>(2) / mScaleRatio;

        // 相机的中心和4周围个点，可视化用
        geometry_msgs::Point msgs_o, msgs_p1, msgs_p2, msgs_p3, msgs_p4;
        msgs_o.x = ow.at<float>(0);
        msgs_o.y = ow.at<float>(1);
        msgs_o.z = ow.at<float>(2);
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

        // 判断第i帧KF是Active/Active
        int count = std::count(vKFsAct.begin(), vKFsAct.end(), pKFi);
        if (count == 0) {  // Negtive
            vKFsNeg.push_back(pKFi);
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
        } else {  // Active
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
        vector<PtrKeyFrame> covKFs = pKFi->getAllCovisibleKFs();
        if (!covKFs.empty()) {
            for (auto it = covKFs.begin(), iend = covKFs.end(); it != iend; ++it) {
                if ((*it)->mIdKF > pKFi->mIdKF)  // 只统计在自己前面的共视KF, 防止重复计入
                    continue;
                Mat Twc2 = cvu::inv((*it)->getPose()) /** Config::Tcb*/;
                geometry_msgs::Point msgs_o2;
                msgs_o2.x = Twc2.at<float>(0, 3) / mScaleRatio;
                msgs_o2.y = Twc2.at<float>(1, 3) / mScaleRatio;
                msgs_o2.z = Twc2.at<float>(2, 3) / mScaleRatio;
                mCovisGraph.points.push_back(msgs_o);
                mCovisGraph.points.push_back(msgs_o2);
            }
        }

        // Feature Graph
        for (auto iter = pKFi->mFtrMeasureFrom.begin(); iter != pKFi->mFtrMeasureFrom.end(); iter++) {
            PtrKeyFrame pKF2 = iter->first;
            Mat Twc2 = cvu::inv(pKF2->getPose()) /* * Config::Tcb*/;
            geometry_msgs::Point msgs_o2;
            msgs_o2.x = Twc2.at<float>(0, 3) / mScaleRatio;
            msgs_o2.y = Twc2.at<float>(1, 3) / mScaleRatio;
            msgs_o2.z = Twc2.at<float>(2, 3) / mScaleRatio;
            mFeatGraph.points.push_back(msgs_o);
            mFeatGraph.points.push_back(msgs_o2);
        }

        // Visual Odometry Graph (estimate). VI轨迹转到odo坐标系下, 方便和odo轨迹对比
        PtrKeyFrame pKFOdoChild = pKFi->mOdoMeasureFrom.first;  // 下一帧
        if (pKFOdoChild != nullptr && !mbIsLocalize) {
            assert(pKFOdoChild->mIdKF - pKFi->mIdKF >= 1);
            Mat Twb = cvu::inv(pKFOdoChild->getPose()) * Config::Tcb;
            geometry_msgs::Point msgs_b2;
            msgs_b2.x = Twb.at<float>(0, 3) / mScaleRatio;
            msgs_b2.y = Twb.at<float>(1, 3) / mScaleRatio;
            msgs_b2.z = Twb.at<float>(2, 3) / mScaleRatio;
            mVIGraph.points.push_back(msgs_b);  // 当前帧的位姿(odo坐标系)
            mVIGraph.points.push_back(msgs_b2);  // 下一帧的位姿
        }
    }

    //! Visual Odometry Graph for Localize only case
    //! 注意位姿访问要带锁, Localizer里对位姿改变时也要上锁, 否则数据不一定会对应上
    if (mbIsLocalize) {
        static geometry_msgs::Point msgsLast;  // 这个必须要静态变量, 要保证在下一帧时可以保存上一帧的位姿
        geometry_msgs::Point msgsCurr;

        auto currState = mpLocalizer->getTrackingState();
        auto lastState = mpLocalizer->getLastTrackingState();
        auto id = mpLocalizer->getKFCurr()->mIdKF;

        if (currState == cvu::OK || currState == cvu::LOST) {
            static bool firstLocatied = true;
            Se2 Twb = mpLocalizer->getCurrKFPose();  // 带锁访问
            msgsCurr.x = Twb.x / mScaleRatio;
            msgsCurr.y = Twb.y / mScaleRatio;

            if (currState == cvu::OK && lastState == cvu::LOST) {
                firstLocatied = true;
                fprintf(stderr, "[MapPublis] #%ld lastState = LOST and currentState = OK\n", id);
            }
            if (firstLocatied) {
                firstLocatied = false;
                msgsLast = msgsCurr;
            }

            mVIGraph.points.push_back(msgsLast);
            mVIGraph.points.push_back(msgsCurr);
            //            printf("[MapPublis] #%ld msgsLast: [%.4f, %.4f], msgsCurr: [%.4f, %.4f],
            //            Twb: [%.4f, %.4f]\n",
            //                   id, msgsLast.x*mScaleRatio/1000., msgsLast.y*mScaleRatio/1000.,
            //                   msgsCurr.x*mScaleRatio/1000., msgsCurr.y*mScaleRatio/1000.,
            //                   Twb.x/1000., Twb.y/1000.);

            msgsLast = msgsCurr;
        }
    }

    mKFsAct.header.stamp = ros::Time::now();
    mKFsNeg.header.stamp = ros::Time::now();
    mCovisGraph.header.stamp = ros::Time::now();
    mFeatGraph.header.stamp = ros::Time::now();
    mVIGraph.header.stamp = ros::Time::now();

    publisher.publish(mKFsNeg);
    publisher.publish(mKFsAct);
    publisher.publish(mCovisGraph);
    publisher.publish(mFeatGraph);
    publisher.publish(mVIGraph);

    cout << "[MapPublisher] KF#" << mpMap->getCurrentKF()->mIdKF
         << " 当前Map的组成: Active/Negtive/All = " << mKFsAct.points.size() / 16 << "/"
         << mKFsNeg.points.size() / 16 << "/" << vKFsAll.size() << ", mVIGraph size = "
         << mVIGraph.points.size() / 2 << endl;

//    cout << "[MapPublisher] KF#" << mpMap->getCurrentKF()->mIdKF << " Active KFs: ";
//    for (size_t i = 0; i < vKFsAct.size(); ++i)
//        cout << vKFsAct[i]->mIdKF << ", ";
//    cout << endl;

//    cout << "[MapPublisher] KF#" << mpMap->getCurrentKF()->mIdKF << " Negtive KFs: ";
//    for (size_t i = 0; i < vKFsNeg.size(); ++i)
//        cout << vKFsNeg[i]->mIdKF << ", ";
//    cout << endl;
}

void MapPublish::publishMapPoints()
{
    mMPsNeg.points.clear();
    mMPsAct.points.clear();
    mMPsNow.points.clear();
    mMPsNoGoodPrl.points.clear();

    const vector<PtrMapPoint> vpMPAll = mpMap->getAllMPs();
    vector<PtrMapPoint> vpMPNow;
    vector<PtrMapPoint> vpMPAct;
    vector<PtrMapPoint> vpMPNeg;
    if (mbIsLocalize) {
        locker lock(mpLocalizer->mMutexLocalMap);
        vpMPAct = mpLocalizer->getLocalMPs();
        vpMPNow = mpLocalizer->mpKFCurr->getObservations(true, false);
    } else {
        vpMPAct = mpMap->getLocalMPs();
        vpMPNow = mpMap->getCurrentKF()->getObservations(true, false);
    }

    // MPsAll 包含了 MPsNeg 和 MPsAct
    for (auto iter = vpMPAll.begin(); iter != vpMPAll.end(); iter++) {
        PtrMapPoint pMPtemp = *iter;
        int count = std::count(vpMPAct.begin(), vpMPAct.end(), pMPtemp);
        if (count == 0)
            vpMPNeg.push_back(pMPtemp);
    }

    // MPsAct 包含了 MPsNow
    vector<PtrMapPoint> vpMPActGood;
    for (auto iter = vpMPAct.begin(); iter != vpMPAct.end(); iter++) {
        PtrMapPoint pMPtemp = *iter;
        int count = std::count(vpMPNow.begin(), vpMPNow.end(), pMPtemp);
        if (count == 0)
            vpMPActGood.push_back(pMPtemp);
    }
    vpMPAct.swap(vpMPActGood);  // MPsAct 去掉 MPsNow

    mMPsNeg.points.reserve(vpMPNeg.size() * 0.6);
    mMPsNoGoodPrl.points.reserve(vpMPNeg.size() * 0.6);
    for (size_t i = 0, iend = vpMPNeg.size(); i != iend; ++i) {
        if (!vpMPNeg[i] || vpMPNeg[i]->isNull())
            continue;

        const Point3f posei = vpMPNeg[i]->getPos();
        geometry_msgs::Point msg_p;
        msg_p.x = posei.x / mScaleRatio;
        msg_p.y = posei.y / mScaleRatio;
        msg_p.z = posei.z / mScaleRatio;

        if (!vpMPNeg[i]->isGoodPrl())
            mMPsNoGoodPrl.points.push_back(msg_p);
        else
            mMPsNeg.points.push_back(msg_p);
    }

    mMPsAct.points.reserve(vpMPAct.size());
    for (size_t i = 0, iend = vpMPAct.size(); i != iend; ++i) {
        if (!vpMPAct[i] || vpMPAct[i]->isNull())
            continue;

        const Point3f posei = vpMPAct[i]->getPos();
        geometry_msgs::Point msg_p;
        msg_p.x = posei.x / mScaleRatio;
        msg_p.y = posei.y / mScaleRatio;
        msg_p.z = posei.z / mScaleRatio;

        mMPsAct.points.push_back(msg_p);
    }

    mMPsNow.points.reserve(vpMPNow.size());
    for (size_t i = 0, iend = vpMPNow.size(); i != iend; ++i) {
        if (!vpMPNow[i] || vpMPNow[i]->isNull())
            continue;

        const Point3f posei = vpMPNow[i]->getPos();
        geometry_msgs::Point msg_p;
        msg_p.x = posei.x / mScaleRatio;
        msg_p.y = posei.y / mScaleRatio;
        msg_p.z = posei.z / mScaleRatio;

        mMPsNow.points.push_back(msg_p);
    }

    mMPsNeg.header.stamp = ros::Time::now();
    mMPsAct.header.stamp = ros::Time::now();
    mMPsNow.header.stamp = ros::Time::now();
    mMPsNoGoodPrl.header.stamp = ros::Time::now();

    publisher.publish(mMPsNeg);
    publisher.publish(mMPsAct);
    publisher.publish(mMPsNow);
    publisher.publish(mMPsNoGoodPrl);
}

void MapPublish::publishCameraCurr(const Mat& Twc)
{
    mKFNow.points.clear();

    tf::Matrix3x3 R(Twc.at<float>(0, 0), Twc.at<float>(0, 1), Twc.at<float>(0, 2),
                    Twc.at<float>(1, 0), Twc.at<float>(1, 1), Twc.at<float>(1, 2),
                    Twc.at<float>(2, 0), Twc.at<float>(2, 1), Twc.at<float>(2, 2));
    tf::Vector3 t(Twc.at<float>(0, 3) / mScaleRatio, Twc.at<float>(1, 3) / mScaleRatio,
                  Twc.at<float>(2, 3) / mScaleRatio);
    tf::Transform tfwTc(R, t);
    tfb.sendTransform(tf::StampedTransform(tfwTc, ros::Time::now(), "se2lam/World", "se2lam/Camera"));

    const float d = mCameraSize;

    // Camera is a pyramid. Define in camera coordinate system
    Mat o = (Mat_<float>(4, 1) << 0, 0, 0, 1);
    Mat p1 = (Mat_<float>(4, 1) << d, d * 0.8, d * 0.5, 1);
    Mat p2 = (Mat_<float>(4, 1) << d, -d * 0.8, d * 0.5, 1);
    Mat p3 = (Mat_<float>(4, 1) << -d, -d * 0.8, d * 0.5, 1);
    Mat p4 = (Mat_<float>(4, 1) << -d, d * 0.8, d * 0.5, 1);

    Mat T = Twc.clone();
    T.at<float>(0, 3) = T.at<float>(0, 3) / mScaleRatio;
    T.at<float>(1, 3) = T.at<float>(1, 3) / mScaleRatio;
    T.at<float>(2, 3) = T.at<float>(2, 3) / mScaleRatio;
    Mat ow = T * o;
    Mat p1w = T * p1;
    Mat p2w = T * p2;
    Mat p3w = T * p3;
    Mat p4w = T * p4;

    geometry_msgs::Point msgs_o, msgs_p1, msgs_p2, msgs_p3, msgs_p4;
    msgs_o.x = ow.at<float>(0);
    msgs_o.y = ow.at<float>(1);
    msgs_o.z = ow.at<float>(2);
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

void MapPublish::publishOdomInformation()
{
    static geometry_msgs::Point msgsLast;
    geometry_msgs::Point msgsCurr;

    // Se2 currOdom = mpTracker->getCurrentFrameOdo(); // odo全部显示
    Se2 currOdom = mpMap->getCurrentKF()->odom;  // 只显示KF的odo
    if (!mbIsLocalize) {
        //! 这里要对齐到首帧的Odom, 位姿从原点开始
        static Mat Tb0w = cvu::inv(currOdom.toCvSE3());

        Mat Twbi = currOdom.toCvSE3();
        Mat Tb0bi = (Tb0w * Twbi).col(3);
        msgsCurr.x = Tb0bi.at<float>(0, 0) / mScaleRatio;
        msgsCurr.y = Tb0bi.at<float>(1, 0) / mScaleRatio;
    } else {
        //! 这里要对齐到首帧的Pose, 位姿不一定从原点开始
        auto currState = mpLocalizer->getTrackingState();
        auto lastState = mpLocalizer->getLastTrackingState();

        if (currState == cvu::OK) {
            PtrKeyFrame pKF = mpLocalizer->getKFCurr();

            static bool isFirstFrame = true;
            static Mat Twc_b = mpLocalizer->getCurrKFPose().toCvSE3();
            static Mat Tb0w = cvu::inv(pKF->odom.toCvSE3());

            Mat Twbi = (pKF->odom).toCvSE3();
            Mat Tb0bi = Tb0w * Twbi;  // 先变换到首帧的odom坐标系原点下
            Mat Twc_bi = (Twc_b * Tb0bi).col(3);  // 再变换到首帧的pose坐标系原点下
            msgsCurr.x = Twc_bi.at<float>(0, 0) / mScaleRatio;
            msgsCurr.y = Twc_bi.at<float>(1, 0) / mScaleRatio;
            if (currState == cvu::OK && lastState == cvu::LOST)
                isFirstFrame = true;
            if (isFirstFrame) {
                msgsLast = msgsCurr;
                isFirstFrame = false;
            }
        } else {
            return;
        }
    }

    mOdomRawGraph.points.push_back(msgsLast);
    mOdomRawGraph.points.push_back(msgsCurr);
    msgsLast = msgsCurr;

    mOdomRawGraph.header.stamp = ros::Time::now();
    publisher.publish(mOdomRawGraph);
}

void MapPublish::requestFinish()
{
    locker lock(mMutexFinish);
    mbFinishRequested = true;
}

bool MapPublish::checkFinish()
{
    locker lock(mMutexFinish);
    return mbFinishRequested;
}

void MapPublish::setFinish()
{
    locker lock(mMutexFinish);
    mbFinished = true;
}

bool MapPublish::isFinished()
{
    locker lock(mMutexFinish);
    return mbFinished;
}

Mat MapPublish::drawCurrentFrameMatches()
{
    locker lock(mMutexPub);

    Mat imgCur, imgRef, imgWarp, imgOut, A21;
    if (mCurrentImage.channels() == 1)
        cvtColor(mCurrentImage, imgCur, CV_GRAY2BGR);
    else
        imgCur = mCurrentImage;
    if (mReferenceImage.channels() == 1)
        cvtColor(mReferenceImage, imgRef, CV_GRAY2BGR);
    else
        imgRef = mReferenceImage;

    drawKeypoints(imgCur, mvCurrentKPs, imgCur, Scalar(255, 0, 0), DrawMatchesFlags::DRAW_OVER_OUTIMG);

    if (!mvReferenceKPs.empty()) {
        drawKeypoints(imgRef, mvReferenceKPs, imgRef, Scalar(255, 0, 0), DrawMatchesFlags::DRAW_OVER_OUTIMG);

        // 取逆得到A21
        invertAffineTransform(mAffineMatrix, A21);
        warpAffine(imgCur, imgWarp, A21, imgCur.size());
        hconcat(imgWarp, imgRef, imgOut);

        for (size_t i = 0, iend = mvMatchIdx.size(); i != iend; ++i) {
            if (mvMatchIdx[i] < 0) {
                continue;
            } else {
                const Point2f& ptRef = mvReferenceKPs[i].pt;
                const Point2f& ptCur = mvCurrentKPs[mvMatchIdx[i]].pt;
                const Mat pt1 = (Mat_<double>(3, 1) << ptCur.x, ptCur.y, 1);
                const Mat pt1W = A21 * pt1;
                const Point2f ptL = Point2f(pt1W.at<double>(0), pt1W.at<double>(1));
                const Point2f ptR = ptRef + Point2f(imgRef.cols, 0);

                if (mvMatchIdxGood[i] < 0) {  // 只有KP匹配对标绿色
                    circle(imgOut, ptL, 3, Scalar(0, 255, 0), -1);
                    circle(imgOut, ptR, 3, Scalar(0, 255, 0), -1);
                } else {  // KP匹配有MP观测或三角化深度符合的标黄色并连线
                    circle(imgOut, ptL, 3, Scalar(0, 255, 255), -1);
                    circle(imgOut, ptR, 3, Scalar(0, 255, 255), -1);
                    line(imgOut, ptL, ptR, Scalar(255, 255, 20));
                }
            }
        }
    } else {  // 说明定位丢失且找不到回环帧
        hconcat(imgCur, imgRef, imgOut);
    }
    putText(imgOut, mImageText, Point(100, 20), 1, 1, Scalar(0, 0, 255), 2);
    Mat imgScalar;
    resize(imgOut, imgScalar, Size2i(imgOut.cols * 2, imgOut.rows * 2));

    return imgScalar.clone();
}


}  // namespace se2lam
