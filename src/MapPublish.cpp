/**
 * This file is part of se2lam
 *
 * Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
 */

#include "MapPublish.h"
#include "FramePublish.h"
#include "Map.h"
#include "converter.h"
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>

namespace se2lam
{
using namespace cv;
using namespace std;

typedef unique_lock<mutex> locker;

MapPublish::MapPublish(Map* pMap)
    : mbIsLocalize(Config::LocalizationOnly), mpMap(pMap), mPointSize(0.2f), mCameraSize(0.5f),
      mScaleRatio(Config::MappubScaleRatio), mbFinishRequested(false), mbFinished(false)
{
    const char* MAP_FRAME_ID = "/se2lam/World";

    // Global (Negtive) KFs, black
    mKFsNeg.header.frame_id = MAP_FRAME_ID;
    mKFsNeg.ns = "KeyFramesNegative";
    mKFsNeg.id = 0;
    mKFsNeg.type = visualization_msgs::Marker::LINE_LIST;
    mKFsNeg.scale.x = 0.1;
    // mKFsNeg.scale.y = 0.1;
    mKFsNeg.pose.orientation.w = 1.0;
    mKFsNeg.action = visualization_msgs::Marker::ADD;
    mKFsNeg.color.r = 0.0;
    mKFsNeg.color.g = 0.0;
    mKFsNeg.color.b = 0.0;
    mKFsNeg.color.a = 0.5;

    // Local (Active) KFs, blue
    mKFsAct.header.frame_id = MAP_FRAME_ID;
    mKFsAct.ns = "KeyFramesActive";
    mKFsAct.id = 1;
    mKFsAct.type = visualization_msgs::Marker::LINE_LIST;
    mKFsAct.scale.x = 0.1;
    // mKFsAct.scale.y = 0.1;
    mKFsAct.pose.orientation.w = 1.0;
    mKFsAct.action = visualization_msgs::Marker::ADD;
    mKFsAct.color.b = 1.0;
    mKFsAct.color.a = 0.5;

    // Configure Current Camera, red
    mKFNow.header.frame_id = MAP_FRAME_ID;
    mKFNow.ns = "Camera";
    mKFNow.id = 2;
    mKFNow.type = visualization_msgs::Marker::LINE_LIST;
    mKFNow.scale.x = 0.1;
    // mKFNow.scale.y = 0.1;
    mKFNow.pose.orientation.w = 1.0;
    mKFNow.action = visualization_msgs::Marker::ADD;
    mKFNow.color.r = 1.0;
    mKFNow.color.a = 1.0;

    // Configure MPs not in local map, black
    mMPsNeg.header.frame_id = MAP_FRAME_ID;
    mMPsNeg.ns = "MapPointsNegative";
    mMPsNeg.id = 3;
    mMPsNeg.type = visualization_msgs::Marker::POINTS;
    mMPsNeg.scale.x = mPointSize;
    mMPsNeg.scale.y = mPointSize;
    mMPsNeg.pose.orientation.w = 1.0;
    mMPsNeg.action = visualization_msgs::Marker::ADD;
    mMPsNeg.color.r = 0.0;
    mMPsNeg.color.g = 0.0;
    mMPsNeg.color.b = 0.0;
    mMPsNeg.color.a = 1.0;

    // Configure MPs in local map, blue
    mMPsAct.header.frame_id = MAP_FRAME_ID;
    mMPsAct.ns = "MapPointsActive";
    mMPsAct.id = 4;
    mMPsAct.type = visualization_msgs::Marker::POINTS;
    mMPsAct.scale.x = mPointSize;
    mMPsAct.scale.y = mPointSize;
    mMPsAct.pose.orientation.w = 1.0;
    mMPsAct.action = visualization_msgs::Marker::ADD;
    mMPsAct.color.r = 0.0;
    mMPsAct.color.g = 0.0;
    mMPsAct.color.b = 1.0;
    mMPsAct.color.a = 1.0;

    // Configure MPs currently observed, red
    mMPsNow.header.frame_id = MAP_FRAME_ID;
    mMPsNow.ns = "MapPointsNow";
    mMPsNow.id = 5;
    mMPsNow.type = visualization_msgs::Marker::POINTS;
    mMPsNow.scale.x = mPointSize * 1.5f;
    mMPsNow.scale.y = mPointSize * 1.5f;
    mMPsNow.pose.orientation.w = 1.0;
    mMPsNow.action = visualization_msgs::Marker::ADD;
    mMPsNow.color.r = 1.0;
    mMPsNow.color.g = 0.0;
    mMPsNow.color.b = 0.0;
    mMPsNow.color.a = 1.0;

    // Configure Covisibility Graph
    mCovisGraph.header.frame_id = MAP_FRAME_ID;
    mCovisGraph.ns = "CovisGraph";
    mCovisGraph.id = 6;
    mCovisGraph.type = visualization_msgs::Marker::LINE_LIST;
    mCovisGraph.scale.x = 0.03;
    // mCovisGraph.scale.y = 0.03;
    mCovisGraph.pose.orientation.w = 1.0;
    mCovisGraph.action = visualization_msgs::Marker::ADD;
    mCovisGraph.color.r = 0.0;
    mCovisGraph.color.g = 1.0;
    mCovisGraph.color.b = 0.0;
    mCovisGraph.color.a = 0.5;

    // Configure Feature Constraint Graph
    mFeatGraph.header.frame_id = MAP_FRAME_ID;
    mFeatGraph.ns = "FeatGraph";
    mFeatGraph.id = 7;
    mFeatGraph.type = visualization_msgs::Marker::LINE_LIST;
    mFeatGraph.scale.x = 0.03;
    // mFeatGraph.scale.y = 0.03;
    mFeatGraph.pose.orientation.w = 1.0;
    mFeatGraph.action = visualization_msgs::Marker::ADD;
    mFeatGraph.color.r = 0.0;
    mFeatGraph.color.g = 0.0;
    mFeatGraph.color.b = 1.0;
    mFeatGraph.color.a = 0.5;

    // Configure Odometry Constraint Graph, black
    mVIGraph.header.frame_id = MAP_FRAME_ID;
    mVIGraph.ns = "VIGraph";
    mVIGraph.id = 8;
    mVIGraph.type = visualization_msgs::Marker::LINE_LIST;
    mVIGraph.scale.x = 0.06;
    // mVIGraph.scale.y = 0.06;
    mVIGraph.pose.orientation.w = 1.0;
    mVIGraph.action = visualization_msgs::Marker::ADD;
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
    // mOdomRawGraph.scale.y = 0.05;
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
    mMPsNoGoodPrl.scale.x = 0.08;
    mMPsNoGoodPrl.scale.y = 0.08;
    mMPsNoGoodPrl.pose.orientation.w = 1.0;
    mMPsNoGoodPrl.action = visualization_msgs::Marker::ADD;
    mMPsNoGoodPrl.color.r = 1.0;
    mMPsNoGoodPrl.color.g = 0.1;
    mMPsNoGoodPrl.color.b = 1.0;
    mMPsNoGoodPrl.color.a = 0.8;

    if (Config::ShowGroundTruth) {
        mGroundTruthGraph.header.frame_id = MAP_FRAME_ID;
        mGroundTruthGraph.ns = "GTGraph";
        mGroundTruthGraph.id = 11;
        mGroundTruthGraph.type = visualization_msgs::Marker::LINE_LIST;
        mGroundTruthGraph.scale.x = 0.06;
        // mGroundTruthGraph.scale.y = 0.06;
        mGroundTruthGraph.pose.orientation.w = 1.0;
        mGroundTruthGraph.action = visualization_msgs::Marker::ADD;
        mGroundTruthGraph.color.r = 1.0;
        mGroundTruthGraph.color.g = 0.0;
        mGroundTruthGraph.color.b = 1.0;
        mGroundTruthGraph.color.a = 1.0;
    }

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

    if (Config::ShowGroundTruth)
        publisher.publish(mGroundTruthGraph);
}

MapPublish::~MapPublish() {}


void MapPublish::run()
{
    assert(Config::NeedVisualization == 1);

    // load ground truth
    if (Config::ShowGroundTruth) {
        mvGroundTruth.reserve(Config::ImgCount);
        int id;
        float x1, y1, t1, x2, y2, t2;
        string gtFile("/home/vance/dataset/se2/DatasetRoom/ground_truth.txt");
        ifstream ifs(gtFile, ios_base::in);
        int i = 0;
        int offset = 0;
        if (ifs.is_open()) {
            string lineData;
            while (!ifs.eof()) {
                i++;
                if (i <= 500)  // 前500个数据紊乱
                    offset = 0;
                else
                    offset = -11680;

                getline(ifs, lineData);
                if (lineData.empty())
                    continue;

                stringstream ss(lineData);
                ss >> id >> x1 >> y1 >> t1 >> x2 >> y2 >> t2;
                Se2 gt(x2 + offset, y2, t2, id);
                mvGroundTruth.push_back(gt);
            }
        } else {
            cerr << "[MapPublisher] Read ground truth file error!" << endl;
        }
        cout << "[MapPublisher] Read  ground truth data: " << mvGroundTruth.size() << endl;
    }

    image_transport::ImageTransport it(nh);
    image_transport::Publisher pub = it.advertise("/camera/framepub", 1);

    int nSaveId = 1;
    ros::Rate rate(Config::FPS);
    while (nh.ok()) {
        if (checkFinish())
            break;
        if (mpMap->empty())
            continue;
        if (mpMap->countMPs() == 0)
            continue;

        cv::Mat img = mpFramePub->drawFrame();
        if (img.empty())
            continue;

        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg();
        pub.publish(msg);

        cv::Mat cTw;
        if (mbIsLocalize) {
            if (mpLocalizer->mpKFCurr == NULL || mpLocalizer->mpKFCurr->isNull()) {
                continue;
            }
            cTw = mpLocalizer->mpKFCurr->getPose();
        } else {
            cTw = mpMap->getCurrentFramePose();
        }

        publishOdomInformation();
        publishCameraCurr(cvu::inv(cTw));
        publishKeyFrames();
        publishMapPoints();
        if (Config::ShowGroundTruth)
            publishGroundTruth();

        rate.sleep();
        ros::spinOnce();
    }
    cerr << "[MapPublisher] Exiting mappublish..." << endl;

    nh.shutdown();

    setFinish();
}

void MapPublish::publishKeyFrames()
{
    mKFsNeg.points.clear();
    mKFsAct.points.clear();

    mCovisGraph.points.clear();
    mFeatGraph.points.clear();
    mVIGraph.points.clear();

    // Camera is a pyramid. Define in camera coordinate system
    float d = mCameraSize;
    cv::Mat o = (cv::Mat_<float>(4, 1) << 0, 0, 0, 1);
    cv::Mat p1 = (cv::Mat_<float>(4, 1) << d, d * 0.8, d * 0.5, 1);
    cv::Mat p2 = (cv::Mat_<float>(4, 1) << d, -d * 0.8, d * 0.5, 1);
    cv::Mat p3 = (cv::Mat_<float>(4, 1) << -d, -d * 0.8, d * 0.5, 1);
    cv::Mat p4 = (cv::Mat_<float>(4, 1) << -d, d * 0.8, d * 0.5, 1);

    vector<PtrKeyFrame> vKFsAll = mpMap->getAllKF();
    if (vKFsAll.empty()) {
        return;
    }

    vector<PtrKeyFrame> vKFsAct;
    if (mbIsLocalize) {
        vKFsAct = mpLocalizer->GetLocalKFs();
    } else {
        vKFsAct = mpMap->getLocalKFs();
    }

    for (int i = 0, iend = vKFsAll.size(); i < iend; i++) {
        if (vKFsAll[i]->isNull())
            continue;

        cv::Mat Twc = vKFsAll[i]->getPose().inv();

        Twc.at<float>(0, 3) = Twc.at<float>(0, 3) / mScaleRatio;
        Twc.at<float>(1, 3) = Twc.at<float>(1, 3) / mScaleRatio;
        Twc.at<float>(2, 3) = Twc.at<float>(2, 3) / mScaleRatio;

        cv::Mat ow = Twc * o;
        cv::Mat p1w = Twc * p1;
        cv::Mat p2w = Twc * p2;
        cv::Mat p3w = Twc * p3;
        cv::Mat p4w = Twc * p4;

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

        if (!covKFs.empty()) {
            for (auto it = covKFs.begin(), iend = covKFs.end(); it != iend; it++) {
                if ((*it)->mIdKF > vKFsAll[i]->mIdKF)
                    continue;
                cv::Mat Twc2 = (*it)->getPose().inv();
                geometry_msgs::Point msgs_o2;
                msgs_o2.x = Twc2.at<float>(0, 3) / mScaleRatio;
                msgs_o2.y = Twc2.at<float>(1, 3) / mScaleRatio;
                msgs_o2.z = Twc2.at<float>(2, 3) / mScaleRatio;
                mCovisGraph.points.push_back(msgs_o);
                mCovisGraph.points.push_back(msgs_o2);
            }
        }

        // Feature Graph
        PtrKeyFrame pKF = vKFsAll[i];
        for (auto iter = pKF->mFtrMeasureFrom.begin(); iter != pKF->mFtrMeasureFrom.end(); iter++) {
            PtrKeyFrame pKF2 = iter->first;
            Mat Twc2 = pKF2->getPose().inv();
            geometry_msgs::Point msgs_o2;
            msgs_o2.x = Twc2.at<float>(0, 3) / mScaleRatio;
            msgs_o2.y = Twc2.at<float>(1, 3) / mScaleRatio;
            msgs_o2.z = Twc2.at<float>(2, 3) / mScaleRatio;
            mFeatGraph.points.push_back(msgs_o);
            mFeatGraph.points.push_back(msgs_o2);
        }

        // Odometry Graph
        PtrKeyFrame pKFOdoChild = pKF->mOdoMeasureFrom.first;
        if (pKFOdoChild != NULL) {
            Mat Twc2 = pKFOdoChild->getPose().inv();
            geometry_msgs::Point msgs_o2;
            msgs_o2.x = Twc2.at<float>(0, 3) / mScaleRatio;
            msgs_o2.y = Twc2.at<float>(1, 3) / mScaleRatio;
            msgs_o2.z = Twc2.at<float>(2, 3) / mScaleRatio;
            mVIGraph.points.push_back(msgs_o);
            mVIGraph.points.push_back(msgs_o2);
        }
    }

    mKFsNeg.header.stamp = ros::Time::now();
    mCovisGraph.header.stamp = ros::Time::now();
    mFeatGraph.header.stamp = ros::Time::now();
    mVIGraph.header.stamp = ros::Time::now();

    publisher.publish(mKFsNeg);
    publisher.publish(mKFsAct);

    publisher.publish(mCovisGraph);
    publisher.publish(mFeatGraph);
    publisher.publish(mVIGraph);
}

void MapPublish::publishMapPoints()
{

    mMPsNeg.points.clear();
    mMPsAct.points.clear();
    mMPsNow.points.clear();

    set<PtrMapPoint> spMPNow;
    vector<PtrMapPoint> vpMPAct;
    if (mbIsLocalize) {
        locker lock(mpLocalizer->mMutexLocalMap);
        vpMPAct = mpLocalizer->GetLocalMPs();
        spMPNow = mpLocalizer->mpKFCurr->getAllObsMPs();
    } else {
        vpMPAct = mpMap->getLocalMPs();
        spMPNow = mpMap->getCurrentKF()->getAllObsMPs();
    }

    vector<PtrMapPoint> vpMPNeg = mpMap->getAllMP();

    vector<PtrMapPoint> vpMPNegGood;
    for (auto iter = vpMPNeg.begin(); iter != vpMPNeg.end(); iter++) {
        PtrMapPoint pMPtemp = *iter;
        int count = std::count(vpMPAct.begin(), vpMPAct.end(), pMPtemp);
        if (count == 0) {
            vpMPNegGood.push_back(pMPtemp);
        }
    }
    vpMPNeg.swap(vpMPNegGood);

    vector<PtrMapPoint> vpMPActGood;
    for (auto iter = vpMPAct.begin(); iter != vpMPAct.end(); iter++) {
        PtrMapPoint pMPtemp = *iter;
        int count = spMPNow.count(pMPtemp);
        if (count == 0) {
            vpMPActGood.push_back(pMPtemp);
        }
    }
    vpMPAct.swap(vpMPActGood);


    mMPsNeg.points.reserve(vpMPNeg.size());
    for (int i = 0, iend = vpMPNeg.size(); i < iend; i++) {
        if (vpMPNeg[i]->isNull() || !vpMPNeg[i]->isGoodPrl()) {
            continue;
        }
        geometry_msgs::Point msg_p;
        msg_p.x = vpMPNeg[i]->getPos().x / mScaleRatio;
        msg_p.y = vpMPNeg[i]->getPos().y / mScaleRatio;
        msg_p.z = vpMPNeg[i]->getPos().z / mScaleRatio;
        mMPsNeg.points.push_back(msg_p);
    }


    mMPsAct.points.reserve(vpMPAct.size());
    for (int i = 0, iend = vpMPAct.size(); i < iend; i++) {
        if (vpMPAct[i]->isNull() || !vpMPAct[i]->isGoodPrl()) {
            continue;
        }
        geometry_msgs::Point msg_p;
        msg_p.x = vpMPAct[i]->getPos().x / mScaleRatio;
        msg_p.y = vpMPAct[i]->getPos().y / mScaleRatio;
        msg_p.z = vpMPAct[i]->getPos().z / mScaleRatio;
        mMPsAct.points.push_back(msg_p);
    }


    mMPsNow.points.reserve(spMPNow.size());
    for (auto iter = spMPNow.begin(); iter != spMPNow.end(); iter++) {
        PtrMapPoint pMPtmp = *iter;
        if (pMPtmp->isNull() || !pMPtmp->isGoodPrl()) {
            continue;
        }
        geometry_msgs::Point msg_p;
        msg_p.x = pMPtmp->getPos().x / mScaleRatio;
        msg_p.y = pMPtmp->getPos().y / mScaleRatio;
        msg_p.z = pMPtmp->getPos().z / mScaleRatio;
        mMPsNow.points.push_back(msg_p);
    }

    mMPsNeg.header.stamp = ros::Time::now();
    mMPsAct.header.stamp = ros::Time::now();
    mMPsNow.header.stamp = ros::Time::now();

    publisher.publish(mMPsNeg);
    publisher.publish(mMPsAct);
    publisher.publish(mMPsNow);
}

void MapPublish::publishCameraCurr(const cv::Mat& Twc)
{
    mKFNow.points.clear();

    tf::Matrix3x3 R(Twc.at<float>(0, 0), Twc.at<float>(0, 1), Twc.at<float>(0, 2),
                    Twc.at<float>(1, 0), Twc.at<float>(1, 1), Twc.at<float>(1, 2),
                    Twc.at<float>(2, 0), Twc.at<float>(2, 1), Twc.at<float>(2, 2));
    tf::Vector3 t(Twc.at<float>(0, 3) / mScaleRatio, Twc.at<float>(1, 3) / mScaleRatio,
                  Twc.at<float>(2, 3) / mScaleRatio);
    tf::Transform tfwTc(R, t);
    tfb.sendTransform(tf::StampedTransform(tfwTc, ros::Time::now(), "se2lam/World", "se2lam/Camera"));

    float d = mCameraSize;

    // Camera is a pyramid. Define in camera coordinate system
    cv::Mat o = (cv::Mat_<float>(4, 1) << 0, 0, 0, 1);
    cv::Mat p1 = (cv::Mat_<float>(4, 1) << d, d * 0.8, d * 0.5, 1);
    cv::Mat p2 = (cv::Mat_<float>(4, 1) << d, -d * 0.8, d * 0.5, 1);
    cv::Mat p3 = (cv::Mat_<float>(4, 1) << -d, -d * 0.8, d * 0.5, 1);
    cv::Mat p4 = (cv::Mat_<float>(4, 1) << -d, d * 0.8, d * 0.5, 1);

    cv::Mat T = Twc.clone();
    T.at<float>(0, 3) = T.at<float>(0, 3) / mScaleRatio;
    T.at<float>(1, 3) = T.at<float>(1, 3) / mScaleRatio;
    T.at<float>(2, 3) = T.at<float>(2, 3) / mScaleRatio;
    cv::Mat ow = T * o;
    cv::Mat p1w = T * p1;
    cv::Mat p2w = T * p2;
    cv::Mat p3w = T * p3;
    cv::Mat p4w = T * p4;

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

//! 可视化原始odo的输入
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
        // auto currState = mpLocalizer->getTrackingState();
        // auto lastState = mpLocalizer->getLastTrackingState();

        // if (currState == cvu::OK) {
        //     PtrKeyFrame pKF = mpLocalizer->getKFCurr();

        //     static bool isFirstFrame = true;
        //     static Mat Twc_b = mpLocalizer->getCurrKFPose().toCvSE3();
        //     static Mat Tb0w = cvu::inv(pKF->odom.toCvSE3());

        //     Mat Twbi = (pKF->odom).toCvSE3();
        //     Mat Tb0bi = Tb0w * Twbi;              // 先变换到首帧的odom坐标系原点下
        //     Mat Twc_bi = (Twc_b * Tb0bi).col(3);  // 再变换到首帧的pose坐标系原点下
        //     msgsCurr.x = Twc_bi.at<float>(0, 0) / mScaleRatio;
        //     msgsCurr.y = Twc_bi.at<float>(1, 0) / mScaleRatio;
        //     if (currState == cvu::OK && lastState == cvu::LOST)
        //         isFirstFrame = true;
        //     if (isFirstFrame) {
        //         msgsLast = msgsCurr;
        //         isFirstFrame = false;
        //     }
        // } else {
        //     return;
        // }
    }

    mOdomRawGraph.points.push_back(msgsLast);
    mOdomRawGraph.points.push_back(msgsCurr);
    msgsLast = msgsCurr;

    mOdomRawGraph.header.stamp = ros::Time::now();
    publisher.publish(mOdomRawGraph);
}

void MapPublish::publishGroundTruth()
{
    static geometry_msgs::Point msgsLast;
    geometry_msgs::Point msgsCurr;

    static int pubId = 0;
    if (pubId == 0) {
        msgsLast.x = mvGroundTruth[0].x / mScaleRatio;
        msgsLast.y = mvGroundTruth[0].y / mScaleRatio;
    }
    if (!mbIsLocalize && pubId < mvGroundTruth.size()) {
        msgsCurr.x = mvGroundTruth[pubId].x / mScaleRatio;
        msgsCurr.y = mvGroundTruth[pubId].y / mScaleRatio;
        mGroundTruthGraph.points.push_back(msgsLast);
        mGroundTruthGraph.points.push_back(msgsCurr);
        msgsLast = msgsCurr;
        pubId += 2;  // GT数据量更多，防止显示不全
    }

    mGroundTruthGraph.header.stamp = ros::Time::now();
    publisher.publish(mGroundTruthGraph);
}

void MapPublish::requestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool MapPublish::checkFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void MapPublish::setFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool MapPublish::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

}  // namespace se2lam
