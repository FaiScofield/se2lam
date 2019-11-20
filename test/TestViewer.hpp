#ifndef TESTVIEWER_HPP
#define TESTVIEWER_HPP

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include "TestTrack.hpp"


class TestViewer
{
public:
    TestViewer() {}
    TestViewer(Map* pMap);
    ~TestViewer() {};

    void run();

    void setMap(Map* pMap) { mpMap = pMap; }
    void setTracker(TestTrack* pTrack) { mpTracker = pTrack; }
    void setLocalMapper(LocalMapper* pLocal) { mpLocalMapper = pLocal; }

    void publishCameraCurr(const cv::Mat& Twc);
    void publishKeyFrames();
    void publishMapPoints();
    void publishOdomInformation();

    void requestFinish() { mbFinishRequested = true; }
public:
    Map* mpMap;
    LocalMapper* mpLocalMapper;
    TestTrack* mpTracker;

private:
    bool checkFinish() { return mbFinishRequested; }
    void setFinish() { mbFinished = true; }

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

    float mPointSize;
    float mCameraSize;
    float mScaleRatio;

    bool mbFinishRequested;
    bool mbFinished;
};

TestViewer::TestViewer(Map* pMap)
    : mpMap(pMap), mPointSize(0.1f), mCameraSize(0.3f), mScaleRatio(Config::MappubScaleRatio),
      mbFinishRequested(false), mbFinished(false)
{
    const char* MAP_FRAME_ID = "/se2lam/World";

    // Global (Negtive) KFs, black
    mKFsNeg.header.frame_id = MAP_FRAME_ID;
    mKFsNeg.ns = "KeyFramesNegative";
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

    // Local (Active) KFs, blue
    mKFsAct.header.frame_id = MAP_FRAME_ID;
    mKFsAct.ns = "KeyFramesActive";
    mKFsAct.id = 1;
    mKFsAct.type = visualization_msgs::Marker::LINE_LIST;
    mKFsAct.scale.x = 0.05;
    mKFsAct.scale.y = 0.05;
    mKFsAct.pose.orientation.w = 1.0;
    mKFsAct.action = visualization_msgs::Marker::ADD;
    mKFsAct.color.b = 1.0;
    mKFsAct.color.a = 1.0;

    // Configure Current Camera, red
    mKFNow.header.frame_id = MAP_FRAME_ID;
    mKFNow.ns = "Camera";
    mKFNow.id = 2;
    mKFNow.type = visualization_msgs::Marker::LINE_LIST;
    mKFNow.scale.x = 0.05;
    mKFNow.scale.y = 0.05;
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
    mMPsNow.scale.x = mPointSize * 2.f;
    mMPsNow.scale.y = mPointSize * 2.f;
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
    mFeatGraph.ns = "FeatGraph";
    mFeatGraph.id = 7;
    mFeatGraph.type = visualization_msgs::Marker::LINE_LIST;
    mFeatGraph.scale.x = 0.05;
    mFeatGraph.scale.y = 0.05;
    mFeatGraph.pose.orientation.w = 1.0;
    mFeatGraph.action = visualization_msgs::Marker::ADD;
    mFeatGraph.color.r = 0.0;
    mFeatGraph.color.g = 0.0;
    mFeatGraph.color.b = 1.0;
    mFeatGraph.color.a = 1.0;

    // Configure Odometry Constraint Graph, black
    mVIGraph.header.frame_id = MAP_FRAME_ID;
    mVIGraph.ns = "VIGraph";
    mVIGraph.id = 8;
    mVIGraph.type = visualization_msgs::Marker::LINE_LIST;
    mVIGraph.scale.x = 0.1;
    mVIGraph.scale.y = 0.1;
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
    mOdomRawGraph.scale.x = 0.07;
    mOdomRawGraph.scale.y = 0.07;
    mOdomRawGraph.pose.orientation.w = 1.0;
    mOdomRawGraph.color.r = 0.95;
    mOdomRawGraph.color.g = 0.0;
    mOdomRawGraph.color.b = 0.0;
    mOdomRawGraph.color.a = 0.7;

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

void TestViewer::run()
{
    image_transport::ImageTransport it(nh);
    image_transport::Publisher pubImgMatches = it.advertise("/camera/imageMatches", 1);

    ros::Rate rate(Config::FPS);
    while (nh.ok()) {
        if (checkFinish())
            break;
        if (mpMap->empty())
            continue;

        Mat imgMatch = mpTracker->drawMatchesForPub();
        if (imgMatch.empty())
            continue;

        sensor_msgs::ImagePtr msgMatch =
            cv_bridge::CvImage(std_msgs::Header(), "bgr8", imgMatch).toImageMsg();
        pubImgMatches.publish(msgMatch);

        cv::Mat Tcw = mpMap->getCurrentFramePose();
        publishCameraCurr(cvu::inv(Tcw));
        publishKeyFrames();
        publishMapPoints();
        publishOdomInformation();

        rate.sleep();
        ros::spinOnce();
    }
    cout << "[Viewe] Exiting TestViewer .." << endl;

    nh.shutdown();
    setFinish();
}

void TestViewer::publishCameraCurr(const cv::Mat& Twc)
{
    mKFNow.points.clear();

    tf::Matrix3x3 R(Twc.at<float>(0, 0), Twc.at<float>(0, 1), Twc.at<float>(0, 2),
                    Twc.at<float>(1, 0), Twc.at<float>(1, 1), Twc.at<float>(1, 2),
                    Twc.at<float>(2, 0), Twc.at<float>(2, 1), Twc.at<float>(2, 2));
    tf::Vector3 t(Twc.at<float>(0, 3) / mScaleRatio, Twc.at<float>(1, 3) / mScaleRatio,
                  Twc.at<float>(2, 3) / mScaleRatio);
    tf::Transform tfwTc(R, t);
    tfb.sendTransform(
        tf::StampedTransform(tfwTc, ros::Time::now(), "se2lam/World", "se2lam/Camera"));

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

void TestViewer::publishKeyFrames()
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

    vector<PtrKeyFrame> vKFsAll = mpMap->getAllKFs();
    if (vKFsAll.empty())
        return;

    vector<PtrKeyFrame> vKFsAct;
    vKFsAct = mpMap->getLocalKFs();

    for (int i = 0, iend = vKFsAll.size(); i != iend; ++i) {
        PtrKeyFrame pKFi = vKFsAll[i];
        if (pKFi->isNull())
            continue;

        // 按比例缩放地图, mScaleRatio = 300 时显示比例为 1:3.333，数据源单位是[mm]
        cv::Mat Tcw = pKFi->getPose();
        cv::Mat Twc = cvu::inv(Tcw);
        cv::Mat Twb = Twc * Config::Tcb;

        Twc.at<float>(0, 3) = Twc.at<float>(0, 3) / mScaleRatio;
        Twc.at<float>(1, 3) = Twc.at<float>(1, 3) / mScaleRatio;
        Twc.at<float>(2, 3) = Twc.at<float>(2, 3) / mScaleRatio;

        cv::Mat ow = Twc * o;  // 第i帧KF的相机中心位姿
        cv::Mat p1w = Twc * p1;
        cv::Mat p2w = Twc * p2;
        cv::Mat p3w = Twc * p3;
        cv::Mat p4w = Twc * p4;

        cv::Mat ob = Twb * o;  // 第i帧KF的Body中心位姿
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

        // 可视化 Negtive/Active 相机位姿
        int count = std::count(vKFsAct.begin(), vKFsAct.end(), pKFi);
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
        std::set<PtrKeyFrame> covKFs = pKFi->getAllCovisibleKFs();
        if (!covKFs.empty()) {
            for (auto it = covKFs.begin(), iend = covKFs.end(); it != iend; ++it) {
                if ((*it)->mIdKF > pKFi->mIdKF)
                    continue;
                Mat Twb = cvu::inv((*it)->getPose()) /** Config::Tcb*/;
                geometry_msgs::Point msgs_o2;
                msgs_o2.x = Twb.at<float>(0, 3) / mScaleRatio;
                msgs_o2.y = Twb.at<float>(1, 3) / mScaleRatio;
                msgs_o2.z = Twb.at<float>(2, 3) / mScaleRatio;
                mCovisGraph.points.push_back(msgs_o);
                mCovisGraph.points.push_back(msgs_o2);
            }
        }

        // Feature Graph
        for (auto iter = pKFi->mFtrMeasureFrom.begin(); iter != pKFi->mFtrMeasureFrom.end(); iter++) {
            PtrKeyFrame pKF2 = iter->first;
            Mat Twb = cvu::inv(pKF2->getPose()) /* * Config::Tcb*/;
            geometry_msgs::Point msgs_o2;
            msgs_o2.x = Twb.at<float>(0, 3) / mScaleRatio;
            msgs_o2.y = Twb.at<float>(1, 3) / mScaleRatio;
            msgs_o2.z = Twb.at<float>(2, 3) / mScaleRatio;
            mFeatGraph.points.push_back(msgs_o);
            mFeatGraph.points.push_back(msgs_o2);
        }

        // Visual Odometry Graph (estimate)
        PtrKeyFrame pKFOdoChild = pKFi->mOdoMeasureFrom.first;
        if (pKFOdoChild != nullptr) {
            Mat Twb1 = cvu::inv(pKFOdoChild->getPose()) * Config::Tcb;
            geometry_msgs::Point msgs_b2;
            msgs_b2.x = Twb1.at<float>(0, 3) / mScaleRatio;
            msgs_b2.y = Twb1.at<float>(1, 3) / mScaleRatio;
            msgs_b2.z = Twb1.at<float>(2, 3) / mScaleRatio;
            mVIGraph.points.push_back(msgs_b2);  // 上一帧的位姿
            mVIGraph.points.push_back(msgs_b);   // 当前帧的位姿
        }
    }

    mKFsNeg.header.stamp = ros::Time::now();
    mKFsAct.header.stamp = ros::Time::now();
    mCovisGraph.header.stamp = ros::Time::now();
    mFeatGraph.header.stamp = ros::Time::now();
    mVIGraph.header.stamp = ros::Time::now();

    publisher.publish(mKFsNeg);
    publisher.publish(mKFsAct);
    publisher.publish(mCovisGraph);
    publisher.publish(mFeatGraph);
    publisher.publish(mVIGraph);
}

void TestViewer::publishMapPoints()
{
    mMPsNeg.points.clear();
    mMPsAct.points.clear();
    mMPsNow.points.clear();
    mMPsNoGoodPrl.points.clear();

    PtrKeyFrame pKFCur = mpMap->getCurrentKF();
    vector<PtrMapPoint> vpMPAll = mpMap->getAllMPs();
    vector<PtrMapPoint> vpMPAct = mpMap->getLocalMPs();
    set<PtrMapPoint> spMPNow= mpMap->getCurrentKF()->getAllObsMPs();
    vector<PtrMapPoint> vpMPNeg;
    fprintf(stdout, "[Viewe] #%ld(KF#%ld) MPsAll = %ld, MPsLocal = %ld, MPsObs = %ld\n",
            pKFCur->id, pKFCur->mIdKF, vpMPAll.size(), vpMPAct.size(), spMPNow.size());

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
        int count = spMPNow.count(pMPtemp);
        if (count == 0)
            vpMPActGood.push_back(pMPtemp);
    }
    vpMPAct.swap(vpMPActGood); // MPsAct 去掉 MPsNow

    fprintf(stdout, "[Viewe] #%ld(KF#%ld) MPsAll = %ld, MPsNeg = %ld, MPsAct = %ld, MPsNow = %ld\n",
            pKFCur->id, pKFCur->mIdKF, vpMPAll.size(), vpMPNeg.size(), vpMPAct.size(), spMPNow.size());

    mMPsNeg.points.reserve(vpMPNeg.size());
    for (int i = 0, iend = vpMPNeg.size(); i != iend; ++i) {
        if (vpMPNeg[i]->isNull()) {
            fprintf(stdout, "[Viewe] #%ld(KF#%ld) Negtive MPs里面有坏点(id%ld)没有析构!\n",
                    pKFCur->id, pKFCur->mIdKF, vpMPNeg[i]->mId);
            continue;
        }

        geometry_msgs::Point msg_p;
        msg_p.x = vpMPNeg[i]->getPos().x / mScaleRatio;
        msg_p.y = vpMPNeg[i]->getPos().y / mScaleRatio;
        msg_p.z = vpMPNeg[i]->getPos().z / mScaleRatio;

        if (!vpMPNeg[i]->isGoodPrl())
            mMPsNoGoodPrl.points.push_back(msg_p);
        else
            mMPsNeg.points.push_back(msg_p);
    }

    mMPsAct.points.reserve(vpMPAct.size());
    for (int i = 0, iend = vpMPAct.size(); i != iend; ++i) {
        if (vpMPAct[i]->isNull()) {
            fprintf(stderr, "[Viewe] #%ld(KF#%ld) Active MPs里面有坏点(id%ld)没有析构!\n",
                    pKFCur->id, pKFCur->mIdKF, vpMPAct[i]->mId);
            continue;
        }

        geometry_msgs::Point msg_p;
        msg_p.x = vpMPAct[i]->getPos().x / mScaleRatio;
        msg_p.y = vpMPAct[i]->getPos().y / mScaleRatio;
        msg_p.z = vpMPAct[i]->getPos().z / mScaleRatio;

//        if (!vpMPAct[i]->isGoodPrl())
//            mMPsNoGoodPrl.points.push_back(msg_p);
//        else
            mMPsAct.points.push_back(msg_p);
    }

    mMPsNow.points.reserve(spMPNow.size());
    for (auto iter = spMPNow.begin(); iter != spMPNow.end(); iter++) {
        PtrMapPoint pMPtmp = *iter;
        if (pMPtmp->isNull()) {
            fprintf(stderr, "[Viewe] #%ld(KF#%ld) Now MPs里面有坏点(id%ld)没有析构!\n",
                    pKFCur->id, pKFCur->mIdKF, pMPtmp->mId);
            continue;
        }

        geometry_msgs::Point msg_p;
        msg_p.x = pMPtmp->getPos().x / mScaleRatio;
        msg_p.y = pMPtmp->getPos().y / mScaleRatio;
        msg_p.z = pMPtmp->getPos().z / mScaleRatio;

//        if (!pMPtmp->isGoodPrl())
//            mMPsNoGoodPrl.points.push_back(msg_p);
//        else
            mMPsNow.points.push_back(msg_p);
    }

    mMPsNeg.header.stamp = ros::Time::now();
    mMPsAct.header.stamp = ros::Time::now();
    mMPsNow.header.stamp = ros::Time::now();
    mMPsNoGoodPrl.header.stamp = ros::Time::now();

//    fprintf("[Viewe] 当前观测MP/局部MP/总MP数量为: %ld/%ld/%ld\n", mMPsNow.points.size(),
//            mMPsAct.points.size(), mMPsNeg.points.size());
    publisher.publish(mMPsNeg);
    publisher.publish(mMPsAct);
    publisher.publish(mMPsNow);
    publisher.publish(mMPsNoGoodPrl);
}

void TestViewer::publishOdomInformation()
{
    static geometry_msgs::Point msgsLast;
    geometry_msgs::Point msgsCurr;

    //! 这里要对齐到首帧的Odom, 位姿从原点开始
    static Se2 firstOdom = mpMap->getCurrentKF()->odom;
    static Mat Tb0w = cvu::inv(firstOdom.toCvSE3());

    Se2 currOdom = mpMap->getCurrentKF()->odom;
    Mat Twbi = currOdom.toCvSE3();
    Mat Tb0bi = (Tb0w * Twbi).col(3);
    msgsCurr.x = Tb0bi.at<float>(0, 0) / mScaleRatio;
    msgsCurr.y = Tb0bi.at<float>(1, 0) / mScaleRatio;

    mOdomRawGraph.points.push_back(msgsLast);
    mOdomRawGraph.points.push_back(msgsCurr);
    msgsLast = msgsCurr;

    mOdomRawGraph.header.stamp = ros::Time::now();
    publisher.publish(mOdomRawGraph);
}

#endif // TESTVIEWER_HPP

