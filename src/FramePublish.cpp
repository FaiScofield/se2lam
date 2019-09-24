/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include "FramePublish.h"
#include "GlobalMapper.h"
#include "Localizer.h"
#include "Track.h"
#include "cvutil.h"
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>

namespace se2lam
{

using namespace cv;
using std::vector;

typedef unique_lock<mutex> locker;

FramePublish::FramePublish()
{
}

FramePublish::FramePublish(Track* pTR, GlobalMapper* pGM)
{
    mpTrack = pTR;
    mpGM = pGM;
    mbIsLocalize = false;
}

FramePublish::~FramePublish()
{
}

//! NOTE 这个线程没有一直跑，相关函数只有在MapPublish里被调用
void FramePublish::run()
{
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Publisher pub = it.advertise("/camera/framepub", 1);

    float fps = Config::FPS;
    ros::Rate rate(fps);

    while (nh.ok() && ros::ok()) {
        cerr << "[FramePb] mbIsLocalize = " << mbIsLocalize << endl;
        if (!mbIsLocalize) {
            if (mpTrack->copyForPub(kpRef, kp, mImgRef, mImg, matches)) {
                WorkTimer timer;
                timer.start();

                Mat imgCurr = drawMatchesInOneImg(kpRef, mImg, kp, matches);
                Mat imgRef = drawKeys(kpRef, mImgRef, matches);
                Mat imgMatch;
                mpGM->mImgMatch.copyTo(imgMatch);
                Size sizeImgCurr = imgCurr.size();
                Size sizeImgMatch = imgMatch.size();

                Mat imgOut(sizeImgCurr.height * 2, sizeImgCurr.width * 2, imgCurr.type(),
                           Scalar(0));

                imgCurr.copyTo(imgOut(cv::Rect(0, 0, sizeImgCurr.width, sizeImgCurr.height)));
                imgRef.copyTo(
                    imgOut(cv::Rect(0, sizeImgCurr.height, sizeImgCurr.width, sizeImgCurr.height)));
                if (sizeImgMatch.width != 0) {
                    imgMatch.copyTo(imgOut(
                        cv::Rect(sizeImgCurr.width, 0, sizeImgMatch.width, sizeImgMatch.height)));
                }

                timer.stop();
                cv::resize(imgOut, imgOut, cv::Size(640, 480));
                sensor_msgs::ImagePtr msg =
                    cv_bridge::CvImage(std_msgs::Header(), "bgr8", imgOut).toImageMsg();
                pub.publish(msg);
            }
        } else {
            locker lockImg(mpLocalizer->mMutexImg);

            if (mpLocalizer == nullptr)
                continue;
            if (mpLocalizer->mpKFCurr == nullptr)
                continue;
            if (mpLocalizer->mImgCurr.cols == 0)
                continue;

            Mat imgCurr;
            mpLocalizer->mImgCurr.copyTo(imgCurr);
            Size sizeImgCurr = imgCurr.size();

            Mat imgOut(sizeImgCurr.height * 2, sizeImgCurr.width * 2, imgCurr.type(), Scalar(0));
            imgCurr.copyTo(imgOut(cv::Rect(0, 0, sizeImgCurr.width, sizeImgCurr.height)));

            if (mpLocalizer->mImgLoop.cols != 0) {
                Mat imgLoop;
                mpLocalizer->mImgLoop.copyTo(imgLoop);
                imgLoop.copyTo(
                    imgOut(cv::Rect(0, sizeImgCurr.height, sizeImgCurr.width, sizeImgCurr.height)));
            }

            Mat imgMatch;
            mpLocalizer->mImgMatch.copyTo(imgMatch);
            Size sizeImgMatch = imgMatch.size();
            if (sizeImgMatch.width != 0) {
                imgMatch.copyTo(imgOut(
                    cv::Rect(sizeImgCurr.width, 0, sizeImgMatch.width, sizeImgMatch.height)));
            }

            sensor_msgs::ImagePtr msg =
                cv_bridge::CvImage(std_msgs::Header(), "bgr8", imgOut).toImageMsg();
            pub.publish(msg);
        }

        rate.sleep();
    }
}

// 可视化图左上角, 画当前帧的特征点和匹配关系
cv::Mat FramePublish::drawMatchesInOneImg(const vector<KeyPoint> queryKeys, const Mat& trainImg,
                                          const vector<KeyPoint> trainKeys,
                                          const vector<int>& matches)
{
    Mat out = trainImg.clone();
    if (trainImg.channels() == 1)
        cvtColor(trainImg, out, CV_GRAY2BGR);

    int nGoodMatchs = 0;
    for (unsigned i = 0; i < matches.size(); ++i) {
        if (matches[i] < 0) {
            continue;
        } else {
            ++nGoodMatchs;
            Point2f ptRef = queryKeys[i].pt;
            Point2f ptCurr = trainKeys[matches[i]].pt;
            circle(out, ptCurr, 5, Scalar(0, 255, 0), 1);
            circle(out, ptRef, 5, Scalar(0, 0, 255), 1);
            line(out, ptRef, ptCurr, Scalar(0, 255, 0));
        }
    }
    putText(out, to_string(nGoodMatchs), Point(15, 15), 1, 1, Scalar(0, 0, 255), 2);
    return out.clone();
}

// 可视化图左下角, 画参考KF的特征点
cv::Mat FramePublish::drawKeys(const vector<KeyPoint> keys, const Mat& img, vector<int> matched)
{
    Mat out = img.clone();
    if (img.channels() == 1)
        cvtColor(img, out, CV_GRAY2BGR);
    for (unsigned i = 0; i < matched.size(); ++i) {
        Point2f pt1 = keys[i].pt;
        if (matched[i] < 0) {
            circle(out, pt1, 5, Scalar(255, 0, 0), 1);
        } else {
            circle(out, pt1, 5, Scalar(0, 0, 255), 1);
        }
    }
    return out.clone();
}

cv::Mat FramePublish::drawFrame()
{
    if (!mbIsLocalize) {
        if (mpTrack->copyForPub(kpRef, kp, mImgRef, mImg, matches)) {
            Mat imgCurr = drawMatchesInOneImg(kpRef, mImg, kp, matches);
            Mat imgRef = drawKeys(kpRef, mImgRef, matches);
            Mat imgMatch;
            mpGM->mImgMatch.copyTo(imgMatch);
            Size sizeImgCurr = imgCurr.size();
            Size sizeImgMatch = imgMatch.size();

            Mat imgOut(sizeImgCurr.height * 2, sizeImgCurr.width * 2, imgCurr.type(), Scalar(0));

            imgCurr.copyTo(imgOut(cv::Rect(0, 0, sizeImgCurr.width, sizeImgCurr.height)));
            imgRef.copyTo(
                imgOut(cv::Rect(0, sizeImgCurr.height, sizeImgCurr.width, sizeImgCurr.height)));
            if (sizeImgMatch.width != 0) {
                imgMatch.copyTo(imgOut(
                    cv::Rect(sizeImgCurr.width, 0, sizeImgMatch.width, sizeImgMatch.height)));
            }

            imgOut.copyTo(mImgOut);
        }
    } else if (mpLocalizer != nullptr && mpLocalizer->mpKFCurr != nullptr &&
               mpLocalizer->mImgCurr.cols != 0) {

        locker lockImg(mpLocalizer->mMutexImg);

        Mat imgCurr;
        mpLocalizer->mImgCurr.copyTo(imgCurr);
        Size sizeImgCurr = imgCurr.size();

        Mat imgOut(sizeImgCurr.height * 2, sizeImgCurr.width * 2, imgCurr.type(), Scalar(0));
        imgCurr.copyTo(imgOut(cv::Rect(0, 0, sizeImgCurr.width, sizeImgCurr.height)));

        if (mpLocalizer->mImgLoop.cols != 0) {
            Mat imgLoop;
            mpLocalizer->mImgLoop.copyTo(imgLoop);
            imgLoop.copyTo(
                imgOut(cv::Rect(0, sizeImgCurr.height, sizeImgCurr.width, sizeImgCurr.height)));
        }

        Mat imgMatch;
        mpLocalizer->mImgMatch.copyTo(imgMatch);
        Size sizeImgMatch = imgMatch.size();
        if (sizeImgMatch.width != 0) {
            imgMatch.copyTo(
                imgOut(cv::Rect(sizeImgCurr.width, 0, sizeImgMatch.width, sizeImgMatch.height)));
        }

        imgOut.copyTo(mImgOut);
    }

    return mImgOut.clone();
}

cv::Mat FramePublish::drawMatch()
{
    if (mbIsLocalize)
        return Mat();

    mpTrack->copyForPub(kpRef, kp, mImgRef, mImg, matches);

    cv::Mat imgCurr, imgRef;
    mImg.copyTo(imgCurr);
    mImgRef.copyTo(imgRef);
    if (imgCurr.channels() == 1)
        cvtColor(imgCurr, imgCurr, CV_GRAY2BGR);
    if (imgRef.channels() == 1)
        cvtColor(imgRef, imgRef, CV_GRAY2BGR);

    cv::Mat res(imgCurr.rows, imgCurr.cols + imgRef.cols, CV_8UC3);
    imgCurr.copyTo(res.colRange(0, imgCurr.cols));
    imgRef.copyTo(res.colRange(imgCurr.cols, imgCurr.cols + imgRef.cols));
    int nMatches = 0;
    for (size_t i = 0; i < matches.size(); ++i) {
        if (matches[i] < 0) {
            continue;
        } else {
            nMatches++;
            Point2f ptRef = kpRef[i].pt + Point2f(imgCurr.cols, 0);
            Point2f ptCurr = kp[matches[i]].pt;
            circle(res, ptCurr, 2, Scalar(0, 255, 0), 1);
            circle(res, ptRef, 2, Scalar(0, 255, 0), 1);
            line(res, ptRef, ptCurr, Scalar(255, 255, 0, 0.8));
        }
    }

    putText(res, to_string(nMatches), Point(15, 15), 1, 1.1, Scalar(0, 0, 255), 3);
    return res.clone();
}

void FramePublish::setLocalizer(Localizer* localizer)
{
    mpLocalizer = localizer;
}


}  // namespace se2lam
