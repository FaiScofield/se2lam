/**
 * This file is part of se2lam
 *
 * Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
 */

#include "Sensors.h"
#include <opencv2/imgproc/imgproc.hpp>

namespace se2lam
{

Sensors::Sensors() : imgUpdated(false), odoUpdated(false) {}

Sensors::~Sensors() {}

bool Sensors::update()
{
    return odoUpdated && imgUpdated;
}

void Sensors::updateImg(const cv::Mat& img_, double time_)
{
    std::unique_lock<std::mutex> lock(mMutexImg);

    while (imgUpdated) {
        cndvSensorUpdate.wait(lock);
    }

    if (img_.channels() == 3)
        cv::cvtColor(img_, mImg, CV_BGR2GRAY);
    else
        img_.copyTo(mImg);

    timeImg = time_;
    imgUpdated = true;
}

void Sensors::updateOdo(float x_, float y_, float theta_, double time_)
{
    std::unique_lock<std::mutex> lock(mMutexOdo);

    while (odoUpdated) {
        cndvSensorUpdate.wait(lock);
    }
    mOdo.x = x_;
    mOdo.y = y_;
    mOdo.theta = theta_;
    mOdo.timeStamp = time_;
    timeOdo = time_;

    odoUpdated = true;
}

void Sensors::readData(Se2& dataOdo_, cv::Mat& dataImg_)
{
    std::unique_lock<std::mutex> lock1(mMutexImg);
    std::unique_lock<std::mutex> lock2(mMutexOdo);

    dataOdo_ = mOdo;
    mImg.copyTo(dataImg_);

    odoUpdated = false;
    imgUpdated = false;
    cndvSensorUpdate.notify_all();
}

void Sensors::forceSetUpdate(bool val)
{
    odoUpdated = val;
    imgUpdated = val;
}


}  // namespace se2lam
