/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include "Sensors.h"
#include <iostream>

namespace se2lam
{

Sensors::Sensors() : imgUpdated(false), odoUpdated(false)
{}

Sensors::~Sensors()
{}

bool Sensors::update()
{
    return odoUpdated && imgUpdated;
}

// 当Sensors类里的当前帧数据没有被读取，则锁住当前线程
// 直到数据被Tracker/Localizer读走，则从system接收新的数据进来
void Sensors::updateImg(const cv::Mat& img_, float time_)
{
    std::unique_lock<std::mutex> lock(mMutexImg);

    while (imgUpdated) {
        cndvSensorUpdate.wait(lock);
    }

    img_.copyTo(mImg);
    timeImg = time_;
    imgUpdated = true;
}

void Sensors::updateOdo(float x_, float y_, float theta_, float time_)
{
    std::unique_lock<std::mutex> lock(mMutexOdo);

    while (odoUpdated) {
        cndvSensorUpdate.wait(lock);
    }
    mOdo.x = x_;
    mOdo.y = y_;
    mOdo.z = theta_;
    timeOdo = time_;
    odoUpdated = true;
}

void Sensors::updateOdo(std::queue<Se2> &odoQue_)
{
    std::unique_lock<std::mutex> lock(mMutexOdo);

    while (odoUpdated) {
        cndvSensorUpdate.wait(lock);
    }

    mvOdoSeq.clear();
    while (!odoQue_.empty()) {
        mvOdoSeq.emplace_back(odoQue_.front());
        odoQue_.pop();
    }
    if (mvOdoSeq.empty())
        std::cerr << "[Sensor] No odom data between two image!" << std::endl;

    odoUpdated = true;
}

void Sensors::readData(cv::Point3f& dataOdo_, cv::Mat& dataImg_)
{
    std::unique_lock<std::mutex> lock1(mMutexImg);
    std::unique_lock<std::mutex> lock2(mMutexOdo);

    dataOdo_ = mOdo;
    mImg.copyTo(dataImg_);

    odoUpdated = false;
    imgUpdated = false;

    cndvSensorUpdate.notify_all();
}

void Sensors::readData(std::vector<Se2>& dataOdoSeq_, cv::Mat& dataImg_, float& timeImg_)
{
    std::unique_lock<std::mutex> lock1(mMutexImg);
    std::unique_lock<std::mutex> lock2(mMutexOdo);

    mImg.copyTo(dataImg_);
    timeImg_ = timeImg;
    dataOdoSeq_ = mvOdoSeq;

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
