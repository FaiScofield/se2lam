/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include "Sensors.h"

namespace se2lam
{

Sensors::Sensors() : imgUpdated(false), odoUpdated(false)
{
}

Sensors::~Sensors()
{
}

bool Sensors::update()
{
    return odoUpdated && imgUpdated;
}

// 当Sensors类里的当前帧数据没有被读取，则锁住当前线程
// 直到数据被Tracker/Localizer读走，则从system接收新的数据进来
void Sensors::updateImg(const cv::Mat& img_, double time_)
{
    std::unique_lock<std::mutex> lock(mMutexImg);

    while (imgUpdated) {
        cndvSensorUpdate.wait(lock);
    }

    img_.copyTo(mImg);
    timeImg = time_;
    imgUpdated = true;
}

void Sensors::updateOdo(double x_, double y_, double theta_, double time_)
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

void Sensors::readData(cv::Point3f& _dataOdo, cv::Mat& _dataImg,
                       float& _timeOdo, float& _timeimg)
{
    std::unique_lock<std::mutex> lock1(mMutexImg);
    std::unique_lock<std::mutex> lock2(mMutexOdo);

    _dataOdo = mOdo;
    mImg.copyTo(_dataImg);
    _timeOdo = timeOdo;
    _timeimg = timeImg;

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
