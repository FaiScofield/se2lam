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

void Sensors::updateImu(double theta_, double time_, bool useCeil_)
{
    std::unique_lock<std::mutex> lock(mutex_Imu);

    while(imuUpdated)
    {
        cndvSensorUpdate.wait(lock);
    }
    use_ceil = useCeil_;
    theta_Imu = theta_;
    time_Imu = time_;
    imuUpdated = true;
}
/*
void Sensors::updateOdoSequence(std::vector<Se2>& odoQue_)
{
    std::unique_lock<std::mutex> lock(mMutexOdo);

    while (odoUpdated) {
        cndvSensorUpdate.wait(lock);
    }

    mvOdoSeq = odoQue_;
    if (mvOdoSeq.empty())
        std::cerr << "[Sensor] No odom data between two image!" << std::endl;

    odoUpdated = true;
}
*/

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

void Sensors::readData(Se2& dataOdo_, cv::Mat& dataImg_, double &Imu_theta, bool &useCeil)
{
    std::unique_lock<std::mutex> lock1(mMutexImg);
    std::unique_lock<std::mutex> lock2(mMutexOdo);
    std::unique_lock<std::mutex> lock3(mutex_Imu);

    useCeil = use_ceil;
    Imu_theta = theta_Imu;
    dataOdo_ = mOdo;
    mImg.copyTo(dataImg_);

    odoUpdated = false;
    imgUpdated = false;
    imuUpdated = false;
    cndvSensorUpdate.notify_all();
}

/*
void Sensors::readDataSequence(std::vector<Se2>& dataOdoSeq_, cv::Mat& dataImg_, double& timeImg_)
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

void Sensors::readDataWithTime(Se2& odo, cv::Mat& img, double& time)
{
    std::unique_lock<std::mutex> lock1(mMutexImg);
    std::unique_lock<std::mutex> lock2(mMutexOdo);

    odo = dataAlignment(mvOdoSeq, timeImg);
    mOdo = odo;
    mImg.copyTo(img);
    time = timeImg;

    odoUpdated = false;
    imgUpdated = false;
    cndvSensorUpdate.notify_all();
}
*/

void Sensors::forceSetUpdate(bool val)
{
    odoUpdated = val;
    imgUpdated = val;
}

/*
Se2 Sensors::dataAlignment(const std::vector<Se2>& dataOdoSeq_, const double& timeImg_)
{
    Se2 res;
    size_t n = dataOdoSeq_.size();
    if (n < 2) {
        std::cerr << "[Sensor][Warni] Less odom sequence input!" << std::endl;
        return res;
    }

    //! 计算单帧图像时间内的平均速度
    Se2 tranSum = dataOdoSeq_[n - 1] - dataOdoSeq_[0];
    float dt = dataOdoSeq_[n - 1].timeStamp - dataOdoSeq_[0].timeStamp;
    float r = (timeImg_ - dataOdoSeq_[n - 1].timeStamp) / dt;

    assert(r >= 0.f);

    Se2 transDelta(tranSum.x * r, tranSum.y * r, tranSum.theta * r);
    res = dataOdoSeq_[n - 1] + transDelta;
    res.timeStamp = timeImg_;

    return res;
}
*/

}  // namespace se2lam
