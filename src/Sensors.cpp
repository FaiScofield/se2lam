/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include "Sensors.h"

namespace se2lam {

Sensors::Sensors() {
    img_updated = false;
    odo_updated = false;
}

Sensors::~Sensors() {

}

bool Sensors::update() {
    return odo_updated && img_updated;
}

// 当Sensors类里的当前帧数据没有被读取，则锁住当前线程
// 直到数据被Localizer读走，则从system接收新的数据进来
void Sensors::updateImg(const cv::Mat &img_, double time_)
{
    std::unique_lock<std::mutex> lock(mutex_img);

    while (img_updated)
    {
        cndvSensorUpdate.wait(lock);
    }

    img_.copyTo(mImg);
    time_img = time_;
    img_updated = true;
}

void Sensors::updateOdo(double x_, double y_, double theta_, double time_)
{
    std::unique_lock<std::mutex> lock(mutex_odo);

    while(odo_updated)
    {
        cndvSensorUpdate.wait(lock);
    }
    mOdo.x = x_;
    mOdo.y = y_;
    mOdo.z = theta_;
    time_odo = time_;
    odo_updated = true;
}

void Sensors::readData(cv::Point3f& dataOdo, cv::Mat& dataImg){
    std::unique_lock<std::mutex> lock1(mutex_img);
    std::unique_lock<std::mutex> lock2(mutex_odo);

    dataOdo = mOdo;
    mImg.copyTo(dataImg);

    odo_updated = false;
    img_updated = false;

    cndvSensorUpdate.notify_all();
}

void Sensors::forceSetUpdate(bool val)
{
    odo_updated = val;
    img_updated = val;
}


}
