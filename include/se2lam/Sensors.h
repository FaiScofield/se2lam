/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#ifndef SENSORS_H
#define SENSORS_H

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <opencv2/core/core.hpp>
#include <thread>

namespace se2lam
{

class Sensors
{

public:
    Sensors();

    ~Sensors();

    bool update();

    void setUpdated(bool val);

    void updateOdo(double x_, double y_, double theta_, double time_ = 0);

    void updateImg(const cv::Mat& img_, double time_ = 0);

    // After readData(), img_updatd and odo_updated would be set false
    void readData(cv::Point3f& dataOdo, cv::Mat& dataImg, float& timeOdo, float& timeimg);

    void forceSetUpdate(bool val);

    cv::Point3f getOdo() { return mOdo; }

protected:
    cv::Mat mImg;
    cv::Point3f mOdo;
    float timeOdo;
    float timeImg;

    std::atomic_bool odoUpdated;
    std::atomic_bool imgUpdated;
    std::condition_variable cndvSensorUpdate;

    std::mutex mMutexOdo;
    std::mutex mMutexImg;
};
}


#endif
