/**
 * This file is part of se2lam
 *
 * Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
 */

#ifndef SENSORS_H
#define SENSORS_H

#include "Config.h"
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

    void updateOdo(float x_, float y_, float theta_, double time_ = 0);
    void updateImg(const cv::Mat& img_, double time_ = 0);

    // After readData(), img_updatd and odo_updated would be set false
    void readData(Se2& dataOdo_, cv::Mat& dataImg);

    void forceSetUpdate(bool val);

protected:
    cv::Mat mImg;
    Se2 mOdo;

    double timeImg;
    double timeOdo;

    std::atomic_bool imgUpdated;
    std::atomic_bool odoUpdated;

    std::condition_variable cndvSensorUpdate;

    std::mutex mMutexOdo;
    std::mutex mMutexImg;
};


}  // namespace se2lam


#endif
