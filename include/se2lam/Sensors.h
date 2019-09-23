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
#include <queue>

namespace se2lam
{

class Sensors
{

public:
    Sensors();

    ~Sensors();

    bool update();

    void setUpdated(bool val);

    void updateOdo(float x_, float y_, float theta_, float time_ = 0);
    void updateOdo(std::queue<Se2>& odoDeque_);

    void updateImg(const cv::Mat& img_, float time_ = 0);

    // After readData(), img_updatd and odo_updated would be set false
    void readData(cv::Point3f& dataOdo_, cv::Mat& dataImg_);
    void readData(std::vector<Se2>& dataOdoSeq_, cv::Mat& dataImg_, float& timeImg_);

    void forceSetUpdate(bool val);

    cv::Point3f getOdo() { return mOdo; }

protected:
    cv::Mat mImg;
    cv::Point3f mOdo;
    std::vector<Se2> mvOdoSeq;
    float timeImg;
    float timeOdo;

    std::atomic_bool imgUpdated;
    std::atomic_bool odoUpdated;
    std::condition_variable cndvSensorUpdate;

    std::mutex mMutexOdo;
    std::mutex mMutexImg;
};

}


#endif
