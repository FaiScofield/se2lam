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
#include <queue>
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
    void updateImu(double theta_, double time_ = 0);
//    void updateOdoSequence(std::vector<Se2>& odoDeque_);
    void updateImg(const cv::Mat& img_, double time_ = 0);

    // After readData(), img_updatd and odo_updated would be set false
    void readData(Se2& dataOdo_, cv::Mat& dataImg_);
    void readData(Se2& dataOdo_, cv::Mat& dataImg_, double& theta);
//    void readDataSequence(std::vector<Se2>& dataOdoSeq_, cv::Mat& dataImg_, double& timeImg_);
//    void readDataWithTime(Se2& odo, cv::Mat& img, double& time);

    void forceSetUpdate(bool val);

    Se2 getOdo() { return mOdo; }
    //    Se2 dataAlignment(const std::vector<Se2>& dataOdoSeq_, const double& timeImg_);

protected:
    cv::Mat mImg;
    Se2 mOdo;

    //    std::vector<Se2> mvOdoSeq;
    double timeImg;
    double timeOdo;
    double timeImu;
    double thetaImu;

    std::atomic_bool imuUpdated;
    std::atomic_bool imgUpdated;
    std::atomic_bool odoUpdated;

    std::condition_variable cndvSensorUpdate;

    std::mutex mMutexOdo;
    std::mutex mMutexImg;
    std::mutex mutex_Imu;
};

}  // namespace se2lam

#endif
