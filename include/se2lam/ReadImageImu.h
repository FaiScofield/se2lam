//
// Created by lmp on 19-10-23.
//

#ifndef READIMAGEIMU_H
#define READIMAGEIMU_H

#include <fstream>
#include <iostream>
#include <iterator>
#include <opencv2/core/core.hpp>
#include <random>

namespace se2lam {

typedef struct {
    unsigned int time_msec;
    double acc[3], rate[3];
    double theta, x, y;
} IMU_DATA;

typedef struct {
    double theta, x, y, dtheta;
} Out_IMU_Data;

class ReadImageImu
{
public:
    bool compute_theta_IMU(int ptime, int ctime, const std::string& fname,
                           const std::string& shape_fname, Out_IMU_Data& outImuData);
    int playback_mydate(const std::string& fname, int t_offset, cv::Mat& gimgf);

    double normalizeAngle(double angle);

private:
    void smooth_image_float(const cv::Mat& img, cv::Mat& gimgf, int w = 1);
};

}  // namespace se2lam

#endif  // READIMAGEIMU_H
