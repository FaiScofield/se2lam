//
// Created by lmp on 19-10-23.
//

#ifndef SE2LAM_READIMAGEIMU_H
#define SE2LAM_READIMAGEIMU_H

#include <iostream>
#include <fstream>
#include <iterator>
#include <random>
#include <opencv/cv.hpp>
#include "opencv2/imgproc/imgproc_c.h"
using namespace std;

typedef struct {
    unsigned int time_msec;
    double acc[3], rate[3];
    double theta, x, y;
} IMU_DATA;

typedef struct {
    double theta, x, y, dtheta;
} Out_IMU_Data;

class readImageImu{
public:
    bool compute_theta_IMU(int ptime, int ctime, const string fname, const string shape_fname,
                           Out_IMU_Data &outImuData);
    int playback_mydate(const string fname, int t_offset, cv::Mat &gimgf);

    double normalizeAngle(double angle);
private:
    void smooth_image_float(cv::Mat img, cv::Mat &gimgf, int w);

};


#endif //SE2LAM_READIMAGEIMU_H
