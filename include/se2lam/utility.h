/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/



#ifndef SUGAR_CV_H
#define SUGAR_CV_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "Config.h"

namespace cvu {


using std::vector;
using cv::Mat;
using cv::Point2f;
using cv::Point3f;

enum eTrackingState {
    NO_READY_YET = -1,
    FIRST_FRAME = 0,
    OK = 1,
    LOST = 2
};

Mat inv(const Mat& T4x4);

Mat sk_sym(const Point3f& _v);

Point3f triangulate(const Point2f &pt1, const Point2f &pt2, const Mat &P1, const Mat &P2);

Point2f camprjc(const Mat& _K, const Point3f& _pt);

bool checkParallax(const Point3f& o1, const Point3f& o2, const Point3f& pt3, int minDegree = 1);

Point3f se3map(const Mat& _Tcw, const Point3f& _pt);

void pts2Ftrs(const vector<cv::KeyPoint>& _orgnFtrs, const vector<Point2f>& _points, vector<cv::KeyPoint>& _features);

double deg2rad(double deg) { return M_PI * deg / 180; }
double rad2deg(double rad) { return 180 * rad / M_PI; }

cv::Mat gamma(const cv::Mat& grayImg, float gamma = 1.2);

cv::Mat sharpping(const cv::Mat& img, float scale = 12);

} // namespace scv

#endif