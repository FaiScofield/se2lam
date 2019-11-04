/**
 * This file is part of se2lam
 *
 * Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
 */


#ifndef SUGAR_CV_H
#define SUGAR_CV_H

#include "Config.h"
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

namespace cvu
{


using cv::Mat;
using cv::Point2f;
using cv::Point3f;
using std::vector;

enum eTrackingState { NO_READY_YET = -1, FIRST_FRAME = 0, OK = 1, LOST = 2 };

Mat inv(const Mat& T4x4);

Mat sk_sym(const Point3f& _v);

Point3f triangulate(const Point2f& pt1, const Point2f& pt2, const Mat& P1, const Mat& P2);

Point2f camprjc(const Mat& _K, const Point3f& _pt);

bool checkParallax(const Point3f& o1, const Point3f& o2, const Point3f& pt3, int minDegree = 1);

Point3f se3map(const Mat& _Tcw, const Point3f& _pt);

void pts2Ftrs(const vector<cv::KeyPoint>& _orgnFtrs, const vector<Point2f>& _points,
              vector<cv::KeyPoint>& _features);

cv::Mat gmmaTransform(const cv::Mat& grayImg, double gamma = 1.2);

cv::Mat edgeSharpping(const cv::Mat& img, float scale = 12.f);

cv::Mat edgeCannyDetection(const cv::Mat& src, int thresholdL, int thresholdT);

}  // namespace cvu

#endif
