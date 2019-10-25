//
// Created by lmp on 19-7-27.
//

#include <opencv/cv.hpp>

#pragma once

class Preprocess
{
public:
    Preprocess();
    ~Preprocess();
    cv::Mat SobelEdgeDetection(cv::Mat img);
    cv::Mat GaMma(cv::Mat grayImg, double gamma);
    cv::Mat edge_canny(cv::Mat src, int threshold1, int threshold2);
    cv::Mat sharpping(cv::Mat img, float scale);
};


