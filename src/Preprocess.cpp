//
// Created by lmp on 19-7-27.
//

#include "Preprocess.h"


Preprocess::Preprocess()
{}

Preprocess::~Preprocess()
{}

// Sobel算子边缘检测
cv::Mat Preprocess::SobelEdgeDetection(cv::Mat img)
{
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y, dst;
    Sobel(img, grad_x, CV_16S, 1, 0, 3, 1, 1);
    convertScaleAbs(grad_x, abs_grad_x);
    // imshow("X方向Sobel", abs_grad_x);

    //
    Sobel(img, grad_y, CV_16S, 0, 1, 3, 1, 1);
    convertScaleAbs(grad_y, abs_grad_y);
    // imshow("Y方向Sobel", abs_grad_y);

    //整合到一起
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);
    return dst;
}

// Gamma变换
cv::Mat Preprocess::GaMma(cv::Mat grayImg, double gamma)
{
    cv::Mat imgGamma, dstImg;
    grayImg.convertTo(imgGamma, CV_32F, 1.0 / 255, 0);
    // gamma = 1.2;
    pow(imgGamma, gamma, dstImg);
    dstImg.convertTo(dstImg, CV_8U, 255, 0);
    return dstImg;
}

// Canny边缘检测
cv::Mat Preprocess::edge_canny(cv::Mat src, int thresholdL, int thresholdT)
{
    cv::Mat edge;
    // blur(src, edge, Size(17, 17));
    Canny(src, edge, thresholdL, thresholdT, 3);
    return edge;
}

// Laplace边缘锐化
cv::Mat Preprocess::sharpping(cv::Mat img, float scale)
{
    cv::Mat imgOut;

    cv::Mat kern = (cv::Mat_<float>(5, 5) << -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 40, -1,
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

    kern = kern / scale;
    // cout << kern << endl;
    cv::filter2D(img, imgOut, img.depth(), kern);
    return imgOut;
}
