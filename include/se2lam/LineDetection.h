/*
 * @Author: Vance.Wu
 * @Date: 2019-11-04 11:03:48
 * @LastEditors: Vance.Wu
 * @LastEditTime: 2019-11-04 11:07:19
 * @Description:
 */
//
// Created by lmp on 19-7-27.
//

#ifndef LINEDETECTION_H
#define LINEDETECTION_H

#include "dependencies/line_descriptor/include/line_descriptor.hpp"
#include "dependencies/lsd_161/include/lsd.h"

#include <Eigen/Eigen>
#include <Eigen/StdList>
#include <Eigen/StdVector>
#include <unsupported/Eigen/NonLinearOptimization>

#include <iostream>
#include <opencv2/core/core.hpp>


namespace se2lam
{

struct img_line {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    cv::Mat imgGray;
    Eigen::Matrix<double, 4, Eigen::Dynamic> linesTh1;
    Eigen::Matrix<double, 4, Eigen::Dynamic> linesTh2;
};

struct lineSort_S {
    cv::Point star;
    cv::Point end;
    double length;
    double k;
    double b;
};

class lineSort_compareClass : public std::binary_function<lineSort_S, lineSort_S, bool>
{
public:
    inline bool operator()(const lineSort_S& t1, const lineSort_S& t2)
    {
        return t1.length > t2.length;
    }
};

cv::Mat getLineMask(const cv::Mat& image, std::vector<lineSort_S>& lines, bool extend);
cv::Mat getLineMask(const cv::Mat& image, bool extend);


#define UNUSE_LINE_FILTER
#define USE_EDLINE

class LineDetection
{
public:
    LineDetection();

    ~LineDetection();


    // std::vector<lineSort_S> lineFeature;

    cv::Mat getLineMaskByHf(const cv::Mat& img, bool extentionLine);

    void ComputeThreeMaxima(const std::vector<std::vector<int>>& rotHist, int lenth, int& ind1,
                            int& ind2, int& ind3, int& ind4);

    void lineStatistics(double theta, int label, std::vector<std::vector<int>>& rotHist, int lenth,
                        float factor);

    void getLineKandB(const cv::Point& starPoint, const cv::Point& endPoint, double& k, double& b);

    void LineDetect(const cv::Mat& image, double thLength1, double thLength2,
                    Eigen::Matrix<double, 4, Eigen::Dynamic>& linesTh1,
                    Eigen::Matrix<double, 4, Eigen::Dynamic>& linesTh2,
                    std::vector<lineSort_S>& linefeatures);

    int pointInwhichLine(double x, double y, double& lenth, std::vector<lineSort_S>& linefeatures);

    cv::Range roiX, roiY;

private:
    int shiftX, shiftY;
    int endX, endY;

    cv::Ptr<cv::line_descriptor::BinaryDescriptor> edlinesDetect;
    std::vector<cv::line_descriptor::KeyLine> edlines;
};


}  // namespace se2lam

#endif  // LINEDETECTION_H
