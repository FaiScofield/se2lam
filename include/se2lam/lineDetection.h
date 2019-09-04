//
// Created by lmp on 19-7-27.
//
#include "Thirdparty/line_descriptor/include/line_descriptor.hpp"
#include "Thirdparty/lsd_161/lsd.h"
#include <Eigen/Eigen>
#include <Eigen/StdList>
#include <Eigen/StdVector>
#include <iostream>
#include <opencv/cv.hpp>
#include <unsupported/Eigen/NonLinearOptimization>

namespace se2lam
{

using namespace std;
using namespace Eigen;

struct img_line {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    cv::Mat imgGray;
    Matrix<double, 4, Dynamic> linesTh1;
    Matrix<double, 4, Dynamic> linesTh2;
};

struct lineSort_S {
    cv::Point star;
    cv::Point end;
    double length;
    double k;
    double b;
};

cv::Mat getLineMask(const cv::Mat image, std::vector<lineSort_S> &linefeatures, bool extentionLine);

#define UNUSE_LINE_FILTER
#define USE_EDLINE

class lineDetection
{
public:
    lineDetection();

    ~lineDetection();

    cv::Range roiX, roiY;
    // std::vector<lineSort_S> lineFeature;

    cv::Mat getLineMaskByHf(cv::Mat img, bool extentionLine);

    void ComputeThreeMaxima(vector<vector<int>> rotHist, int lenth, int &ind1, int &ind2, int &ind3,
                            int &ind4);

    void lineStatistics(double theta, int label, vector<vector<int>> &rotHist, int lenth,
                        float factor);

    void getLineKandB(cv::Point starPoint, cv::Point endPoint, double &k, double &b);

    void LineDetect(const cv::Mat &image, double thLength1, double thLength2,
                    Matrix<double, 4, Dynamic> &linesTh1, Matrix<double, 4, Dynamic> &linesTh2,
                    std::vector<lineSort_S> &linefeatures);
    int pointInwhichLine(double x, double y, double &lenth, std::vector<lineSort_S> &linefeatures);

private:
    int shiftX, shiftY;
    int endX, endY;
    cv::Ptr<cv::line_descriptor::BinaryDescriptor> edlinesDetect;
    std::vector<cv::line_descriptor::KeyLine> edlines;
};
}
