/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include "cvutil.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
namespace cvu
{

using namespace cv;
using namespace std;

//! 变换矩阵的逆
Mat inv(const Mat &T4x4)
{
    assert(T4x4.cols == 4 && T4x4.rows == 4);
    Mat RT = T4x4.rowRange(0, 3).colRange(0, 3).t();
    Mat t = -RT * T4x4.rowRange(0, 3).col(3);
    Mat T = Mat::eye(4, 4, CV_32FC1);
    RT.copyTo(T.rowRange(0, 3).colRange(0, 3));
    t.copyTo(T.rowRange(0, 3).col(3));
    return T;
}

void pts2Ftrs(const vector<KeyPoint> &_orgnFtrs, const vector<Point2f> &_points,
              vector<KeyPoint> &_features)
{
    _features.resize(_points.size());
    for (size_t i = 0; i < _points.size(); i++) {
        _features[i] = _orgnFtrs[i];
        _features[i].pt = _points[i];
    }
}

//! 求向量的反对称矩阵^
Mat sk_sym(const Point3f _v)
{
    Mat mat(3, 3, CV_32FC1, Scalar(0));
    mat.at<float>(0, 1) = -_v.z;
    mat.at<float>(0, 2) = _v.y;
    mat.at<float>(1, 0) = _v.z;
    mat.at<float>(1, 2) = -_v.x;
    mat.at<float>(2, 0) = -_v.y;
    mat.at<float>(2, 1) = _v.x;
    return mat;
}

/**
 * @brief triangulate 特征点三角化
 * @param pt1   参考帧KP1
 * @param pt2   当前帧KP2
 * @param P1    投影矩阵P1 = Kcam * I_3x4
 * @param P2    投影矩阵P2 = Kcam * mFrame.tcr
 * @return      三维点坐标
 *
 * @see         Multiple View Geometry in Computer Vision - 12.2 Linear triangulation methods p312
 */
Point3f triangulate(const Point2f &pt1, const Point2f &pt2, const Mat &P1, const Mat &P2)
{
    Mat A(4, 4, CV_32FC1);

    A.row(0) = pt1.x * P1.row(2) - P1.row(0);
    A.row(1) = pt1.y * P1.row(2) - P1.row(1);
    A.row(2) = pt2.x * P2.row(2) - P2.row(0);
    A.row(3) = pt2.y * P2.row(2) - P2.row(1);

    Mat u, w, vt, x3D;
    SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | SVD::FULL_UV);
    x3D = vt.row(3).t();
    x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);    // 第四行置1, 归一化

    return Point3f(x3D);
}

Point2f camprjc(const Mat &_K, const Point3f &_pt)
{
    Point3f uvw = Matx33f(_K) * _pt;
    return Point2f(uvw.x / uvw.z, uvw.y / uvw.z);
}


// 计算视差角余弦值
bool checkParallax(const Point3f &o1, const Point3f &o2, const Point3f &pt3, int minDegree)
{
    float minCos[4] = {0.9998, 0.9994, 0.9986, 0.9976};
    Point3f p1 = pt3 - o1;
    Point3f p2 = pt3 - o2;
    float cosParallax = cv::norm(p1.dot(p2)) / (cv::norm(p1) * cv::norm(p2));
    return cosParallax < minCos[minDegree - 1];
}

Point3f se3map(const Mat &_Tcw, const Point3f &_pt)
{
    Matx33f R(_Tcw.rowRange(0, 3).colRange(0, 3));
    Point3f t(_Tcw.rowRange(0, 3).col(3));
    return (R * _pt + t);
}

//void normalizeYawAngle(se2lam::Se2 &odom)
//{
//    if (odom.theta < -M_PI)
//        odom.theta += M_PI;
//    else if (odom.theta > M_PI)
//        odom.theta -= M_PI;
//}

// Gamma变换 gamma = 1.2 越小越亮
cv::Mat gamma(const cv::Mat &grayImg, float gamma)
{
    cv::Mat imgGamma, imgOut;
    grayImg.convertTo(imgGamma, CV_32F, 1.0 / 255, 0);
    pow(imgGamma, gamma, imgOut);
    imgOut.convertTo(imgOut, CV_8U, 255, 0);

    return imgOut;
}

// Laplace边缘锐化 scale = 6~10  越小越强
cv::Mat sharpping(const cv::Mat &img, float scale)
{
    cv::Mat imgOut;

    cv::Mat kern = (cv::Mat_<float>(5, 5) << -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 40, -1,
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

    kern = kern / scale;
    cv::filter2D(img, imgOut, img.depth(), kern);

    return imgOut;
}


}  // namespace scv
