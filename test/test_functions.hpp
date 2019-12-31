#ifndef TEST_FUNCTIONS_HPP
#define TEST_FUNCTIONS_HPP

#include "Config.h"
#include "Frame.h"
#include "GlobalMapper.h"
#include "KeyFrame.h"
#include "Map.h"
#include "ORBVocabulary.h"
#include "ORBextractor.h"
#include "ORBmatcher.h"
#include "Sensors.h"
#include "converter.h"
#include "cvutil.h"

#include <ros/ros.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/video.hpp>
//#include <boost/algorithm/string.hpp>
//#include <boost/algorithm/string_regex.hpp>
#include <boost/filesystem.hpp>
#include <iostream>

using namespace se2lam;
using namespace std;
using namespace cv;
namespace bf = boost::filesystem;

struct RK_IMAGE {
    RK_IMAGE(const string& s, const double& t) : fileName(s), timeStamp(t) {}

    string fileName;
    double timeStamp;
};

bool lessThen(const RK_IMAGE& r1, const RK_IMAGE& r2)
{
    return r1.timeStamp < r2.timeStamp;
}

void readImagesSe2(const string& dataFolder, vector<RK_IMAGE>& files)
{
    vector<RK_IMAGE> allImages;
    allImages.reserve(3108);
    for (int i = 0; i < 3108; ++i) {
        string ni = dataFolder + to_string(i) + ".bmp";
        allImages.emplace_back(ni, 0);
    }

    if (allImages.empty()) {
        cerr << "[ERROR] Not image data in the folder! " << dataFolder << endl;
        return;
    } else {
        cout << "[INFO ] Read " << allImages.size() << " files in the folder." << endl;
    }

    files = allImages;
}

void readImagesRK(const string& dataFolder, vector<RK_IMAGE>& files)
{
    bf::path path(dataFolder);
    if (!bf::exists(path)) {
        cerr << "[ERROR] Folder doesn't exist! " << dataFolder << endl;
        return;
    }

    vector<RK_IMAGE> allImages;
    bf::directory_iterator end_iter;
    for (bf::directory_iterator iter(path); iter != end_iter; ++iter) {
        if (bf::is_directory(iter->status()))
            continue;
        if (bf::is_regular_file(iter->status())) {
            // format: /frameRaw12987978101.jpg
            string s = iter->path().string();
            auto i = s.find_last_of('w');
            auto j = s.find_last_of('.');
            double t = atoll(s.substr(i + 1, j - i - 1).c_str()) * 1e-6;
            allImages.emplace_back(s, t);
        }
    }

    if (allImages.empty()) {
        cerr << "[ERROR] Not image data in the folder! " << dataFolder << endl;
        return;
    } else {
        cout << "[INFO ] Read " << allImages.size() << " files in the folder." << endl;
    }

    //! 应该根据后面的时间戳数值来排序
    sort(allImages.begin(), allImages.end(), lessThen);
    files = allImages;
}

void readOdomDatas(const string& odomFile, vector<Se2>& odoData)
{
    ifstream rec(odomFile);
    if (!rec.is_open()) {
        cerr << "[ERROR] Opening file error in " << odomFile << endl;
        rec.close();
        return;
    }

    vector<Se2> vOdoData;
    vOdoData.reserve(Config::ImgCount);

    string line;
    while (std::getline(rec, line) && !line.empty()) {
        istringstream iss(line);
        Se2 odo;
        iss >> odo.x >> odo.y >> odo.theta;
        vOdoData.emplace_back(odo);
    }
    rec.close();

    if (vOdoData.empty()) {
        cerr << "[ERROR] Not odom data in the file!" << endl;
        return;
    } else {
        cout << "[INFO ] Read " << vOdoData.size() << " odom datas from the file." << endl;
    }

    odoData = vOdoData;
}

Mat drawKPMatches(const PtrKeyFrame KFRef, const PtrKeyFrame KFCur, const map<int, int>& matches)
{
    Mat outImg;
    vconcat(KFCur->mImage, KFRef->mImage, outImg);

    for (auto& m : matches) {
        Point2f p1 = KFCur->mvKeyPoints[m.second].pt;
        Point2f p2 = KFRef->mvKeyPoints[m.first].pt + Point2f(0, KFCur->mImage.rows);
        circle(outImg, p1, 3, Scalar(0, 255, 0));
        circle(outImg, p2, 3, Scalar(0, 255, 0));
        line(outImg, p1, p2, Scalar(0, 0, 255));
    }
    return outImg;
}

Mat drawKPMatches(const Frame* frameRef, const Frame* frameCur, const vector<int>& matchIdx12)
{
    Mat outImg;
    vconcat(frameRef->mImage, frameCur->mImage, outImg);

    for (size_t i = 0, iend = frameRef->N; i < iend; ++i) {
        if (matchIdx12[i] < 0) {
            continue;
        } else {
            Point2f p1 = frameCur->mvKeyPoints[matchIdx12[i]].pt;
            Point2f p2 = frameRef->mvKeyPoints[i].pt + Point2f(0, frameCur->mImage.rows);

            circle(outImg, p1, 3, Scalar(0, 255, 0));
            circle(outImg, p2, 3, Scalar(0, 255, 0));
            line(outImg, p1, p2, Scalar(0, 0, 255));
        }
    }
    return outImg;
}

Mat drawKPMatchesH(const Frame* frameRef, const Frame* frameCur, const vector<int>& matchIdx12,
                   const Mat& H21, double& projError)
{
    if (!H21.data)  // H_3x3
        cerr << "Empty H matrix!!" << endl;
    Mat H12 = H21.inv(DECOMP_SVD);

    Mat imgCur, imgRef, imgWarp, outImg;
    cvtColor(frameRef->mImage, imgRef, CV_GRAY2BGR);
    cvtColor(frameCur->mImage, imgCur, CV_GRAY2BGR);

    warpPerspective(imgCur, imgWarp, H12, imgCur.size());
    hconcat(imgWarp, imgRef, outImg);

    projError = 0;
    int n = 0;
    for (size_t i = 0, iend = frameRef->N; i < iend; ++i) {
        if (matchIdx12[i] < 0) {
            continue;
        } else {
            const Point2f& ptRef = frameRef->mvKeyPoints[i].pt;
            const Point2f& ptCur = frameCur->mvKeyPoints[matchIdx12[i]].pt;
            const Mat pt2 = (Mat_<double>(3, 1) << ptCur.x, ptCur.y, 1);
            Mat pt2W = H12 * pt2;
            pt2W /= pt2W.at<double>(2);

            const Point2f ptL = Point2f(pt2W.at<double>(0), pt2W.at<double>(1));
            const Point2f ptR = ptRef + Point2f(imgCur.cols, 0);

            n++;
            projError += sqrt(norm(ptRef - ptL));
            circle(outImg, ptL, 3, Scalar(0, 255, 0));
            circle(outImg, ptR, 3, Scalar(0, 255, 0));
            if (i % 3 == 0)
                line(outImg, ptL, ptR, Scalar(0, 0, 255));
        }
    }
    projError /= n * 1.f;
    return outImg.clone();
}

Mat drawKPMatchesA(const Frame* frameRef, const Frame* frameCur, const vector<int>& matchIdx12,
                   const Mat& A21, double& projError)
{
    if (!A21.data)
        cerr << "Empty A matrix!!" << endl;

    Mat A12;
    invertAffineTransform(A21, A12);  // A21和A12都是2x3

    Mat imgCur, imgRef, imgWarp, outImg;
    cvtColor(frameRef->mImage, imgRef, CV_GRAY2BGR);
    cvtColor(frameCur->mImage, imgCur, CV_GRAY2BGR);

    warpAffine(imgCur, imgWarp, A12, frameCur->mImage.size());
    hconcat(imgWarp, imgRef, outImg);

    projError = 0;
    int n = 0;
    for (size_t i = 0, iend = frameRef->N; i < iend; ++i) {
        if (matchIdx12[i] < 0) {
            continue;
        } else {
            const Point2f& ptRef = frameRef->mvKeyPoints[i].pt;
            const Point2f& ptCur = frameCur->mvKeyPoints[matchIdx12[i]].pt;
            const Mat pt2 = (Mat_<double>(3, 1) << ptCur.x, ptCur.y, 1);
            const Mat pt2W = A12 * pt2;

            const Point2f ptL = Point2f(pt2W.at<double>(0), pt2W.at<double>(1));
            const Point2f ptR = ptRef + Point2f(imgCur.cols, 0);

            n++;
            projError += sqrt(norm(ptRef - ptL));
            circle(outImg, ptL, 3, Scalar(0, 255, 0));
            circle(outImg, ptR, 3, Scalar(0, 255, 0));
            if (i % 5 == 0)
                line(outImg, ptL, ptR, Scalar(0, 0, 255));
        }
    }
    projError /= n * 1.f;
    return outImg;
}

Mat drawKPMatchesAGood(const Frame* frameRef, const Frame* frameCur, const vector<int>& matchIdx12,
                       const vector<int>& matchIdx12Good, const Mat& A21, double& projError)
{
    if (!A21.data)
        cerr << "Empty A matrix!!" << endl;

    Mat A12;
    invertAffineTransform(A21, A12);  // A21和A12都是2x3

    Mat imgCur, imgRef, imgWarp, outImg;
    cvtColor(frameRef->mImage, imgRef, CV_GRAY2BGR);
    cvtColor(frameCur->mImage, imgCur, CV_GRAY2BGR);

    warpAffine(imgCur, imgWarp, A12, frameCur->mImage.size());
    hconcat(imgWarp, imgRef, outImg);

    projError = 0;
    int n = 0;
    for (size_t i = 0, iend = frameRef->N; i < iend; ++i) {
        if (matchIdx12[i] < 0) {
            continue;
        } else {
            const Point2f& ptRef = frameRef->mvKeyPoints[i].pt;
            const Point2f& ptCur = frameCur->mvKeyPoints[matchIdx12[i]].pt;
            const Mat pt2 = (Mat_<double>(3, 1) << ptCur.x, ptCur.y, 1);
            const Mat pt2W = A12 * pt2;

            const Point2f ptL = Point2f(pt2W.at<double>(0), pt2W.at<double>(1));
            const Point2f ptR = ptRef + Point2f(imgCur.cols, 0);

            if (matchIdx12Good[i] < 0) {
                circle(outImg, ptL, 3, Scalar(0, 255, 0));
                circle(outImg, ptR, 3, Scalar(0, 255, 0));
                if (i % 5 == 0)
                    line(outImg, ptL, ptR, Scalar(0, 255, 0));
            } else {
                n++;
                projError += sqrt(norm(ptRef - ptL));
                circle(outImg, ptL, 3, Scalar(0, 0, 255));
                circle(outImg, ptR, 3, Scalar(0, 0, 255));
                if (i % 5 == 0)
                    line(outImg, ptL, ptR, Scalar(0, 0, 255));
            }
        }
    }
    projError /= n * 1.f;
    return outImg;
}

// void calculateAffineMatrixSVD(const vector<Point2f>& ptRef, const vector<Point2f>& ptCur,
//                              const vector<uchar>& inlineMask, Mat& Affine)
//{
//    assert(ptRef.size() == ptCur.size());
//    assert(ptRef.size() == inlineMask.size());

//    Affine = Mat::eye(2, 3, CV_64FC1);

//    // 1.求质心
//    Point2f p1_c(0.f, 0.f), p2_c(0.f, 0.f);
//    int N = 0;
//    for (size_t i = 0, iend = inlineMask.size(); i < iend; ++i) {
//        if (inlineMask[i] < 0)
//            continue;
//        p1_c += ptRef[i];
//        p2_c += ptCur[i];
//        N++;
//    }
//    if (N == 0)
//        return;

//    p1_c.x /= N;
//    p1_c.y /= N;
//    p2_c.x /= N;
//    p2_c.y /= N;

//    // 2.构造超定矩阵A
//    Mat A = Mat::zeros(2, 2, CV_32FC1);
//    for (size_t i = 0; i < N; ++i) {
//        Mat p1_i = Mat(ptRef[i] - p1_c);
//        Mat p2_i = Mat(ptCur[i] - p2_c);
//        A += p1_i * p2_i.t();
//    }

//    // 3.用SVD分解求得R,t
//    Mat U, W, Vt;
//    SVD::compute(A, W, U, Vt);
//    Mat R = U * Vt;
//    Mat t = Mat(p1_c) - R * Mat(p2_c);

//    // 求得是A12, 需要A21, 故求逆
//    R.copyTo(Affine.colRange(0, 2));
//    t.copyTo(Affine.col(2));
//    //invertAffineTransform(Affine, Affine);
//}


int removeOutliersWithH(const vector<KeyPoint>& kpRef, const vector<KeyPoint>& kpCur,
                        vector<int>& matches12, Mat& H21)
{
    assert(kpRef.size() == kpCur.size());
    assert(!matches12.empty());

    vector<Point2f> ptRef, ptCur;
    vector<size_t> idx;
    ptRef.reserve(kpRef.size());
    ptCur.reserve(kpCur.size());
    idx.reserve(kpRef.size());

    for (size_t i = 0, iend = kpRef.size(); i < iend; ++i) {
        if (matches12[i] < 0)
            continue;
        idx.push_back(i);
        ptRef.push_back(kpRef[i].pt);
        ptCur.push_back(kpCur[matches12[i]].pt);
    }
    size_t nPoints = ptRef.size();

    // 默认RANSAC法计算H矩阵. 天花板处于平面,最好用H矩阵, H_33 = 1
    vector<uchar> inlineMask;
    if (nPoints >= 4)
        H21 = findHomography(ptRef, ptCur, RANSAC, 3.0, inlineMask);
    else
        cerr << "Too less points (%ld) for calculate H!" << endl;
    assert(nPoints == inlineMask.size());

    int nInlier = 0;
    for (size_t i = 0; i < nPoints; ++i) {
        if (!inlineMask[i])
            matches12[idx[i]] = -1;
        else
            nInlier++;
    }

    return nInlier;
}

int removeOutliersWithA(const vector<KeyPoint>& kpRef, const vector<KeyPoint>& kpCur,
                        vector<int>& matches12, Mat& A21, const int flag = 0)
{
    assert(kpRef.size() == kpCur.size());
    assert(!matches12.empty());

    vector<Point2f> ptRef, ptCur;
    vector<size_t> idxRef;
    idxRef.reserve(kpRef.size());
    ptRef.reserve(kpRef.size());
    ptCur.reserve(kpCur.size());
    for (size_t i = 0, iend = kpRef.size(); i < iend; ++i) {
        if (matches12[i] < 0)
            continue;
        idxRef.push_back(i);
        ptRef.push_back(kpRef[i].pt);
        ptCur.push_back(kpCur[matches12[i]].pt);
    }

    vector<uchar> inlineMask;
    switch (flag) {
    case 0:
        // 旋转 + 平移 + 尺度 + 错切
        A21 = estimateAffine2D(ptRef, ptCur, inlineMask, RANSAC, 3.0);
        break;
    case 1: {
        // 旋转 + 平移 + 尺度
        A21 = estimateAffinePartial2D(ptRef, ptCur, inlineMask, RANSAC, 3.0);
        //        if (abs(A12.at<double>(0, 1)) <= 1.0) {  // 去掉尺度变换 TODO 需要decompose
        //            double c_theta = cos(asin(A12.at<double>(0, 1)));
        //            A12.at<double>(0, 0) = A12.at<double>(1, 1) = c_theta;
        //        } else {
        //            A12.at<double>(0, 0) = A12.at<double>(1, 1) = 0.;
        //            A12.at<double>(0, 1) = A12.at<double>(1, 0) = 1.;
        //        }
        break;
    }
    case 2: {
        //! NOTE 内点数<50%后无法计算
        A21 = estimateRigidTransform(ptRef, ptCur, false);
        inlineMask.resize(ptRef.size(), 1);
        if (A21.empty()) {
            cerr << "Warning! 内点数太少无法计算RigidTransform!" << endl;
            A21 = Mat::eye(2, 3, CV_64FC1);
        }
        break;
    }
    default:
        A21 = estimateAffinePartial2D(ptRef, ptCur, inlineMask, RANSAC, 3.0);
        break;
    }

    int nInlier = 0;
    for (size_t i = 0, iend = inlineMask.size(); i < iend; ++i) {
        if (!inlineMask[i])
            matches12[idxRef[i]] = -1;
        else
            nInlier++;
    }

    return nInlier;
}

// int removeOutliersWithRansac(const vector<KeyPoint>& kpRef, const vector<KeyPoint>& kpCur,
//                             vector<int>& matches12, Mat& Asvd, int inlineTh = 3)
//{
//    assert(kpRef.size() == kpCur.size());
//    assert(kpRef.size() >= 10);
//    assert(!matches12.empty());

//    vector<Point2f> ptRef, ptCur;
//    vector<int> idxRef;
//    idxRef.reserve(kpRef.size());
//    ptRef.reserve(kpRef.size());
//    ptCur.reserve(kpCur.size());
//    for (int i = 0, iend = kpRef.size(); i < iend; ++i) {
//        if (matches12[i] < 0)
//            continue;
//        idxRef.push_back(i);
//        ptRef.push_back(kpRef[i].pt);
//        ptCur.push_back(kpCur[matches12[i]].pt);
//    }

//    // Ransac
//    int inliers = 0, lastInliers = 0;
//    double error = 0, lastError = 99999;
//    Mat Affine, lastAffine;
//    size_t N = ptRef.size(), i = 0;
//    vector<uchar> inlineMask(ptRef.size(), 1);
//    for (; i < 10; ++i) {
//        inliers = 0;
//        error = 0;
//        calculateAffineMatrixSVD(ptRef, ptCur, inlineMask, Affine);
//        for (size_t j = 0; j < N; ++j) {
//            const Mat pt2 = (Mat_<double>(3, 1) << ptCur[j].x, ptCur[j].y, 1);
//            const Mat pt2W = Affine * pt2;
//            const Point2f ptCurWarpj = Point2f(pt2W.at<double>(0), pt2W.at<double>(1));

//            double ej = norm(ptRef[j] - ptCurWarpj);
//            if (ej < inlineTh) {
//                error += ej;
//                inliers++;
//            } else {
//                inlineMask[j] = 0;
//                matches12[idxRef[j]] = -1;
//            }
//        }
//        error /= inliers * 1.0;

//        if (error > lastError) {  // 如果误差变大就用上一次的结果
//            Affine = lastAffine;
//            error = lastError;
//            inliers = lastInliers;
//            break;
//        }

//        if (lastError - error < 1e-3)  // 如果误差下降不明显则退出
//            break;

//        lastError = error;
//        lastAffine = Affine;
//        lastInliers = inliers;
//    }
//    cout << "迭代次数 = " << i << ", 内点数 = " << inliers << "/" << N << ", 平均误差 = " << error
//    << endl;

//    Asvd = Affine.clone();

//    return inliers;
//}

cv::Mat predictAffineMatrix(const Se2& dOdom)
{
    Point2f rotationCenter;
    rotationCenter.x = 160.5827 - 0.01525;  // cx - Tbc.y / 12
    rotationCenter.y = 117.7329 - 3.6984;  // cy - Tbc.x / 12
    // rotationCenter.x = Config::cx - Config::Tbc.at<float>(0, 3) / 12;
    // rotationCenter.y =  Config::cy - Config::Tbc.at<float>(1, 3) / 12;

    Mat R = getRotationMatrix2D(rotationCenter, dOdom.theta * 180.f / CV_PI, 1.);
    // cout << "Affine Matrix of R3 = " << endl << R << endl;

    return R.clone();
}

// dOdom = Ref.odo - Cur.odo
cv::Mat calculateAffineMatrix(const Se2& dOdom)
{
    Mat Tc1c2 = Config::Tcb * dOdom.inv().toCvSE3() * Config::Tbc;
    Mat Rc1c2 = Tc1c2.rowRange(0, 3).colRange(0, 3).clone();
    Mat R1 = Config::Kcam * Rc1c2 * (Config::Kcam).inv();
    cout << "Affine Matrix of R1 = " << endl << R1.rowRange(0, 2) << endl;

    Point2f rotationCenter;
    rotationCenter.x = Config::cx + Config::Tbc.at<float>(1, 3) / 12;
    rotationCenter.y = Config::cy + Config::Tbc.at<float>(0, 3) / 12;
    Mat R2 = getRotationMatrix2D(rotationCenter, dOdom.theta * 180.f / CV_PI, 1.);
    cout << "Affine Matrix of R2 = " << endl << R2 << endl;

    return R1.rowRange(0, 2).clone();
}


#endif  // TEST_FUNCTIONS_HPP
