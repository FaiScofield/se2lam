#ifndef TEST_FUNCTIONS_HPP
#define TEST_FUNCTIONS_HPP

#include "Config.h"
#include "Frame.h"
#include "KeyFrame.h"
#include "Map.h"
#include "ORBVocabulary.h"
#include "ORBextractor.h"
#include "ORBmatcher.h"
#include "Sensors.h"
#include "GlobalMapper.h"
#include "converter.h"
#include "cvutil.h"

#include <ros/ros.h>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
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

void readImagesSe2(const string& dataFolder, vector<string>& files)
{
    vector<string> allImages;
    allImages.reserve(3108);
    for (int i = 0; i < 3108; ++i) {
        string ni = dataFolder + to_string(i) + ".bmp";
        allImages.push_back(ni);
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
            allImages.push_back(RK_IMAGE(s, t));
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

void readOdomsRK(const string& odomFile, vector<Se2>& odoData)
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

Mat drawKPMatches(const Frame* frameRef, const Frame* frameCur, const Mat& imgRef,
                  const Mat& imgCur, const vector<int>& matchIdx12)
{
    Mat outImg;
    vconcat(imgCur, imgRef, outImg);

    for (size_t i = 0, iend = frameRef->N; i < iend; ++i) {
        if (matchIdx12[i] < 0) {
            continue;
        } else {
            Point2f p1 = frameCur->mvKeyPoints[matchIdx12[i]].pt;
            Point2f p2 = frameRef->mvKeyPoints[i].pt + Point2f(0, imgCur.rows);

            circle(outImg, p1, 3, Scalar(0, 255, 0));
            circle(outImg, p2, 3, Scalar(0, 255, 0));
            line(outImg, p1, p2, Scalar(0, 0, 255));
        }
    }
    return outImg;
}

Mat drawKPMatchesH(const Frame* frameRef, const Frame* frameCur, const Mat& imgRef, const Mat& imgCur,
                  const vector<int>& matchIdx12, const Mat& H12, double& projError)
{
    if (!H12.data)  // H_3x3
        cerr << "Empty H matrix!!" << endl;

    Mat H21 = H12.inv(DECOMP_SVD);

    Mat imgWarp, outImg;
    warpPerspective(imgCur, imgWarp, H21, imgCur.size());
    hconcat(imgWarp, imgRef, outImg);

    projError = 0;
    int n = 0;
    for (size_t i = 0, iend = frameRef->N; i < iend; ++i) {
        if (matchIdx12[i] < 0) {
            continue;
        } else {
            const Point2f& ptRef = frameRef->mvKeyPoints[i].pt;
            const Point2f& ptCur = frameCur->mvKeyPoints[matchIdx12[i]].pt;
            const Mat pt1 = (Mat_<double>(3, 1) << ptCur.x, ptCur.y, 1);
            Mat pt1W = H21 * pt1;
            pt1W /= pt1W.at<double>(2);

            const Point2f ptL = Point2f(pt1W.at<double>(0), pt1W.at<double>(1));
            const Point2f ptR = ptRef + Point2f(imgCur.cols, 0);

            n++;
            projError += norm(ptRef - ptL);
            circle(outImg, ptL, 3, Scalar(0, 255, 0));
            circle(outImg, ptR, 3, Scalar(0, 255, 0));
            if (i % 3 == 0)
                line(outImg, ptL, ptR, Scalar(0, 0, 255));
        }
    }
    projError /= n * 1.f;
    return outImg.clone();
}

Mat drawKPMatchesA(const Frame* frameRef, const Frame* frameCur, const Mat& imgRef, const Mat& imgCur,
                  const vector<int>& matchIdx12, const Mat& A12, double& projError)
{
    if (!A12.data)  // A12_2x3
        cerr << "Empty A matrix!!" << endl;

    Mat A21;
    invertAffineTransform(A12, A21);  // A21_2x3

    Mat imgWarp, outImg;
    warpAffine(imgCur, imgWarp, A21, imgCur.size());
    hconcat(imgWarp, imgRef, outImg);

    projError = 0;
    int n = 0;
    for (size_t i = 0, iend = frameRef->N; i < iend; ++i) {
        if (matchIdx12[i] < 0) {
            continue;
        } else {
            const Point2f& ptRef = frameRef->mvKeyPoints[i].pt;
            const Point2f& ptCur = frameCur->mvKeyPoints[matchIdx12[i]].pt;
            const Mat pt1 = (Mat_<double>(3, 1) << ptCur.x, ptCur.y, 1);
            const Mat pt1W = A21 * pt1;

            const Point2f ptL = Point2f(pt1W.at<double>(0), pt1W.at<double>(1));
            const Point2f ptR = ptRef + Point2f(imgCur.cols, 0);

            n++;
            projError += norm(ptRef - ptL);
            circle(outImg, ptL, 3, Scalar(0, 255, 0));
            circle(outImg, ptR, 3, Scalar(0, 255, 0));
            if (i % 5 == 0)
                line(outImg, ptL, ptR, Scalar(0, 0, 255));
        }
    }
    projError /= n * 1.f;
    return outImg;
}

Mat drawKPMatches(const PtrKeyFrame KFRef, const PtrKeyFrame KFCur, const Mat& imgRef,
                 const Mat& imgCur, const map<int, int>& matches)
{
    Mat outImg;
    vconcat(imgCur, imgRef, outImg);

    for (auto& m : matches) {
        Point2f p1 = KFCur->mvKeyPoints[m.second].pt;
        Point2f p2 = KFRef->mvKeyPoints[m.first].pt + Point2f(0, imgCur.rows);
        circle(outImg, p1, 3, Scalar(0, 255, 0));
        circle(outImg, p2, 3, Scalar(0, 255, 0));
        line(outImg, p1, p2, Scalar(0, 0, 255));
    }
    return outImg;
}

/**
 * @brief 根据F/H矩阵剔除外点，利用了RANSAC算法
 * @param kpRef     参考帧KP
 * @param kpCur     当前帧KP
 * @param matches12 参考帧到当前帧的匹配索引
 * @param H12       参考帧到当前帧的变换H/F矩阵
 * @param flag      0表示使用F矩阵, 1表示使用H矩阵
 * @return          返回内点数
 */
int removeOutliersWithHF(const vector<KeyPoint>& kpRef, const vector<KeyPoint>& kpCur,
                         vector<int>& matches12, Mat& H12, int flag = 1)
{
    assert(kpRef.size() == kpCur.size());
    assert(!matches12.empty());

    vector<Point2f> ptRef, ptCur;
    vector<int> idx;
    ptRef.reserve(kpRef.size());
    ptCur.reserve(kpCur.size());
    idx.reserve(kpRef.size());

    for (int i = 0, iend = kpRef.size(); i < iend; ++i) {
        if (matches12[i] < 0)
            continue;
        idx.push_back(i);
        ptRef.push_back(kpRef[i].pt);
        ptCur.push_back(kpCur[matches12[i]].pt);
    }

    vector<unsigned char> mask;
    size_t nPoints = ptRef.size();
    // 默认RANSAC法计算H矩阵. 天花板处于平面,最好用H矩阵, H_33 = 1
    if (flag) {
        if (nPoints >= 4)
            H12 = findHomography(ptRef, ptCur, RANSAC, 3.0, mask);
        else
            fprintf(stderr, "Too less points (%ld) for calculate H!\n", nPoints);
    } else {
        if (nPoints >= 8)
            H12 = findFundamentalMat(ptRef, ptCur, FM_RANSAC, 3.0, 0.99, mask);
        else
            fprintf(stderr, "Too less points (%ld) for calculate F!\n", nPoints);
    }

    int nInlier = 0;
    for (size_t i = 0, iend = mask.size(); i < iend; ++i) {
        if (!mask[i])
            matches12[idx[i]] = -1;
        else
            nInlier++;
    }

    return nInlier;
}

int removeOutliersWithHF(const vector<KeyPoint>& kpRef, const vector<KeyPoint>& kpCur,
                         map<int, int>& matches12, Mat& H12, int flag = 1)
{
    assert(!matches12.empty());

    vector<Point2f> ptRef, ptCur;
    vector<int> idx;
    ptRef.reserve(kpRef.size());
    ptCur.reserve(kpCur.size());
    idx.reserve(kpRef.size());

    for (auto it = matches12.begin(), iend = matches12.end(); it != iend; ++it) {
        idx.push_back(it->first);
        ptRef.push_back(kpRef[it->first].pt);
        ptCur.push_back(kpCur[it->second].pt);
    }

    vector<unsigned char> mask;
    size_t nPoints = ptRef.size();
    // 默认RANSAC法计算H矩阵. 天花板处于平面,最好用H矩阵, H_33 = 1
    if (flag) {
        if (nPoints >= 4)
            H12 = findHomography(ptRef, ptCur, RANSAC, 3.0, mask);
        else
            fprintf(stderr, "Too less points (%ld) for calculate H!\n", nPoints);
    }

    int nInlier = 0;
    for (size_t i = 0, iend = mask.size(); i < iend; ++i) {
        if (!mask[i])
            matches12.erase(idx[i]);
        else
            nInlier++;
    }

    return nInlier;
}

/**
 * @brief removeOutliersWithA
 * @param flag
 *  0:  estimateAffine2D(), full A
 *  1:  estimateAffinePartial2D(),  partial A
 *  2:  estimateRigidTransform(),   rigid transform
 *
 * @return
 */
int removeOutliersWithA(const vector<KeyPoint>& kpRef, const vector<KeyPoint>& kpCur,
                        vector<int>& matches12, Mat& A12, const int flag = 0)
{
    assert(kpRef.size() == kpCur.size());
    assert(!matches12.empty());

    vector<Point2f> ptRef, ptCur;
    vector<int> idxRef;
    idxRef.reserve(kpRef.size());
    ptRef.reserve(kpRef.size());
    ptCur.reserve(kpCur.size());

    for (int i = 0, iend = kpRef.size(); i < iend; ++i) {
        if (matches12[i] < 0)
            continue;
        idxRef.push_back(i);
        ptRef.push_back(kpRef[i].pt);
        ptCur.push_back(kpCur[matches12[i]].pt);
    }

    vector<unsigned char> mask;
    switch (flag) {
    case 0:
        A12 = estimateAffine2D(ptRef, ptCur, mask, RANSAC, 3.0);
        break;
    case 1:
        A12 = estimateAffinePartial2D(ptRef, ptCur, mask, RANSAC, 3.0);
        break;
    case 2:
        //! NOTE 内点数<50%后无法计算
        A12 = estimateRigidTransform(ptRef, ptCur, false);
        mask.resize(ptRef.size(), 1);
        if (A12.empty()) {
            cerr << "Warning! 内点数太少无法计算RigidTransform!" << endl;
            A12 = Mat::eye(2, 3, CV_64FC1);
        }
        break;
    default:
        A12 = estimateAffine2D(ptRef, ptCur, mask, RANSAC, 3.0);
        break;
    }

    int nInlier = 0;
    for (size_t i = 0, iend = mask.size(); i < iend; ++i) {
        if (!mask[i])
            matches12[idxRef[i]] = -1;
        else
            nInlier++;
    }

    return nInlier;
}

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
    rotationCenter.y =  Config::cy + Config::Tbc.at<float>(0, 3) / 12;
    Mat R2 = getRotationMatrix2D(rotationCenter, dOdom.theta * 180.f / CV_PI, 1.);
    cout << "Affine Matrix of R2 = " << endl << R2 << endl;

    return R1.rowRange(0, 2).clone();
}

#endif  // TEST_FUNCTIONS_HPP
