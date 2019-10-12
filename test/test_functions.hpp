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

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string_regex.hpp>
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
        cerr << "[ERROR] Opening file error in '%s'" << odomFile << endl;
        rec.close();
        return;
    }

    odoData.clear();
    string line;
    while (std::getline(rec, line) && !line.empty()) {
        istringstream iss(line);
        Se2 odo;
        iss >> odo.timeStamp >> odo.x >> odo.y >> odo.theta;  // theta 可能会比超过pi,使用前要归一化
        odo.timeStamp *= 1e-6;
        odo.x *= 1000.f;
        odo.y *= 1000.f;
        odoData.emplace_back(odo);
    }
    rec.close();

    if (odoData.empty()) {
        cerr << "[ERROR] Not odom data in the file!" << endl;
        return;
    } else {
        cout << "[ INFO] Read " << odoData.size() << " odom datas from the file." << endl;
    }
}

void dataAlignment(vector<RK_IMAGE>& allImages, const vector<Se2>& allOdoms,
                   vector<Se2>& alignedOdoms)
{
    // 去除掉没有odom数据的图像
    Se2 firstOdo = allOdoms[0];
    auto iter = allImages.begin();
    int cut = 0;
    for (auto iend = allImages.end(); iter != iend; ++iter) {
        if (iter->timeStamp < firstOdo.timeStamp) {
            cut++;
            continue;
        } else {
            break;
        }
    }
    allImages.erase(allImages.begin(), iter);
    printf("[ INFO] Cut %d images for timestamp too earlier, now image size is %ld\n", cut, allImages.size());

    // 数据对齐
    auto ite = allOdoms.begin(), its = allOdoms.begin();
    for (size_t i = 0, iend = allImages.size(); i < iend; ++i) {
        double imgTime = allImages[i].timeStamp;

        while (ite != allOdoms.end()) {
            if (ite->timeStamp <= imgTime)
                ite++;
            else
                break;
        }
        vector<Se2> odoDeq = vector<Se2>(its, ite);
        its = ite;

        // 插值计算对应的odom
        Se2 res;
        size_t n = odoDeq.size();
        if (n < 2) {
            cerr << "[Track] ** WARNING ** Less odom sequence input!" << endl;
            alignedOdoms.push_back(alignedOdoms.back());
        }

        //! 计算单帧图像时间内的平均速度
        Se2 tranSum = odoDeq[n - 1] - odoDeq[0];
        double dt = odoDeq[n - 1].timeStamp - odoDeq[0].timeStamp;
        double r = (imgTime - odoDeq[n - 1].timeStamp) / dt;
        assert(r >= 0.f);

        Se2 transDelta(tranSum.x * r, tranSum.y * r, tranSum.theta * r);
        res = odoDeq[n - 1] + transDelta;

        alignedOdoms.push_back(res);
    }
    assert(alignedOdoms.size() == allImages.size());
}

Mat drawKPMatches(const Frame* frameRef, const Frame* frameCur, const Mat& imgRef, const Mat& imgCur,
                  const vector<int>& matchIdx12, const Mat& H12 = Mat())
{
    Mat H21, imgCurWarp, outImg;
    if (H12.data) {
        H21 = H12.inv(DECOMP_SVD);  // 内点数为0时H可能是空矩阵, 无法求逆.
        warpPerspective(imgCur, imgCurWarp, H21, imgCur.size());
        vconcat(imgCurWarp, imgRef, outImg);
    } else {
        vconcat(imgCur, imgRef, outImg);
    }

    for (size_t i = 0, iend = frameRef->mvKeyPoints.size(); i < iend; ++i) {
        if (matchIdx12[i] < 0) {
            continue;
        } else {
            Point2f p1, p2;
            if (H12.data) {
                Point2f p = frameCur->mvKeyPoints[matchIdx12[i]].pt;
                Mat pt1 = (Mat_<double>(3, 1) << p.x, p.y, 1);
                Mat pt2 = H21 * pt1;
                pt2 /= pt2.at<double>(2);
                p1 = Point2f(pt2.at<double>(0), pt2.at<double>(1));
            } else {
                p1 = frameCur->mvKeyPoints[matchIdx12[i]].pt;
            }
            p2 = frameRef->mvKeyPoints[i].pt + Point2f(0, imgCur.rows);

            // 外点红色, 内点绿色, 没有匹配的点蓝色
            if (matchIdx12[i] == -2) {
                circle(outImg, p1, 3, Scalar(0, 255, 255));
                circle(outImg, p2, 3, Scalar(0, 255, 255));
                line(outImg, p1, p2, Scalar(0, 255, 255));
            } else {
                circle(outImg, p1, 3, Scalar(0, 255, 0));
                circle(outImg, p2, 3, Scalar(0, 255, 0));
                line(outImg, p1, p2, Scalar(0, 0, 255));
            }
        }
    }
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
 * @param flag      0表示使用H矩阵, 1表示使用F矩阵
 * @return          返回内点数
 */
int removeOutliersWithHF(const vector<KeyPoint>& kpRef, const vector<KeyPoint>& kpCur,
                         vector<int>& matches12, Mat& H12, const int flag = 0)
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
        if (nPoints >= 8)
            H12 = findFundamentalMat(ptRef, ptCur, FM_RANSAC, 2.0, 0.99, mask);
        else
            fprintf(stderr, "Too less points (%ld) for calculate F!\n", nPoints);
    } else {
        if (nPoints >= 4)
            H12 = findHomography(ptRef, ptCur, RANSAC, 2.0, mask);
        else
            fprintf(stderr, "Too less points (%ld) for calculate H!\n", nPoints);
    }

    // Affine Transform
//    vector<Point2f> ptR = {ptRef.begin(), ptRef.begin() + 3};
//    vector<Point2f> ptC = {ptCur.begin(), ptCur.begin() + 3};
//    H12 = getAffineTransform(ptR, ptC);  // 2x3

    int nInlier = 0;
    for (size_t i = 0, iend = mask.size(); i < iend; ++i) {
        if (!mask[i])
            matches12[idx[i]] = -2;
        else
            nInlier++;
    }

    return nInlier;
}

int removeOutliersWithHF(const vector<KeyPoint>& kpRef, const vector<KeyPoint>& kpCur,
                         map<int, int>& matches12, Mat& H12, const int flag = 0)
{
    assert(kpRef.size() == kpCur.size());
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
        if (nPoints >= 8)
            H12 = findFundamentalMat(ptRef, ptCur, FM_RANSAC, 2.0, 0.99, mask);
        else
            fprintf(stderr, "Too less points (%ld) for calculate F!\n", nPoints);
    } else {
        if (nPoints >= 4)
            H12 = findHomography(ptRef, ptCur, RANSAC, 2.0, mask);
        else
            fprintf(stderr, "Too less points (%ld) for calculate H!\n", nPoints);
    }

    // Affine Transform
//    vector<Point2f> ptR = {ptRef.begin(), ptRef.begin() + 3};
//    vector<Point2f> ptC = {ptCur.begin(), ptCur.begin() + 3};
//    H12 = getAffineTransform(ptR, ptC);  // 2x3

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
 * @brief 从特征点匹配求homography（normalized DLT）
 *
 * @param vP1   归一化后的点, in reference frame
 * @param vP2   归一化后的点, in current frame
 * @return      单应矩阵
 * @see         Multiple View Geometry in Computer Vision - Algorithm 4.2 p109
 */
cv::Mat ComputeH21(const vector<cv::Point2f>& vP1, const vector<cv::Point2f>& vP2)
{
    const int N = vP1.size();

    cv::Mat A(2 * N, 9, CV_32F);  // 2N*9

    for (int i = 0; i < N; i++) {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(2 * i, 0) = 0.0;
        A.at<float>(2 * i, 1) = 0.0;
        A.at<float>(2 * i, 2) = 0.0;
        A.at<float>(2 * i, 3) = -u1;
        A.at<float>(2 * i, 4) = -v1;
        A.at<float>(2 * i, 5) = -1;
        A.at<float>(2 * i, 6) = v2 * u1;
        A.at<float>(2 * i, 7) = v2 * v1;
        A.at<float>(2 * i, 8) = v2;

        A.at<float>(2 * i + 1, 0) = u1;
        A.at<float>(2 * i + 1, 1) = v1;
        A.at<float>(2 * i + 1, 2) = 1;
        A.at<float>(2 * i + 1, 3) = 0.0;
        A.at<float>(2 * i + 1, 4) = 0.0;
        A.at<float>(2 * i + 1, 5) = 0.0;
        A.at<float>(2 * i + 1, 6) = -u2 * u1;
        A.at<float>(2 * i + 1, 7) = -u2 * v1;
        A.at<float>(2 * i + 1, 8) = -u2;
    }

    cv::Mat u, w, vt;

    cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    return vt.row(8).reshape(0, 3);  // v的最后一列
}

/**
 * @brief 从特征点匹配求fundamental matrix（normalized 8点法）
 *
 * @param vP1   归一化后的点, in reference frame
 * @param vP2   归一化后的点, in current frame
 * @return      基础矩阵
 * @see         Multiple View Geometry in Computer Vision - Algorithm 11.1 p282 (中文版 p191)
 */
cv::Mat ComputeF21(const vector<cv::Point2f>& vP1, const vector<cv::Point2f>& vP2)
{
    const int N = vP1.size();

    cv::Mat A(N, 9, CV_32F);  // N*9

    for (int i = 0; i < N; i++) {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(i, 0) = u2 * u1;
        A.at<float>(i, 1) = u2 * v1;
        A.at<float>(i, 2) = u2;
        A.at<float>(i, 3) = v2 * u1;
        A.at<float>(i, 4) = v2 * v1;
        A.at<float>(i, 5) = v2;
        A.at<float>(i, 6) = u1;
        A.at<float>(i, 7) = v1;
        A.at<float>(i, 8) = 1;
    }

    cv::Mat u, w, vt;

    cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    cv::Mat Fpre = vt.row(8).reshape(0, 3);  // v的最后一列

    cv::SVDecomp(Fpre, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    w.at<float>(2) = 0;  // 秩2约束，将第3个奇异值设为0

    return u * cv::Mat::diag(w) * vt;
}

/**
 * @brief 对给定的homography matrix打分
 *
 * @see
 * - Author's paper - IV. AUTOMATIC MAP INITIALIZATION （2）
 * - Multiple View Geometry in Computer Vision - symmetric transfer errors: 4.2.2 Geometric distance
 * - Multiple View Geometry in Computer Vision - model selection 4.7.1 RANSAC

float CheckHomography(const cv::Mat& H21, const cv::Mat& H12, vector<bool>& vbMatchesInliers,
                      float sigma)
{
    const int N = mvMatches12.size();

    // |h11 h12 h13|
    // |h21 h22 h23|
    // |h31 h32 h33|
    const float h11 = H21.at<float>(0, 0);
    const float h12 = H21.at<float>(0, 1);
    const float h13 = H21.at<float>(0, 2);
    const float h21 = H21.at<float>(1, 0);
    const float h22 = H21.at<float>(1, 1);
    const float h23 = H21.at<float>(1, 2);
    const float h31 = H21.at<float>(2, 0);
    const float h32 = H21.at<float>(2, 1);
    const float h33 = H21.at<float>(2, 2);

    // |h11inv h12inv h13inv|
    // |h21inv h22inv h23inv|
    // |h31inv h32inv h33inv|
    const float h11inv = H12.at<float>(0, 0);
    const float h12inv = H12.at<float>(0, 1);
    const float h13inv = H12.at<float>(0, 2);
    const float h21inv = H12.at<float>(1, 0);
    const float h22inv = H12.at<float>(1, 1);
    const float h23inv = H12.at<float>(1, 2);
    const float h31inv = H12.at<float>(2, 0);
    const float h32inv = H12.at<float>(2, 1);
    const float h33inv = H12.at<float>(2, 2);

    vbMatchesInliers.resize(N);

    float score = 0;

    // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
    const float th = 5.991;

    //信息矩阵，方差平方的倒数
    const float invSigmaSquare = 1.0 / (sigma * sigma);

    // N对特征匹配点
    for (int i = 0; i < N; i++) {
        bool bIn = true;

        const cv::KeyPoint& kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint& kp2 = mvKeys2[mvMatches12[i].second];

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // Reprojection error in first image
        // x2in1 = H12*x2
        // 将图像2中的特征点单应到图像1中
        // |u1|   |h11inv h12inv h13inv||u2|
        // |v1| = |h21inv h22inv h23inv||v2|
        // |1 |   |h31inv h32inv h33inv||1 |
        const float w2in1inv = 1.0 / (h31inv * u2 + h32inv * v2 + h33inv);
        const float u2in1 = (h11inv * u2 + h12inv * v2 + h13inv) * w2in1inv;
        const float v2in1 = (h21inv * u2 + h22inv * v2 + h23inv) * w2in1inv;

        // 计算重投影误差
        const float squareDist1 = (u1 - u2in1) * (u1 - u2in1) + (v1 - v2in1) * (v1 - v2in1);

        // 根据方差归一化误差
        const float chiSquare1 = squareDist1 * invSigmaSquare;

        if (chiSquare1 > th)
            bIn = false;
        else
            score += th - chiSquare1;

        // Reprojection error in second image
        // x1in2 = H21*x1
        // 将图像1中的特征点单应到图像2中
        const float w1in2inv = 1.0 / (h31 * u1 + h32 * v1 + h33);
        const float u1in2 = (h11 * u1 + h12 * v1 + h13) * w1in2inv;
        const float v1in2 = (h21 * u1 + h22 * v1 + h23) * w1in2inv;

        const float squareDist2 = (u2 - u1in2) * (u2 - u1in2) + (v2 - v1in2) * (v2 - v1in2);

        const float chiSquare2 = squareDist2 * invSigmaSquare;

        if (chiSquare2 > th)
            bIn = false;
        else
            score += th - chiSquare2;

        if (bIn)
            vbMatchesInliers[i] = true;
        else
            vbMatchesInliers[i] = false;
    }

    return score;
}
 */
/**
 * @brief 对给定的fundamental matrix打分
 *
 * @see
 * - Author's paper - IV. AUTOMATIC MAP INITIALIZATION （2）
 * - Multiple View Geometry in Computer Vision - symmetric transfer errors: 4.2.2 Geometric distance
 * - Multiple View Geometry in Computer Vision - model selection 4.7.1 RANSAC

float CheckFundamental(const cv::Mat& F21, vector<bool>& vbMatchesInliers, float sigma)
{
    const int N = mvMatches12.size();

    const float f11 = F21.at<float>(0, 0);
    const float f12 = F21.at<float>(0, 1);
    const float f13 = F21.at<float>(0, 2);
    const float f21 = F21.at<float>(1, 0);
    const float f22 = F21.at<float>(1, 1);
    const float f23 = F21.at<float>(1, 2);
    const float f31 = F21.at<float>(2, 0);
    const float f32 = F21.at<float>(2, 1);
    const float f33 = F21.at<float>(2, 2);

    vbMatchesInliers.resize(N);

    float score = 0;

    // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
    const float th = 3.841;
    const float thScore = 5.991;

    const float invSigmaSquare = 1.0 / (sigma * sigma);

    for (int i = 0; i < N; i++) {
        bool bIn = true;

        const cv::KeyPoint& kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint& kp2 = mvKeys2[mvMatches12[i].second];

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // Reprojection error in second image
        // l2=F21x1=(a2,b2,c2)
        // F21x1可以算出x1在图像中x2对应的线l
        const float a2 = f11 * u1 + f12 * v1 + f13;
        const float b2 = f21 * u1 + f22 * v1 + f23;
        const float c2 = f31 * u1 + f32 * v1 + f33;

        // x2应该在l这条线上:x2点乘l = 0
        const float num2 = a2 * u2 + b2 * v2 + c2;

        const float squareDist1 = num2 * num2 / (a2 * a2 + b2 * b2);  // 点到线的几何距离 的平方

        const float chiSquare1 = squareDist1 * invSigmaSquare;

        if (chiSquare1 > th)
            bIn = false;
        else
            score += thScore - chiSquare1;

        // Reprojection error in second image
        // l1 =x2tF21=(a1,b1,c1)

        const float a1 = f11 * u2 + f21 * v2 + f31;
        const float b1 = f12 * u2 + f22 * v2 + f32;
        const float c1 = f13 * u2 + f23 * v2 + f33;

        const float num1 = a1 * u1 + b1 * v1 + c1;

        const float squareDist2 = num1 * num1 / (a1 * a1 + b1 * b1);

        const float chiSquare2 = squareDist2 * invSigmaSquare;

        if (chiSquare2 > th)
            bIn = false;
        else
            score += thScore - chiSquare2;

        if (bIn)
            vbMatchesInliers[i] = true;
        else
            vbMatchesInliers[i] = false;
    }

    return score;
}
 */
#endif  // TEST_FUNCTIONS_HPP
