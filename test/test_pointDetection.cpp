#include "cvutil.h"
#include "ORBextractor.h"
#include "ORBmatcher.h"
#include "ORBVocabulary.h"
#include "Frame.h"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/line_descriptor.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string_regex.hpp>
#include <boost/filesystem.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace se2lam;
namespace bf = boost::filesystem;

string g_orbVocFile = "/home/vance/slam_ws/ORB_SLAM2/Vocabulary/ORBvoc.bin";


struct RK_IMAGE {
    RK_IMAGE(const string& s, const long long int t) : fileName(s), timeStamp(t) {}

    string fileName;
    long long int timeStamp;
};


bool lessThen(const RK_IMAGE& r1, const RK_IMAGE& r2)
{
    return r1.timeStamp < r2.timeStamp;
}


void readImagesRK(const string& dataFolder, vector<string>& files)
{
    bf::path path(dataFolder);
    if (!bf::exists(path)) {
        cerr << "[Main] Data folder doesn't exist!" << endl;
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
            auto t = atoll(s.substr(i + 1, j - i - 1).c_str());
            allImages.push_back(RK_IMAGE(s, t));
        }
    }

    if (allImages.empty()) {
        cerr << "[Main] Not image data in the folder!" << endl;
        return;
    } else
        cout << "[Main] Read " << allImages.size() << " files in the folder." << endl;

    //! 应该根据后面的时间戳数值来排序
    sort(allImages.begin(), allImages.end(), lessThen);

    files.clear();
    for (size_t i = 0; i < allImages.size(); ++i)
        files.push_back(allImages[i].fileName);
}

int removeOutliers(const vector<KeyPoint> &kp1, const vector<KeyPoint> &kp2, vector<int> &matches)
{
    vector<Point2f> pt1, pt2;
    vector<int> idx;
    pt1.reserve(kp1.size());
    pt2.reserve(kp2.size());
    idx.reserve(kp1.size());

    for (int i = 0, iend = kp1.size(); i < iend; i++) {
        if (matches[i] < 0)
            continue;
        idx.push_back(i);
        pt1.push_back(kp1[i].pt);
        pt2.push_back(kp2[matches[i]].pt);
    }

    vector<unsigned char> mask;
    if (pt1.size() != 0)
        findFundamentalMat(pt1, pt2, FM_RANSAC, 2, 0.99, mask);  // 默认RANSAC法计算F矩阵

    int nInlier = 0;
    for (int i = 0, iend = mask.size(); i < iend; i++) {
        if (!mask[i])
            matches[idx[i]] = -1;
        else
            nInlier++;
    }

    // If too few match inlier, discard all matches. The enviroment might not be
    // suitable for image tracking.
//    if (nInlier < 10) {
//        nInlier = 0;
//        cerr << "内点数少于10!!!" << endl;
//    }

    return nInlier;
}


int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: lineDetection <rk_dataPath>");
        return -1;
    }

    Mat K = (Mat_<double>(3, 3) << 207.9359613169054, 0., 160.5827136112504, 0., 207.4159055585876,
             117.7328673795551, 0., 0., 1.);
    Mat D = (Mat_<double>(5, 1) << 3.77044e-02, -3.261434e-02, -9.238135e-04, 5.452823e-04, 0.0);

    string dataFolder = string(argv[1]) + "slamimg";
    vector<string> imgFiles;
    readImagesRK(dataFolder, imgFiles);

    ORBextractor *kpExtractor = new ORBextractor(800, 1, 1, 1, 7);
    ORBmatcher *kpMatcher = new ORBmatcher();
    ORBVocabulary *vocabulary = new ORBVocabulary();
    bool bVocLoad = vocabulary->loadFromBinaryFile(g_orbVocFile);
    if(!bVocLoad) {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Falied to open at: " << g_orbVocFile << endl;
        exit(-1);
    }
    cout << "Vocabulary loaded!" << endl << endl;

    Frame frameCur, frameRef;
    vector<MapPoint *> mapPoints;
    Mat imgColor, imgGray, imgCur, imgRef, imgJoint;
    Mat outImgORBMatch, outImgKPs;
    bool firstFrame = true;
    for (size_t i = 0; i < imgFiles.size(); ++i) {
        imgColor = imread(imgFiles[i], CV_LOAD_IMAGE_COLOR);
        if (imgColor.data == nullptr)
            continue;
        cv::undistort(imgColor, imgCur, K, D);
        cvtColor(imgCur, imgGray, CV_BGR2GRAY);

        //! 限制对比度自适应直方图均衡
        Mat imgClahe;
        Ptr<CLAHE> clahe = createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(imgGray, imgClahe);
        vconcat(imgGray, imgClahe, imgJoint);  // 垂直拼接
        imshow("Image Clahe", imgJoint);

        //! ORB提取特征点
        frameCur = Frame(imgClahe, Se2(), kpExtractor, K, D);

        if (firstFrame) {
            imgCur.copyTo(imgRef);
            frameRef = frameCur;
            firstFrame = false;
            continue;
        }

        //! Match
        vector<Point2f> prevMatched;
        vector<int> matchIdx;
        KeyPoint::convert(frameRef.keyPointsUn, prevMatched);
        int nMatched = kpMatcher->MatchByWindow(frameRef, frameCur, prevMatched, 25, matchIdx);
        int nInlines = removeOutliers(frameRef.keyPointsUn, frameCur.keyPointsUn, matchIdx);
        printf("提取了%d个点, 匹配上了%d个点, 内点数为:%d\n", frameCur.N, nMatched, nInlines);

        //! Show Matche
        vconcat(imgRef, imgCur, outImgORBMatch);
        for (int i = 0, iend = frameRef.keyPointsUn.size(); i < iend; i++) {
            if (matchIdx[i] < 0)
                continue;
            circle(outImgORBMatch, frameRef.keyPointsUn[i].pt, 3, Scalar(0, 255, 0));
            circle(outImgORBMatch, frameCur.keyPointsUn[matchIdx[i]].pt + Point2f(0, imgRef.rows),
                   3, Scalar(0, 255, 0));
            line(outImgORBMatch, frameRef.keyPointsUn[i].pt,
                 frameCur.keyPointsUn[matchIdx[i]].pt + Point2f(0, imgRef.rows), Scalar(0, 0, 255));
        }
        imshow("ORB Match", outImgORBMatch);


        waitKey(50);

        // 参考帧每隔5帧更新,模拟KF
        if (i % 10 == 0) {
            imgCur.copyTo(imgRef);
            frameRef = frameCur;
        }
    }

    delete kpExtractor;
    delete kpMatcher;
    delete vocabulary;
    return 0;
}
