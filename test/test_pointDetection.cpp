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
string g_matchResult = "/home/vance/output/rk_se2lam/test_matchResult.txt";

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

// 根据F/H矩阵剔除外点，利用了RANSAC算法
int removeOutliers(const vector<KeyPoint> &kpRef, const vector<KeyPoint> &kpCur,
                   vector<int> &matches12, Mat& H12, Mat& A12)
{
    assert(kpRef.size() == kpCur.size());

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
    if (ptRef.size() != 0) {
//        F = findFundamentalMat(pt1, pt2, FM_RANSAC, 2, 0.99, mask);  // 默认RANSAC法计算F矩阵
        H12 = findHomography(ptRef, ptCur, RANSAC, 2, mask);  // 天花板处于平面,最好用H矩阵, H_33 = 1
        vector<Point2f> ptR = {ptRef.begin(), ptRef.begin() + 3};
        vector<Point2f> ptC = {ptCur.begin(), ptCur.begin() + 3};
        A12 = getAffineTransform(ptR, ptC); // 2x3
    }

    int nInlier = 0;
    for (int i = 0, iend = mask.size(); i < iend; ++i) {
        if (!mask[i])
            matches12[idx[i]] = -1;
        else
            nInlier++;
    }

    return nInlier;
}

void writeMatchData(const string outFile,
                    const vector<vector<int>>& vvMatches,
                    const vector<vector<int>>& vvInliners)
{
    assert(vvMatches.size() == vvInliners.size());

    ofstream ofs(outFile);
    if (!ofs.is_open()) {
        cerr << "Open file error: " << outFile << endl;
        return;
    }

    int n = vvMatches.size();
    int sum[n] = {0}, sumIn[n] = {0};
    for (int i = 0; i < n; ++i) {
        ofs << i << " ";
        for (size_t j = 0; j < vvMatches[i].size(); ++j) {
            ofs << vvMatches[i][j] << " ";
            sum[i] += vvMatches[i][j];
        }
        ofs << endl;
    }
    ofs << endl;
    for (int i = 0; i < n; ++i)
        cout << "Tatal Match Average " << i << ": " << 1.0*sum[i]/vvMatches[i].size() << endl;

    for (int i = 0; i < n; ++i) {
        ofs << i << " ";
        for (size_t j = 0; j < vvInliners[i].size(); ++j) {
            ofs << vvInliners[i][j] << " ";
            sumIn[i] += vvInliners[i][j];
        }
        ofs << endl;
    }
    ofs.close();
    for (int i = 0; i < n; ++i)
        cout << "Tatal Inliners Average " << i << ": " << 1.0*sumIn[i]/vvInliners[i].size() << endl;

    cerr << "Write match data to file: " << outFile << endl;
}


int main(int argc, char** argv)
{

    //! check input
    if (argc < 2) {
        fprintf(stderr, "Usage: lineDetection <rk_dataPath> [number_frames_to_process]");
        return -1;
    }
    int num = 999999;
    if (argc == 3) {
        num = atoi(argv[2]);
        cout << " - set number_frames_to_process = " << num << endl << endl;
    }


    //! initialization
    Mat K = (Mat_<double>(3, 3) << 207.9359613169054, 0., 160.5827136112504, 0., 207.4159055585876,
             117.7328673795551, 0., 0., 1.);
    Mat D = (Mat_<double>(5, 1) << 3.77044e-02, -3.261434e-02, -9.238135e-04, 5.452823e-04, 0.0);

    string dataFolder = string(argv[1]) + "slamimg";
    vector<string> imgFiles;
    readImagesRK(dataFolder, imgFiles);

    ORBextractor *kpExtractor = new ORBextractor(500, 1, 1, 1, 7);
    ORBmatcher *kpMatcher = new ORBmatcher();
    ORBVocabulary *vocabulary = new ORBVocabulary();
    bool bVocLoad = vocabulary->loadFromBinaryFile(g_orbVocFile);
    if(!bVocLoad) {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Falied to open at: " << g_orbVocFile << endl;
        exit(-1);
    }
    cout << "Vocabulary loaded!" << endl << endl;


    //! main loop
    bool firstFrame = true;
    int idKF = 0;
    const int deltaKF = 10;
    vector<vector<int>> vvMatches(deltaKF);
    vector<vector<int>> vvInliners(deltaKF);

    Frame frameCur, frameRef;
//    vector<MapPoint *> mapPoints;
    Mat imgColor, imgGray, imgCur, imgRef, imgWithFeatureCur, imgWithFeatureRef;
    Mat outImgORBMatch, outImgWarp, outImgAffine, imgCurAffine;
    num = min(num, (int)imgFiles.size());
    int skipFrames = 100;
    Mat H12 = Mat::eye(3, 3, CV_64F);
    for (int i = skipFrames; i < num + skipFrames; ++i) {
        imgColor = imread(imgFiles[i], CV_LOAD_IMAGE_COLOR);
        if (imgColor.data == nullptr)
            continue;
        cv::undistort(imgColor, imgCur, K, D);
        cvtColor(imgCur, imgGray, CV_BGR2GRAY);

        //! 限制对比度自适应直方图均衡
        Mat imgClahe;
        Ptr<CLAHE> clahe = createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(imgGray, imgClahe);

        //! ORB提取特征点
        float imgTime;
        frameCur = Frame(imgClahe, imgTime, Se2(), kpExtractor, K, D);
        imgWithFeatureCur = imgCur.clone();
        for (int i = 0, iend = frameCur.N; i < iend; ++i) {
            circle(imgWithFeatureCur, frameCur.mvKeyPoints[i].pt, 2, Scalar(255, 0, 0));
        }

        if (firstFrame) {
            imgCur.copyTo(imgRef);
            imgWithFeatureCur.copyTo(imgWithFeatureRef);
            frameRef = frameCur;
            firstFrame = false;
            idKF++;
            continue;
        }

        assert(imgCur.data);
        assert(imgRef.data);
        assert(imgWithFeatureCur.data);
        assert(imgWithFeatureRef.data);

        //! 特征点匹配
        Mat A12, imgCurWarp;
        vector<int> matchIdx12;
//        vector<Point2f> prevMatched;
//        KeyPoint::convert(frameRef.keyPoints, prevMatched);
//        int nMatched = kpMatcher->MatchByWindow(frameRef, frameCur, prevMatched, 25, matchIdx12);
        int nMatched = kpMatcher->MatchByWindowWarp(frameRef, frameCur, H12, matchIdx12, 25);
        int nInlines = removeOutliers(frameRef.mvKeyPoints, frameCur.mvKeyPoints, matchIdx12, H12, A12);

        //! 存储匹配结果作分析H
        int idx = i % deltaKF - 1;
        if (idx == -1)
            idx = deltaKF - 1;
        assert(idx < deltaKF);
        vvMatches[idx].push_back(nMatched);
        vvInliners[idx].push_back(nInlines);
        printf("#%d-%d 内点数/匹配点数: %d/%d\n", idKF, idx + 1, nInlines, nMatched);

        //! 匹配情况可视化
        if (nInlines > 0 && H12.data) {
            Mat H21 = H12.inv(DECOMP_SVD);  // 内点数为0时H可能是空矩阵, 无法求逆.
            warpPerspective(imgWithFeatureCur, imgCurWarp, H21, imgCur.size());
            warpAffine(imgWithFeatureCur, imgCurAffine, A12, imgCur.size(), INTER_CUBIC);
            vconcat(imgCurWarp, imgWithFeatureRef, outImgWarp);
            vconcat(imgCurAffine, imgWithFeatureRef, outImgAffine);
            vconcat(imgWithFeatureCur, imgWithFeatureRef, outImgORBMatch);
            for (int i = 0, iend = frameRef.mvKeyPoints.size(); i < iend; ++i) {
                if (matchIdx12[i] < 0) {
                    continue;
                } else {
                    circle(outImgORBMatch, frameRef.mvKeyPoints[i].pt + Point2f(0, imgCur.rows),
                           3, Scalar(0, 255, 0));
                    circle(outImgORBMatch, frameCur.mvKeyPoints[matchIdx12[i]].pt,
                           3, Scalar(0, 255, 0));
                    line(outImgORBMatch, frameRef.mvKeyPoints[i].pt + Point2f(0, imgCur.rows),
                         frameCur.mvKeyPoints[matchIdx12[i]].pt, Scalar(0, 0, 255));

                    Mat pt1 = (Mat_<double>(3,1) << frameCur.mvKeyPoints[matchIdx12[i]].pt.x,
                                                    frameCur.mvKeyPoints[matchIdx12[i]].pt.y, 1);
                    Mat pt2 = H21 * pt1;
                    pt2 /= pt2.at<double>(2);
                    Point2f p1(pt2.at<double>(0), pt2.at<double>(1));
                    Point2f p2 = frameRef.mvKeyPoints[i].pt + Point2f(0, imgCur.rows);
                    circle(outImgWarp, p1, 2, Scalar(0, 255, 0));
                    circle(outImgWarp, p2, 2, Scalar(0, 255, 0));
                    line(outImgWarp, p1, p2, Scalar(0, 0, 255));

                    Mat A21 = A12.inv(DECOMP_SVD);
                    Mat pt3 = (Mat_<double>(2,1) << frameCur.mvKeyPoints[matchIdx12[i]].pt.x,
                                                    frameCur.mvKeyPoints[matchIdx12[i]].pt.y);
                    Mat pt4 = A21 * pt3;    // A12(3*2)
                    pt4 /= pt4.at<double>(3);
                    Point2f p3(pt3.at<double>(0), pt3.at<double>(1));
                    circle(outImgAffine, p3, 2, Scalar(0, 255, 0));
                    circle(outImgAffine, p2, 2, Scalar(0, 255, 0));
                    line(outImgAffine, p3, p2, Scalar(0, 0, 255));
                }
            }
            string text = to_string(idKF) + "-" + to_string(idx + 1) + ": " + to_string(nInlines);
            putText(outImgORBMatch, text, Point(15, 15), 1, 1, Scalar(0,0,255));
            putText(outImgWarp, text, Point(15, 15), 1, 1, Scalar(0,0,255));
            imshow("ORB Match", outImgORBMatch);
            imshow("Image Warp", outImgWarp);
            imshow("Image Affine", outImgAffine);
//            string fileWarp = "/home/vance/output/rk_se2lam/warp/warp-" + text + ".bmp";
//            string fileMatch = "/home/vance/output/rk_se2lam/warp/match-" + text + ".bmp";
//            imwrite(fileWarp, outImgWarp);
//            imwrite(fileMatch, outImgORBMatch);
            waitKey(200);
        }

        //! 参考帧每隔deltaKF帧更新一次,模拟实际情况
        if (i % deltaKF == 0) {
            imgCur.copyTo(imgRef);
            imgWithFeatureCur.copyTo(imgWithFeatureRef);
            frameRef = frameCur;
            idKF++;
            H12 = Mat::eye(3, 3, CV_64F);
        }
    }

    writeMatchData(g_matchResult, vvMatches, vvInliners);

    delete kpExtractor;
    delete kpMatcher;
    delete vocabulary;
    return 0;
}
