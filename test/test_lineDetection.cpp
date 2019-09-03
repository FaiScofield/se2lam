//#include <ros/ros.h>
//#include <image_transport/image_transport.h>
#include "cvutil.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/line_descriptor.hpp>

#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string_regex.hpp>

using namespace std;
using namespace cv;
using namespace line_descriptor;
namespace bf =  boost::filesystem;

struct RK_IMAGE
{
    RK_IMAGE(const string& s, const long long int t)
        : fileName(s), timeStamp(t) {}

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
            auto t = atoll(s.substr(i+1, j-i-1).c_str());
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


int main(int argc, char **argv)
{
    if (argc < 2){
        fprintf(stderr, "Usage: lineDetection <rk_dataPath>");
        return -1;
    }

    Mat K = (Mat_<double>(3, 3) << 219.9359613169054, 0., 161.5827136112504, 0., 219.4159055585876,
         117.7128673795551, 0., 0., 1.);
    Mat D = (Mat_<double>(5, 1) << 0.064610443232716, -0.086814339668420, -0.0009238134627751219,
         0.0005452823230733891, 0.000000000000000);

    string dataFolder = string(argv[1]) + "slamimg";
    vector<string> imgFiles;
    readImagesRK(dataFolder, imgFiles);

    cv::Mat imgGray, imgCur, imgRef, imgColor, imgJoint;
    Mat outImageLSD, outImageLsdMatch, outImageLsdMatchGood;
    Mat desCur, desRef;
    vector<KeyLine> keyLinesCur, keyLinesRef;
    bool firstFrame = true;
    for (size_t i = 0; i < imgFiles.size(); ++i) {
        printf("Reading image: %s\n", imgFiles[i].c_str());
        imgColor = imread(imgFiles[i], CV_LOAD_IMAGE_COLOR);
        if (imgColor.data == nullptr)
            continue;
        cv::undistort(imgColor, imgCur, K, D);
        Mat imgTmp;
        cvtColor(imgColor, imgTmp, CV_BGR2GRAY);

        //! 限制对比度自适应直方图均衡
        cvtColor(imgCur, imgGray, CV_BGR2GRAY);
        Ptr<CLAHE> clahe = createCLAHE(10.0, cv::Size(8, 8));
        clahe->apply(imgGray, imgGray);
        vconcat(imgTmp, imgGray, imgJoint);   // 垂直拼接
        imshow("Image clahe", imgJoint);

        //! LSDDetector 提取的直线很多，比较杂
        // LSD: A fast line segment detector with a false detection control, 2010
        vector<KeyLine> keyLines;
        Ptr<LSDDetector> lsd = LSDDetector::createLSDDetector();
        lsd->detect(imgGray, keyLines, 1, 1);
//        cvtColor(imgCur, outImageLSD, CV_GRAY2BGR);
        drawKeylines(imgCur, keyLines, outImageLSD, Scalar(0,255,0));
        imshow("LSD lines", outImageLSD);


        //! BinaryDescriptor 提取的直线相对少一些，但杂线更少
        keyLinesCur.clear();
        Ptr<BinaryDescriptor> bd = BinaryDescriptor::createBinaryDescriptor();
        bd->detect(imgGray, keyLinesCur);
        bd->compute(imgGray, keyLinesCur, desCur);

        if (firstFrame) {
            imgCur.copyTo(imgRef);
            desCur.copyTo(desRef);
            keyLinesRef.swap(keyLinesCur);
            firstFrame = false;
            continue;
        }

        //! Match
        vector<DMatch> matches, matchesKnn;
        Ptr<BinaryDescriptorMatcher> bdm = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
        bdm->match(desCur, desRef, matches);

        vector<vector<DMatch>> vMatchesKnn;
        bdm->knnMatch(desCur, desRef, vMatchesKnn, 3);
        for (auto& ms : vMatchesKnn)
            if (ms.size() > 2 && ms[0].distance < 20 && ms[0].distance < 0.8*ms[1].distance)
                matchesKnn.push_back(ms[0]);

        //! Show Matches
        vector<char> mask(matches.size(), 1);
//        cvtColor(imgCur, outImageLsdMatch, CV_GRAY2BGR);
        drawLineMatches(imgCur, keyLinesCur, imgRef, keyLinesRef, matches,
                        outImageLsdMatch, Scalar(0,0,255), Scalar(0,255,0), mask);
        imshow("LSD Matches", outImageLsdMatch);

        vector<char> mask2(matchesKnn.size(), 1);
//        cvtColor(imgCur, outImageLsdMatchGood, CV_GRAY2BGR);
        drawLineMatches(imgCur, keyLinesCur, imgRef, keyLinesRef, matchesKnn,
                        outImageLsdMatchGood, Scalar(0,0,255), Scalar(0,255,0), mask2);
        imshow("LSD Matches Knn", outImageLsdMatchGood);

        waitKey(100);

        imgCur.copyTo(imgRef);
        desCur.copyTo(desRef);
        keyLinesRef.swap(keyLinesCur);
    }

    return 0;
}
