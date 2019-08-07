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
namespace fs =  boost::filesystem;

const char *g_vocFile = "/home/vance/dataset/se2/ORBvoc.bin";

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
    fs::path path(dataFolder);
    if (!fs::exists(path)) {
        cerr << "[Main] Data folder doesn't exist!" << endl;
        return;
    }

    vector<RK_IMAGE> allImages;
    fs::directory_iterator end_iter;
    for (fs::directory_iterator iter(path); iter != end_iter; ++iter) {
        if (fs::is_directory(iter->status()))
            continue;
        if (fs::is_regular_file(iter->status())) {
            // format: /frameRaw12987978101.jpg
            string s = iter->path().string();
            auto i = s.find_last_of('/');
            auto j = s.find_last_of('.');
            auto t = atoll(s.substr(i+8+1, j-i-8-1).c_str());
            cout << s << endl;
            cout << t << endl;
            allImages.push_back(RK_IMAGE(s, t));
        }
    }

    if (allImages.empty()) {
        cerr << "[Main] Not image data in the folder!" << endl;
        return;
    } else
        cout << "[Main] Read " << allImages.size() << " files in the folder." << endl;

    //! 这里sort string不对
//    sort(files.begin(), files.end());
    //! 应该根据后面的时间戳数值来排序
    sort(allImages.begin(), allImages.end(), lessThen);

    files.clear();
    for (int i = 0; i < allImages.size(); ++i)
        files.push_back(allImages[i].fileName);
}


int main(int argc, char **argv)
{
    if (argc < 2){
        fprintf(stderr, "Usage: lineDetection <rk_dataPath>");
        return -1;
    }

//    ros::init(argc, argv, "lineDetection");
//    ros::NodeHandle nh;
//    ros::Publisher chatter_pub = nh.advertise<std_msgs::String>("chatter", 1000);

    string dataFolder = string(argv[1]) + "slamimg";
    vector<string> imgFiles;
    readImagesRK(dataFolder, imgFiles);

    cv::Mat imgCur, imgRef;
    Mat desCur, desRef;
    vector<KeyLine> keyLinesCur, keyLinesRef;
    Mat outImageCanny, outImageLSD, outImageLsdMatch, outImageLsdMatchGood;
    cvtColor(outImageLSD, outImageLSD, COLOR_GRAY2BGR);
    cvtColor(outImageLsdMatch, outImageLsdMatch, COLOR_GRAY2BGR);
    cvtColor(outImageLsdMatchGood, outImageLsdMatchGood, COLOR_GRAY2BGR);
    bool firstFrame = true;
    for (int i = 0; i < imgFiles.size(); ++i) {
        printf("Reading image: %s\n", imgFiles[i].c_str());
        imgCur = imread(imgFiles[i], 1);
        if (imgCur.data == nullptr)
            continue;

        Mat imgShap, imgGamma;
        imgShap = cvu::sharpping(imgCur, 12);
        imgGamma = cvu::gamma(imgShap, 1.2);
        imshow("imgShap", imgShap);
        imshow("imgGamma", imgGamma);

        //! Canny
        Canny(imgCur, outImageCanny, 50, 200, 3); // 输出图像会变黑白
//        imshow("Canny", outImageCanny);

        //! Hough
//        HoughLinesP();

        //! LineSegmentDetector 这个效果不如LSDDetector
//        Ptr<LineSegmentDetector> pLSD  = createLineSegmentDetector();

        //! LSDDetector
        // LSD: A fast line segment detector with a false detection control, 2010
        vector<KeyLine> keyLines;
        Ptr<LSDDetector> lsd = LSDDetector::createLSDDetector();
        lsd->detect(imgCur, keyLines, 2, 2);
        drawKeylines(imgCur, keyLines, outImageLSD, Scalar(0,255,0));
//        imshow("LSD lines", outImageLSD);

        Mat outOneImg(2*outImageLSD.rows, outImageLSD.cols, CV_8UC3);
        cvtColor(outImageCanny, outImageCanny, COLOR_GRAY2BGR);
        assert(outImageLSD.rows == outImageCanny.rows);
        assert(outImageLSD.cols == outImageCanny.cols);
        outImageCanny.copyTo(outOneImg(Rect(0,0,outImageLSD.cols,outImageLSD.rows)));
        outImageLSD.copyTo(outOneImg(Rect(0,outImageLSD.rows,outImageLSD.cols,outImageLSD.rows)));
        imshow("Canny & LSD lines", outOneImg);


        //! BinaryDescriptor
        BinaryDescriptor::Params param; // 参数设置

        keyLinesCur.clear();
        Ptr<BinaryDescriptor> bd = BinaryDescriptor::createBinaryDescriptor();
        bd->detect(imgCur, keyLinesCur);
        bd->compute(imgCur, keyLinesCur, desCur);

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
        drawLineMatches(imgCur, keyLinesCur, imgRef, keyLinesRef, matches,
                        outImageLsdMatch, Scalar(0,0,255), Scalar(0,255,0), mask);
        imshow("LSD Matches", outImageLsdMatch);

        vector<char> mask2(matchesKnn.size(), 1);
        drawLineMatches(imgCur, keyLinesCur, imgRef, keyLinesRef, matchesKnn,
                        outImageLsdMatchGood, Scalar(0,0,255), Scalar(0,255,0), mask2);
        imshow("LSD Matches Knn", outImageLsdMatchGood);

        waitKey(200);

        imgCur.copyTo(imgRef);
        desCur.copyTo(desRef);
        keyLinesRef.swap(keyLinesCur);
    }

    return 0;
}
