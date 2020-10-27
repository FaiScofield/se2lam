#include "Config.h"
#include "ORBextractor.h"
#include "ORBmatcher.h"
#include "gms_matcher.h"
#include "test_funcs.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define IMAGE_WIDTH 752
#define IMAGE_HEIGHT 480
#define IMAGE_SUBREGION_NUM_COL 6
#define IMAGE_SUBREGION_NUM_ROW 4
#define IMAGE_BORDER 16
#define MAX_FEATURE_NUM 500
#define MAX_PYRAMID_LEVEL 3
#define PYRAMID_SCALE_FATOR 1.5

enum FeatureType {
    ORB = 0,
    FAST = 1,
    CV_FAST = 4,
    CV_ORB = 5,
    CV_SURF = 6,
    CV_SIFT = 7,
    CV_AKAZE = 8
};


using namespace std;
using namespace cv;
using namespace se2lam;

// drawMatches();


int main(int argc, char* argv[])
{
    //    const String keys = "{help h usage ? |      | print this message   }"
    //                        "{t type         | ORB  | value input type: ORB, CV_ORB, CV_SURF, CV_AKAZE}"
    //                        "{f folder       |<none>| data folder}"
    //                        "{b begin        |0     | start index for image sequence}"
    //                        "{e end          |-1    | end index for image sequence}";
    /// parse input arguments
    CommandLineParser parser(argc, argv,
                             "{type      t|ORB|value input type: ORB, CV_ORB, CV_SURF, CV_AKAZE}"
                             "{folder    f| |data folder}"
                             "{begin     a|0|start index for image sequence}"
                             "{end       e|-1|end index for image sequence}"
                             "{help      h|false|show help message}");

    if (argc < 2 || parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }

    const String str_type = parser.get<String>("type");
    const String str_folder = parser.get<String>("folder");
    const int beginIdx = parser.get<int>("begin");
    const int endIdx = parser.get<int>("end");
    const int feature = atoi(str_type.c_str());

    /// read images
    vector<String> vImgFiles;
    cv::glob(str_folder, vImgFiles);
    //    ReadImageNamesFromFolder(str_folder, vImgFiles);
    const int nImgSize = vImgFiles.size();
    if (nImgSize == 0)
        return -1;
    LOGI("Read " << nImgSize << " Images in the folder: " << str_folder);

    /// subregion params
    const int nImgWid = IMAGE_WIDTH;
    const int nImgHgt = IMAGE_HEIGHT;
    const int nImgBrd = IMAGE_BORDER;
    const int nSubregionNumCol = IMAGE_SUBREGION_NUM_COL;
    const int nSubregionNumRow = IMAGE_SUBREGION_NUM_ROW;
    const int nSubregionWid = (nImgWid - 2 * nImgBrd) / nSubregionNumCol;
    const int nSubregionHgt = (nImgHgt - 2 * nImgBrd) / nSubregionNumRow;
    vector<Rect> vSubregins(nSubregionNumCol * nSubregionNumRow);
    for (size_t r = 0; r < nSubregionNumRow; ++r) {
        const int y = nImgBrd + r * nSubregionHgt;
        int h = nSubregionHgt;
        if (r == nSubregionNumRow - 1) {
            h = max(h, nImgHgt - nImgBrd - y);
            LOGT("Subregin height of row " << r << " is: " << h);
        }

        for (size_t c = 0; c < nSubregionNumCol; ++c) {
            const int idx = r * nSubregionNumCol + c;
            const int x = nImgBrd + c * nSubregionWid;
            int w = nSubregionWid;
            if (c == nSubregionNumCol - 1) {
                w = max(h, nImgWid - nImgBrd - x);
                LOGT("Subregin width of col " << c << " is: " << w);
            }
            vSubregins[idx] = Rect(x, y, w, h);
        }
    }

    // auto pDetector = cv::ORB::create(MAX_FEATURE_NUM, PYRAMID_SCALE_FATOR, MAX_PYRAMID_LEVEL);
    Ptr<ORBextractor> pDetector =
        makePtr<ORBextractor>(ORBextractor(MAX_FEATURE_NUM, PYRAMID_SCALE_FATOR, MAX_PYRAMID_LEVEL));
    Ptr<DescriptorMatcher> pMatcher = DescriptorMatcher::create("BruteForce-Hamming");

    /// data
    size_t nKPs1, nKPs2;
    Mat image1, image2, imageOut;
    Mat descriptors1, descriptors2;
    vector<KeyPoint> vFeatures1, vFeatures2;
    vector<DMatch> vRoughMatches, vFineMatches;
    //    vFeatures1.reserve(MAX_FEATURE_NUM);
    //    vFeatures2.reserve(MAX_FEATURE_NUM);
    //    vRoughMatches.reserve(MAX_FEATURE_NUM);
    //    vFineMatches.reserve(MAX_FEATURE_NUM);

    //    const size_t x_start = IMAGE_WIDTH;
    cv::Mat K = cv::Mat::eye(3, 3, CV_32FC1);
    cv::Mat D = cv::Mat::zeros(4, 1, CV_32FC1);
    K.at<float>(0, 0) = 231.976033627644090;
    K.at<float>(1, 1) = 232.157224036901510;
    K.at<float>(0, 2) = 326.923920970539310;
    K.at<float>(1, 2) = 227.838488395348380;
    D.at<float>(0, 0) = -0.207406506100898;
    D.at<float>(1, 0) = 0.032194071349429;
    D.at<float>(2, 0) = 0.001120166051888;
    D.at<float>(3, 0) = 0.000859411522110;
    cout << endl << "K = " << K << endl;
    cout << "D = " << D << endl;
    for (int k = 0; k < nImgSize; ++k) {
        image2 = imread(vImgFiles[k], IMREAD_GRAYSCALE);
        if (image2.empty()) {
            LOGW("Empty image #" << k << ": " << vImgFiles[k]);
            continue;
        }

        Mat imgUn;
        cv::undistort(image2, imgUn, K, D);

        // pDetector->detect(image2, vFeatures2);
        // pDetector->compute(image2, vFeatures2, descriptors2);
        // pDetector->detectAndCompute(image2, noArray(), vFeatures2, descriptors2);
        (*pDetector)(imgUn, cv::noArray(), vFeatures2, descriptors2);

        if (k > 0) {
            pMatcher->match(descriptors1, descriptors2, vRoughMatches);
            drawMatches(image1, vFeatures1, imgUn, vFeatures2, vRoughMatches, imageOut);
            imshow("Match", imageOut);
            waitKey(200);
        }

        // swap data
        image1 = imgUn.clone();
        descriptors1 = descriptors2.clone();
        vFeatures1.swap(vFeatures2);
    }

    return 0;
}
