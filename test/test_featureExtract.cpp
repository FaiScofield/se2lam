#include "Config.h"
#include "ORBextractor.h"
#include "ORBmatcher.h"
#include "test_funcs.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define IMAGE_WIDTH 752
#define IMAGE_HEIGHT 480
#define IMAGE_SUBREGION_NUM_COL 6
#define IMAGE_SUBREGION_NUM_ROW 4
#define IMAGE_BORDER 16
#define MAX_FEATURE_NUM 500
#define MAX_PYRAMID_LEVEL 3
#define PYRAMID_SCALE_FATOR 1.2

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

//drawMatches();


int main(int argc, char* argv[])
{
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
    const size_t nImgSize = vImgFiles.size();
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

    /// feature extractor
//    void* pExtractor;
//    switch (feature) {
//    case 0:
//        cv::ORB* pOrbExtractor = new cv::ORB(MAX_FEATURE_NUM, PYRAMID_SCALE_FATOR,
//         MAX_PYRAMID_LEVEL);
//        pExtractor = (void*)pOrbExtractor;
//        break;
//    case 1:
//    default:
//       break;
//    }
    Ptr<FeatureDetector> pDetector = cv::ORB::create(MAX_FEATURE_NUM, PYRAMID_SCALE_FATOR, MAX_PYRAMID_LEVEL);
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
    for (int k = 0; k < nImgSize; ++k) {
        image2 = imread(vImgFiles[k], IMREAD_GRAYSCALE);
        if (image2.empty()) {
            LOGW("Empty image #" << k << ": " << vImgFiles[k]);
            continue;
        }

        pDetector->detect(image2, vFeatures2);
        pDetector->compute(image2, vFeatures2, descriptors2);

        if (k > 0) {
            pMatcher->match(descriptors1, descriptors2, vRoughMatches);
            drawMatches(image1, vFeatures1, image2, vFeatures2, vRoughMatches, imageOut);
            imshow("Match", imageOut);
            waitKey(200); 
        }

        // swap data
        image1 = image2.clone();
        descriptors1 = descriptors2.clone();
        vFeatures1.swap(vFeatures2);
    }

    return 0;
}
