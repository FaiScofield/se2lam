#include <iostream>
#include <vector>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/filesystem.hpp>

using namespace std;
using namespace cv;
namespace bf = boost::filesystem;


vector<string> readFolderFiles(const string& folder) {
    vector<string> files;

    bf::path folderPath(folder);
    if (!bf::exists(folderPath))
        return files;

    bf::directory_iterator end_iter;
    for (bf::directory_iterator iter(folderPath); iter != end_iter; ++iter) {
        if (bf::is_directory(iter->status()))
            continue;

        if (bf::is_regular_file(iter->status()))
            files.push_back(iter->path().string());
    }

    return files;
}


int main(int argc, char *argv[])
{
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <calib_image_folder>" << endl;
        return -1;
    }
    string folder(argv[1]);
    if (!bf::exists(bf::path(folder))) {
        cerr << "Folder does'n eixst! - " << folder << endl;
        return -1;
    }
    vector<string> images = readFolderFiles(folder);
    if (images.size() < 1) {
        cerr << " No images in this folder!" << endl;
        return -1;
    }

    float K[3][3] = {1., 0., 0., 0, 2., 0., 0, 0, 1}; //摄像机内参数矩阵K

    cv::Mat mK = cv::Mat(3, 3, CV_32FC1, K); //内参数K Mat类型变量

    Mat M(640, 480, CV_8UC1);

    Mat Kb =  (cv::Mat_<double>(3,3) <<
               4.2328803236708745e+02, 0., 3.1286415405574957e+02,
               0., 4.2328803236708745e+02, 2.5238408563665234e+02,
               0., 0., 1.);

    Mat Db =  (cv::Mat_<double>(5,1) <<
               2.0038273473223407e-02, -2.5165970048240040e-02,
               9.7101647177130195e-03, -3.6291586958263873e-03, 0.);

    cout << "640*480 内参:\n" << Kb << endl;
    cout << "640*480 畸变参数:\n" << Db << endl;

    Mat Ks =  (cv::Mat_<double>(3,3) <<
               219.9359613169054, 0., 161.5827136112504,
               0., 219.4159055585876e+02, 117.7128673795551,
               0., 0., 1. );

    Mat Ds =  (cv::Mat_<double>(5,1) <<
               0.064610443232716, -0.086814339668420,
               -0.0009238134627751219, 0.0005452823230733891, 0.);

    Mat Ks2; Kb.copyTo(Ks2);
    Ks2.at<double>(0,0) *= 1./2;
    Ks2.at<double>(0,2) *= 1./2;
    Ks2.at<double>(1,1) *= 1./2;
    Ks2.at<double>(1,2) *= 1./2;

    Mat Ds2 =  Db.clone();
    Ds2.at<double>(0,0) *= 1./4;    // k1
    Ds2.at<double>(1,0) *= 1./16;   // k2
    Ds2.at<double>(2,0) *= 1./2;    // p1
    Ds2.at<double>(3,0) *= 1./2;    // p1

    cout << "320*240 内参:\n" << Ks2 << endl;
    cout << "320*240 畸变参数:\n" << Ds2 << endl;

    Mat imageUndistorted;
    for (const auto& img : images) {
        Mat image = imread(img, IMREAD_GRAYSCALE);
        undistort(image, imageUndistorted, Kb, Db);
        imshow("Original Image (640x480)", image);
        imshow("Undistorted Image (640x480)", imageUndistorted);

        Mat sImage, sImageUndis;
        resize(image, sImage, Size(image.cols/2, image.rows/2));
        undistort(sImage, sImageUndis, Ks, Ds);
        imshow("Original Image (320x240)", sImage);
        imshow("Undistorted Image (320x240)", sImageUndis);

        Mat sImageUndis2;
        undistort(sImage, sImageUndis2, Ks2, Ds2);
        imshow("Undistorted Image2 (320x240)", sImageUndis2);

        waitKey(0);
    }


    return 0;
}
