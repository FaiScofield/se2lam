/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/


#include "OdoSLAM.h"
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

using namespace std;
using namespace cv;
using namespace Eigen;
namespace fs = boost::filesystem;

const char *vocFile = "/home/vance/dataset/se2/ORBvoc.bin";

void readImagesRK(const string& dataFolder, vector<string>& files) {
    files.clear();

    fs::path path(dataFolder);
    if (!fs::exists(path)) {
        cerr << "Data folder doesn't exist!" << endl;
        return;
    }

    fs::directory_iterator end_iter;
    for (fs::directory_iterator iter(path); iter != end_iter; ++iter) {
        if (fs::is_directory(iter->status()))
            continue;
        if (fs::is_regular_file(iter->status()))
            files.push_back(iter->path().string());
    }

    if (files.empty()) {
        cerr << "Not image data in the folder!" << endl;
        return;
    } else
        cout << "Read " << files.size() << " files in the folder." << endl;

    sort(files.begin(), files.end());
}


int main(int argc, char **argv)
{
    //! ROS Initialize
    ros::init(argc, argv, "test_vn");
    ros::start();

    if (argc < 2){
        cerr << "Usage: rosrun se2lam test_rk rk_dataPath" << endl;
        ros::shutdown();
        return -1;
    }

    se2lam::OdoSLAM system;

    system.setVocFileBin(vocFile);
    system.setDataPath(argv[1]);
    system.start();

    string fullOdoName = se2lam::Config::DataPath + "/odo_raw.txt";
    ifstream rec(fullOdoName);
    float x,y,theta;
    string line;

    ros::Rate rate(se2lam::Config::FPS);

    size_t n = static_cast<size_t>(se2lam::Config::ImgIndex);
    size_t i = 0;

    string imageFolder = se2lam::Config::DataPath + "/slamimg";
    vector<string> allImages;
    readImagesRK(imageFolder, allImages);
    n = allImages.size();
    for(; i < n && system.ok(); i++) {
        string fullImgName = allImages[i];
        cout << " ===> reading image: " << fullImgName << endl;
        Mat img = imread(fullImgName, CV_LOAD_IMAGE_GRAYSCALE);
        if (!img.data) {
            cerr << "No image data for image " << fullImgName << endl;
            continue;
        }
        std::getline(rec, line);
        istringstream iss(line);
        iss >> x >> y >> theta;

        system.receiveOdoData(x, y, theta);
        system.receiveImgData(img);

        rate.sleep();
    }
    cout << "Finish test..." << endl;

    system.requestFinish();
    system.waitForFinish();

    ros::shutdown();

    cout << "Rec close..." << endl;
    rec.close();
    cout << "Exit test..." << endl;
    return 0;

}

