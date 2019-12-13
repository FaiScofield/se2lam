/**
 * This file is part of se2lam
 *
 * Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
 */


#include "OdoSLAM.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace Eigen;

int main(int argc, char** argv)
{
    //! Initialize
    ros::init(argc, argv, "test_vn");
    ros::start();

    if (argc < 2) {
        cerr << "Usage: rosrun se2lam test_vn <dataPath>" << endl;
        ros::shutdown();
        return -1;
    }

    const string vocFile = string(argv[1]) + "../config/ORBvoc.bin";

    se2lam::OdoSLAM system;
    system.setVocFileBin(vocFile.c_str());
    system.setDataPath(argv[1]);
    system.start();

    string fullOdoName = se2lam::Config::DataPath + "/odo_raw.txt";
    ifstream rec(fullOdoName);
    if (!rec.is_open()) {
        cerr << "[Main ][Error] Please check if the 'odo_raw.txt' file exists!" << endl;
        rec.close();
        ros::shutdown();
        exit(-1);
    }

    float x, y, theta;
    string line;
    size_t m = static_cast<size_t>(se2lam::Config::ImgStartIndex);
    size_t n = static_cast<size_t>(se2lam::Config::ImgCount);

    ros::Rate rate(se2lam::Config::FPS);
    for (size_t i = 0; i < n && system.ok(); i++) {
        if (i < m) {
            std::getline(rec, line);
            continue;
        }
        std::getline(rec, line);
        istringstream iss(line);
        iss >> x >> y >> theta;

        string fullImgName = se2lam::Config::DataPath + "/image/" + to_string(i) + ".bmp";
        Mat img = imread(fullImgName, CV_LOAD_IMAGE_GRAYSCALE);
        if (!img.data) {
            cerr << "[Main ][Warni] No image data for image " << fullImgName << endl;
            continue;
        }

        system.receiveOdoData(x, y, theta);
        system.receiveImgData(img);

        rate.sleep();
    }
    cerr << "[Main ][Info ] Finish test..." << endl;

    rec.close();
    system.requestFinish();
    system.waitForFinish();

    ros::shutdown();

    cerr << "[Main ][Info ] Exit test..." << endl;
    return 0;
}
