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

const char *vocFile = "/home/vance/dataset/se2/ORBvoc.bin";

int main(int argc, char **argv)
{
    //! Initialize
    ros::init(argc, argv, "test_vn");
    ros::start();

    if (argc != 2) {
        cerr << "Usage: rosrun se2lam test_vn dataPath" << endl;
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

    int n = se2lam::Config::ImgIndex;
    int m = se2lam::Config::ImgStartIndex;

    for(int i=m; i < m+n && system.ok(); i++) {
        string fullImgName = se2lam::Config::DataPath + "/image/" + to_string(i) + ".bmp";
        Mat img = imread(fullImgName, CV_LOAD_IMAGE_GRAYSCALE);
        std::getline(rec, line);
        istringstream iss(line);
        iss >> x >> y >> theta;

        system.receiveOdoData(x, y, theta);
        system.receiveImgData(img);

        rate.sleep();
    }
    cout << "[Main] Finish test..." << endl;

    system.requestFinish();
    system.waitForFinish();

    ros::shutdown();

    cout << "[Main] Ros close..." << endl;
    rec.close();
    cout << "[Main] Exit test..." << endl;
    return 0;

}

