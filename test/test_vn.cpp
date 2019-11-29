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
        cerr << "Usage: rosrun se2lam test_vn <dataPath>" << endl;
        ros::shutdown();
        exit(-1);
    }

    se2lam::OdoSLAM system;
    system.setVocFileBin(vocFile);
    system.setDataPath(argv[1]);
    system.start();

    string odomRawFile = se2lam::Config::DataPath + "/odo_raw.txt";
    ifstream rec(odomRawFile);
    if (!rec.is_open()) {
        cerr << "[Main ][Error] Please check if the file exists!" << endl;
        rec.close();
        ros::shutdown();
        exit(-1);
    }

    float x, y, theta;
    string line;
    int n = se2lam::Config::ImgCount;
    int m = se2lam::Config::ImgStartIndex;

    ros::Rate rate(se2lam::Config::FPS);
    for(int i = 0; i < n && system.ok(); ++i) {
        // 起始帧不为0的时候保证odom数据跟image对应
        if (i < m) {
            std::getline(rec, line);
            continue;
        }
        std::getline(rec, line);
        istringstream iss(line);
        iss >> x >> y >> theta;

        string fullImgName = se2lam::Config::DataPath + "/image/" + to_string(i) + ".bmp";
        Mat img = imread(fullImgName, CV_LOAD_IMAGE_GRAYSCALE);
        if (img.empty()) {
            cerr << "[Main ][Error] No image data for image " << fullImgName << endl;
            continue;
        }

        system.receiveOdoData(x, y, theta);
        system.receiveImgData(img);

        rate.sleep();
    }
    cout << "[Main ][Info ] Finish test_vn..." << endl;

    rec.close();
    system.requestFinish();  // 让系统给其他线程发送结束指令
    system.waitForFinish();

    ros::shutdown();

    cerr << "[Main ][Info ] System shutdown..." << endl;
    cerr << "[Main ][Info ] Exit test..." << endl;
    return 0;
}

