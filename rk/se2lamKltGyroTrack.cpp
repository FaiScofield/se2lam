//
// Created by lmp on 19-10-23.
//

/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include "Frame.h"
#include "OdoSLAM.h"
#include "readImageImu.h"
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

using namespace std;
using namespace cv;
using namespace Eigen;
namespace fs = boost::filesystem;

int imgRows = 240;
int imgCols = 320;
int t_offset = 0;
const string imu_list_name = "/home/lmp/klt_se2lam/mypicture/dibea6500_2/OdomRaw.txt";//IMU数据路径
const string imu_shape_name = "/home/lmp/klt_se2lam/mypicture/dibea6500_2/desk_imu_shape.txt";//IMU误差路径
const string img_list_name = "/home/lmp/klt_se2lam/mypicture/dibea6500_2/list.txt";//图片文件名

const char *vocFile = "/home/lmp/wse2lam/data/ORBvoc.bin";


int main(int argc, char **argv)
{
    //! ROS Initialize
    ros::init(argc, argv, "test_vn");
    ros::start();

    if (argc > 2){
        cerr << "Usage: rosrun se2lam test_rk rk_dataPath" << endl;
        ros::shutdown();
        return -1;
    }

    se2lam::OdoSLAM system;

    argv[1] = "/home/lmp/klt_se2lam/mypicture/dibea6500_2";
    system.setVocFileBin(vocFile);
    system.setDataPath(argv[1]);
    system.start();

    string fullOdoName = se2lam::Config::DataPath + "/odo_raw.txt";
    ifstream rec(fullOdoName);
    float x,y,theta;
    string line;

    ros::Rate rate(se2lam::Config::FPS);

    size_t n = static_cast<size_t>(se2lam::Config::ImgCount);
    size_t i = 0;

    Out_IMU_Data outImuData;
    readImageImu imageimu;
    int ctime = 0, ptime = 0, frame = 0;
    for(; i < n && system.ok(); i++) {


        cv::Mat img(imgRows, imgCols, CV_8U);
        ctime = imageimu.playback_mydate(img_list_name, t_offset, img);
        frame++;
        Mat gray;
        if (img.channels() != 1)
            cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);
        else
            gray = img.clone();

        if (ctime < 0) break;

        if (!gray.data) {
            cerr << "No image data for image "<< endl;
            continue;
        }

        //读取相邻两帧间的角度
        if (ptime != 0) {
            if (!imageimu.compute_theta_IMU(ptime, ctime, imu_list_name, imu_shape_name, outImuData)) break;
        }

        //读取IMU数据
        std::getline(rec, line);
        istringstream iss(line);
        iss >> x >> y >> theta;

        system.receiveOdoData(outImuData.x, outImuData.y, outImuData.theta);
        system.receiveImgData(gray);
        system.receiveImuTheta(outImuData.dtheta,ctime);

        rate.sleep();

        ptime = ctime;
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
