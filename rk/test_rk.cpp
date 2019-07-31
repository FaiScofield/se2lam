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
            allImages.push_back(RK_IMAGE(s, t));
        }
    }

    if (allImages.empty()) {
        cerr << "[Main] Not image data in the folder!" << endl;
        return;
    } else
        cout << "[Main] Read " << allImages.size() << " files in the folder." << endl;


    //! 注意不能直接对string排序
    sort(allImages.begin(), allImages.end(), lessThen);

    files.clear();
    for (int i = 0; i < allImages.size(); ++i)
        files.push_back(allImages[i].fileName);
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
    size_t m = static_cast<size_t>(se2lam::Config::ImgStartIndex);

    string imageFolder = se2lam::Config::DataPath + "/slamimg";
    vector<string> allImages;
    readImagesRK(imageFolder, allImages);
    n = min(allImages.size(), n);
    for(size_t i = 0; i < n && system.ok(); i++) {
        // 起始帧不为0的时候保证odom数据跟image对应
        if (i < m) {
            std::getline(rec, line);
            continue;
        }

        string fullImgName = allImages[i];
//        cout << "[Main] reading image: " << fullImgName << endl;
        Mat img = imread(fullImgName, CV_LOAD_IMAGE_GRAYSCALE);
        if (!img.data) {
            cerr << "[Main] No image data for image " << fullImgName << endl;
            continue;
        }
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

    cout << "[Main] Rec close..." << endl;
    rec.close();
    cout << "[Main] Exit test..." << endl;
    return 0;

}

