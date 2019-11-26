/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include "Frame.h"
#include "Config.h"
#include "OdoSLAM.h"
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace se2lam;
namespace bf = boost::filesystem;

struct RK_IMAGE {
    RK_IMAGE(const string& s, const double& t) : fileName(s), timeStamp(t) {}

    string fileName;
    double timeStamp;

    // for map container
    bool operator <(const RK_IMAGE& that) const
    {
        return timeStamp < that.timeStamp;
    }
};

inline bool lessThen(const RK_IMAGE& r1, const RK_IMAGE& r2)
{
    return r1.timeStamp < r2.timeStamp;
}

void readImagesRK(const string& dataFolder, vector<RK_IMAGE>& files)
{
    bf::path path(dataFolder);
    if (!bf::exists(path)) {
        cerr << "[Main ][Error] Data folder doesn't exist!" << endl;
        ros::shutdown();
        return;
    }

    vector<RK_IMAGE> allImages;
    allImages.reserve(3000);
    bf::directory_iterator end_iter;
    for (bf::directory_iterator iter(path); iter != end_iter; ++iter) {
        if (bf::is_directory(iter->status()))
            continue;
        if (bf::is_regular_file(iter->status())) {
            // format: /frameRaw12987978101.jpg
            string s = iter->path().string();
            size_t i = s.find_last_of('w');
            size_t j = s.find_last_of('.');
            if (i == string::npos || j == string::npos)
                continue;
            auto t = atoll(s.substr(i + 1, j - i - 1).c_str());
            allImages.emplace_back(RK_IMAGE(s, t * 1e-6));
        }
    }

    if (allImages.empty()) {
        cerr << "[Main ][Error] Not image data in the folder!" << endl;
        ros::shutdown();
        return;
    } else {
        cout << "[Main ][Info ] Read " << allImages.size() << " image files in the folder." << endl;
    }

    //! 注意不能直接对string排序
    sort(allImages.begin(), allImages.end(), lessThen);
    files = allImages;
}


int main(int argc, char** argv)
{
    //! ROS Initialize
    ros::init(argc, argv, "se2_pipeline");
    ros::start();

    if (argc > 2) {
        cerr << "Usage: rosrun se2lam se2lamPipeline <dataPath>" << endl;
        ros::shutdown();
        exit(-1);
    }

    const string vocFile = string(argv[1]) + "../se2_config/ORBvoc.bin";

    OdoSLAM system;
    system.setVocFileBin(vocFile.c_str());
    system.setDataPath(argv[1]);
    system.start();

    string imageFolder = Config::DataPath + "/slamimg";
    vector<RK_IMAGE> allImages;
    readImagesRK(imageFolder, allImages);

    string odomRawFile = Config::DataPath + "/odo_raw.txt"; // [mm]
    ifstream rec(odomRawFile);
    if (!rec.is_open()) {
        cerr << "[Main ][Error] Please check if the file exists!" << endl;
        rec.close();
        ros::shutdown();
        exit(-1);
    }

    float x, y, theta;
    string line;
    size_t m = static_cast<size_t>(Config::ImgStartIndex);
    size_t n = static_cast<size_t>(Config::ImgCount);
    n = min(allImages.size(), n);

    ros::Rate rate(Config::FPS);
    for (size_t i = 0; i != n && system.ok(); ++i) {
        // 起始帧不为0的时候保证odom数据跟image对应
        if (i < m) {
            std::getline(rec, line);
            continue;
        }
        std::getline(rec, line);
        istringstream iss(line);
        iss >> x >> y >> theta;

        string fullImgName = allImages[i].fileName;
        Mat img = imread(fullImgName, CV_LOAD_IMAGE_GRAYSCALE);
        if (!img.data) {
            cerr << "[Main ][Error] No image data for image " << fullImgName << endl;
            continue;
        }

        system.receiveOdoData(x, y, theta);
        system.receiveImgData(img);

        rate.sleep();
    }
    cerr << "[Main ][Info ] Finish se2lamPipeline..." << endl;

    rec.close();
    system.requestFinish();  // 让系统给其他线程发送结束指令
    system.waitForFinish();

    ros::shutdown();

    cerr << "[Main ][Info ] System shutdown..." << endl;
    cerr << "[Main ][Info ] Exit test..." << endl;
    return 0;
}
