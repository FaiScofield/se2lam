/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include "Config.h"
#include "OdoSLAM.h"
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace se2lam;
namespace bf = boost::filesystem;

const char* vocFile = "/home/vance/dataset/se2/ORBvoc.bin";

struct RK_IMAGE {
    RK_IMAGE(const string& s, const float& t) : fileName(s), timeStamp(t) {}

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
        cerr << "[main ][Error] Data folder doesn't exist!" << endl;
        ros::shutdown();
        return;
    }

    vector<RK_IMAGE> allImages;
    bf::directory_iterator end_iter;
    for (bf::directory_iterator iter(path); iter != end_iter; ++iter) {
        if (bf::is_directory(iter->status()))
            continue;
        if (bf::is_regular_file(iter->status())) {
            // format: /frameRaw12987978101.jpg
            string s = iter->path().string();
            auto i = s.find_last_of('w');
            auto j = s.find_last_of('.');
            auto t = atoll(s.substr(i + 1, j - i - 1).c_str());
            allImages.emplace_back(RK_IMAGE(s, t * 1e-6));
        }
    }

    if (allImages.empty()) {
        cerr << "[main ][Error] Not image data in the folder!" << endl;
        ros::shutdown();
        return;
    } else {
        cout << "[main ][Info ] Read " << allImages.size() << " image files in the folder." << endl;
    }

    //! 注意不能直接对string排序
    sort(allImages.begin(), allImages.end(), lessThen);
    files = allImages;
}

void readOdomsRK(const string& odomFile, vector<Se2>& odoData)
{
    ifstream rec(odomFile);
    if (!rec.is_open()) {
        cerr << "[main ][Error] Error in opening file: " << odomFile << endl;
        rec.close();
        ros::shutdown();
        return;
    }

    odoData.clear();
    string line;
    while (std::getline(rec, line) && !line.empty()) {
        istringstream iss(line);
        Se2 odo;
        iss >> odo.timeStamp >> odo.x >> odo.y >> odo.theta;  // theta 可能会比超过pi,使用前要归一化
        odo.timeStamp *= 1e-6;
        odo.x *= 1000.f;
        odo.y *= 1000.f;
        odoData.emplace_back(odo);
    }
    rec.close();

    if (odoData.empty()) {
        cerr << "[main ][Error] Not odom data in the file!" << endl;
        ros::shutdown();
        return;
    } else {
        cout << "[main ][Info ] Read " << odoData.size() << " odom datas from the file." << endl;
    }
}

void dataAlignment(vector<RK_IMAGE>& allImages, const vector<Se2>& allOdoms,
                   map<RK_IMAGE, vector<Se2>>& dataAligned)
{
    // 去除掉没有odom数据的图像
    Se2 firstOdo = allOdoms[0];
    auto iter = allImages.begin();
    for (auto iend = allImages.end(); iter != iend; ++iter) {
        if (iter->timeStamp < firstOdo.timeStamp)
            continue;
        else
            break;
    }
    allImages.erase(allImages.begin(), iter);
    cout << "[main ][Info ] Cut some images for timestamp too earlier, now image size: "
         << allImages.size() << endl;

    // 数据对齐
    auto ite = allOdoms.begin(), its = allOdoms.begin();
    for (size_t i = 0, iend = allImages.size(); i < iend; ++i) {
        double imgTime = allImages[i].timeStamp;

        while (ite != allOdoms.end()) {
            if (ite->timeStamp <= imgTime)
                ite++;
            else
                break;
        }
        vector<Se2> odoDeq = vector<Se2>(its, ite);
        its = ite;

        dataAligned[allImages[i]] = odoDeq;
    }
}

int main(int argc, char** argv)
{
    //! ROS Initialize
    ros::init(argc, argv, "test_vn");
    ros::start();

    if (argc < 2) {
        cerr << "Usage: rosrun se2lam test_rk rk_dataPath" << endl;
        ros::shutdown();
        return -1;
    }

    OdoSLAM system;
    system.setVocFileBin(vocFile);
    system.setDataPath(argv[1]);
    system.start();

    string imageFolder = Config::DataPath + "/slamimg";
    vector<RK_IMAGE> allImages;
    readImagesRK(imageFolder, allImages);

//    string odomRawFile = Config::DataPath + "/OdomRaw.txt";
//    vector<Se2> allOdoms;
//    readOdomsRK(odomRawFile, allOdoms);

//    map<RK_IMAGE, vector<Se2>> dataAligned;
//    dataAlignment(allImages, allOdoms, dataAligned);
//    assert(allImages.size() == dataAligned.size());

    string odomRawFile = Config::DataPath + "/odo_raw.txt";
    ifstream rec(odomRawFile);
    if (!rec.is_open()) {
        cerr << "[main ][Error] Please check file if exists!" << endl;
        rec.close();
        ros::shutdown();
        return -1;
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
        double imgTime = allImages[i].timeStamp;
        Mat img = imread(fullImgName, CV_LOAD_IMAGE_GRAYSCALE);
        if (!img.data) {
            cerr << "[main ][Error] No image data for image " << fullImgName << endl;
            continue;
        }

//        system.receiveOdoDatas(dataAligned[allImages[i]]);
//        system.receiveImgData(img, imgTime);
        system.receiveOdoData(x, y, theta);
        system.receiveImgData(img);

        rate.sleep();
    }
    cout << "[main ][Info ] Finish test_rk..." << endl;

    system.requestFinish();  // 让系统给其他线程发送结束指令
    system.waitForFinish();

    ros::shutdown();

    cout << "[main ][Info ] System shutdown..." << endl;
    cout << "[main ][Info ] Exit test..." << endl;
    return 0;
}
