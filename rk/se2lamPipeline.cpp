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
namespace bf = boost::filesystem;

const char* vocFile = "/home/vance/dataset/se2/ORBvoc.bin";

struct RK_IMAGE {
    RK_IMAGE(const string& s, const float& t) : fileName(s), timeStamp(t) {}

    string fileName;
    float timeStamp;
};

bool lessThen(const RK_IMAGE& r1, const RK_IMAGE& r2)
{
    return r1.timeStamp < r2.timeStamp;
}

void readImagesRK(const string& dataFolder, vector<RK_IMAGE>& files)
{
    bf::path path(dataFolder);
    if (!bf::exists(path)) {
        cerr << "[main ] Data folder doesn't exist!" << endl;
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
            allImages.emplace_back(RK_IMAGE(s, t/1000000.f));
        }
    }

    if (allImages.empty()) {
        cerr << "[main ] Not image data in the folder!" << endl;
        ros::shutdown();
        return;
    } else {
        cout << "[main ] Read " << allImages.size() << " files in the folder." << endl;
    }

    //! 注意不能直接对string排序
    sort(allImages.begin(), allImages.end(), lessThen);
    files = allImages;
}

void readOdomsRK(const string& odomFile, vector<se2lam::Se2>& odoData)
{
    ifstream rec(odomFile);
    if (!rec.is_open()) {
        cerr << "[main ] Error in opening file: " << odomFile << endl;
        rec.close();
        ros::shutdown();
        return;
    }

    odoData.clear();
    string line;
    while (std::getline(rec, line) && !line.empty()) {
        istringstream iss(line);
        se2lam::Se2 odo;
        iss >> odo.timeStamp >> odo.x >> odo.y >> odo.theta;  // theta 可能会比超过pi,使用前要归一化
        odoData.emplace_back(odo);
    }
    rec.close();

    if (odoData.empty()) {
        cerr << "[main ] Not odom data in the file!" << endl;
        ros::shutdown();
        return;
    } else {
        cout << "[main ] Read " << odoData.size() << " datas from the file." << endl;
    }
}

void dataAlignment(const vector<RK_IMAGE>& allImages, const vector<se2lam::Se2>& allOdoms,
                   map<RK_IMAGE, queue<se2lam::Se2>>& dataAligned)
{
    auto iter = allOdoms.begin();
    for (size_t i = 0, iend = allImages.size(); i < iend; ++i) {
        float imgTime = allImages[i].timeStamp;
        queue<se2lam::Se2> odoDeq;
        while (iter != allOdoms.end()) {
            if (iter->timeStamp <= imgTime) {
                odoDeq.emplace_back(*iter);
                iter++;
            } else {
                break;
            }
        }
        dataAligned.insert(make_pair<RK_IMAGE, queue<se2lam::Se2>>(allImages[i], odoDeq));
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

    se2lam::OdoSLAM system;
    system.setVocFileBin(vocFile);
    system.setDataPath(argv[1]);
    system.start();

    string imageFolder = se2lam::Config::DataPath + "/slamimg";
    vector<RK_IMAGE> allImages;
    readImagesRK(imageFolder, allImages);

    string odomRawFile = se2lam::Config::DataPath + "/OdomRaw.txt";
    vector<se2lam::Se2> allOdoms;
    readOdomsRK(odomRawFile, allOdoms);

    map<RK_IMAGE, queue<se2lam::Se2>> dataAligned;
    dataAlignment(allImages, allOdoms, dataAligned);
    assert(allImages.size() == dataAligned.size());

    size_t m = static_cast<size_t>(se2lam::Config::ImgStartIndex);
    size_t n = static_cast<size_t>(se2lam::Config::ImgCount);
    n = min(allImages.size(), n);

    ros::Rate rate(se2lam::Config::FPS);
    for (size_t i = m; i < n && system.ok(); ++i) {
        string fullImgName = allImages[i].fileName;
        float imgTime = allImages[i].timeStamp;
        Mat img = imread(fullImgName, CV_LOAD_IMAGE_GRAYSCALE);
        if (!img.data) {
            cerr << "[main ] No image data for image " << fullImgName << endl;
            continue;
        }

        system.receiveOdoData(dataAligned[allImages[i]]);
        system.receiveImgData(img, imgTime);

        rate.sleep();
    }
    cout << "[main ] Finish test_rk..." << endl;

    system.requestFinish();  // 让系统给其他线程发送结束指令
    system.waitForFinish();

    ros::shutdown();

    cout << "[main ] System shutdown..." << endl;
    cout << "[main ] Exit test..." << endl;
    return 0;
}
