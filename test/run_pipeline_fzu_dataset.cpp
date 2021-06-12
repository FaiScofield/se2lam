#include "Config.h"
#include "OdoSLAM.h"
#include <algorithm>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace se2lam;
namespace bf = boost::filesystem;

void readImagesFZU(const string& strImagePath, vector<string>& vstrImages, vector<double>& vTimeStamps);


int main(int argc, char** argv)
{
    //! ROS Initialize
    ros::init(argc, argv, "run_pipeline_fzu_dataset");
    ros::start();

    const string vocFile = "/home/vance/slam_ws/ORB_SLAM3/Vocabulary/ORBvoc.bin";
    string path = "/home/vance/dataset/fzu/201224_hall_1/";
    if (argc < 2) {
        cout << "Usage: rosrun se2lam " << argv[0] << " <dataPath>" << endl;
        cout << "use default path: " << path << endl;
    } else {
        path = string(argv[1]);
        cout << "set path to: " << path << endl;
    }

    OdoSLAM system;
    system.setVocFileBin(vocFile.c_str());
    system.setDataPath(path.c_str());
    system.start();

    string imageFolder = Config::DataPath + "/image/";
    vector<string> allImages;
    vector<double> timestamps;
    readImagesFZU(imageFolder, allImages, timestamps);

    string odomRawFile = Config::DataPath + "/odom_sync.txt";  // [m]
    ifstream rec(odomRawFile);
    if (!rec.is_open()) {
        cerr << "[Main ][Error] Please check if the 'odom_sync.txt' file exists!" << endl;
        rec.close();
        ros::shutdown();
        exit(-1);
    }

    double time;
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
        iss >> time >> x >> y >> theta;

        string fullImgName = allImages[i];
        Mat img = imread(fullImgName, CV_LOAD_IMAGE_GRAYSCALE);
        if (!img.data) {
            cerr << "[Main ][Error] No image data for image " << fullImgName << endl;
            continue;
        }

        system.receiveOdoData(x, y, theta, time);
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



void readImagesFZU(const string& strImagePath, vector<string>& vstrImages, vector<double>& vTimeStamps)
{
    bf::path path(strImagePath);
    if (!bf::exists(path)) {
        cerr << "[Main ][Error] Data folder doesn't exist!" << endl;
        return;
    }

    vector<pair<string, double>> vstrImgTime;
    vstrImgTime.reserve(2000);

    bf::directory_iterator end_iter;
    for (bf::directory_iterator iter(path); iter != end_iter; ++iter) {
        if (bf::is_directory(iter->status()))
            continue;
        if (bf::is_regular_file(iter->status())) {
            // format: /s.ns.jpg
            string s = iter->path().string();
            size_t i = s.find_last_of('/');
            size_t j = s.find_last_of('.');
            if (i == string::npos || j == string::npos)
                continue;
            string strTimeStamp = s.substr(i + 1, j);
            double t = atof(strTimeStamp.c_str());
            vstrImgTime.emplace_back(s, t); // us
        }
    }

    sort(vstrImgTime.begin(), vstrImgTime.end(),
         [&](const pair<string, double>& lf, const pair<string, double>& rf) {
            return lf.second < rf.second;}
    );

    const size_t numImgs = vstrImgTime.size();
    vTimeStamps.resize(numImgs);
    vstrImages.resize(numImgs);
    for (size_t k = 0; k < numImgs; ++k) {
        vstrImages[k] = vstrImgTime[k].first;
        vTimeStamps[k] = vstrImgTime[k].second;
    }

    if (vstrImages.empty()) {
        cerr << "[Main ][Error] Not image data in the folder!" << endl;
        return;
    } else {
        cout << "[Main ][Info ] Read " << vstrImages.size() << " image files in the folder." << endl;
    }
}
