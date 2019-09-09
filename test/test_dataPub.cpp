/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>

#include <boost/filesystem.hpp>

using namespace cv;
using namespace std;
namespace bf = boost::filesystem;

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
    bf::path path(dataFolder);
    if (!bf::exists(path)) {
        cerr << "[main ] Data folder doesn't exist!" << endl;
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
            auto t = atoll(s.substr(i+1, j-i-1).c_str());
            allImages.push_back(RK_IMAGE(s, t));
        }
    }

    if (allImages.empty()) {
        cerr << "[main ] Not image data in the folder!" << endl;
        return;
    } else
        cout << "[main ] Read " << allImages.size() << " files in the folder." << endl;


    //! 注意不能直接对string排序
    sort(allImages.begin(), allImages.end(), lessThen);

    files.clear();
    for (size_t i = 0; i < allImages.size(); ++i)
        files.push_back(allImages[i].fileName);
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "datapub");
    ros::start();

    if (argc != 2) {
        cerr << "Usage: rosrun se2lam test_dataPab <dataset_folder> <Number_of_images>" << endl;
        ros::shutdown();
        return -1;
    }

    const char *ImgTopic = "/camera/image_gray";
    const char *OdoTopic = "/odo_raw";

    std::string path = argv[1];
    int N = min(100, atoi(argv[2]));  // Number of images

    string fullOdoName = path + "/odo_raw.txt";
    ifstream rec(fullOdoName);
    float x, y, theta;
    string line;

    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Publisher img_pub = it.advertise(ImgTopic, 1);
    ros::Publisher odo_pub = nh.advertise<geometry_msgs::Vector3Stamped>(OdoTopic, 100);


    ros::Rate rate(10);
    for (int i = 0; i < N && ros::ok(); i++) {
        string fullImgName = path + "/image/" + to_string(i) + ".bmp";
        Mat img = imread(fullImgName, CV_LOAD_IMAGE_GRAYSCALE);
        getline(rec, line);
        istringstream iss(line);
        iss >> x >> y >> theta;
        sensor_msgs::ImagePtr img_msg =
            cv_bridge::CvImage(std_msgs::Header(), "mono8", img).toImageMsg();
        geometry_msgs::Vector3Stamped odo_msg;

        img_msg->header.stamp = ros::Time::now();
        odo_msg.header.stamp = img_msg->header.stamp;
        odo_msg.vector.x = x;
        odo_msg.vector.y = y;
        odo_msg.vector.z = theta;

        img_pub.publish(img_msg);
        odo_pub.publish(odo_msg);
        rate.sleep();
    }

    ros::shutdown();
    return 0;
}
