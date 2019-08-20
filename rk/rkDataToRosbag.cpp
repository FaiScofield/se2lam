#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <sensor_msgs/Image.h>

#include <opencv2/highgui/highgui.hpp>

#include <boost/filesystem.hpp>
#include <fstream>
#include <sstream>

using namespace std;
namespace bf = boost::filesystem;

//string dataPath = "/home/vance/dataset/rk/dibeaDataSet/dibea6500VI_1/";
string dataPath = "/home/vance/dataset/rk/calibration/extrinsic-0819/extrinsic_good/";
string bagFile = "/home/vance/dataset/rosbags/rk_extrinsic_1.bag";

struct RK_IMAGE {
    RK_IMAGE(const string &s, const long long int t) : fileName(s), timestamp(t) {}

    string fileName;
    long long int timestamp;
};

struct OdomRaw {
    long long int timestamp;
    double x, y, theta;
    double linearVelX, AngularVelZ;
    double deltaDistance, deltaTheta;

    OdomRaw()
    {
        timestamp = 0;
        x = y = theta = 0.0;
        linearVelX = AngularVelZ = 0.0;
        deltaDistance = deltaTheta = 0.0;
    }
};

vector<OdomRaw> readOdomeRaw(const string &file)
{
    vector<OdomRaw> result;

    ifstream reader;
    reader.open(file.c_str());
    if (!reader) {
        fprintf(stderr, "%s file open error!\n", file.c_str());
        return result;
    }

    // get data
    string lineData;
    while (getline(reader, lineData) && !lineData.empty()) {
        OdomRaw oraw;
        stringstream ss(lineData);
        ss >> oraw.timestamp >> oraw.x >> oraw.y >> oraw.theta >> oraw.linearVelX >>
            oraw.AngularVelZ >> oraw.deltaDistance >> oraw.deltaTheta;
        result.push_back(oraw);
    }

    reader.close();

    return result;
}

bool lessThen(const RK_IMAGE &r1, const RK_IMAGE &r2)
{
    return r1.timestamp < r2.timestamp;
}

vector<RK_IMAGE> readImagesRK(const string &dataFolder)
{
    vector<RK_IMAGE> allImages;

    bf::path path(dataFolder);
    if (!bf::exists(path)) {
        cerr << "[main ] Data folder doesn't exist!" << endl;
        return allImages;
    }

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
            allImages.push_back(RK_IMAGE(s, t));
        }
    }

    if (allImages.empty()) {
        cerr << "[main ] Not image data in the folder!" << endl;
        return allImages;
    } else {
        cout << "[main ] Read " << allImages.size() << " files in the folder." << endl;
    }

    //! 注意不能直接对string排序
    sort(allImages.begin(), allImages.end(), lessThen);

    return allImages;
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "rkDataToRosbag");
    ros::NodeHandle nh;

    ROS_INFO("Start converting data, please wait...");

    vector<RK_IMAGE> fullImages = readImagesRK(dataPath + "slamimg");
    vector<OdomRaw> fullOdoms = readOdomeRaw(dataPath + "OdomRaw.txt");
    ROS_INFO("Get %ld image and %ld odom messages.", fullImages.size(), fullOdoms.size());

    rosbag::Bag bag;
    bag.open(bagFile, rosbag::bagmode::Write);

    cv_bridge::CvImage bridge;
    for (auto &image : fullImages) {
        double t = image.timestamp * 1e-6;
        ros::Time time(t);

        cv::Mat imgMat = cv::imread(image.fileName, CV_LOAD_IMAGE_UNCHANGED);
        if (!imgMat.data) {
            cerr << "No data in the image: " << image.fileName << endl;
            continue;
        }
//        cv::imshow("current image", imgMat);
        bridge.image = imgMat;
        bridge.encoding = sensor_msgs::image_encodings::MONO8;
        bridge.header.stamp = time;

        bag.write("/image", time, bridge.toImageMsg());
//        cv::waitKey(190);
    }

    vector<double> lastOdom(4, 0);  // time, x, y, theta
    bool firstFrame = true;
    for (auto &odom : fullOdoms) {
        nav_msgs::Odometry odomMsgs;
        double t = odom.timestamp * 1e-6;
        ros::Time time(t);
        odomMsgs.header.stamp = time;
        odomMsgs.header.frame_id = "odom";
        odomMsgs.child_frame_id = "base_link";
        odomMsgs.pose.pose.position.x = odom.x;
        odomMsgs.pose.pose.position.y = odom.y;
        odomMsgs.pose.pose.position.z = 0.0;
        odomMsgs.pose.pose.orientation.x = 0.0;
        odomMsgs.pose.pose.orientation.y = 0.0;
        odomMsgs.pose.pose.orientation.z = odom.theta;

        if (firstFrame) {
            odomMsgs.twist.twist.linear.x = 0.0;
            odomMsgs.twist.twist.linear.y = 0.0;
            odomMsgs.twist.twist.angular.z = 0.0;
            firstFrame = false;
        } else {
            double dt = (odom.timestamp - lastOdom[0]) * 1e6;
            odomMsgs.twist.twist.linear.x = (odom.x - lastOdom[1]) / dt;
            odomMsgs.twist.twist.linear.y = (odom.y - lastOdom[2]) / dt;
            odomMsgs.twist.twist.angular.z = (odom.theta - lastOdom[3]) / dt;
        }

        lastOdom[0] = odom.timestamp;
        lastOdom[1] = odom.x;
        lastOdom[2] = odom.y;
        lastOdom[3] = odom.theta;

        bag.write("/odom", time, odomMsgs);
    }
    bag.close();

    ROS_INFO(" END ");
    ROS_INFO("Successfully creat the bag: %s", bagFile.c_str());

    return 0;
}
