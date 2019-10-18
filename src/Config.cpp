/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include "Config.h"
#include <cmath>
#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>


namespace se2lam
{

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

std::string Config::DataPath;

cv::Size Config::ImgSize;
cv::Mat Config::Tbc;   // camera extrinsic
cv::Mat Config::Tcb;   // inv of bTc
cv::Mat Config::Kcam;  // camera intrinsic
cv::Mat Config::Dcam;  // camera distortion
float Config::fx;
float Config::fy;
float Config::cx;
float Config::cy;

int Config::FPS = 10;
int Config::ImgStartIndex = 0;
int Config::ImgCount = 2000;

float Config::UpperDepth = 10000;  // 10m
float Config::LowerDepth = 500;    // 0.5m

float Config::MaxLinearSpeed = 2000;  // [mm/s]
float Config::MaxAngularSpeed = 200;  // [degree/s]

float Config::ScaleFactor = 1.2;   // scalefactor in detecting features
float Config::FeatureSigma = 0.5;  //! useless for now
int Config::MaxLevel = 6;          // level number of pyramid in detecting features
int Config::MaxFtrNumber = 500;    // max feature number to detect

float Config::ThHuber = 5.991;
int Config::LocalIterNum = 10;
int Config::GlobalIterNum = 12;
bool Config::LocalVerbose = false;
bool Config::GlobalVerbose = false;

float Config::OdoNoiseX = 1.0;
float Config::OdoNoiseY = 1.0;
float Config::OdoNoiseTheta = 1.0;
float Config::OdoUncertainX = 0.01;
float Config::OdoUncertainY = 0.01;
float Config::OdoUncertainTheta = 0.01;

float Config::PlaneMotionInfoXrot = 1e6;
float Config::PlaneMotionInfoYrot = 1e6;
float Config::PlaneMotionInfoZ = 1;

int Config::MaxLocalFrameNum = 20;         //! TODO
float Config::LocalFrameSearchRadius = 5;  //! TODO

float Config::MinScoreBest = 0.005;
float Config::MinMPMatchRatio = 0.05;
int Config::MinMPMatchNum = 15;
int Config::MinKPMatchNum = 30;
int Config::MinKFidOffset = 25;  // 回环间隔

bool Config::UsePrevMap = false;
bool Config::SaveNewMap = false;
bool Config::LocalizationOnly = false;
std::string Config::MapFileStorePath;
std::string Config::ReadMapFileName;
std::string Config::WriteMapFileName = "se2lam.map";
std::string Config::WriteTrajFileName = "se2lam.traj";

bool Config::NeedVisulization = true;
int Config::MappubScaleRatio = 300;

cv::Mat Config::PrjMtrxEye;
float Config::ThDepthFilter;  //! TODO

//! for debug
bool Config::LocalPrint = false;
bool Config::GlobalPrint = false;
bool Config::SaveMatchImage = false;
std::string Config::MatchImageStorePath = "/home/vance/output/se2/";


void Config::readConfig(const std::string& path)
{
    std::cout << "[Config][Info ] Loading config file..." << std::endl;

    DataPath = path;

    //! read camera config
    std::string camParaPath = path + "../se2_config/CamConfig.yml";
    cv::FileStorage camPara(camParaPath, cv::FileStorage::READ);
    assert(camPara.isOpened());

    cv::Mat K, D, rvec, tvec, r, R, t;
    float height, width;
    camPara["image_height"] >> height;
    camPara["image_width"] >> width;
    camPara["camera_matrix"] >> K;
    camPara["distortion_coefficients"] >> D;
    camPara["rvec_b_c"] >> rvec;  // rad
    camPara["tvec_b_c"] >> tvec;  // [mm]

    // convert double(CV_64FC1) to float(CV_32FC1)
    cv::Rodrigues(rvec, r);
    K.convertTo(Kcam, CV_32FC1);
    D.convertTo(Dcam, CV_32FC1);
    r.convertTo(R, CV_32FC1);
    tvec.convertTo(t, CV_32FC1);

    ImgSize.height = height;
    ImgSize.width = width;
    fx = Kcam.at<float>(0, 0);
    fy = Kcam.at<float>(1, 1);
    cx = Kcam.at<float>(0, 2);
    cy = Kcam.at<float>(1, 2);
    Tbc = cv::Mat::eye(4, 4, CV_32FC1);
    R.copyTo(Tbc.rowRange(0, 3).colRange(0, 3));
    t.copyTo(Tbc.rowRange(0, 3).col(3));

    Tcb = cv::Mat::eye(4, 4, CV_32FC1);
    cv::Mat Rcb = R.t();
    cv::Mat tcb = -Rcb * t;
    Rcb.copyTo(Tcb.rowRange(0, 3).colRange(0, 3));
    tcb.copyTo(Tcb.rowRange(0, 3).col(3));

    camPara.release();

    //! read setting
    std::string settingsPath = path + "../se2_config/Settings.yml";
    cv::FileStorage settings(settingsPath, cv::FileStorage::READ);
    assert(settings.isOpened());

    settings["fps"] >> FPS;
    settings["img_start_idx"] >> ImgStartIndex;
    settings["img_count"] >> ImgCount;

    settings["upper_depth"] >> UpperDepth;
    settings["lower_depth"] >> LowerDepth;

    settings["max_linear_speed"] >> MaxLinearSpeed;
    settings["max_angular_speed"] >> MaxAngularSpeed;

    settings["scale_factor"] >> ScaleFactor;
    settings["max_level"] >> MaxLevel;
    settings["max_feature_num"] >> MaxFtrNumber;
    settings["feature_sigma"] >> FeatureSigma;

    ThHuber = sqrt(static_cast<float>(settings["th_huber2"]));
    settings["local_iter"] >> LocalIterNum;
    settings["global_iter"] >> GlobalIterNum;
    settings["local_verbose"] >> LocalVerbose;
    settings["global_verbose"] >> GlobalVerbose;

    settings["odo_x_uncertain"] >> OdoUncertainX;
    settings["odo_y_uncertain"] >> OdoUncertainY;
    settings["odo_theta_uncertain"] >> OdoUncertainTheta;
    settings["odo_x_steady_noise"] >> OdoNoiseX;
    settings["odo_y_steady_noise"] >> OdoNoiseY;
    settings["odo_theta_steady_noise"] >> OdoNoiseTheta;
    if (!settings["plane_motion_xrot_info"].empty())
        settings["plane_motion_xrot_info"] >> PlaneMotionInfoXrot;
    if (!settings["plane_motion_yrot_info"].empty())
        settings["plane_motion_yrot_info"] >> PlaneMotionInfoYrot;
    if (!settings["plane_motion_z_info"].empty())
        settings["plane_motion_z_info"] >> PlaneMotionInfoZ;

    settings["max_local_frame_num"] >> MaxLocalFrameNum;
    settings["local_frame_search_radius"] >> LocalFrameSearchRadius;

    settings["gm_vcl_num_min_match_mp"] >> MinMPMatchNum;
    settings["gm_vcl_num_min_match_kp"] >> MinKPMatchNum;
    settings["gm_vcl_ratio_min_match_kp"] >> MinMPMatchRatio;
    settings["gm_dcl_min_kfid_offset"] >> MinKFidOffset;
    settings["gm_dcl_min_score_best"] >> MinScoreBest;

    settings["use_prev_map"] >> UsePrevMap;
    settings["save_new_map"] >> SaveNewMap;
    settings["localization_only"] >> LocalizationOnly;
    settings["map_file_store_path"] >> MapFileStorePath;
    settings["read_map_file_name"] >> ReadMapFileName;
    settings["write_map_file_name"] >> WriteMapFileName;
    settings["write_traj_file_name"] >> WriteTrajFileName;

    settings["need_visulization"] >> NeedVisulization;
    settings["mappub_scale_ratio"] >> MappubScaleRatio;

    PrjMtrxEye = Kcam * cv::Mat::eye(3, 4, CV_32FC1);
    settings["depth_filter_thresh"] >> ThDepthFilter;

    //! NOTE for debug
    settings["local_print"] >> LocalPrint;
    settings["global_print"] >> GlobalPrint;
    settings["save_match_image"] >> SaveMatchImage;
    settings["match_image_store_path"] >> MatchImageStorePath;

    settings.release();

    checkParamValidity();
}

bool Config::acceptDepth(float depth)
{
    return (depth >= LowerDepth && depth <= UpperDepth);
}

void Config::checkParamValidity()
{
    std::cout << "[Config][Info ] Camera paramters below:" << std::endl
              << " - Camera matrix: " << std::endl
              << " " << Kcam << std::endl
              << " - Camera distortion: " << std::endl
              << " " << Dcam << std::endl
              << " - Image size: " << std::endl
              << " " << ImgSize << std::endl
              << " - Camera extrinsic Tbc (Body to Camera): " << std::endl
              << " " << Tbc << std::endl
              << std::endl;
    std::cout << "[Config][Info ] Setting paramters below:" << std::endl
              << " - FPS: " << FPS << std::endl
              << " - Image start index: " << ImgStartIndex << std::endl
              << " - Image count: " << ImgCount << std::endl
              << " - MP upper depth[mm]: " << UpperDepth << std::endl
              << " - MP lower depth[mm]: " << LowerDepth << std::endl
              << " - Max linear speed[mm/s]: " << MaxLinearSpeed << std::endl
              << " - Max angular speed[deg/s]: " << MaxAngularSpeed << std::endl
              << " - Scale factor: " << ScaleFactor << std::endl
              << " - Max Pyramid level: " << MaxLevel << std::endl
              << " - Max feature extraction: " << MaxFtrNumber << std::endl
              << " - Threshold Huber kernel(square): " << ThHuber * ThHuber << std::endl
              << " - Local iteration time: " << LocalIterNum << std::endl
              << " - Global iteration time: " << GlobalIterNum << std::endl
              << " - Max local frame num: " << MaxLocalFrameNum << std::endl
              << " - Local KF search radius[mm]: " << LocalFrameSearchRadius << std::endl
              << " - Use prev map: " << UsePrevMap << std::endl
              << " - Save new map: " << SaveNewMap << std::endl
              << " - Map file store path: " << MapFileStorePath << std::endl
              << " - Map file name(read): " << ReadMapFileName << std::endl
              << " - Map file name(write): " << WriteMapFileName << std::endl
              << " - Trajectory file name(write): " << WriteTrajFileName << std::endl
              << " - Mappub scale ratio: " << MappubScaleRatio << std::endl
              << " - Need visulization: " << NeedVisulization << std::endl
              << " - Local print(debug): " << LocalPrint << std::endl
              << " - Global print(debug): " << GlobalPrint << std::endl
              << " - Save match images(debug): " << SaveMatchImage << std::endl
              << std::endl;

    assert(Kcam.data);
    assert(Kcam.rows == 3 && Kcam.cols == 3);
    assert(Dcam.data);
    assert(Tbc.data);
    assert(ImgSize.width > 0);
    assert(ImgSize.height > 0);

    assert(FPS > 0);
    assert(ImgStartIndex >= 0);
    assert(ImgCount > 0);
    assert(LowerDepth > 0.f && UpperDepth > 0.f && LowerDepth < UpperDepth);
    assert(MaxLinearSpeed > 0.f);
    assert(MaxAngularSpeed > 0.f);
    assert(ScaleFactor > 0.f);
    assert(MaxLevel >= 1);
    assert(MaxFtrNumber > 0);
    assert(ThHuber > 0.f);
    assert(LocalIterNum >= 0);
    assert(GlobalIterNum >= 0);
    assert(MaxLocalFrameNum > 0);
    assert(LocalFrameSearchRadius > 0.f);
    assert(MappubScaleRatio >= 1);
}

Se2::Se2() : x(0.f), y(0.f), theta(0.f), timeStamp(0.)
{
}

Se2::Se2(float _x, float _y, float _theta, double _time)
    : x(_x), y(_y), theta(normalize_angle(_theta)), timeStamp(_time)
{}

Se2::Se2(const Se2& that) : x(that.x), y(that.y), theta(that.theta), timeStamp(that.timeStamp)
{}

Se2::~Se2()
{}

Se2 Se2::inv() const
{
    float c = std::cos(theta);
    float s = std::sin(theta);
    return Se2(-c * x - s * y, s * x - c * y, -theta);
}

Se2 Se2::operator+(const Se2& that) const
{
    float c = std::cos(theta);
    float s = std::sin(theta);
    float _x = x + that.x * c - that.y * s;
    float _y = y + that.x * s + that.y * c;
    float _theta = normalize_angle(theta + that.theta);
    return Se2(_x, _y, _theta);
}

// Same as: that.inv() + *this
Se2 Se2::operator-(const Se2& that) const
{
    float dx = x - that.x;
    float dy = y - that.y;
    float dth = normalize_angle(theta - that.theta);

    float c = std::cos(that.theta);
    float s = std::sin(that.theta);
    return Se2(c * dx + s * dy, -s * dx + c * dy, dth);
}

Se2& Se2::operator=(const Se2& that)
{
    x = that.x;
    y = that.y;
    theta = that.theta;
    timeStamp = that.timeStamp;

    return *this;
}

cv::Mat Se2::toCvSE3() const
{
    float c = cos(theta);
    float s = sin(theta);

    return (cv::Mat_<float>(4, 4) << c, -s, 0, x, s, c, 0, y, 0, 0, 1, 0, 0, 0, 0, 1);
}


Se2& Se2::fromCvSE3(const cv::Mat& mat)
{
    float yaw = std::atan2(mat.at<float>(1, 0), mat.at<float>(0, 0));
    theta = normalize_angle(yaw);
    x = mat.at<float>(0, 3);
    y = mat.at<float>(1, 3);
    return *this;
}

}  // namespace se2lam
