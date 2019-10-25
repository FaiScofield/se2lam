/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/


#ifndef FRAME_H
#define FRAME_H

#include "Config.h"
#include "ORBextractor.h"
#include <memory>
#include <mutex>
#include <vector>

namespace se2lam
{

// class MapPoint;

struct PreSE2 {
public:
    double meas[3];
    double cov[9];  // 3*3, RowMajor
};

const int FRAME_GRID_ROWS = 48;  // default 48 for 480, 23
const int FRAME_GRID_COLS = 64;  // default 64 for 640, 31

class Frame
{
public:
    Frame();
    Frame(const cv::Mat& imgGray, const Se2& odo, ORBextractor* extractor, const cv::Mat& K,
          const cv::Mat& distCoef);
    Frame(const cv::Mat& imgGray, const double& time, const Se2& odo, ORBextractor* extractor,
          const cv::Mat& K, const cv::Mat& distCoef);

    Frame(const Frame& f);
    Frame& operator=(const Frame& f);
    ~Frame();


    //klt_gyro 10.23日添加
    Frame(const cv::Mat &im, const Se2& odo, double Imu_theta, ORBextractor *extractor, const cv::Mat &K, const cv::Mat &distCoef);//首帧创建Frame
    Frame(const cv::Mat &im, const Se2& odo, std::vector<cv::KeyPoint> mckeyPoints, ORBextractor *extractor, const cv::Mat &K, const cv::Mat &distCoef);//跟踪过程中创建Frame
    //klt补充变量
    int MIN_DIST;//mask建立时的特征点周边半径
    int MAX_CNT; //最大特征点数量
    cv::Mat img,prev_img, cur_img, forw_img;//prev_img是预测上一次帧的图像数据，cur_img是光流跟踪的前一帧的图像数据，forw_img是光流跟踪的后一帧的图像数据
    std::vector<cv::Point2f> n_pts;//每一帧中新提取的特征点
    std::vector<cv::Point2f> prev_pts, cur_pts, forw_pts;//对应的图像特征点
    std::vector<int> ids;//能够被跟踪到的特征点的id
    std::vector<int> track_cnt;//当前帧forw_img中每个特征点被追踪的时间次数
    std::vector<int> track_midx;//当前帧与关键帧匹配点的对应关系;
    //klt补充函数
    void addPoints();//更新跟踪点



    void computeBoundUn(const cv::Mat& K, const cv::Mat& D);
//    void undistortKeyPoints(const cv::Mat& K, const cv::Mat& D);

    Se2 getTwb();
    cv::Mat getTcr();
    cv::Mat getPose();
    cv::Point3f getCameraCenter();
    void setPose(const cv::Mat& _Tcw);
    void setPose(const Se2& _Twb);
    void setTcr(const cv::Mat& _Tcr);
    void setTrb(const Se2& _Trb);

    bool PosInGrid(cv::KeyPoint& kp, int& posX, int& posY);
    bool inImgBound(cv::Point2f pt);
    std::vector<size_t> GetFeaturesInArea(const float& x, const float& y, const float& r,
                                          const int minLevel = -1, const int maxLevel = -1) const;

    void copyImgTo(cv::Mat& imgRet) { mImage.copyTo(imgRet); }
    void copyDesTo(cv::Mat& desRet) { mDescriptors.copyTo(desRet); }


public:
    //KLT添加
    std::vector<cv::KeyPoint> keyPoints;
    std::vector<cv::KeyPoint> keyPointsUn;
    // pose info: pose to ref KF, pose to World, odometry raw.
    cv::Mat Tcr;    //
    cv::Mat Tcw;    //
    Se2 Trb;     // ref KF to body
    Se2 Twb;     // world to body
//    // Scale Pyramid Info
//    int mnScaleLevels;
//    float mfScaleFactor;
//    std::vector<float> mvScaleFactors;
//    std::vector<float> mvLevelSigma2;
//    std::vector<float> mvInvLevelSigma2;

    //! static variable
    static bool bNeedVisulization;
    static bool bIsInitialComputations;  // 首帧畸变校正后会重新计算图像边界,然后此flag取反
    static float minXUn;                 // 图像边界
    static float minYUn;
    static float maxXUn;
    static float maxYUn;
    // 坐标乘以gridElementWidthInv和gridElementHeightInv就可以确定在哪个格子
    static float gridElementWidthInv;
    static float gridElementHeightInv;
    static unsigned long nextId;

    ORBextractor* mpORBExtractor;

    //! frame information
    double mTimeStamp;
    unsigned long id;  // 图像序号
    Se2 odom;          // 原始里程计输入

    size_t N;  // 特征总数
    cv::Mat mImage;
    cv::Mat mDescriptors;
    std::vector<cv::KeyPoint> mvKeyPoints;
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    // 图像金字塔相关
    int mnScaleLevels;
    float mfScaleFactor;
    std::vector<float> mvScaleFactors;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

    //! 以下信息需要加锁访问/修改
protected:
    // 位姿信息
//    cv::Mat Tcr;  // Current Camera frame to Reference Camera frame, 三角化和此有关
//    cv::Mat Tcw;  // Current Camera frame to World frame
//    Se2 Trb;      // reference KF body to current frame body
//    Se2 Twb;      // world to body

    std::mutex mMutexPose;
};

typedef std::shared_ptr<Frame> PtrFrame;

}  // namespace se2lam
#endif  // FRAME_H
