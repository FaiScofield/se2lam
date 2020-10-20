/**
 * This file is part of se2lam
 *
 * Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
 */


#ifndef CONFIG_H
#define CONFIG_H

#include <opencv2/core/core.hpp>
#include <ostream>

namespace se2lam
{

// LOG define
#define LOGT(msg) (std::cout << "\033[32m" << "[TRACE] " << msg << "\033[0m" << std::endl)
#define LOGI(msg) (std::cout << "\033[0m"  << "[INFO ] " << msg << "\033[0m" << std::endl)
#define LOGW(msg) (std::cerr << "\033[33m" << "[WARN ] " << msg << "\033[0m" << std::endl)
#define LOGE(msg) (std::cerr << "\033[31m" << "[ERROR] " << msg << "\033[0m" << std::endl)
#define LOGF(msg) (std::cerr << "\033[35m" << "[FATAL] " << msg << "\033[0m" << std::endl)


// -pi ~ +pi
inline double normalizeAngle(double theta)
{
    if (theta >= -M_PI && theta < M_PI)
        return theta;

    double multiplier = floor(theta / (2 * M_PI));
    theta = theta - multiplier * 2 * M_PI;
    if (theta >= M_PI)
        theta -= 2 * M_PI;
    if (theta < -M_PI)
        theta += 2 * M_PI;

    return theta;
}

struct Se2 {
    float x;
    float y;
    float theta;
    double timeStamp;  // for odo message

    Se2();
    Se2(float _x, float _y, float _theta, double _time = -1.);
    Se2(const Se2& that);
    ~Se2();
    Se2 inv() const;
    Se2 operator-(const Se2& that) const;
    Se2 operator+(const Se2& that) const;
    Se2& operator=(const Se2& that);
    Se2 operator*(double scale) const;
    cv::Mat toCvSE3() const;
    Se2& fromCvSE3(const cv::Mat& mat);

    friend std::ostream& operator<<(std::ostream& os, const Se2& that)
    {
        os << " [" << that.x * 1e-3 << ", " << that.y * 1e-3 << ", " << that.theta << "]";
        return os;
    }
};

class WorkTimer
{
private:
    int64 tickBegin, tickEnd;

public:
    WorkTimer() { start(); }
    ~WorkTimer() {}
    double time;

    void start() { tickBegin = cv::getTickCount(); }
    void stop()
    {
        tickEnd = cv::getTickCount();
        time = double(tickEnd - tickBegin) * 1000.0 / cv::getTickFrequency();  // [ms]
    }

    double count()
    {
        stop();
        return time;
    }
};


class Config
{
public:
    //! data path
    static std::string DataPath;

    //! camera config
    static cv::Size ImgSize;
    static cv::Mat Tbc;   // camera extrinsic
    static cv::Mat Tcb;   // inv of bTc
    static cv::Mat Kcam;  // camera intrinsic
    static cv::Mat Dcam;  // camera distortion
    static float fx, fy, cx, cy;

    //! setting
    // frequency & image sequence
    static int FPS;
    static int ImgStartIndex;
    static int ImgCount;

    // depth acception
    static float UpperDepth;
    static float LowerDepth;

    // lose detection
    static float MaxLinearSpeed;   // unit [mm/s]
    static float MaxAngularSpeed;  // unit [degree/s]

    // ferture detection
    static float ScaleFactor;   // scalefactor in detecting features
    static float FeatureSigma;  //! useless for now
    static int MaxLevel;        // level number of pyramid in detecting features
    static int MaxFtrNumber;    // max feature number to detect

    // optimization
    static float ThHuber;  // robust kernel for 2d MapPoint constraint
    static int LocalIterNum;
    static int GlobalIterNum;
    static bool LocalVerbose;
    static bool GlobalVerbose;

    // noise & uncertainty
    static float OdoNoiseX, OdoNoiseY, OdoNoiseTheta;
    static float OdoUncertainX, OdoUncertainY, OdoUncertainTheta;

    // plane motion
    static float PlaneMotionInfoZ;
    static float PlaneMotionInfoXrot;
    static float PlaneMotionInfoYrot;

    // local graph
    static int MaxLocalFrameNum;  //! TODO
    static int LocalFrameSearchLevel;
    static float LocalFrameSearchRadius;

    // global map loopclose
    static float MinScoreBest;     // 场景相似度得分阈值
    static float MinMPMatchRatio;  // 回环验证中MP最少匹配比率
    static int MinMPMatchNum;      // 回环验证中MP最少匹配数
    static int MinKPMatchNum;      // 回环验证中KP最少匹配数
    static int MinKFidOffset;      // 回环搜索帧间隔数

    // map storage
    static bool UsePrevMap;
    static bool SaveNewMap;
    static bool LocalizationOnly;
    static std::string MapFileStorePath;
    static std::string ReadMapFileName;
    static std::string WriteMapFileName;
    static std::string WriteTrajFileName;

    // visulization
    static bool NeedVisualization;
    static int MappubScaleRatio;
    static float CameraSize, PointSize;

    //! other
    static cv::Mat PrjMtrxEye;
    static float ThDepthFilter;  //! TODO 深度滤波阈值    

    //! debug
    static bool ShowGroundTruth;
    static bool LocalPrint;
    static bool GlobalPrint;
    static bool SaveMatchImage;
    static std::string MatchImageStorePath;

public:
    //! functions
    static void readConfig(const std::string& path);
    static bool acceptDepth(float depth);
    static void checkParamValidity();
};

}  // namespace se2lam

#endif  // CONFIG_H
