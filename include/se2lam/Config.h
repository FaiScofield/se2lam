/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/


#ifndef CONFIG_H
#define CONFIG_H

#include <fstream>
#include <opencv2/core/core.hpp>
#include <ostream>

namespace se2lam
{

struct Se2 {
    float x;
    float y;
    float theta;
    float timeStamp;    // for odo message

    Se2();
    Se2(float _x, float _y, float _theta, float _time = 0.f);
    Se2(const Se2& that);
    ~Se2();
    Se2 inv() const;
    Se2 operator-(const Se2& that) const;
    Se2 operator+(const Se2& that) const;
    Se2& operator=(const Se2& that);
    cv::Mat toCvSE3() const;
    Se2& fromCvSE3(const cv::Mat& mat);

    friend std::ostream& operator<<(std::ostream& os, const Se2& that)
    {
        os << " [" << that.x / 1000 << ", " << that.y / 1000 << ", " << that.theta << "]";
        return os;
    }
};

inline double normalize_angle(double theta)
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

class WorkTimer
{
private:
    int64 tickBegin, tickEnd;

public:
    WorkTimer() {}
    ~WorkTimer() {}
    double time;

    void start() { tickBegin = cv::getTickCount(); }
    void stop()
    {
        tickEnd = cv::getTickCount();
        time = (tickEnd - tickBegin) / (cv::getTickFrequency() * 1000.);
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
    static float UpperDepth;  // unit mm
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
    static int MaxLocalFrameNum;          //! TODO
    static float LocalFrameSearchRadius;  //! TODO

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
    static bool NeedVisulization; //! TODO 是否需要可视化, 不可视化时内存占用小, 可在嵌入式平台上跑
    static int MappubScaleRatio;  // 地图可视化比例

    //! other
    static cv::Mat PrjMtrxEye;
    static float ThDepthFilter;  //! TODO 深度滤波阈值

    //! debug
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
