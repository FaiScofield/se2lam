/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/


#ifndef FRAME_H
#define FRAME_H
#pragma once

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

//! 分块数目
const int GRID_ROWS = 12;  // default 48 for 480, 23
const int GRID_COLS = 16;  // default 64 for 640, 31

class Frame
{
public:
    Frame();
    Frame(const cv::Mat& im, const Se2& odo, ORBextractor* extractor, const cv::Mat& K,
          const cv::Mat& distCoef);
    Frame(const cv::Mat& im, const double time, const Se2& odo, ORBextractor* extractor,
          const cv::Mat& K, const cv::Mat& distCoef);
    Frame(const cv::Mat& im, const Se2& odo, const std::vector<cv::KeyPoint>& vKPs,
          ORBextractor* extractor); // klt创建Frame, 10.23日添加

    Frame(const Frame& f);
    Frame& operator=(const Frame& f);
    ~Frame();

    void setPose(const cv::Mat& _Tcw);
    void setPose(const Se2& _Twb);
    void setTcr(const cv::Mat& _Tcr);
    void setTrb(const Se2& _Trb);
    Se2 getTwb();
    Se2 getTrb();
    cv::Mat getTcr();
    cv::Mat getPose();
    cv::Point3f getCameraCenter();

    bool inImgBound(const cv::Point2f& pt);
    void computeBoundUn(const cv::Mat& K, const cv::Mat& D);
    bool posInGrid(cv::KeyPoint& kp, int& posX, int& posY);
    std::vector<size_t> getFeaturesInArea(const float& x, const float& y, const float& r,
                                          const int minLevel = -1, const int maxLevel = -1) const;

    void copyImgTo(cv::Mat& imgRet) { mImage.copyTo(imgRet); }
    void copyDesTo(cv::Mat& desRet) { mDescriptors.copyTo(desRet); }

    //! static variable
    static bool bNeedVisualization;
    static bool bIsInitialComputation;  // 首帧畸变校正后会重新计算图像边界,然后此flag取反
    static float minXUn;                // 图像校正畸变后的边界
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
    std::vector<std::size_t> mGrid[GRID_COLS][GRID_ROWS];

    // 图像金字塔相关
    int mnScaleLevels;
    float mfScaleFactor;
    std::vector<float> mvScaleFactors;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

protected:
    //! 位姿信息需要加锁访问/修改
    cv::Mat Tcr;  // Current Camera frame to Reference Camera frame, 三角化和此有关
    cv::Mat Tcw;  // Current Camera frame to World frame
    Se2 Trb;      // reference KF body to current frame body
    Se2 Twb;      // world to body

    std::mutex mMutexPose;
};

typedef std::shared_ptr<Frame> PtrFrame;

}  // namespace se2lam

#endif  // FRAME_H
