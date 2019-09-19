/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/


#ifndef FRAME_H
#define FRAME_H

#include "Config.h"
#include "MapPoint.h"
#include "ORBextractor.h"
#include <memory>
#include <mutex>
#include <vector>

namespace se2lam
{


struct PreSE2 {
public:
    double meas[3];
    double cov[9];  // 3*3, RowMajor
};

const int FRAME_GRID_ROWS = 36;  // default 48 for 480
const int FRAME_GRID_COLS = 48;  // default 64 for 640

class KeyFrame;

class Frame
{
public:
    Frame();
    Frame(const cv::Mat& im, const Se2& odo, ORBextractor* extractor, const cv::Mat& K,
          const cv::Mat& distCoef);

    Frame(const Frame& f);
    Frame& operator=(const Frame& f);
    ~Frame();

    float getTime() const;
    void setTime(float time);

    void undistortKeyPoints(const cv::Mat& K, const cv::Mat& D);
    void computeBoundUn(const cv::Mat& K, const cv::Mat& D);

    // Compute the cell of a keypoint (return false if outside the grid)
    bool PosInGrid(cv::KeyPoint& kp, int& posX, int& posY);
    bool inImgBound(cv::Point2f pt);

    std::vector<size_t> GetFeaturesInArea(const float& x, const float& y, const float& r,
                                          const int minLevel = -1, const int maxLevel = -1) const;

    void copyImgTo(cv::Mat& imgRet);
    void copyDesTo(cv::Mat& desRet);

public:
    //! static variable
    static bool bIsInitialComputations;  // 首帧畸变校正后会重新计算图像边界,然后此flag取反
    static float minXUn;    // 图像边界
    static float minYUn;
    static float maxXUn;
    static float maxYUn;
    // 坐标乘以gridElementWidthInv和gridElementHeightInv就可以确定在哪个格子
    static float gridElementWidthInv;
    static float gridElementHeightInv;
    static int nextId;

    //! frame information
    int id; // 图像序号
    int N;  // 特征总数
    cv::Mat mImage;
    std::vector<cv::KeyPoint> mvKeyPoints;
    cv::Mat mDescriptors;
    std::vector<PtrMapPoint> mvpMapPoints;  // 每个特征点对应的MapPoint
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    // 图像金字塔相关
    int mnScaleLevels;
    float mfScaleFactor;
    std::vector<float> mvScaleFactors;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

    // 位姿信息
    cv::Mat Tcr;  // Current Camera frame to Reference Camera frame, 三角化和此有关
    cv::Mat Tcw;  // Current Camera frame to World frame

    Se2 Trb;   // reference KF body to current frame body
    Se2 Twb;   // world to body
    Se2 odom;  // odometry raw

    // 观测不到Map中的3D点
    std::vector<bool> mvbOutlier;

    ORBextractor* mpORBExtractor;

    std::mutex mMutexImg;
    std::mutex mMutexDes;

protected:
    float mTime;

};

typedef std::shared_ptr<Frame> PtrFrame;

}  // namespace se2lam
#endif  // FRAME_H
