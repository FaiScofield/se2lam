/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/


#ifndef FRAME_H
#define FRAME_H

#include <vector>
#include <mutex>
#include <memory>
#include "Config.h"
#include "ORBextractor.h"
#include "MapPoint.h"

namespace se2lam {


struct PreSE2{
public:
    double meas[3];
    double cov[9]; // 3*3, RowMajor
};

const int FRAME_GRID_ROWS = 36; // default 48 for 480
const int FRAME_GRID_COLS = 48; // default 64 for 640

class KeyFrame;
class Frame
{
public:
    Frame();
    Frame(const cv::Mat &im, const Se2& odo, ORBextractor* extractor,
          const cv::Mat &K, const cv::Mat &distCoef);

    Frame(const Frame& f);
    Frame& operator=(const Frame& f);
    ~Frame();

    float getTime() const;
    void setTime(float time);

    void undistortKeyPoints(const cv::Mat &K, const cv::Mat &D);
    void computeBoundUn(const cv::Mat &K, const cv::Mat &D);

    // Image Info
    cv::Mat img;
    static float minXUn;
    static float minYUn;
    static float maxXUn;
    static float maxYUn;
    bool inImgBound(cv::Point2f pt);
    ORBextractor* mpORBExtractor;
    std::vector<cv::KeyPoint> keyPoints;
    std::vector<cv::KeyPoint> keyPointsUn;
    cv::Mat descriptors;
    int N;
    std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y,
        const float  &r, const int minLevel=-1, const int maxLevel=-1) const;

    // Scale Pyramid Info
    int mnScaleLevels;
    float mfScaleFactor;
    std::vector<float> mvScaleFactors;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

    static bool mbInitialComputations;

    // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints
    // 坐标乘以mnvfGridElementWidthI和mfGridElementHeightInv就可以确定在哪个格子
    static float mfGridElementWidthInv;
    static float mfGridElementHeightInv;
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    // Compute the cell of a keypoint (return false if outside the grid)
    bool PosInGrid(cv::KeyPoint &kp, int &posX, int &posY);


    // pose info: pose to ref KF, pose to World, odometry raw.
    cv::Mat Tcr;    //!@Vance: Current Camera frame to Reference Camera frame, 三角化和此有关
    cv::Mat Tcw;    //!@Vance: Current Camera frame to World frame

    Se2 Trb;     // reference KF body to current frame body
    Se2 Twb;     // world to body
    Se2 odom;    // odometry raw

    int id;
    static int nextId;

    //! Multi-thread processing
    std::mutex mMutexImg;
    void copyImgTo(cv::Mat & imgRet);

    std::mutex mMutexDes;
    void copyDesTo(cv::Mat & desRet);

    // 每个特征点对应的MapPoint
    std::vector<PtrMapPoint> mvpMapPoints;
    // 观测不到Map中的3D点
    std::vector<bool> mvbOutlier;

    std::vector<lineSort_S> lineFeature;                // 线特征信息
    std::vector<pointLineLable> pointAndLineLable;      // 每个点对应的线特征
    std::vector<std::vector<int>> lineIncludePoints;    // 每条线中包含的点有哪些


protected:
    float mTime;
};

typedef std::shared_ptr<Frame> PtrFrame;

} // namespace se2lam
#endif // FRAME_H
