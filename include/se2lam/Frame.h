/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/


#ifndef FRAME_H
#define FRAME_H

#include "Config.h"
#include "ORBVocabulary.h"
#include "ORBextractor.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include <eigen3/Eigen/Core>
#include <memory>
#include <mutex>
#include <vector>

namespace se2lam
{

class MapStorage;
class MapPoint;
typedef std::shared_ptr<MapPoint> PtrMapPoint;

struct PreSE2 {
public:
    double meas[3];  // x, y, theta
    double cov[9];  // 3*3, RowMajor
};


//! 分块数目
const int GRID_ROWS = 12;  // default 48 for 480, 23
const int GRID_COLS = 16;  // default 64 for 640, 31

class Frame
{
public:
    Frame();
    Frame(const cv::Mat& im, const Se2& odo, ORBextractor* extractor, double time = 0.);
    Frame(const cv::Mat& im, const Se2& odo, const std::vector<cv::KeyPoint>& vKPs,
          ORBextractor* extractor, double time = 0.);  // klt创建Frame, 10.23日添加

    Frame(const Frame& f);
    Frame& operator=(const Frame& f);
    ~Frame();

    bool isNull() const { return mbNull; }
    virtual void setNull();

    //! Pose Operations
    void setPose(const cv::Mat& _Tcw);
    void setPose(const Se2& _Twb);
    void setTcr(const cv::Mat& _Tcr);
    void setTrb(const Se2& _Trb);
    Se2 getTwb();
    Se2 getTrb();
    cv::Mat getTcr();
    cv::Mat getPose();
    cv::Point3f getCameraCenter();

    bool inImgBound(const cv::Point2f& pt) const;
    void computeBoundUn(const cv::Mat& K, const cv::Mat& D);
    bool posInGrid(const cv::KeyPoint& kp, int& posX, int& posY) const;
    std::vector<size_t> getFeaturesInArea(const float x, const float y, const float r,
                                          const int minLevel = -1, const int maxLevel = -1) const;

    void computeBoW(const ORBVocabulary* _pVoc);

    //! Observations Operations
    cv::Point3f getMPPoseInCamareFrame(size_t idx);
    PtrMapPoint getObservation(size_t idx);
    std::vector<PtrMapPoint> getObservations(bool checkValid = false, bool checkParallax = false);
    size_t countObservations();
    void setObservation(const PtrMapPoint& pMP, size_t idx);
    void updateObservationsAfterOpt();

    void clearObservations();
    bool hasObservationByIndex(size_t idx);
    void eraseObservationByIndex(size_t idx);
    virtual bool hasObservationByPointer(const PtrMapPoint& pMP);
    virtual void eraseObservationByPointer(const PtrMapPoint& pMP);

    friend class MapStorage;

public:
    //! static variable
    static unsigned long nextId;
    static bool bNeedVisualization;
    static bool bIsInitialComputation;  // 首帧畸变校正后会重新计算图像边界,然后此flag取反
    static float minXUn;                // 图像校正畸变后的边界
    static float minYUn;
    static float maxXUn;
    static float maxYUn;
    static float gridElementWidthInv;  // 确定在哪个格子
    static float gridElementHeightInv;

    //! ORB BoW by THB:
    ORBextractor* mpORBExtractor;
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;
    bool mbBowVecExist;

    //! frame information
    double mTimeStamp;
    unsigned long id;  // 图像序号
    Se2 odom;          // 原始里程计输入

    size_t N;        // 特征总数
    cv::Mat mImage;  // bNeedVisualization = false, 图像无数据
    cv::Mat mDescriptors;
    std::vector<cv::KeyPoint> mvKeyPoints;  // N个KP

    std::vector<bool> mvbMPOutlier;  // 优化的时候用
    std::vector<std::size_t> mGrid[GRID_COLS][GRID_ROWS];

    //! 图像金字塔相关
    int mnScaleLevels;
    float mfScaleFactor;
    std::vector<float> mvScaleFactors;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

protected:
    bool mbNull;

    //! 位姿信息需要加锁访问/修改
    cv::Mat Tcr;  // Current Camera frame to Reference Camera frame, 三角化和此有关
    cv::Mat Tcw;  // Current Camera frame to World frame
    Se2 Trb;      // reference KF body to current frame body
    Se2 Twb;      // world to body
    std::mutex mMutexPose;

    std::vector<PtrMapPoint> mvpMapPoints;  // KP对应的MP
    std::mutex mMutexObs;
};

typedef std::shared_ptr<Frame> PtrFrame;

}  // namespace se2lam

#endif  // FRAME_H
