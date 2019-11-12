// Created by lmp on 19-10-28.
//
/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#ifndef TRACKKLT_H
#define TRACKKLT_H

#include "Config.h"
#include "Frame.h"
#include "GlobalMapper.h"
#include "Sensors.h"
#include "cvutil.h"
#include <g2o/types/sba/types_six_dof_expmap.h>

#define USEKLT

namespace se2lam
{

class KeyFrame;
class Map;
class LocalMapper;
class GlobalMapper;

typedef std::shared_ptr<KeyFrame> PtrKeyFrame;

class TrackKlt
{
public:
    TrackKlt();
    ~TrackKlt();

    void run();

    void setMap(Map* pMap) { mpMap = pMap; }
    void setLocalMapper(LocalMapper* pLocalMapper) { mpLocalMapper = pLocalMapper; }
    void setGlobalMapper(GlobalMapper* pGlobalMapper) { mpGlobalMapper = pGlobalMapper; }
    void setSensors(Sensors* pSensors) { mpSensors = pSensors; }

    //! NOTE 锁?
    cv::Mat getCurrentFramePose() { return mCurrentFrame.getPose(); }
    Se2 getCurrentFrameOdo() { return mCurrentFrame.odom; }

    static void calcOdoConstraintCam(const Se2& dOdo, cv::Mat& cTc, g2o::Matrix6d& Info_se3);
    static void calcSE3toXYZInfo(cv::Point3f xyz1, const cv::Mat& Tcw1, const cv::Mat& Tcw2,
                                 Eigen::Matrix3d& info1, Eigen::Matrix3d& info2);

    // for visulization message publisher
    size_t copyForPub(cv::Mat& img1, cv::Mat& img2, std::vector<cv::Point2f>& kp1,
                      std::vector<cv::Point2f>& kp2, std::vector<int>& vMatches12);
    cv::Mat getImageMatches();


    bool isFinished();
    void requestFinish();

public:
    // Tracking states
    cvu::eTrackingState mState;
    cvu::eTrackingState mLastState;

    int N1 = 0, N2 = 0, N3 = 0;  // for debug print

private:
    void relocalization(const cv::Mat& img, double imgTime, const Se2& odo) {}
    void updateFramePose();

    bool needNewKF(int trackedRef);
    int doTriangulate();

    void resetLocalTrack();
    bool checkFinish();
    void setFinish();

    // klt跟踪添加函数
    void createFrameFirstKlt(const cv::Mat& img, const Se2& odo);
    void trackKlt(const cv::Mat& img, const Se2& odo, double imuTheta);
    void trackRefKlt(const cv::Mat& img, const Se2& odo, double imuTheta);
    void predictPointsAndImage(double angle);
    void detectFeaturePointsCell(const cv::Mat& image, const cv::Mat& mask);
    bool inBorder(const cv::Point2f& pt);
    void removeOutliers();
    void reduceVectorCell(const std::vector<unsigned char>& status);
    void setMaskCell();
    void addNewPoints();
    void getRotatedPoints(const std::vector<cv::Point2f>& Points,
                         std::vector<cv::Point2f>& dstPoints, const cv::Point& center,
                         double angle);
    void segImageToCells(const cv::Mat& image, std::vector<cv::Mat>& cellImgs);
    void resetKltData();

    cv::Mat drawMatchesPointsToLastFrame(const std::string& title = "");
    cv::Mat drawMatchesPointsToRefFrame(const std::string& title = "");

private:
    static bool mbUseOdometry;  //! TODO 冗余变量
    bool mbPrint;
    bool mbNeedVisualization;

    // set in OdoSLAM class
    Map* mpMap;
    LocalMapper* mpLocalMapper;
    GlobalMapper* mpGlobalMapper;
    Sensors* mpSensors;
    ORBextractor* mpORBextractor;  // 这里有new
    ORBmatcher* mpORBmatcher;

    // local map
    Frame mCurrentFrame, mLastFrame;
    PtrKeyFrame mpReferenceKF;
    std::vector<cv::Point3f> mLocalMPs;     // 参考帧的MP观测(Pc非Pw), 每帧处理会更新此变量
    std::vector<int> mvMatchIdxToRefKF;   // Matches12, 参考帧到当前帧的KP匹配索引
    std::vector<bool> mvbGoodPrl;
    int mnGoodPrl;  // count number of mLocalMPs with good parallax
    int mnInliers, mnMatchSum, mnTrackedOld;
    int mnLostFrames;

    // New KeyFrame rules (according to fps)
    int nMinFrames, nMaxFrames;
    double mMaxAngle, mMaxDistance;

    // preintegration on SE2
    PreSE2 preSE2;
    Se2 mLastOdom;

    cv::Mat mK, mD;
    cv::Mat mAffineMatrix, mHomograpy;
    cv::Mat mImgOutMatch;

    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexForPub;
    std::mutex mMutexFinish;

    // klt跟踪添加变量
    int mnImgRows, mnImgCols;  // 图像尺寸
    int mnMaskRadius;          // mask建立时的特征点周边半径
    int mnMaxCnt;              // 总最大特征点数量

    cv::Mat mMask;  // 图像掩码
    // prevImg是预测上一次帧的图像数据，curImg是光流跟踪的前一帧的图像数据，forwImg是光流跟踪的后一帧的图像数据
    cv::Mat mPrevImg, mCurrImg, mForwImg;
    std::vector<cv::Point2f> mvPrevPts, mvCurrPts, mvForwPts;  // 对应的图像特征点
    std::vector<cv::Point2f> mvNewPts;                         // 每一帧中新提取的特征点

    // 以下变量针对当前帧(Forw)
    std::vector<unsigned long> mvIdFirstAdded;  // 追踪上的KP是在哪一帧中生成的
    std::vector<size_t> mvIdxToFirstAdded;      // 与关键帧匹配点的对应索引;
    std::vector<int> mvTrackCount;              // 每个特征点被追踪的时间次数
    std::vector<int> mvNumPtsInCell;            // 每个分块当前检测到的KP数
    std::vector<int> mvCellLabel;               // 每个KP所属的分块索引号

    // klt分块添加
    int mnCellWidth, mnCellHeight;        // 分块尺寸
    int mnCellsY, mnCellsX, mnCells;  // 分块数
    int mnMaxNumPtsInCell;                // 与分块检点的最大点数
};

}  // namespace se2lam

#endif  // TRACKKLT_H
