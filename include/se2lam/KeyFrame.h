/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/


#ifndef KEYFRAME_H
#define KEYFRAME_H

#pragma once

#include "Frame.h"
#include <set>

namespace se2lam
{

//#define WEIGHT_COVISIBLITIES

struct SE3Constraint {
public:
    cv::Mat measure;
    cv::Mat info;
    SE3Constraint() {}
    SE3Constraint(const cv::Mat& _mea, const cv::Mat& _info)
    {
        _mea.copyTo(measure);
        _info.copyTo(info);
    }
    ~SE3Constraint() {}
};
typedef std::shared_ptr<SE3Constraint> PtrSE3Cstrt;

class Map;
class MapPoint;
typedef std::shared_ptr<MapPoint> PtrMapPoint;

class KeyFrame : public Frame, public std::enable_shared_from_this<KeyFrame>
{
public:
    KeyFrame();  // MapStorage类加载地图时需要
    KeyFrame(const Frame& frame);
    ~KeyFrame();

    struct IdLessThan {
        bool operator()(const std::shared_ptr<KeyFrame>& lhs, const std::shared_ptr<KeyFrame>& rhs) const
        {
            return lhs->mIdKF < rhs->mIdKF;
        }
    };

    // 用于带权重的共视关系排序
    struct SortByValueGreater {
        bool operator()(const std::pair<std::shared_ptr<KeyFrame>, int>& lhs,
                        const std::pair<std::shared_ptr<KeyFrame>, int>& rhs)
        {
            return lhs.second > rhs.second;
        }
    };

    void setMap(Map* pMap) { mpMap = pMap; }
    void setNull();

    //! 共视关系的维护函数
    std::vector<std::shared_ptr<KeyFrame>> getAllCovisibleKFs();
    void eraseCovisibleKF(const std::shared_ptr<KeyFrame>& pKF);
    void addCovisibleKF(const std::shared_ptr<KeyFrame>& pKF);
    size_t countCovisibleKFs();
#ifdef WEIGHT_COVISIBLITIES
    std::vector<std::shared_ptr<KeyFrame>> getBestCovisibleKFs(size_t n = 0);
    std::vector<std::shared_ptr<KeyFrame>> getCovisibleKFsByWeight(int w);
    std::map<std::shared_ptr<KeyFrame>, int> getAllCovisibleKFsAndWeights();
    void addCovisibleKF(const std::shared_ptr<KeyFrame>& pKF, int weight);
    void sortCovisibleKFs();
    void updateCovisibleGraph();
#endif


    //! MP观测的维护函数
    int getFeatureIndex(const PtrMapPoint& pMP); // 返回MP对应的KP的索引
    void setObsAndInfo(const PtrMapPoint& pMP, size_t idx, const Eigen::Matrix3d& info);
    bool hasObservationByPointer(const PtrMapPoint& pMP);
    void eraseObservationByPointer(const PtrMapPoint& pMP);

    //! 特征约束关系的维护函数
    void addFtrMeasureFrom(const std::shared_ptr<KeyFrame>& pKF, const cv::Mat& _mea, const cv::Mat& _info);
    void addFtrMeasureTo(const std::shared_ptr<KeyFrame>& pKF, const cv::Mat& _mea, const cv::Mat& _info);
    void eraseFtrMeasureFrom(const std::shared_ptr<KeyFrame>& pKF);
    void eraseFtrMeasureTo(const std::shared_ptr<KeyFrame>& pKF);
    void setOdoMeasureFrom(const std::shared_ptr<KeyFrame>& pKF, const cv::Mat& _mea, const cv::Mat& _info);
    void setOdoMeasureTo(const std::shared_ptr<KeyFrame>& pKF, const cv::Mat& _mea, const cv::Mat& _info);


public:
    static unsigned long mNextIdKF;

    unsigned long mIdKF;

    std::vector<bool> mvbViewMPsInfoExist;
    std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> mvViewMPsInfo;

    // KeyFrame contraints: From this or To this
    std::map<std::shared_ptr<KeyFrame>, SE3Constraint> mFtrMeasureFrom;  // 特征图中后面的连接
    std::map<std::shared_ptr<KeyFrame>, SE3Constraint> mFtrMeasureTo;  // 特征图中前面的连接
    std::pair<std::shared_ptr<KeyFrame>, SE3Constraint> mOdoMeasureFrom;  // 和后一帧的里程约束
    std::pair<std::shared_ptr<KeyFrame>, SE3Constraint> mOdoMeasureTo;  // 和前一帧的里程约束

    //! TODO 暂时没用，待添加预积分
    std::pair<std::shared_ptr<KeyFrame>, PreSE2> preOdomFromSelf;
    std::pair<std::shared_ptr<KeyFrame>, PreSE2> preOdomToSelf;

    //! 以下信息需要加锁访问
protected:
    Map* mpMap;

#ifdef WEIGHT_COVISIBLITIES
    std::map<std::shared_ptr<KeyFrame>, int> mCovisibleKFsWeight;
    std::vector<std::shared_ptr<KeyFrame>> mvpCovisibleKFsSorted;
    std::vector<int> mvOrderedWeights;
#else
    std::set<std::shared_ptr<KeyFrame>> mspCovisibleKFs;
#endif
    std::mutex mMutexCovis;

//    std::mutex mMutexObs;

};

typedef std::shared_ptr<KeyFrame> PtrKeyFrame;

}  // namespace se2lam

#endif
