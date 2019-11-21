/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/


#ifndef KEYFRAME_H
#define KEYFRAME_H
#pragma once

#include "Config.h"
#include "Frame.h"
#include "ORBVocabulary.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "converter.h"

namespace se2lam
{

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
    KeyFrame();
    KeyFrame(const Frame& frame);
    ~KeyFrame();

    struct IdLessThan {
        bool operator()(const std::shared_ptr<KeyFrame>& lhs,
                        const std::shared_ptr<KeyFrame>& rhs) const
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
    //bool isNull() { return mbNull; }
    void setNull();

    //! 共视关系的维护函数
    std::set<std::shared_ptr<KeyFrame>> getAllCovisibleKFs();
    std::vector<std::shared_ptr<KeyFrame>> getBestCovisibleKFs(size_t n = 0);
    void addCovisibleKF(const std::shared_ptr<KeyFrame>& pKF);
    void addCovisibleKF(const std::shared_ptr<KeyFrame>& pKF, int weight);
    void eraseCovisibleKF(const std::shared_ptr<KeyFrame>& pKF);
    void sortCovisibleKFs();
    void updateCovisibleKFs();
    size_t countCovisibleKFs();

    //! MP观测的维护函数
    std::set<PtrMapPoint> getAllObsMPs(bool checkParallax = true);
    std::map<PtrMapPoint, size_t> getObservations();  // 返回所有的MP
    std::vector<PtrMapPoint> getMapPointMatches();    // 返回KP对应的MP
    PtrMapPoint getObservation(size_t idx);           // 返回索引id对应的MP
    int getFeatureIndex(const PtrMapPoint& pMP);      // 返回MP对应的KP的索引
    void addObservation(const PtrMapPoint& pMP, size_t idx);
    void eraseObservation(const PtrMapPoint& pMP);
    void eraseObservation(size_t idx);
    bool hasObservation(const PtrMapPoint& pMP);
    bool hasObservation(size_t idx);
    void setObservation(const PtrMapPoint& pMP, size_t idx);
    size_t countObservations();  // 计算MP的观测数

    // NOTE 关键函数，在LocalMapper和MapPoint里会给KF添加观测
    void setViewMP(const cv::Point3f& pt3f, size_t idx, const Eigen::Matrix3d& info);
    cv::Point3f getViewMPPoseInCamareFrame(size_t idx);

    //! 特征约束关系的维护函数
    void addFtrMeasureFrom(const std::shared_ptr<KeyFrame>& pKF, const cv::Mat& _mea,
                           const cv::Mat& _info);
    void addFtrMeasureTo(const std::shared_ptr<KeyFrame>& pKF, const cv::Mat& _mea,
                         const cv::Mat& _info);
    void eraseFtrMeasureFrom(const std::shared_ptr<KeyFrame>& pKF);
    void eraseFtrMeasureTo(const std::shared_ptr<KeyFrame>& pKF);
    void setOdoMeasureFrom(const std::shared_ptr<KeyFrame>& pKF, const cv::Mat& _mea,
                           const cv::Mat& _info);
    void setOdoMeasureTo(const std::shared_ptr<KeyFrame>& pKF, const cv::Mat& _mea,
                         const cv::Mat& _info);

    //! 词向量相关
    void computeBoW(const ORBVocabulary* _pVoc);
    DBoW2::FeatureVector getFeatureVector() { return mFeatVec; }
    DBoW2::BowVector getBowVector() { return mBowVec; }

public:
    static unsigned long mNextIdKF;

    unsigned long mIdKF;

    //! TODO 此变量的作用和变化还需要探究一下, 是否需要加锁访问? 是否可以改成PtrMapPoint?
    std::vector<cv::Point3f> mvViewMPs;  // MP在当前KF相机坐标系下的坐标, 即Pc
    std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> mvViewMPsInfo;

    // KeyFrame contraints: From this or To this
    std::map<std::shared_ptr<KeyFrame>, SE3Constraint> mFtrMeasureFrom;  // 特征图中后面的连接
    std::map<std::shared_ptr<KeyFrame>, SE3Constraint> mFtrMeasureTo;  // 特征图中前面的连接
    std::pair<std::shared_ptr<KeyFrame>, SE3Constraint> mOdoMeasureFrom;  // 和后一帧的里程约束
    std::pair<std::shared_ptr<KeyFrame>, SE3Constraint> mOdoMeasureTo;  // 和前一帧的里程约束

    //! TODO 暂时没用，待添加预积分
    std::pair<std::shared_ptr<KeyFrame>, PreSE2> preOdomFromSelf;
    std::pair<std::shared_ptr<KeyFrame>, PreSE2> preOdomToSelf;

    // ORB BoW by THB:
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;
    bool mbBowVecExist;

    //! 以下信息需要加锁访问
protected:
    Map* mpMap;

    //bool mbNull;

    std::map<PtrMapPoint, size_t> mObservations;  // size_t为MP在此KF中对应的特征点的索引
    std::map<size_t, PtrMapPoint> mDualObservations;

    std::set<std::shared_ptr<KeyFrame>> mspCovisibleKFs;
//    std::map<std::shared_ptr<KeyFrame>, int> mCovisibleKFsWeight;
//    std::vector<std::shared_ptr<KeyFrame>> mvpCovisibleKFsSorted;

    std::mutex mMutexObs;
    std::mutex mMutexCovis;
};

typedef std::shared_ptr<KeyFrame> PtrKeyFrame;

}  // namespace se2lam

#endif
