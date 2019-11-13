/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/


#ifndef MAPPOINT_H
#define MAPPOINT_H
#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <set>

namespace se2lam
{

class Map;
class KeyFrame;
typedef std::shared_ptr<KeyFrame> PtrKeyFrame;

class MapPoint : public std::enable_shared_from_this<MapPoint>
{
public:
    MapPoint();
    MapPoint(const cv::Point3f& pos, bool goodPrl);
    ~MapPoint();

    struct IdLessThan {
        bool operator()(const std::shared_ptr<MapPoint>& lhs,
                        const std::shared_ptr<MapPoint>& rhs) const
        {
            return lhs->mId < rhs->mId;
        }
    };

    void setMap(Map* pMap) { mpMap = pMap; }
    void setNull();

    //! 观测属性
    std::set<PtrKeyFrame> getObservations();
    void addObservation(const PtrKeyFrame& pKF, size_t idx); // Do pKF.setViewMP() before use this
    void eraseObservation(const PtrKeyFrame& pKF);
    bool hasObservation(const PtrKeyFrame& pKF);
    size_t countObservations();

    cv::Point3f getNormalVector();
    cv::Point2f getMainMeasureProjection();
    cv::Mat getDescriptor();
    PtrKeyFrame getMainKF();
    int getOctave(const PtrKeyFrame& pKF);
    int getIndexInKF(const PtrKeyFrame& pKF);
    int getMainOctave();

    //! 自身属性
    bool isGoodPrl() { return mbGoodParallax; }
    void setGoodPrl(bool value) { mbGoodParallax = value; }
    bool isNull() { return mbNull; }
    void setPos(const cv::Point3f& pt3f);
    cv::Point3f getPos();

    //! Map调用
    bool acceptNewObserve(const cv::Point3f& posKF, const cv::KeyPoint& kp);
    void mergedInto(const std::shared_ptr<MapPoint>& pMP);

    unsigned long mId;
    static unsigned long mNextId;

private:
    //! 内部使用的成员函数, 不需要加锁, 因为调用它的函数已经加了锁
    void setNullSelf();
    void updateMeasureInKFs();  // setPos()后调用
    void updateParallax(const PtrKeyFrame& pKF);  // addObservation()后调用, 更新视差
    void updateMainKFandDescriptor();  // addObservation()后调用, 更新相关参数
    bool updateParallaxCheck(const PtrKeyFrame& pKF1, const PtrKeyFrame& pKF2); // 更新视差里调用

    Map* mpMap;

    bool mbNull;
    bool mbGoodParallax;

    float mMinDist;  // Scale invariance distances
    float mMaxDist;

    //! 以下成员变量需加锁访问
    cv::Point3f mPos;  // 三维空间坐标
    std::mutex mMutexPos;

    // first = 观测到此MP的KF, second = 在其KP中的索引
    std::map<PtrKeyFrame, size_t> mObservations;  // 最重要的成员变量
    cv::Point3f mNormalVector;  // 平均观测方向
    PtrKeyFrame mMainKF;
    cv::Mat mMainDescriptor;    // 最优描述子, 到其他描述子平均距离最小
    int mMainOctave;
    float mLevelScaleFactor;
    std::mutex mMutexObs;
};

typedef std::shared_ptr<MapPoint> PtrMapPoint;


}  // namespace se2lam

#endif  // MAPPOINT_H
