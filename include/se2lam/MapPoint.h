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

class KeyFrame;
class Map;
typedef std::shared_ptr<KeyFrame> PtrKeyFrame;

class MapPoint : public std::enable_shared_from_this<MapPoint>
{
public:
    MapPoint();
    MapPoint(cv::Point3f pos, bool goodPrl);
    ~MapPoint();

    //! 观测属性
    bool acceptNewObserve(cv::Point3f posKF, const cv::KeyPoint kp);
    std::set<PtrKeyFrame> getObservations();
    // Do pKF.setViewMP() before use this
    void addObservation(const PtrKeyFrame& pKF, size_t idx);
    void addObservations(const std::map<PtrKeyFrame, size_t>& obsCandidates);
    void eraseObservation(const PtrKeyFrame& pKF);
    void eraseObservations(const std::map<PtrKeyFrame, size_t>& obsCandidates);
    bool hasObservation(const PtrKeyFrame& pKF);
    size_t countObservations();

    cv::Point3f getNormalVector();
    cv::Point2f getMainMeasure();
    cv::Mat getDescriptor();
    PtrKeyFrame getMainKF();

    //    float getInvLevelSigma2(const PtrKeyFrame &pKF);

    int getOctave(const PtrKeyFrame& pKF);
    int getIndexInKF(const PtrKeyFrame& pKF);

    void updateMeasureInKFs();

    //! 自身属性
    bool isGoodPrl() { return mbGoodParallax; }
    void setGoodPrl(bool value) { mbGoodParallax = value; }

    bool isNull() { return mbNull; }
    void setNull(std::shared_ptr<MapPoint>& pThis);

    cv::Point3f getPos();
    void setPos(const cv::Point3f& pt3f);

    struct IdLessThan {
        bool operator()(const std::shared_ptr<MapPoint>& lhs,
                        const std::shared_ptr<MapPoint>& rhs) const
        {
            return lhs->mId < rhs->mId;
        }
    };

    // This MP would be replaced and abandoned later by
    void mergedInto(const std::shared_ptr<MapPoint>& pMP);
    void setMap(Map* pMap) { mpMap = pMap; }

    int mMainOctave;
    float mLevelScaleFactor;

    unsigned long mId;
    static unsigned long mNextId;

protected:
    //! 内部使用的成员函数, 不需要加锁, 因为调用它的函数已经加了锁
    void setNull();
    void updateParallax(const PtrKeyFrame& pKF);
    void updateMainKFandDescriptor();  // 更新mainKF,desccriptor,normalVector
    bool updateParallaxCheck(const PtrKeyFrame& pKF1, const PtrKeyFrame& pKF2);

    Map* mpMap;

    //! 以下成员变量需加锁访问
    cv::Point3f mPos;  // 三维空间坐标

    // first = 观测到此MP的KF, second = 在其KP中的索引
    std::map<PtrKeyFrame, size_t> mObservations;  // 最重要的成员变量

    PtrKeyFrame mMainKF;
    cv::Point3f mNormalVector;  // 平均观测方向
    cv::Mat mMainDescriptor;    // 最优描述子, 到其他描述子平均距离最小

    bool mbNull;
    bool mbGoodParallax;

    float mMinDist;  // Scale invariance distances
    float mMaxDist;

    std::mutex mMutexPos;
    std::mutex mMutexObs;
};

typedef std::shared_ptr<MapPoint> PtrMapPoint;


}  // namespace se2lam

#endif  // MAPPOINT_H
