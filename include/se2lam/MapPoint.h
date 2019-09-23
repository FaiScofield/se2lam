/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/


#ifndef MAPPOINT_H
#define MAPPOINT_H

#include "KeyFrame.h"
#include <mutex>
#include <map>
#include <memory>
#include <set>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

namespace se2lam{

class KeyFrame;
typedef std::shared_ptr<KeyFrame> PtrKeyFrame;

class MapPoint
{
public:
    MapPoint();
    MapPoint(cv::Point3f pos, bool goodPrl);
    ~MapPoint();

    std::set<PtrKeyFrame> getObservations();
    // Do pKF.setViewMP() before use this
    void addObservation(const PtrKeyFrame &pKF, size_t idx);
    void addObservations(const std::vector<std::pair<PtrKeyFrame, size_t>>& obsCandidates);
    void eraseObservation(const PtrKeyFrame& pKF);
    void eraseObservations(const std::vector<std::pair<PtrKeyFrame, size_t>>& obsCandidates);
    bool hasObservation(const PtrKeyFrame& pKF);
    int countObservation();

    int getOctave(const PtrKeyFrame pKF);
    size_t getFtrIdx(const PtrKeyFrame& pKF);

    bool isGoodPrl();
    void setGoodPrl(bool value);
    void updateParallax(const PtrKeyFrame& pKF);

    cv::Point3f getNormalVector();

    bool isNull();
    void setNull(const std::shared_ptr<MapPoint> &pThis);

    cv::Point3f getPos();
    void setPos(const cv::Point3f& pt3f);

    float getInvLevelSigma2(const PtrKeyFrame &pKF);


    cv::Point2f getMainMeasure();
    void updateMeasureInKFs();

    cv::Mat getDescriptor();
    void updateMainKFandDescriptor(); // 更新mainKF,desccriptor,normalVector

    bool acceptNewObserve(cv::Point3f posKF, const cv::KeyPoint kp);

//    void increaseVisibleCount(int n = 1);

    struct IdLessThan{
        bool operator() (const std::shared_ptr<MapPoint>& lhs, const std::shared_ptr<MapPoint>& rhs) const{
            return lhs->mId < rhs->mId;
        }
    };

    // This MP would be replaced and abandoned later by
    void mergedInto(const std::shared_ptr<MapPoint>& pMP);



public:
    PtrKeyFrame mMainKF;
    int mMainOctave;
    float mLevelScaleFactor;

    unsigned long mId;
    static unsigned long mNextId;

//! 以下成员变量需加锁访问
protected:
    void setNull();

    //! first = 观测到此MP的KF, second = 在其KP中的索引, 按KFid从小到大排序
//    std::map<PtrKeyFrame, size_t> mObservations;  // 最重要的成员变量
    std::map<PtrKeyFrame, size_t, KeyFrame::IdLessThan> mObservations;

    cv::Point3f mPos;   // 三维空间坐标
    cv::Point3f mNormalVector;  // Mean view direction
    cv::Mat mMainDescriptor;    // The descriptor with least median distance to the rest

//    int mVisibleCount; // 被观测数

    bool mbNull;
    bool mbGoodParallax;

    float mMinDist; // Scale invariance distances
    float mMaxDist;

    std::mutex mMutexPos;
    std::mutex mMutexObs;
};

typedef std::shared_ptr<MapPoint> PtrMapPoint;


} //namespace se2lam

#endif // MAPPOINT_H
