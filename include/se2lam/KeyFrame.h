/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/


#ifndef KEYFRAME_H
#define KEYFRAME_H

#include "Config.h"
#include "Frame.h"
#include "converter.h"
#include <opencv2/calib3d/calib3d.hpp>

#include "ORBVocabulary.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

namespace se2lam
{

class MapPoint;
typedef std::shared_ptr<MapPoint> PtrMapPoint;

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


class KeyFrame : public Frame
{
public:
    KeyFrame();
    KeyFrame(const Frame& frame);
    ~KeyFrame();

    //    cv::Mat getPose();
    //    void setPose(const cv::Mat &_Tcw);
    //    void setPose(const Se2 &_Twb);

    struct IdLessThan {
        bool operator()(const std::shared_ptr<KeyFrame>& lhs,
                        const std::shared_ptr<KeyFrame>& rhs) const
        {
            return lhs->mIdKF < rhs->mIdKF;
        }
    };

    void setNull(const std::shared_ptr<KeyFrame>& pThis);
    bool isNull();

    std::set<std::shared_ptr<KeyFrame>> getAllCovisibleKFs();
    void eraseCovisibleKF(const std::shared_ptr<KeyFrame> pKF);
    void addCovisibleKF(const std::shared_ptr<KeyFrame> pKF);

    //! NOTE 关键函数，在LocalMapper和MapPoint里会给KF添加观测
    void setViewMP(cv::Point3f pt3f, int idx, Eigen::Matrix3d info);

    //! Functions for observation operations
    std::set<PtrMapPoint> getAllObsMPs(bool checkParallax = true);
    // Return all observations as a std::map
    std::map<PtrMapPoint, size_t> getObservations();

    void addObservation(PtrMapPoint pMP, size_t idx);
    void eraseObservation(const PtrMapPoint pMP);
    void eraseObservation(size_t idx);

    size_t getSizeObsMP();  // Count how many observed MP

    // Whether a MP is observed by this KF.
    bool hasObservation(const PtrMapPoint& pMP);
    // Whether the index in image KeyPoints corresponds to an observed MP
    bool hasObservation(size_t idx);

    // Get an observed MP by an index. 从图像特征点id获取观测点MP
    PtrMapPoint getObservation(size_t id);
    // Get the corresponding index of an observed MP
    int getFeatureIndex(const PtrMapPoint& pMP);

    // Set a new MP in location index (used in MP merge)
    void setObservation(const PtrMapPoint& pMP, size_t idx);

    void ComputeBoW(ORBVocabulary* _pVoc);
    DBoW2::FeatureVector GetFeatureVector();
    DBoW2::BowVector GetBowVector();

    vector<PtrMapPoint> GetMapPointMatches();

    void addFtrMeasureFrom(std::shared_ptr<KeyFrame> pKF, const cv::Mat& _mea,
                           const cv::Mat& _info);
    void addFtrMeasureTo(std::shared_ptr<KeyFrame> pKF, const cv::Mat& _mea, const cv::Mat& _info);
    void eraseFtrMeasureFrom(std::shared_ptr<KeyFrame> pKF);
    void eraseFtrMeasureTo(std::shared_ptr<KeyFrame> pKF);

    void setOdoMeasureFrom(std::shared_ptr<KeyFrame> pKF, const cv::Mat& _mea,
                           const cv::Mat& _info);
    void setOdoMeasureTo(std::shared_ptr<KeyFrame> pKF, const cv::Mat& _mea, const cv::Mat& _info);

public:
    static unsigned long mNextIdKF;

    unsigned long mIdKF;

    //! TODO 此变量的作用和变化还需要探究一下, 是否需要加锁访问? 是否可以改成PtrMapPoint?
    vector<cv::Point3f> mViewMPs;  // MP在当前KF相机坐标系下的坐标, 即Pc
    vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> mViewMPsInfo;

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
    bool mbNull;

    std::map<PtrMapPoint, size_t> mObservations;  // size_t为MP在此KF中对应的特征点的索引
    std::map<size_t, PtrMapPoint> mDualObservations;

    std::set<std::shared_ptr<KeyFrame>> mCovisibleKFs;

//    std::mutex mMutexImg; // 这两个在Frame里已经有了
//    std::mutex mMutexPose;
    std::mutex mMutexObs;
    std::mutex mMutexCovis;

};

typedef std::shared_ptr<KeyFrame> PtrKeyFrame;

// bool operator <(const PtrKeyFrame& lhs, const PtrKeyFrame& rhs)
//{
//    return lhs->mIdKF < rhs->mIdKF;
//}

//class KFIdCompareClass : public std::binary_function<PtrKeyFrame, PtrKeyFrame, bool>
//{
//    public:
//        inline bool operator()(const PtrKeyFrame &f1, const PtrKeyFrame &f2)
//        {
//            return f1->mIdKF < f2->mIdKF;
//        }
//};


}  // namespace se2lam
#endif
