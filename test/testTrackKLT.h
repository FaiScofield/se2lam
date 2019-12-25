#ifndef TESTTRACKKLT_H
#define TESTTRACKKLT_H


#include "Config.h"
#include "Frame.h"
#include "KeyFrame.h"
#include "Map.h"
#include "MapPoint.h"
#include "ORBmatcher.h"
#include "cvutil.h"

namespace se2lam
{

#define USE_KLT      1

using namespace cv;
using namespace std;

typedef std::unique_lock<std::mutex> locker;

class MapPublish;

//! 单线程的Track. 测试用!
class TestTrack
{
public:
    TestTrack();
    ~TestTrack();

    bool checkReady();
    void setMap(Map* pMap) { mpMap = pMap; }
    void setMapPublisher(MapPublish* pMapPublisher) { mpMapPublisher = pMapPublisher; }

    void run(const cv::Mat& img, const Se2& odo, const double time);
    int removeOutliers(const PtrKeyFrame& pKFRef, const Frame* pFCur, std::vector<int>& vKPMatchIdx,
                       cv::Mat& A12);
    cv::Mat getAffineMatrix(const Se2& dOdo);

    // for visulization
    unsigned long getCurrentFrameID() { return mCurrentFrame.id; }
    Se2 getCurrentFrameOdo() { return mCurrentFrame.odom; }
    cv::Mat getCurrentFramePose() { return mCurrentFrame.getPose(); }
    void copyForPub();

    cvu::eTrackingState mState;
    cvu::eTrackingState mLastState;
    std::string mImageText;

    // klt
    int mImageRows, mImageCols;
    int mCellWidth, mCellHeight;      // 分块尺寸
    int mnCellsW, mnCellsH, mnCells;  // 分块数
    int mnMaxNumPtsInCell;            // 与分块检点的最大点数
    int mnMaskRadius;                 // mask建立时的特征点周边半径

    cv::Mat mMask;                    // 图像掩码
    cv::Mat mPrevImg, mCurrImg, mForwImg;
    std::vector<cv::Point2f> mvPrevPts, mvCurrPts, mvForwPts;  // 对应的图像特征点
    std::vector<cv::Point2f> mvNewPts;                         // 每一帧中新提取的特征点

    // 以下变量针对当前帧(Forw)
    std::vector<size_t> mvIdxToFirstAdded;      // 与关键帧匹配点的对应索引;
    std::vector<int> mvCellLabel;               // 每个KP所属的分块索引号
    std::vector<int> mvNumPtsInCell;            // 每个分块当前检测到的KP数

    // debug
    unsigned long mLastRefKFid = 0;
    double trackTimeTatal = 0;

private:
    void processFirstFrame(const cv::Mat& img, const Se2& odo, double time);
    bool trackReferenceKF(const cv::Mat& img, const Se2& odo, double time);
//    bool trackLocalMap(const cv::Mat& img, const Se2& odo, double time);
//    bool doRelocalization(const cv::Mat& img, const Se2& odo, double time);
    void predictPointsAndImage(const Se2& dOdo);
    void detectFeaturePointsCell(const cv::Mat& image, const cv::Mat& mask);
    bool inBorder(const cv::Point2f& pt);
    void segImageToCells(const cv::Mat& image, std::vector<cv::Mat>& cellImgs);
    void updateAffineMatix();

    cv::Mat poseOptimize(Frame* pFrame);

    void updateFramePoseFromRef();
    int doTriangulate(PtrKeyFrame& pKF, Frame* frame);
    void resetLocalTrack();
    bool needNewKF();

    // Relocalization & GlobalMap functions
    bool doRelocalization();
    bool detectLoopClose(Frame* frame);
    bool verifyLoopClose(Frame* frame);
    bool detectIfLost();
    bool detectIfLost(Frame& f, const cv::Mat& Tcw_opt);
    void startNewTrack();
    void globalBA();
    int doTriangulate_Global(PtrKeyFrame& pKFLoop, PtrKeyFrame& pKFCurr,
                             std::vector<int>& vKPMatchIdx);
    bool detectLoopClose_Global(PtrKeyFrame& pKF);
    bool verifyLoopClose_Global(PtrKeyFrame& pKF);

    // LocalMap functions
    void addNewKF(PtrKeyFrame& pKF, const map<size_t, MPCandidate>& MPCandidates);
    void findCorresponds(const map<size_t, MPCandidate>& MPCandidates);
    void updateLocalGraph();
    void pruneRedundantKFinMap();
    void removeOutlierChi2();
    void localBA();
    void loadLocalGraph(SlamOptimizer& optimizer);

    // for debug
    void localBA_test();

    Map* mpMap;
    MapPublish* mpMapPublisher;
    ORBextractor* mpORBextractor;  // 这三个有new
    ORBmatcher* mpORBmatcher;
    ORBVocabulary* mpORBvoc;

    // local map
    Frame mCurrentFrame, mLastFrame;
    PtrKeyFrame mpReferenceKF, mpNewKF, mpLoopKF;

    std::map<size_t, MPCandidate> mMPCandidates;  // 候选MP容器
    std::map<int, int> mKPMatchesLoop;            // 和回环帧的BoW匹配情况
    std::vector<int> mvKPMatchIdx, mvKPMatchIdxGood;  // KP内点匹配对/三角化结果都好的匹配
    int mnMPsCandidate;                               // 潜在MP总数
    int mnKPMatches, mnKPsInline, mnKPMatchesGood;  // KP匹配对数/内点数/具有MP的KP匹配对数
    int mnMPsTracked, mnMPsNewAdded, mnMPsInline;   // 关联上MP数/新增MP数/总MP配对数(前两者之和)
    int mnLostFrames;  // 连续追踪失败的帧数
    double mLoopScore;

    // New KeyFrame rules (according to fps)
    int nMinFrames, nMaxFrames, nMinMatches;
    float mMaxAngle, mMaxDistance;

    cv::Mat mAffineMatrix; // A21
    PreSE2 preSE2;  // 参考帧到当前帧的se2积分

    std::mutex mMutexForPub;
};
}



#endif // TESTTRACKKLT_H
