#ifndef TESTTRACK_HPP
#define TESTTRACK_HPP

#include "Config.h"
#include "Frame.h"
#include "KeyFrame.h"
#include "Map.h"
#include "MapPoint.h"
#include "ORBmatcher.h"
#include "cvutil.h"

namespace se2lam
{

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

    unsigned long mLastRefKFid = 0;
    double trackTimeTatal = 0;

private:
    void processFirstFrame();
    bool trackReferenceKF();
    bool trackLocalMap();

    void updateFramePoseFromRef();
    void doTriangulate(PtrKeyFrame& pKF);
    void resetLocalTrack();
    bool needNewKF();

    // LocalMap functions
    void addNewKF(PtrKeyFrame& pKF, const map<size_t, MPCandidate>& MPCandidates);
    void findCorresponds(const map<size_t, MPCandidate>& MPCandidates);
    void updateLocalGraph();
    void pruneRedundantKFinMap();
    void removeOutlierChi2();
    void localBA();

    // Relocalization
    bool doRelocalization();
    bool detectLoopClose();
    bool verifyLoopClose();
    bool detectIfLost();
    bool detectIfLost(int cros, double projError);
    void doLocalBA(Frame& pKF);
    void startNewTrack();

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
    std::map<int, int> mKPMatchesLoop;  // 和回环帧的BoW匹配情况
    std::vector<int> mvKPMatchIdx, mvKPMatchIdxGood;  // KP内点匹配对/三角化结果都好的匹配
    int mnMPsCandidate;  // 潜在MP总数
    int mnKPMatches, mnKPsInline, mnKPMatchesGood;  // KP匹配对数/内点数/三角化结果不符而剔除的匹配对数
    int mnMPsTracked, mnMPsNewAdded, mnMPsInline;  // 三角化中关联上MP数/新增MP数/总MP配对数(前两者之和)
    int mnLostFrames;  // 连续追踪失败的帧数
    double mLoopScore;

    // New KeyFrame rules (according to fps)
    int nMinFrames, nMaxFrames, nMinMatches;
    float mMaxAngle, mMaxDistance;
    float mCurrRatioGoodDepth, mCurrRatioGoodParl;
    float mLastRatioGoodDepth, mLastRatioGoodParl;

    cv::Mat mAffineMatrix;

    std::mutex mMutexForPub;
};
}

#endif  // TESTTRACK_HPP
