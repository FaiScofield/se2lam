#ifndef TESTTRACK_HPP
#define TESTTRACK_HPP

#include "Config.h"
#include "Frame.h"
#include "KeyFrame.h"
#include "Map.h"
#include "MapPoint.h"
#include "ORBmatcher.h"
#include "cvutil.h"

using namespace cv;
using namespace std;
using namespace se2lam;

typedef std::unique_lock<std::mutex> locker;

class TestTrack
{
public:
    TestTrack();
    ~TestTrack() { delete mpORBextractor; }

    void setMap(Map* pMap) { mpMap = pMap; }

    Se2 getCurrentFrameOdo() { return mCurrentFrame.odom; }

    bool isFinished() { return mbFinished; }
    void requestFinish() { mbFinishRequested = true; }

    void createFirstFrame(const cv::Mat& img, const double& imgTime, const Se2& odo);
    void trackReferenceKF(const cv::Mat& img, const double& imgTime, const Se2& odo);
    void addNewKF(PtrKeyFrame& pKF);
    void findCorrespd();
    void updateLocalGraph();

    cv::Mat drawMatchesForPub();

public:
    // Tracking states
    cvu::eTrackingState mState;
    cvu::eTrackingState mLastState;

    int N1 = 0, N2 = 0, N3 = 0;  // for debug print

public:
    void resetLocalTrack();
    void updateFramePose();
    int removeOutliers();

    bool needNewKF(int nTrackedOldMP, int nMatched);
    int doTriangulate();

    bool checkFinish() { return mbFinishRequested; }
    void setFinish() { mbFinished = true; }

public:
    Map* mpMap;
    ORBextractor* mpORBextractor;  // 这里有new

    // local map
    Frame mCurrentFrame;
    PtrKeyFrame mpReferenceKF, mpCurrentKF;

    //    std::vector<PtrMapPoint> mvpLocalMPs;
//    std::map<int, int> mMatchIdx;
    std::vector<int> mvMatchIdx;  // Matches12, 参考帧到当前帧的KP匹配索引
    std::vector<bool> mvbGoodPrl;
    int mnGoodPrl;  // count number of mLocalMPs with good parallax
    int mnInliers, mnMatchSum;

    cv::Mat K, D;
    cv::Mat Homography;

    bool mbPrint;
    bool mbFinishRequested;
    bool mbFinished;

    std::mutex mMutexForPub;
    std::mutex mMutexFinish;
};


TestTrack::TestTrack()
    : mState(cvu::NO_READY_YET), mLastState(cvu::NO_READY_YET), mnGoodPrl(0), mnInliers(0),
      mnMatchSum(0), mbFinishRequested(false), mbFinished(false)
{
    mpORBextractor = new ORBextractor(Config::MaxFtrNumber, 1.f, 1, 1, 20);

    //    mLocalMPs = vector<Point3f>(Config::MaxFtrNumber, Point3f(-1.f, -1.f, -1.f));

    K = Config::Kcam;
    D = Config::Dcam;
    Homography = cv::Mat::eye(3, 3, CV_64FC1);  // double

    mbPrint = Config::GlobalPrint;
}

void TestTrack::createFirstFrame(const Mat& img, const double& imgTime, const Se2& odo)
{
    locker lock(mMutexForPub);
    mCurrentFrame = Frame(img, imgTime, odo, mpORBextractor, K, D);

    if (mCurrentFrame.mvKeyPoints.size() > 200) {
        cout << "========================================================" << endl;
        cout << "[Track] Create first frame with " << mCurrentFrame.N << " features. "
             << "And the start odom is: " << odo << endl;
        cout << "========================================================" << endl;

        mCurrentFrame.setPose(Se2(0, 0, 0));
        mpCurrentKF = make_shared<KeyFrame>(mCurrentFrame);  // 首帧为关键帧
        mpMap->insertKF(mpCurrentKF);  // 首帧的KF直接给Map,没有给LocalMapper
        mpMap->updateLocalGraph();     // 首帧添加到LocalMap里

        resetLocalTrack();

        mState = cvu::OK;
    } else {
        cout << "[Track] Failed to create first frame for too less keyPoints: "
             << mCurrentFrame.mvKeyPoints.size() << endl;
        Frame::nextId = 1;

        mState = cvu::FIRST_FRAME;
    }
}

void TestTrack::trackReferenceKF(const Mat& img, const double& imgTime, const Se2& odo)
{
    locker lock(mMutexForPub);
    mCurrentFrame = Frame(img, imgTime, odo, mpORBextractor, K, D);
    updateFramePose();

    ORBmatcher matcher(0.8);
    mnMatchSum = matcher.MatchByWindowWarp(*mpReferenceKF, mCurrentFrame, Homography, mvMatchIdx, 20);
//    mnMatchSum = matcher.MatchByWindowWarp(*mpReferenceKF, mCurrentFrame, Homography, mMatchIdx, 20);
    mnInliers = removeOutliers();  // H在此更新, 内点数小于10时清除所有匹配点对

    int nTrackedOld = 0;
    if (mnInliers >= 10) { // 内点数大于10进行三角化
        nTrackedOld = doTriangulate();  // 更新viewMPs
        printf("[Track] #%ld Get trackedOld/inliers/matchedSum points = %d/%d/%d\n",
                mCurrentFrame.id, nTrackedOld, mnInliers, mnMatchSum);
    } else {
        fprintf(stderr, "[Track] #%ld Get ZERO inliers! matchedSum = %d.\n",
                mCurrentFrame.id, mnMatchSum);
    }

    N1 += nTrackedOld;
    N2 += mnInliers;
    N3 += mnMatchSum;
    if (mCurrentFrame.id % 10 == 0) {  // 每隔10帧输出一次平均匹配情况
        float nFrames = mCurrentFrame.id * 1.f;
        printf("[Track] #%ld tracked/inliers/matchedSum points average: %.2f/%.2f/%.2f .\n",
               mCurrentFrame.id, N1 * 1.f / nFrames, N2 * 1.f / nFrames, N3 * 1.f / nFrames);
    }

    if (needNewKF(nTrackedOld, mnInliers)) {
        PtrKeyFrame pKF = make_shared<KeyFrame>(mCurrentFrame);
        assert(mpMap->getCurrentKF()->mIdKF == mpReferenceKF->mIdKF);

        addNewKF(pKF);
        mpMap->insertKF(mpCurrentKF);
        mpMap->updateLocalGraph();

        resetLocalTrack();
    }
}

void TestTrack::addNewKF(PtrKeyFrame& pKF)
{
    mpCurrentKF = pKF;

    //! 以下代码等于findCorrespd()函数;
    bool bNoMP = (mpMap->countMPs() == 0);

    // TODO to delete, for debug.
    printf("[Track] #%ld 正在添加新的KF, 参考帧的MP观测数量: %ld, 已有MP数量: %ld\n",
           mpCurrentKF->id, mpReferenceKF->countObservations(), mpMap->countMPs());

    vector<unsigned long> vCrosMPsWithRefKF, vCrosMPsWithLocalMPs, vGeneratedMPs;
    if (!bNoMP) {
        // 1.如果参考帧的第i个特征点有对应的MP，且和当前帧KP有对应的匹配，就给当前帧对应的KP关联上MP
        for (int i = 0, iend = mpReferenceKF->N; i != iend; ++i) {
            if (mvMatchIdx[i] >= 0 && mpReferenceKF->hasObservation(i)) {
                PtrMapPoint pMP = mpReferenceKF->getObservation(i);
                if (pMP->isNull())
                    continue;
                mpCurrentKF->addObservation(pMP, mvMatchIdx[i]);
                pMP->addObservation(mpCurrentKF, mvMatchIdx[i]);
                vCrosMPsWithRefKF.push_back(pMP->mId);
            }
        }

        // 2.和局部地图匹配，关联局部地图里的MP, 其中已经和参考KF有关联的MP不会被匹配, 故不会重复关联
        vector<PtrMapPoint> vLocalMPs = mpMap->getLocalMPs();
        vector<int> vMatchedIdxMPs; // matched index of LocalMPs
        ORBmatcher matcher;
        matcher.MatchByProjection(mpCurrentKF, vLocalMPs, 20, 0, vMatchedIdxMPs);   // 15
        for (int i = 0, iend = mpCurrentKF->N; i != iend; ++i) {
            if (vMatchedIdxMPs[i] < 0)
                continue;

            PtrMapPoint pMP = vLocalMPs[vMatchedIdxMPs[i]];

            // We do triangulation here because we need to produce constraint of
            // mNewKF to the matched old MapPoint.
            Mat Tcw = mpCurrentKF->getPose();
            Point3f x3d = cvu::triangulate(pMP->getMainMeasureProjection(), mpCurrentKF->mvKeyPoints[i].pt,
                                           Config::Kcam * pMP->getMainKF()->getPose().rowRange(0, 3),
                                           Config::Kcam * Tcw.rowRange(0, 3));
            Point3f posNewKF = cvu::se3map(Tcw, x3d);
            if (posNewKF.z > Config::UpperDepth || posNewKF.z < Config::LowerDepth)
                continue;
            if (!pMP->acceptNewObserve(posNewKF, mpCurrentKF->mvKeyPoints[i]))
                continue;

            mpCurrentKF->addObservation(pMP, i);
            pMP->addObservation(mpCurrentKF, i);
            vCrosMPsWithLocalMPs.push_back(pMP->mId);
            printf("[Track] #%ld 局部MP#%ld通过投影与当前帧进行了关联!\n", mpCurrentKF->id, pMP->mId);
        }
    }

    // 3.根据匹配情况给新的KF添加MP
    // 首帧没有处理到这，第二帧进来有了参考帧，但还没有MPS，就会直接执行到这，生成MPs，所以第二帧才有MP
    for (size_t i = 0, iend = mpReferenceKF->N; i != iend; ++i) {
        // 参考帧的特征点i没有对应的MP，且与当前帧KP存在匹配(也没有对应的MP)，则给他们创造MP
        if (mvMatchIdx[i] >= 0 && !mpReferenceKF->hasObservation(i)) {
            if (pKF->hasObservation(mvMatchIdx[i])) {
                PtrMapPoint pMP = pKF->getObservation(mvMatchIdx[i]);

                pMP->addObservation(mpReferenceKF, i);
                mpReferenceKF->addObservation(pMP, i);

                continue;
            }

            Point3f posW = cvu::se3map(cvu::inv(mpReferenceKF->getPose()), mpReferenceKF->mvViewMPs[i]);

            //! TODO to delete, for debug.
            //! 这个应该会出现. 内点数不多的时候没有三角化, 则虽有匹配, 但mvViewMPs没有更新, 故这里不能生成MP
            if (posW.z < 0.f) {
                fprintf(stderr, "[LocalMap] KF#%ld的mvViewMPs[%ld].z < 0. \n", mpReferenceKF->mIdKF, i);
                cerr << "[LocalMap] 此点在成为MP之前的坐标Pc是: " << mpReferenceKF->mvViewMPs[i] << endl;
                cerr << "[LocalMap] 此点在成为MP之后的坐标Pw是: " << posW << endl;
                continue;
            }

            Point3f Pc2 = cvu::se3map(mpCurrentKF->getTcr(), mpReferenceKF->mvViewMPs[i]);
            mpCurrentKF->mvViewMPs[mvMatchIdx[i]] = Pc2;

            PtrMapPoint pMP = std::make_shared<MapPoint>(posW, mvbGoodPrl[i]);

            pMP->addObservation(mpReferenceKF, i);
            pMP->addObservation(mpCurrentKF, mvMatchIdx[i]);
            mpReferenceKF->addObservation(pMP, i);
            mpCurrentKF->addObservation(pMP, mvMatchIdx[i]);

            mpMap->insertMP(pMP);
            vGeneratedMPs.push_back(pMP->mId);
        }
    }
    printf("[Track] #%ld 关联了%ld个MPs, 和局部地图匹配了%ld个MPs, 新生成了%ld个MPs\n", mpCurrentKF->id,
           vCrosMPsWithRefKF.size(), vCrosMPsWithLocalMPs.size(), vGeneratedMPs.size());
    printf("[Track] #%ld 当前帧共有%ld个MP观测, 目前MP总数为: %ld\n",
           mpCurrentKF->id, mpCurrentKF->countObservations(), mpMap->countMPs());

    mpMap->updateCovisibility(mpCurrentKF);
}

void TestTrack::updateFramePose()
{
    Se2 Tb1b2 = mCurrentFrame.odom - mpReferenceKF->odom;
    Se2 Tb2b1 = mpReferenceKF->odom - mCurrentFrame.odom;
    Mat Tc2c1 = Config::Tcb * Tb2b1.toCvSE3() * Config::Tbc;
    Mat Tc1w = mpReferenceKF->getPose();

    mCurrentFrame.setTrb(Tb1b2);
    mCurrentFrame.setTcr(Tc2c1);
    mCurrentFrame.setPose(Tc2c1 * Tc1w);
}

void TestTrack::resetLocalTrack()
{
    mpReferenceKF = mpCurrentKF;

    mnGoodPrl = 0;
    mvMatchIdx.clear();
    Homography = Mat::eye(3, 3, CV_64FC1);
}

cv::Mat TestTrack::drawMatchesForPub()
{
    locker lock(mMutexForPub);
//    if (mCurrentFrame.id == mpReferenceKF->id)
//        return Mat();

    Mat imgWarp, imgOut;
    Mat imgCur = mCurrentFrame.mImage.clone();
    Mat imgRef = mpReferenceKF->mImage.clone();
    if (imgCur.channels() == 1)
        cvtColor(imgCur, imgCur, CV_GRAY2BGR);
    if (imgRef.channels() == 1)
        cvtColor(imgRef, imgRef, CV_GRAY2BGR);

    // 所有KP先上蓝色
    drawKeypoints(imgCur, mCurrentFrame.mvKeyPoints, imgCur, Scalar(255, 0, 0),
                  DrawMatchesFlags::DRAW_OVER_OUTIMG);
    drawKeypoints(imgRef, mpReferenceKF->mvKeyPoints, imgRef, Scalar(255, 0, 0),
                  DrawMatchesFlags::DRAW_OVER_OUTIMG);

    Mat H21 = Homography.inv(DECOMP_SVD);
    warpPerspective(imgCur, imgWarp, H21, imgCur.size());
    hconcat(imgWarp, imgRef, imgOut);

    char strMatches[64];
    std::snprintf(strMatches, 64, "KF: %ld-%ld, M: %d/%d", mpReferenceKF->mIdKF,
                  mCurrentFrame.id - mpReferenceKF->id, mnInliers, mnMatchSum);
    putText(imgOut, strMatches, Point(245, 15), 1, 1, Scalar(0, 0, 255), 2);

    for (size_t i = 0, iend = mvMatchIdx.size(); i != iend; ++i) {
        if (mvMatchIdx[i] < 0) {
            continue;
        } else {
            Point2f ptRef = mpReferenceKF->mvKeyPoints[i].pt;
            Point2f ptCur = mCurrentFrame.mvKeyPoints[mvMatchIdx[i]].pt;

            Mat pt1 = (Mat_<double>(3, 1) << ptCur.x, ptCur.y, 1);
            Mat pt2 = H21 * pt1;
            pt2 /= pt2.at<double>(2);
            Point2f pc(pt2.at<double>(0), pt2.at<double>(1));
            Point2f pr = ptRef + Point2f(imgCur.cols, 0);

            circle(imgOut, pc, 3, Scalar(0, 255, 0));  // 匹配上的为绿色
            circle(imgOut, pr, 3, Scalar(0, 255, 0));
            line(imgOut, pc, pr, Scalar(255, 255, 0, 0.6));
        }
    }

    return imgOut;
}

bool TestTrack::needNewKF(int nTrackedOldMP, int nInliers)
{
    int nMPObs = mpReferenceKF->countObservations();
    bool c1 = (float)nTrackedOldMP <= nMPObs * 0.5f;
    bool c2 = nInliers < 0.1 * Config::MaxFtrNumber;
    bool c3 = mnGoodPrl < 20;
    bool bNeedNewKF = c1 && (c2 /*|| c3*/);
    if (c2)
        printf("[Track] #%ld 增加一个KF, 因为内点数不足10%%! (MP关联数%d)\n", mCurrentFrame.id, nTrackedOldMP);
//    else if (c3)
//        printf("[Track] #%ld 增加一个KF, 因为视差不好! (MP关联数%d)\n", mCurrentFrame.id, nTrackedOldMP);


    return bNeedNewKF;
}

int TestTrack::doTriangulate()
{
    Mat Tcr = mCurrentFrame.getTcr();
    Mat Tc1c2 = cvu::inv(Tcr);
    Point3f Ocam1 = Point3f(0.f, 0.f, 0.f);
    Point3f Ocam2 = Point3f(Tc1c2.rowRange(0, 3).col(3));
    mvbGoodPrl = vector<bool>(mpReferenceKF->N, false);
    mnGoodPrl = 0;

    // 相机1和2的投影矩阵
    cv::Mat Proj1 = Config::PrjMtrxEye;                 // P1 = K * cv::Mat::eye(3, 4, CV_32FC1)
    cv::Mat Proj2 = Config::Kcam * Tcr.rowRange(0, 3);  // P2 = K * Tc2c1(3*4)

    // 1.遍历参考帧的KP
    int nTrackedOld = 0, nGoodDepth = 0;
    for (int i = 0, iend = mpReferenceKF->N; i < iend; ++i) {
        if (mvMatchIdx[i] < 0)
            continue;

        // 2.如果参考帧的KP与当前帧的KP有匹配,且参考帧KP已经有对应的MP观测了，则局部地图点更新为此MP
        if (mpReferenceKF->hasObservation(i)) {
//            mLocalMPs[i] = mpReferenceKF->mViewMPs[i];
            Point3f Pw = mpReferenceKF->getObservation(i)->getPos();
            Point3f Pc = cvu::se3map(mpReferenceKF->getPose(), Pw);
            mpReferenceKF->mvViewMPs[i] = Pc;
            nTrackedOld++;
            continue;
        }

        // 3.如果参考帧KP没有对应的MP，则为此匹配对KP三角化计算深度(相对参考帧的坐标)
        Point2f pt1 = mpReferenceKF->mvKeyPoints[i].pt;
        Point2f pt2 = mCurrentFrame.mvKeyPoints[mvMatchIdx[i]].pt;
        Point3f Pc1 = cvu::triangulate(pt1, pt2, Proj1, Proj2);

        // 3.如果深度计算符合预期，就将有深度的KP更新到ViewMPs里, 其中视差较好的会被标记
        if (Config::acceptDepth(Pc1.z)) {
            nGoodDepth++;
            mpReferenceKF->mvViewMPs[i] = Pc1;
//            mLocalMPs[i] = Pc1;
            // 检查视差
            if (cvu::checkParallax(Ocam1, Ocam2, Pc1, 2)) {
                mnGoodPrl++;
                mvbGoodPrl[i] = true;
            }
        } else {  // 3.如果深度计算不符合预期，就剔除此匹配点对
            mvMatchIdx[i] = -1;
        }
    }
    printf("[Track] #%ld Generate MPs with good parallax/depth: %d/%d\n",
           mCurrentFrame.id, mnGoodPrl, nGoodDepth);

    return nTrackedOld;
}

int TestTrack::removeOutliers()
{
    vector<Point2f> ptRef, ptCur;
    vector<size_t> idxRef;
    idxRef.reserve(mpReferenceKF->N);
    ptRef.reserve(mpReferenceKF->N);
    ptCur.reserve(mCurrentFrame.N);

    for (size_t i = 0, iend = mpReferenceKF->N; i < iend; ++i) {
        if (mvMatchIdx[i] < 0)
            continue;
        idxRef.push_back(i);
        ptRef.push_back(mpReferenceKF->mvKeyPoints[i].pt);
        ptCur.push_back(mCurrentFrame.mvKeyPoints[mvMatchIdx[i]].pt);
    }

    if (ptRef.size() == 0)
        return 0;

    vector<unsigned char> mask;
    Homography = findHomography(ptRef, ptCur, RANSAC, 2.0, mask);

    int nInlier = 0;
    for (size_t i = 0, iend = mask.size(); i < iend; ++i) {
        if (!mask[i])
            mvMatchIdx[idxRef[i]] = -1;
        else
            nInlier++;
    }

//    if (nInlier < 10) {
//        nInlier = 0;
//        std::fill(mvMatchIdx.begin(), mvMatchIdx.end(), -1);
//        Homography = Mat::eye(3, 3, CV_32FC1);
//    }

    return nInlier;
}


#endif  // TESTTRACK_HPP
