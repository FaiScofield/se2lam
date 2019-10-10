#ifndef TESTTRACK_HPP
#define TESTTRACK_HPP

#include "Config.h"
#include "Frame.h"
#include "KeyFrame.h"
#include "Map.h"
#include "ORBmatcher.h"
#include "cvutil.h"
#include "utility.h"

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

    size_t copyForPub(cv::Mat& img1, cv::Mat& img2, std::vector<cv::KeyPoint>& kp1,
                      std::vector<cv::KeyPoint>& kp2, std::vector<int>& vMatches12);
    void drawFrameForPub(cv::Mat& imgLeft);
    void drawMatchesForPub(cv::Mat& imgMatch);

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
    Homography = cv::Mat::eye(3, 3, CV_64F);  // double

    mbPrint = Config::GlobalPrint;
}

void TestTrack::createFirstFrame(const Mat& img, const double& imgTime, const Se2& odo)
{
    mCurrentFrame = Frame(img, imgTime, odo, mpORBextractor, K, D);

    if (mCurrentFrame.mvKeyPoints.size() > 200) {
        cout << "========================================================" << endl;
        cout << "[Track] Create first frame with " << mCurrentFrame.N << " features. "
             << "And the start odom is: " << odo << endl;
        cout << "========================================================" << endl;

        mCurrentFrame.setPose(Se2(0, 0, 0));
        mpCurrentKF = make_shared<KeyFrame>(mCurrentFrame);  // 首帧为关键帧
        mpMap->insertKF(mpCurrentKF);  // 首帧的KF直接给Map,没有给LocalMapper

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
    mCurrentFrame = Frame(img, imgTime, odo, mpORBextractor, K, D);
    updateFramePose();

    ORBmatcher matcher(0.8);
    mnMatchSum = matcher.MatchByWindowWarp(*mpReferenceKF, mCurrentFrame, Homography, mvMatchIdx, 20);

    mnInliers = removeOutliers();
    int nTrackedOld = 0;
    if (mnInliers) {
        nTrackedOld = doTriangulate();
        if (mbPrint)
            printf("[Track] #%ld Get trackedOld/inliers/matchedSum points = %d/%d/%d.\n",
                   mCurrentFrame.id, nTrackedOld, mnInliers, mnMatchSum);
    } else {
        if (mbPrint)
            fprintf(stderr, "[Track] #%ld Get ZERO inliers! matchedSum = %d.\n",
                    mCurrentFrame.id, mnMatchSum);
    }

    N1 += nTrackedOld;
    N2 += mnInliers;
    N3 += mnMatchSum;
    if (mbPrint && mCurrentFrame.id % 10 == 0) {  // 每隔10帧输出一次平均匹配情况
        float sum = mCurrentFrame.id - 1.0;
        printf("[Track] #%ld tracked/matched/matchedSum points average: %.2f/%.2f/%.2f .\n",
               mCurrentFrame.id, N1 * 1.f / sum, N2 * 1.f / sum, N3 * 1.f / sum);
    }

    if (needNewKF(nTrackedOld, mnInliers)) {
        PtrKeyFrame pKF = make_shared<KeyFrame>(mCurrentFrame);
        assert(mpMap->getCurrentKF()->mIdKF == mpReferenceKF->mIdKF);

        addNewKF(pKF);
        mpMap->insertKF(mpCurrentKF);

        resetLocalTrack();

        if (mbPrint)
            printf("[Track] #%ld Add new KF at #%ld(KF#%ld)\n", mCurrentFrame.id,
                   mCurrentFrame.id, pKF->mIdKF);
    }
}

void TestTrack::addNewKF(PtrKeyFrame& pKF)
{
    mpCurrentKF = pKF;

    //! 以下代码等于findCorrespd()函数;
    bool bNoMP = (mpMap->countMPs() == 0);

    // TODO to delete, for debug.
    printf("#KF%ld findCorrespd() Count MPs: %ld\n", mpCurrentKF->mIdKF, mpMap->countMPs());
    printf("#KF%ld findCorrespd() Count observations of reference KF: %ld\n",
           mpCurrentKF->mIdKF, mpReferenceKF->countObservation());


    if (!bNoMP) {
        // 1.如果参考帧的第i个特征点有对应的MP，且和当前帧KP有对应的匹配，就给当前帧对应的KP关联上MP
        for (int i = 0, iend = mpReferenceKF->N; i != iend; ++i) {
            if (mvMatchIdx[i] >= 0 && mpReferenceKF->hasObservation(i)) {
                PtrMapPoint pMP = mpReferenceKF->getObservation(i);
                if (!pMP) {
                    //! TODO to delete, for debug.
                    cerr << "[LocalMap] This is NULL. 这不应该发生啊！！！" << endl;
                    continue;
                }
                mpCurrentKF->addObservation(pMP, mvMatchIdx[i]);
                pMP->addObservation(mpCurrentKF, mvMatchIdx[i]);
            }
        }

        // 2.和局部地图匹配，关联局部地图里的MP, 其中已经和参考KF有关联的MP不会被匹配, 故不会重复关联
        vector<PtrMapPoint> vLocalMPs = mpMap->getLocalMPs();
        vector<int> vMatchedIdxMPs;
        ORBmatcher matcher;
        matcher.MatchByProjection(mpCurrentKF, vLocalMPs, 20, 2, vMatchedIdxMPs);   // 15
        for (int i = 0, iend = mpCurrentKF->N; i != iend; ++i) {
            if (vMatchedIdxMPs[i] < 0)
                continue;
            PtrMapPoint pMP = vLocalMPs[vMatchedIdxMPs[i]];

            if (mpCurrentKF->hasObservation(pMP)) {
                //! TODO to delete, for debug. 这个应该不会出现,
                cerr << "[LocalMap] 重复关联了!! 这不应该出现! pMP->id: " << pMP->mId << endl;
                continue;
            }

            // We do triangulation here because we need to produce constraint of
            // mNewKF to the matched old MapPoint.
            Mat Tcw = mpCurrentKF->getPose();
            Point3f x3d = cvu::triangulate(pMP->getMainMeasure(), mpCurrentKF->mvKeyPoints[i].pt,
                                           Config::Kcam * pMP->getMainKF()->getPose().rowRange(0, 3),
                                           Config::Kcam * Tcw.rowRange(0, 3));
            Point3f posNewKF = cvu::se3map(Tcw, x3d);
            if (posNewKF.z > Config::UpperDepth || posNewKF.z < Config::LowerDepth)
                continue;
            if (!pMP->acceptNewObserve(posNewKF, mpCurrentKF->mvKeyPoints[i]))
                continue;
            Eigen::Matrix3d infoNew, infoOld;
            Track::calcSE3toXYZInfo(posNewKF, Tcw, pMP->getMainKF()->getPose(), infoNew, infoOld);
            mpCurrentKF->setViewMP(posNewKF, i, infoNew);
            mpCurrentKF->addObservation(pMP, i);
            pMP->addObservation(mpCurrentKF, i);
        }
    }


    if (!bNoMP) {
        vector<PtrMapPoint> vLocalMPs = mpMap->getLocalMPs();
        vector<int> vMatchedIdxMPs;
        ORBmatcher matcher;
        matcher.MatchByProjection(mpCurrentKF, vLocalMPs, 20, 2, vMatchedIdxMPs);   // 15
        for (int i = 0, iend = mpCurrentKF->N; i != iend; ++i) {
            if (vMatchedIdxMPs[i] < 0)
                continue;
            PtrMapPoint pMP = vLocalMPs[vMatchedIdxMPs[i]];

            if (mpCurrentKF->hasObservation(pMP)) {
                //! TODO to delete, for debug. 这个应该不会出现,
                cerr << "[LocalMap] 重复关联了!! 这不应该出现! pMP->id: " << pMP->mId << endl;
                continue;
            }

            // We do triangulation here because we need to produce constraint of
            // mNewKF to the matched old MapPoint.
            Mat Tcw = mpCurrentKF->getPose();
            Point3f x3d = cvu::triangulate(pMP->getMainMeasure(), mpCurrentKF->mvKeyPoints[i].pt,
                                           Config::Kcam * pMP->getMainKF()->getPose().rowRange(0, 3),
                                           Config::Kcam * Tcw.rowRange(0, 3));
            Point3f posNewKF = cvu::se3map(Tcw, x3d);
            if (posNewKF.z > Config::UpperDepth || posNewKF.z < Config::LowerDepth)
                continue;
            if (!pMP->acceptNewObserve(posNewKF, mpCurrentKF->mvKeyPoints[i]))
                continue;
            Eigen::Matrix3d infoNew, infoOld;
            Track::calcSE3toXYZInfo(posNewKF, Tcw, pMP->getMainKF()->getPose(), infoNew, infoOld);
            mpCurrentKF->setViewMP(posNewKF, i, infoNew);
            mpCurrentKF->addObservation(pMP, i);
            pMP->addObservation(mpCurrentKF, i);
        }
    }

    // 3.根据匹配情况给新的KF添加MP
    // 首帧没有处理到这，第二帧进来有了参考帧，但还没有MPS，就会直接执行到这，生成MPs，所以第二帧才有MP
    int nAddNewMP = 0;
    assert(mpReferenceKF->N == localMPs.size());
    for (size_t i = 0, iend = localMPs.size(); i != iend; ++i) {
        // 参考帧的特征点i没有对应的MP，且与当前帧KP存在匹配(也没有对应的MP)，则给他们创造MP
        if (mvMatchIdx[i] >= 0 && !mpReferenceKF->hasObservation(i)) {
            if (mpCurrentKF->hasObservation(mvMatchIdx[i])) {
                //! TODO to delete, for debug. 这个应该很可能会出现,是否需要给参考帧关联MP?
                cerr << "[LocalMap] 这个可能会出现, 参考帧KP1无观测但当前帧KP2有观测! 是否需要给参考帧关联MP?" << endl;
                continue;
            }

            Point3f posW = cvu::se3map(cvu::inv(mpReferenceKF->getPose()), localMPs[i]);

            //! TODO to delete, for debug. 这个应该很可能会出现
            //! 照理说 localMPs[i] 有一个正常的值的话, 那么就应该有观测出现啊???
            //! 有匹配点对的情况下不会出现! 可以删了. 20191010
            if (localMPs[i].z < 0.f) {
                cerr << "[LocalMap] localMPs[i].z < 0. 这个可能会出现, Pc1的深度有负的情况, 代表此点没有观测" << endl;
                cerr << "[LocalMap] 此点在成为MP之前的坐标Pc是: " << localMPs[i] << endl;
                cerr << "[LocalMap] 此点在成为MP之后的坐标Pw是: " << posW << endl;
            }

            Point3f Pc2 = cvu::se3map(mpCurrentKF->getTcr(), localMPs[i]);
            Eigen::Matrix3d xyzinfo1, xyzinfo2;
            Track::calcSE3toXYZInfo(localMPs[i], mpReferenceKF->getPose(), mpCurrentKF->getPose(), xyzinfo1, xyzinfo2);

            mpReferenceKF->setViewMP(localMPs[i], i, xyzinfo1);
            mpCurrentKF->setViewMP(Pc2, mvMatchIdx[i], xyzinfo2);
            PtrMapPoint pMP = std::make_shared<MapPoint>(posW, vbGoodPrl[i]);

            pMP->addObservation(mpReferenceKF, i);
            pMP->addObservation(mpCurrentKF, mvMatchIdx[i]);
            mpReferenceKF->addObservation(pMP, i);
            mpCurrentKF->addObservation(pMP, mvMatchIdx[i]);

            mpMap->insertMP(pMP);
            nAddNewMP++;
        }
    }
    printf("[Local] #%ld(#KF%ld) findCorrespd() Add new MPs: %d, now tatal MPs = %ld\n",
           mpCurrentKF->id, mpCurrentKF->mIdKF, nAddNewMP, mpMap->countMPs());

    mpMap->updateCovisibility(mpCurrentKF);
}

void TestTrack::updateFramePose()
{
    Se2 Trb = mCurrentFrame.odom - mpReferenceKF->odom;
    Se2 dOdo = mpReferenceKF->odom - mCurrentFrame.odom;
    Mat Tc2c1 = Config::Tcb * dOdo.toCvSE3() * Config::Tbc;
    Mat Tc1w = mpReferenceKF->getPose();

    mCurrentFrame.setTrb(Trb);
    mCurrentFrame.setTcr(Tc2c1);
    mCurrentFrame.setPose(Tc2c1 * Tc1w);
}

void TestTrack::resetLocalTrack()
{
    mpReferenceKF = mpCurrentKF;

    mnGoodPrl = 0;
    mvMatchIdx.clear();
    Homography = Mat::eye(3, 3, CV_64F);
}

size_t TestTrack::copyForPub(Mat& img1, Mat& img2, vector<KeyPoint>& kp1, vector<KeyPoint>& kp2,
                             vector<int>& vMatches12)
{
    locker lock(mMutexForPub);

    if (mvMatchIdx.empty())
        return 0;

    mpReferenceKF->copyImgTo(img1);
    mCurrentFrame.copyImgTo(img2);

    kp1 = mpReferenceKF->mvKeyPoints;
    kp2 = mCurrentFrame.mvKeyPoints;
    vMatches12 = mvMatchIdx;

    return vMatches12.size();
}

void TestTrack::drawFrameForPub(Mat& imgLeft)
{
    locker lock(mMutexForPub);

    //! 画左侧两幅图
    Mat imgUp = mCurrentFrame.mImage.clone();
    Mat imgDown = mpReferenceKF->mImage.clone();
    if (imgUp.channels() == 1)
        cvtColor(imgUp, imgUp, CV_GRAY2BGR);
    if (imgDown.channels() == 1)
        cvtColor(imgDown, imgDown, CV_GRAY2BGR);

    for (size_t i = 0, iend = mvMatchIdx.size(); i != iend; ++i) {
        Point2f ptRef = mpReferenceKF->mvKeyPoints[i].pt;
        if (mvMatchIdx[i] < 0) {
            circle(imgDown, ptRef, 3, Scalar(255, 0, 0), 1);  // 未匹配上的为蓝色
            continue;
        } else {
            circle(imgDown, ptRef, 3, Scalar(0, 255, 0), 1);  // 匹配上的为绿色

            Point2f ptCurr = mCurrentFrame.mvKeyPoints[mvMatchIdx[i]].pt;
            circle(imgUp, ptCurr, 3, Scalar(0, 255, 0), 1);  // 当前KP为绿色
            circle(imgUp, ptRef, 3, Scalar(0, 0, 255), 1);   // 参考KP为红色
            line(imgUp, ptRef, ptCurr, Scalar(0, 255, 0));
        }
    }
    putText(imgUp, to_string(mnInliers), Point(15, 15), 1, 1, Scalar(0, 0, 255), 2);
    putText(imgDown, to_string(mpReferenceKF->mIdKF), Point(15, 15), 1, 1, Scalar(0, 0, 255), 2);
    vconcat(imgUp, imgDown, imgLeft);
}

void TestTrack::drawMatchesForPub(Mat& imgMatch)
{
    locker lock(mMutexForPub);

    Mat imgL = mCurrentFrame.mImage.clone();
    Mat imgR = mpReferenceKF->mImage.clone();
    if (imgL.channels() == 1)
        cvtColor(imgL, imgL, CV_GRAY2BGR);
    if (imgR.channels() == 1)
        cvtColor(imgR, imgR, CV_GRAY2BGR);

    drawKeypoints(imgL, mCurrentFrame.mvKeyPoints, imgL, Scalar(255, 0, 0),
                  DrawMatchesFlags::DRAW_OVER_OUTIMG);
    drawKeypoints(imgR, mpReferenceKF->mvKeyPoints, imgR, Scalar(255, 0, 0),
                  DrawMatchesFlags::DRAW_OVER_OUTIMG);

    Mat H21 = Homography.inv(DECOMP_SVD);
    warpPerspective(imgL, imgL, H21, imgR.size());

    double dt = abs(normalize_angle(mCurrentFrame.odom.theta - mpReferenceKF->odom.theta));
    dt = dt * 180 / M_PI;
    char dt_char[16];
    std::snprintf(dt_char, 16, "%.2f", dt);
    string strTheta = "dt: " + string(dt_char) + "deg";
    putText(imgL, strTheta, Point(15, 15), 1, 1, Scalar(0, 0, 255), 2);
    putText(imgR, to_string(mpReferenceKF->mIdKF), Point(15, 15), 1, 1, Scalar(0, 0, 255), 2);
    hconcat(imgL, imgR, imgMatch);

    for (size_t i = 0, iend = mvMatchIdx.size(); i != iend; ++i) {
        if (mvMatchIdx[i] < 0) {
            continue;
        } else {
            Point2f ptRef = mpReferenceKF->mvKeyPoints[i].pt;
            Point2f ptCur = mCurrentFrame.mvKeyPoints[mvMatchIdx[i]].pt;

            Mat pt1 = (Mat_<double>(3, 1) << ptCur.x, ptCur.y, 1);
            Mat pt2 = Homography * pt1;
            pt2 /= pt2.at<double>(2);
            Point2f pc(pt2.at<double>(0), pt2.at<double>(1));
            Point2f pr = ptRef + Point2f(imgL.cols, 0);

            circle(imgMatch, pc, 3, Scalar(0, 255, 0));  // 匹配上的为绿色
            circle(imgMatch, pr, 3, Scalar(0, 255, 0));
            line(imgMatch, pc, pr, Scalar(255, 255, 0, 0.6));
        }
    }
}

bool TestTrack::needNewKF(int nTrackedOldMP, int nInliers)
{
    int nMPObs = mpReferenceKF->countObservation();
    bool c1 = (float)nTrackedOldMP <= nMPObs * 0.5f;
    bool c2 = nInliers < 10;
    bool bNeedNewKF = c1 && c2;

    return bNeedNewKF;
}

int TestTrack::doTriangulate()
{
    if (mCurrentFrame.id == mpReferenceKF->id)
        return 0;

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
            nTrackedOld++;
            continue;
        }

        // 3.如果参考帧KP没有对应的MP，则为此匹配对KP三角化计算深度(相对参考帧的坐标)
        Point2f pt1 = mpReferenceKF->mvKeyPoints[i].pt;
        Point2f pt2 = mCurrentFrame.mvKeyPoints[mvMatchIdx[i]].pt;
        Point3f Pc1 = cvu::triangulate(pt1, pt2, Proj1, Proj2);

        // 3.如果深度计算符合预期，就将有深度的KP更新到LocalMPs里, 其中视差较好的会被标记
        if (Config::acceptDepth(Pc1.z)) {
            nGoodDepth++;
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
    printf("[Track] #%ld Generate %d points and %d with good parallax.\n",
           mCurrentFrame.id, nGoodDepth, mnGoodPrl);

    return nTrackedOld;
}

int TestTrack::removeOutliers()
{
    vector<Point2f> pt1, pt2;
    vector<size_t> idx;
    pt1.reserve(mpReferenceKF->mvKeyPoints.size());
    pt2.reserve(mCurrentFrame.mvKeyPoints.size());
    idx.reserve(mpReferenceKF->mvKeyPoints.size());

    for (size_t i = 0, iend = mpReferenceKF->mvKeyPoints.size(); i < iend; ++i) {
        if (mvMatchIdx[i] < 0)
            continue;
        idx.push_back(i);
        pt1.push_back(mpReferenceKF->mvKeyPoints[i].pt);
        pt2.push_back(mCurrentFrame.mvKeyPoints[mvMatchIdx[i]].pt);
    }

    if (pt1.size() == 0)
        return 0;

    vector<unsigned char> mask;
    Homography = findHomography(pt1, pt2, RANSAC, 3, mask);

    int nInlier = 0;
    for (size_t i = 0, iend = mask.size(); i < iend; ++i) {
        if (!mask[i])
            mvMatchIdx[idx[i]] = -1;
        else
            nInlier++;
    }

    if (nInlier < 10) {
        nInlier = 0;
        std::fill(mvMatchIdx.begin(), mvMatchIdx.end(), -1);
        Homography = Mat::eye(3, 3, CV_64F);
    }

    return nInlier;
}


#endif  // TESTTRACK_HPP
