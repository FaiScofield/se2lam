/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include "Localizer.h"
#include "ORBmatcher.h"
#include "cvutil.h"
#include "optimizer.h"
#include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>

namespace se2lam
{

using namespace std;
using namespace cv;
using namespace g2o;

typedef std::unique_lock<mutex> locker;

Localizer::Localizer()
{
    mpKFCurr = static_cast<PtrKeyFrame>(nullptr);
    mpKFRef  = static_cast<PtrKeyFrame>(nullptr);

    mpORBextractor = new ORBextractor(Config::MaxFtrNumber, Config::ScaleFactor, Config::MaxLevel);

    mbFinished = false;
    mbFinishRequested = false;

    mState = cvu::NO_READY_YET;
    nLostFrames = 0;

    thMaxDistance = Config::maxLinearSpeed / Config::FPS;
    thMaxAngular = (float)g2o::deg2rad(Config::maxAngularSpeed / Config::FPS);
}

Localizer::~Localizer()
{
    delete mpORBextractor;
    mpORBextractor = nullptr;
}


//! BUG 跑时经常出现SegFault, 有内存泄露/或访问越界！需要检查！
//! 在Release模式和Debug模式下均会出现, 在起始位置离建图时的起始位置比较接近时出现频率较高
void Localizer::run()
{
    //! Init
    static bool bMapOK = !mpMap->empty();
    if (bMapOK) {
        mState = cvu::FIRST_FRAME;
    } else {
        fprintf(stderr, "[Localizer] Map is empty!\n");
        mState = cvu::NO_READY_YET;
        return;
    }
    assert(mState == cvu::FIRST_FRAME);

    WorkTimer timer;
    timer.start();

    ComputeBowVecAll();

    timer.stop();
    fprintf(stderr, "[Localizer] Compute Bow Vectors all cost time: %fms\n", timer.time);
    cerr << "[Localizer] Tracking Start ..." << endl;

    // traj log
    //    ofstream fileOutTraj(se2lam::Config::WRITE_TRAJ_FILE_PATH +
    //    se2lam::Config::WRITE_TRAJ_FILE_NAME);
    // traj log

    //! Main loop
    ros::Rate rate(Config::FPS);
    while (ros::ok()) {
        setLastTrackingState(mState);

        WorkTimer timer;
        timer.start();

        //! Get new measurement: image and odometry
        cv::Mat img;
        Se2 odo;
        Point3f odo_3f;
        bool sensorUpdated = mpSensors->update();
        if (!sensorUpdated)
            continue;

        mpSensors->readData(odo_3f, img);
        odo = Se2(odo_3f.x, odo_3f.y, odo_3f.z);
        ReadFrameInfo(img, odo);  // 每一帧都是KF，mpKFRef数据赋值

        //! 定位成功后会更新Tcw, 并由Tcw更新Twb
        if (getTrackingState() == cvu::FIRST_FRAME) {
            bool bIfRelocalized = relocalization();
            if (bIfRelocalized) {
                setTrackingState(cvu::OK);
                fprintf(stderr, "[Localizer] #%d relocalization successed! Set pose to: [%.4f, %.4f]\n",
                       mpKFCurr->mIdKF, mpKFCurr->Twb.x / 1000, mpKFCurr->Twb.y / 1000);
            }
            continue;
        }

        //! TODO 丢失后,先通过LocalMap计算位姿, 如果位姿不对, 再用回环
        //! TODO 回环验证需要更严格的条件
        if (getTrackingState() == cvu::LOST) {
            bool bIfRelocalized = false;
            bool bIfLocalMatched = false;

//            bIfLocalMatched = TrackLocalMap();
//            if (bIfLocalMatched) {
//                setTrackingState(cvu::OK);
//                fprintf(stderr, "[Localizer] #%d Track LocalMap successed! Set pose to: [%.4f, %.4f]\n",
//                       mpKFCurr->mIdKF, mpKFCurr->Twb.x / 1000, mpKFCurr->Twb.y / 1000);
//                continue;
//            }

            bIfRelocalized = relocalization();
            if (bIfRelocalized) {
                setTrackingState(cvu::OK);
                nLostFrames = 0;
                fprintf(stderr, "[Localizer] #%d Relocalization successed! Set pose to: [%.4f, %.4f]\n",
                       mpKFCurr->mIdKF, mpKFCurr->Twb.x / 1000, mpKFCurr->Twb.y / 1000);
            } else {
                nLostFrames++;
                UpdatePoseCurr(); // 重定位失败,暂时用odom信息更新位姿
                fprintf(stderr, "[Localizer] #%d Relocalization failed! Set pose to: [%.4f, %.4f]\n",
                       mpKFCurr->mIdKF, mpKFCurr->Twb.x / 1000, mpKFCurr->Twb.y / 1000);
            }
            continue;
        }

        if (getTrackingState() == cvu::OK) {
            // 根据mpKFRef的odom信息更新Tcw, 作为初始估计
            UpdatePoseCurr();

            MatchLocalMap();

            int numMPCurr = mpKFCurr->getSizeObsMP();
            if (numMPCurr > 30) {
                DoLocalBA();  // 用局部图优化更新Tcw, 并以此更新Twb
            }

            UpdateCovisKFCurr();

            UpdateLocalMap(1);

            DrawImgCurr();

            DetectIfLost();
        }

        mpKFCurrRefined = mpKFCurr;
//        WriteTrajFile(fileOutTraj);

        timer.stop();
        printf("[Localizer] #%d Localization tracking time: %.2fms, Pose: [%.4f, %.4f]\n", mpKFCurr->mIdKF,
               timer.time, mpKFCurr->Twb.x / 1000, mpKFCurr->Twb.y / 1000);

        if (checkFinish()) {
            break;
        }

        rate.sleep();
    }

    cerr << "[Localizer] Exiting locaizer .." << endl;

    setFinish();
    ros::shutdown();
}

void Localizer::WriteTrajFile(ofstream& file)
{
    if (mpKFCurrRefined == NULL || mpKFCurrRefined->isNull()) {
        return;
    }

    Mat wTb = cvu::inv(se2lam::Config::bTc * mpKFCurrRefined->getPose());
    Mat wRb = wTb.rowRange(0, 3).colRange(0, 3);
    g2o::Vector3D euler = g2o::internal::toEuler(se2lam::toMatrix3d(wRb));

    file << mpKFCurrRefined->id << "," << wTb.at<float>(0, 3) << "," << wTb.at<float>(1, 3) << ","
         << euler(2) << endl;
}

void Localizer::ReadFrameInfo(const Mat& img, const Se2& odo)
{
    locker lock(mMutexKFLocal);

    mFrameRef = mFrameCurr;
    mpKFRef = mpKFCurr;

    mFrameCurr = Frame(img, odo, mpORBextractor, Config::Kcam, Config::Dcam);
    mFrameCurr.Tcw = Config::cTb.clone();
    mFrameCurr.Twb = mFrameRef.Twb + (odo - mFrameRef.odom);

    mpKFCurr = make_shared<KeyFrame>(mFrameCurr);
    mpKFCurr->ComputeBoW(mpORBVoc);
}

void Localizer::MatchLocalMap()
{
    //! Match in local map
    vector<PtrMapPoint> vpMPLocal = GetLocalMPs();
    vector<int> vIdxMPMatched;
    ORBmatcher matcher;
    int numMPMatched = matcher.MatchByProjection(mpKFCurr, vpMPLocal, 15, 2, vIdxMPMatched);

    //! Renew KF observation
    for (int idxKPCurr = 0, idend = vIdxMPMatched.size(); idxKPCurr < idend; idxKPCurr++) {
        int idxMPLocal = vIdxMPMatched[idxKPCurr];

        if (idxMPLocal == -1)
            continue;

        PtrMapPoint pMP = vpMPLocal[idxMPLocal];
        mpKFCurr->addObservation(pMP, idxKPCurr);
    }
}

void Localizer::DoLocalBA()
{
    locker lock(mutex mMutexKFLocal);  //!@Vance: 20190729新增,解决MapPub闪烁问题

    SlamOptimizer optimizer;
    SlamLinearSolver* linearSolver = new SlamLinearSolver();
    SlamBlockSolver* blockSolver = new SlamBlockSolver(linearSolver);
    SlamAlgorithm* solver = new SlamAlgorithm(blockSolver);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    int camParaId = 0;
    addCamPara(optimizer, Config::Kcam, camParaId);

    // Add KFCurr
    addVertexSE3Expmap(optimizer, toSE3Quat(mpKFCurr->getPose()), mpKFCurr->mIdKF, false);
    addPlaneMotionSE3Expmap(optimizer, toSE3Quat(mpKFCurr->getPose()), mpKFCurr->mIdKF, Config::bTc);
    int maxKFid = mpKFCurr->mIdKF + 1;

    // Add MPs in local map as fixed
    const float delta = Config::TH_HUBER;
    set<PtrMapPoint> setMPs = mpKFCurr->getAllObsMPs();

    map<PtrMapPoint, int> Observations = mpKFCurr->getObservations();

    // Add Edges
    for (auto iter = setMPs.begin(); iter != setMPs.end(); iter++) {
        PtrMapPoint pMP = *iter;
        if (pMP->isNull() || !pMP->isGoodPrl())
            continue;

        bool marginal = false;
        bool fixed = true;
        addVertexSBAXYZ(optimizer, toVector3d(pMP->getPos()), maxKFid + pMP->mId, marginal, fixed);

        int ftrIdx = Observations[pMP];
        int octave = pMP->getOctave(mpKFCurr);
        const float invSigma2 = mpKFCurr->mvInvLevelSigma2[octave];
        Eigen::Vector2d uv = toVector2d(mpKFCurr->mvKeyPoints[ftrIdx].pt);
        Eigen::Matrix2d info = Eigen::Matrix2d::Identity() * invSigma2;

        EdgeProjectXYZ2UV* ei = new EdgeProjectXYZ2UV();
        ei->setVertex(
            0, dynamic_cast<OptimizableGraph::Vertex*>(optimizer.vertex(maxKFid + pMP->mId)));
        ei->setVertex(1,
                      dynamic_cast<OptimizableGraph::Vertex*>(optimizer.vertex(mpKFCurr->mIdKF)));
        ei->setMeasurement(uv);
        ei->setParameterId(0, camParaId);
        ei->setInformation(info);
        RobustKernelHuber* rk = new RobustKernelHuber;
        ei->setRobustKernel(rk);
        rk->setDelta(delta);
        ei->setLevel(0);
        optimizer.addEdge(ei);
    }

    WorkTimer timer;
    timer.start();

    optimizer.initializeOptimization(0);
    optimizer.optimize(20);

    timer.stop();

    Mat Tcw = toCvMat(estimateVertexSE3Expmap(optimizer, mpKFCurr->mIdKF));
    mpKFCurr->setPose(Tcw);  // 更新Tcw和Twb

    printf("[Localizer] #%d localBA Time = %.2fms, set pose to [%.4f, %.4f]\n", mpKFCurr->mIdKF,
           timer.time, mpKFCurr->Twb.x / 1000, mpKFCurr->Twb.y / 1000);
}


cv::Mat Localizer::DoPoseGraphOptimization(int iterNum)
{
    locker lock(mutex mMutexKFLocal);  // 加锁, 解决MapPub闪烁问题

    SlamOptimizer optimizer;
    SlamLinearSolver* linearSolver = new SlamLinearSolver();
    SlamBlockSolver* blockSolver = new SlamBlockSolver(linearSolver);
    SlamAlgorithm* solver = new SlamAlgorithm(blockSolver);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    int camParaId = 0;
    addCamPara(optimizer, Config::Kcam, camParaId);

    // Add KFCurr
    addVertexSE3Expmap(optimizer, toSE3Quat(mpKFCurr->getPose()), 0, false); // mpKFCurr->mIdKF
    addPlaneMotionSE3Expmap(optimizer, toSE3Quat(mpKFCurr->getPose()), 0, Config::bTc);

    // Add MPs in local map as fixed
    const float delta = Config::TH_HUBER;
    set<PtrMapPoint> setMPs = mpKFCurr->getAllObsMPs();
    map<PtrMapPoint, int> Observations = mpKFCurr->getObservations();

    // Add MP Vertex and Edges
    for (auto iter = setMPs.begin(); iter != setMPs.end(); iter++) {
        PtrMapPoint pMP = *iter;
        if (pMP->isNull() || !pMP->isGoodPrl())
            continue;

        bool marginal = false;
        bool fixed = true;
        addVertexSBAXYZ(optimizer, toVector3d(pMP->getPos()), 1 + pMP->mId, marginal, fixed);

        int ftrIdx = Observations[pMP]; // 对应的KP索引
        int octave = pMP->getOctave(mpKFCurr);
        const float invSigma2 = mpKFCurr->mvInvLevelSigma2[octave];
        Eigen::Vector2d uv = toVector2d(mpKFCurr->mvKeyPoints[ftrIdx].pt);
        Eigen::Matrix2d info = Eigen::Matrix2d::Identity() * invSigma2;

        EdgeProjectXYZ2UV* ei = new EdgeProjectXYZ2UV();
        ei->setVertex(0, dynamic_cast<OptimizableGraph::Vertex*>(optimizer.vertex(1 + pMP->mId)));
        ei->setVertex(1, dynamic_cast<OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        ei->setMeasurement(uv);
        ei->setParameterId(0, camParaId);
        ei->setInformation(info);
        RobustKernelHuber* rk = new RobustKernelHuber;
        ei->setRobustKernel(rk);
        rk->setDelta(delta);
        ei->setLevel(0);
        optimizer.addEdge(ei);
    }

    WorkTimer timer;
    timer.start();

    optimizer.initializeOptimization(0);
    optimizer.optimize(iterNum);

    timer.stop();

    Mat Tcw = toCvMat(estimateVertexSE3Expmap(optimizer, mpKFCurr->mIdKF));

    Se2 Twb;
    Twb.fromCvSE3(cvu::inv(Tcw) * Config::cTb);
    printf("[Localizer] #%d localBA Time = %.2fms, pose after optimization is: [%.4f, %.4f]\n",
           mpKFCurr->mIdKF, timer.time, Twb.x / 1000, Twb.y / 1000);

    return Tcw;
}

void Localizer::DetectIfLost()
{
    locker lock(mMutexState);

    bool haveKFLocal = GetLocalKFs().size() > 0;
    if (!haveKFLocal) {
        fprintf(stderr, "[Localizer] #%d Lost because with no local KFs!", mpKFCurr->mIdKF);
        mState = cvu::LOST;
        return;
    }

    Se2 dvo = mpKFCurr->Twb - mpKFRef->Twb;
    Se2 dodom = mpKFCurr->odom - mpKFRef->odom;
    float distVO = cv::norm(cv::Point2f(dvo.x, dvo.y));
    float distOdom = cv::norm(cv::Point2f(dodom.x, dodom.y));

    bool distanceOK = distVO <= (1.5 * distOdom + 50);
    bool angleOK = abs(normalize_angle(dvo.theta)) < (1.5 * abs(normalize_angle(dodom.theta)) + 0.05);  // rad
    if (!distanceOK) {
        fprintf(stderr, "[Localizer] #%d Lost because too large distance: %f compared to odom: %f\n",
                mpKFCurr->mIdKF, distVO, distOdom);
        mState = cvu::LOST;
        return;
    }
    else if (!angleOK) {
        fprintf(stderr, "[Localizer] #%d Lost because too large angle: %f degree compared to odom: %f\n",
                mpKFCurr->mIdKF, g2o::rad2deg(abs(normalize_angle(dvo.theta))), g2o::rad2deg(abs(normalize_angle(dvo.theta))));
        mState = cvu::LOST;
        return;
    }

    mState = cvu::OK;
}

void Localizer::setMap(Map* pMap)
{
    mpMap = pMap;
}

void Localizer::setORBVoc(ORBVocabulary* pORBVoc)
{
    mpORBVoc = pORBVoc;
}

void Localizer::ComputeBowVecAll()
{
    // Compute BowVector for all KFs, when BowVec does not exist
    vector<PtrKeyFrame> vpKFs;
    vpKFs = mpMap->getAllKF();
    int numKFs = vpKFs.size();
    for (int i = 0; i < numKFs; i++) {
        PtrKeyFrame pKF = vpKFs[i];
        if (pKF->mbBowVecExist) {
            continue;
        }
        pKF->ComputeBoW(mpORBVoc);
    }
}

bool Localizer::DetectLoopClose()
{
    mvScores.clear();
    mvLocalScores.clear();

    // Loop closure detection with ORB-BOW method
    bool bDetected = false;
    double minScoreBest = 0.05;

    PtrKeyFrame pKFCurr = mpKFCurr;
    if (pKFCurr == NULL) {
        return false;
    }

    DBoW2::BowVector BowVecCurr = pKFCurr->mBowVec;

    vector<PtrKeyFrame> vpKFsAll = mpMap->getAllKF();
    int numKFs = vpKFsAll.size();
    PtrKeyFrame pKFBest;
    double scoreBest = 0;
    int bestIndex = -1;

    for (int i = 0; i < numKFs; i++) {
        PtrKeyFrame pKF = vpKFsAll[i];
        DBoW2::BowVector BowVec = pKF->mBowVec;

        double score = mpORBVoc->score(BowVecCurr, BowVec);
        if (score > scoreBest) {
            scoreBest = score;
            pKFBest = pKF;
            bestIndex = i;
        }
        mvScores.push_back(score);
    }
    // 记录最佳得分KF前后各5帧的KF匹配得分
    int leftIndex = max(0, bestIndex - 5);
    int rightIndex = min(numKFs - 1, bestIndex + 5);
    for (int j = leftIndex; j <= rightIndex; ++j)
        mvLocalScores.push_back(mvScores[j]);

    //! TODO 提升判定条件
    // Loop CLosing Threshold ...
    if (pKFBest != NULL && scoreBest > minScoreBest) {
        mpKFLoop = pKFBest;
        bDetected = true;
    } else {
        mpKFLoop.reset();
    }

    //! DEBUG: Print loop closing info
    if (bDetected && Config::LOCAL_PRINT) {
        fprintf(stderr, "[Localizer] #%d Detect a loop close to #%d, bestScore = %f\n",
               pKFCurr->mIdKF, mpKFLoop->mIdKF, scoreBest);
    } /*else {
        printf("[Localizer] #%d NO good loop close detected!\n", pKFCurr->mIdKF);
    }*/

    return bDetected;
}

bool Localizer::VerifyLoopClose(map<int, int>& mapMatchMP, map<int, int>& mapMatchGood,
                                map<int, int>& mapMatchRaw)
{
    if (mpKFCurr == NULL || mpKFLoop == NULL) {
        return false;
    }

    mapMatchMP.clear();
    mapMatchGood.clear();
    mapMatchRaw.clear();
    map<int, int> mapMatch;

    bool bVerified = false;
    int numMinMatch = 45;

    //! Match ORB KPs
    ORBmatcher matcher;
    bool bIfMatchMPOnly = false;
    matcher.SearchByBoW(mpKFCurr, mpKFLoop, mapMatch, bIfMatchMPOnly);
    mapMatchRaw = mapMatch;

    //! Remove Outliers: by RANSAC of Fundamental
    RemoveMatchOutlierRansac(mpKFCurr, mpKFLoop, mapMatch);
    mapMatchGood = mapMatch;
    int numGoodMatch = mapMatch.size();

    if (numGoodMatch >= numMinMatch) {
        fprintf(stderr, "[Localizer] #%d Loop close verification PASSED! numGoodMatch = %d >= %d\n",
               mpKFCurr->mIdKF, numGoodMatch, numMinMatch);
        bVerified = true;
    } else {
        fprintf(stderr, "[Localizer] #%d Loop close verification FAILED! numGoodMatch = %d < %d\n",
                mpKFCurr->mIdKF, numGoodMatch, numMinMatch);
    }
    return bVerified;
}

vector<PtrKeyFrame> Localizer::GetLocalKFs()
{
    locker lock(mutex mMutexKFLocal);
    return vector<PtrKeyFrame>(mspKFLocal.begin(), mspKFLocal.end());
}

vector<PtrMapPoint> Localizer::GetLocalMPs()
{
    locker lock(mutex mMutexMPLocal);
    return vector<PtrMapPoint>(mspMPLocal.begin(), mspMPLocal.end());
}

void Localizer::DrawImgCurr()
{
    locker lockImg(mMutexImg);

    if (mpKFCurr == NULL)
        return;

    mpKFCurr->copyImgTo(mImgCurr);
    if (mImgCurr.channels() == 1)
        cvtColor(mImgCurr, mImgCurr, CV_GRAY2BGR);

    for (int i = 0, iend = mpKFCurr->mvKeyPoints.size(); i < iend; i++) {
        KeyPoint kpCurr = mpKFCurr->mvKeyPoints[i];
        Point2f ptCurr = kpCurr.pt;

        bool ifMPCurr = mpKFCurr->hasObservation(i);
        Scalar colorCurr;
        if (ifMPCurr) {
            colorCurr = Scalar(0, 255, 0);
        } else {
            colorCurr = Scalar(255, 0, 0);
        }

        circle(mImgCurr, ptCurr, 3, colorCurr, 1);
    }
}

void Localizer::DrawImgMatch(const map<int, int>& mapMatch)
{
    locker lockImg(mMutexImg);

    //! Renew images
    if (mpKFCurr == NULL || mpKFLoop == NULL) {
        return;
    }
    cv::Mat curr = mImgCurr.clone();
    cv::Mat loop = mImgLoop.clone();
    if (mpKFLoop != NULL) {
        mpKFLoop->copyImgTo(mImgLoop);
    } else {
        mImgCurr.copyTo(mImgLoop);
        mImgLoop.setTo(cv::Scalar(0));
    }

    if (mImgCurr.channels() == 1)
        cvtColor(mImgCurr, mImgCurr, CV_GRAY2BGR);
    if (mImgLoop.channels() == 1)
        cvtColor(mImgLoop, mImgLoop, CV_GRAY2BGR);
    if (mImgMatch.channels() == 1)
        cvtColor(mImgMatch, mImgMatch, CV_GRAY2BGR);
    vconcat(mImgCurr, mImgLoop, mImgMatch);  // 垂直拼接

    //! Draw Features
    for (int i = 0, iend = mpKFCurr->mvKeyPoints.size(); i < iend; i++) {
        KeyPoint kpCurr = mpKFCurr->mvKeyPoints[i];
        Point2f ptCurr = kpCurr.pt;
        bool ifMPCurr = mpKFCurr->hasObservation(i);
        Scalar colorCurr;
        if (ifMPCurr) {
            colorCurr = Scalar(0, 255, 0);
        } else {
            colorCurr = Scalar(255, 0, 0);
        }
        circle(mImgMatch, ptCurr, 3, colorCurr, 1);
    }
    for (int i = 0, iend = mpKFLoop->mvKeyPoints.size(); i < iend; i++) {
        KeyPoint kpLoop = mpKFLoop->mvKeyPoints[i];
        Point2f ptLoop = kpLoop.pt;
        Point2f ptLoopMatch = ptLoop;
        ptLoopMatch.y += mImgCurr.rows;

        bool ifMPLoop = mpKFLoop->hasObservation(i);
        Scalar colorLoop;
        if (ifMPLoop) {
            colorLoop = Scalar(0, 255, 0);
        } else {
            colorLoop = Scalar(255, 0, 0);
        }
        circle(mImgMatch, ptLoopMatch, 3, colorLoop, 1);
    }

    //! Draw Matches
    for (auto iter = mapMatch.begin(); iter != mapMatch.end(); iter++) {
        int idxCurr = iter->first;
        KeyPoint kpCurr = mpKFCurr->mvKeyPoints[idxCurr];
        Point2f ptCurr = kpCurr.pt;

        int idxLoop = iter->second;
        KeyPoint kpLoop = mpKFLoop->mvKeyPoints[idxLoop];
        Point2f ptLoop = kpLoop.pt;
        Point2f ptLoopMatch = ptLoop;
        ptLoopMatch.y += mImgCurr.rows;

        bool ifMPCurr = mpKFCurr->hasObservation(idxCurr);
        bool ifMPLoop = mpKFLoop->hasObservation(idxLoop);

        Scalar colorCurr, colorLoop;
        if (ifMPCurr) {
            colorCurr = Scalar(0, 255, 0);
        } else {
            colorCurr = Scalar(255, 0, 0);
        }
        if (ifMPLoop) {
            colorLoop = Scalar(0, 255, 0);
        } else {
            colorLoop = Scalar(255, 0, 0);
        }

        circle(mImgMatch, ptCurr, 3, colorCurr, 1);
        circle(mImgMatch, ptLoopMatch, 3, colorLoop, 1);
        if (ifMPCurr && ifMPLoop) {
            line(mImgMatch, ptCurr, ptLoopMatch, Scalar(0, 97, 255), 2);
        } else {
            line(mImgMatch, ptCurr, ptLoopMatch, colorLoop, 1);
        }
    }

    //! text frame id
    string idCurr = to_string(mpKFCurr->mIdKF), idLoop = to_string(mpKFLoop->mIdKF);
    string score = to_string(mvScores[0]*100), nMatches = to_string(mapMatch.size());
    putText(mImgMatch, idCurr, Point(20, 15), 1, 1.1, Scalar(0, 255, 0), 2);
    putText(mImgMatch, idLoop, Point(20, mImgMatch.rows - 15), 1, 1.1, Scalar(0, 255, 0), 2);
    putText(mImgMatch, nMatches, Point(mImgMatch.cols - 60, 15), 1, 1.1, Scalar(0, 255, 0), 2);
    putText(mImgMatch, score, Point(mImgMatch.cols - 60, mImgMatch.rows - 15), 1, 1.1, Scalar(0, 255, 0), 2);

    if (Config::SAVE_MATCH_IMAGE) {
        string fileName = Config::SAVE_MATCH_IMAGE_PATH + "../loop/" + to_string(mpKFCurr->mIdKF) + ".bmp";
        imwrite(fileName, mImgMatch);
        fprintf(stderr, "[Localizer] #%d Save image to %s\n", mpKFCurr->mIdKF, fileName.c_str());
    }
}

void Localizer::RemoveMatchOutlierRansac(PtrKeyFrame _pKFCurr, PtrKeyFrame _pKFLoop,
                                         map<int, int>& mapMatch)
{
    int numMinMatch = 10;

    // Initialize
    int numMatch = mapMatch.size();
    if (numMatch < numMinMatch) {
        mapMatch.clear();
        return;  // return when small number of matches
    }

    map<int, int> mapMatchGood;
    vector<int> vIdxCurr, vIdxLoop;
    vector<Point2f> vPtCurr, vPtLoop;

    for (auto iter = mapMatch.begin(); iter != mapMatch.end(); iter++) {

        int idxCurr = iter->first;
        int idxLoop = iter->second;

        vIdxCurr.push_back(idxCurr);
        vIdxLoop.push_back(idxLoop);

        vPtCurr.push_back(_pKFCurr->mvKeyPoints[idxCurr].pt);
        vPtLoop.push_back(_pKFLoop->mvKeyPoints[idxLoop].pt);
    }

    // RANSAC with fundemantal matrix
    vector<uchar> vInlier;  // 1 when inliers, 0 when outliers
    findFundamentalMat(vPtCurr, vPtLoop, FM_RANSAC, 3.0, 0.99, vInlier);
    for (unsigned int i = 0; i < vInlier.size(); i++) {
        int idxCurr = vIdxCurr[i];
        int idxLoop = vIdxLoop[i];
        if (vInlier[i] == true) {
            mapMatchGood[idxCurr] = idxLoop;
        }
    }

    // Return good Matches
    mapMatch = mapMatchGood;
}

void Localizer::UpdatePoseCurr()
{
    locker lock(mMutexKFLocal);

    Se2 dOdo = mpKFRef->odom - mpKFCurr->odom;

    mpKFCurr->Tcr = Config::cTb * Se2(dOdo.x, dOdo.y, dOdo.theta).toCvSE3() * Config::bTc;
    mpKFCurr->Tcw = mpKFCurr->Tcr * mpKFRef->Tcw;
    // 更新Twb
    mpKFCurr->Twb.fromCvSE3(cvu::inv(mpKFCurr->Tcw) * Config::cTb);
}

void Localizer::ResetLocalMap()
{
    mspKFLocal.clear();
    mspMPLocal.clear();
}

void Localizer::UpdateLocalMapTrack()
{
//    mspKFLocal.clear();
//    mspMPLocal.clear();
}

void Localizer::UpdateLocalMap(int searchLevel)
{
    locker lock(mMutexLocalMap);
    mspKFLocal.clear();
    mspMPLocal.clear();

    addLocalGraphThroughKdtree(mspKFLocal); // 注意死锁问题
//    size_t m = mspKFLocal.size();
//    std::cout << "addLocalGraphThroughKdtree() size: " << m << std::endl;

//    mspKFLocal = mpKFCurr->getAllCovisibleKFs();
//    auto covisibleKFs = mpKFCurr->getAllCovisibleKFs();
//    mspKFLocal.insert(covisibleKFs.begin(), covisibleKFs.end());
//    size_t n = mspKFLocal.size();
//    std::cout << "getAllCovisibleKFs() size: " << n - m << std::endl;

    while (searchLevel > 0) {
        std::set<PtrKeyFrame> currentLocalKFs = mspKFLocal;
        for (auto iter = currentLocalKFs.begin(); iter != currentLocalKFs.end(); iter++) {
            PtrKeyFrame pKF = *iter;
            std::set<PtrKeyFrame> spKF = pKF->getAllCovisibleKFs();
            mspKFLocal.insert(spKF.begin(), spKF.end());
        }
        searchLevel--;
    }
//    std::cout << "searchLevel size: " << mspKFLocal.size() - n << std::endl;
//    std::cout << "Tatal local KF size: " << mspKFLocal.size() << std::endl;

    for (auto iter = mspKFLocal.begin(), iend = mspKFLocal.end(); iter != iend; iter++) {
        PtrKeyFrame pKF = *iter;
        set<PtrMapPoint> spMP = pKF->getAllObsMPs();
        mspMPLocal.insert(spMP.begin(), spMP.end());
    }
//    std::cout << "get MPs size: " << mspMPLocal.size() << std::endl;
}


void Localizer::MatchLoopClose(map<int, int> mapMatchGood)
{
    // mapMatchGood: KP index map from KFCurr to KFLoop

    //! Set MP observation in KFCurr
    for (auto iter = mapMatchGood.begin(); iter != mapMatchGood.end(); iter++) {
        int idxCurr = iter->first;
        int idxLoop = iter->second;
        bool isMPLoop = mpKFLoop->hasObservation(idxLoop);

        if (isMPLoop) {
            PtrMapPoint pMP = mpKFLoop->getObservation(idxLoop);
            mpKFCurr->addObservation(pMP, idxCurr);
        }
    }
}

void Localizer::UpdateCovisKFCurr()
{
    for (auto iter = mspKFLocal.begin(); iter != mspKFLocal.end(); iter++) {
        set<PtrMapPoint> spMPs;
        PtrKeyFrame pKF = *iter;

        FindCommonMPs(mpKFCurr, pKF, spMPs);

        if (spMPs.size() > 0.1 * mpKFCurr->getSizeObsMP()) {
            mpKFCurr->addCovisibleKF(pKF);
        }
    }
}

int Localizer::FindCommonMPs(const PtrKeyFrame pKF1, const PtrKeyFrame pKF2,
                             set<PtrMapPoint>& spMPs)
{
    spMPs.clear();
    mpMap->compareViewMPs(pKF1, pKF2, spMPs);
    return spMPs.size();
}

void Localizer::setSensors(Sensors* pSensors)
{
    mpSensors = pSensors;
}

void Localizer::requestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool Localizer::checkFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

bool Localizer::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void Localizer::setFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

cv::Point3f Localizer::getCurrentFrameOdom()
{
    locker lock(mMutexKFLocal);
    return mpSensors->getOdo();
}

Se2 Localizer::getCurrKFPose()
{
    locker lock(mMutexKFLocal);
    return mpKFCurr->Twb;
}

Se2 Localizer::getRefKFPose()
{
    locker lock(mMutexKFLocal);
    return mpKFRef->Twb;
}

PtrKeyFrame Localizer::getKFCurr()
{
    locker lock(mMutexKFLocal);
    return mpKFCurr;
}

bool Localizer::relocalization()
{
    bool bIfLoopCloseDetected = false;
    bool bIfLoopCloseVerified = false;

    bIfLoopCloseDetected = DetectLoopClose();

    if (bIfLoopCloseDetected) {
        map<int, int> mapMatchMP, mapMatchGood, mapMatchRaw;
        bIfLoopCloseVerified = VerifyLoopClose(mapMatchMP, mapMatchGood, mapMatchRaw);

        if (bIfLoopCloseVerified) {
            locker lock(mMutexKFLocal);

            mpKFCurr->setPose(mpKFLoop->getPose());
            mpKFCurr->addCovisibleKF(mpKFLoop);

            // Update local map from KFLoop. 更新LocalMap里的KF和MP
            lock.unlock();
            UpdateLocalMap(2);
            lock.lock();

            // Set MP observation of KFCurr from KFLoop. KFLoop里的MP关联到当前KF观测中
            MatchLoopClose(mapMatchGood);

            // Do Local BA and Do outlier rejection. 根据观测的MP做位姿图优化
            DoLocalBA();

            // Set MP observation of KFCurr from local map. LocalMap里的MP添加到当前KF观测中
            MatchLocalMap();

            // Do local BA again and do outlier rejection
            DoLocalBA();

            // output scores
            string file = Config::SAVE_MATCH_IMAGE_PATH + "../loop/scores.txt";
            ofstream ofs(file, ios::app | ios::out);
            if (Config::GLOBAL_PRINT) {
                sort(mvScores.begin(), mvScores.end(), [](double a, double b){ return a > b; });
//                int m = std::min(15, (int)mvScores.size());
//                ofs << mpKFCurr->mIdKF << " LoopClose Scores: ";
//                for (int i = 0; i < m; ++i)
//                    ofs << mvScores[i] << " ";
                for (size_t i = 0; i < mvLocalScores.size(); ++i)
                    ofs << mvLocalScores[i] << " ";
                ofs << std::endl;
            }
            ofs.close();

            // draw after sorted
            DrawImgCurr();
            DrawImgMatch(mapMatchGood);
        } else {
            DrawImgCurr();
            ResetLocalMap();
        }
    }

    return bIfLoopCloseVerified;
}

void Localizer::setTrackingState(const cvu::eTrackingState &s)
{
    locker lock(mMutexState);
    mState = s;
}

void Localizer::setLastTrackingState(const cvu::eTrackingState &s)
{
    locker lock(mMutexState);
    mLastState = s;
}

cvu::eTrackingState Localizer::getTrackingState()
{
    locker lock(mMutexState);
    return mState;
}

cvu::eTrackingState Localizer::getLastTrackingState()
{
    locker lock(mMutexState);
    return mLastState;
}

void Localizer::addLocalGraphThroughKdtree(std::set<PtrKeyFrame>& setLocalKFs)
{
    vector<PtrKeyFrame> vKFsAll = mpMap->getAllKF();
    vector<Point3f> vKFPoses;
    for (size_t i = 0; i < vKFsAll.size(); ++i) {
        Mat Twb = cvu::inv(vKFsAll[i]->getPose()) * Config::cTb;
        Point3f pose(Twb.at<float>(0, 3)/1000.f, Twb.at<float>(1, 3)/1000.f, Twb.at<float>(2, 3)/1000.f);
        vKFPoses.push_back(pose);
    }

    cv::flann::KDTreeIndexParams kdtreeParams;
    cv::flann::Index kdtree(Mat(vKFPoses).reshape(1), kdtreeParams);

    Se2 pose = getCurrKFPose();
    std::vector<float> query = {pose.x/1000.f, pose.y/1000.f, 0.f};
    int size = std::min(vKFsAll.size(), static_cast<size_t>(15));    // 最近的5个KF
    std::vector<int> indices;
    std::vector<float> dists;
    kdtree.knnSearch(query, indices, dists, size, cv::flann::SearchParams());
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] > 0 && dists[i] < 0.3 && vKFsAll[indices[i]]) // 距离在0.3m以内
            setLocalKFs.insert(vKFsAll[indices[i]]);
    }
}

bool Localizer::TrackLocalMap()
{
    UpdatePoseCurr();

    MatchLocalMap();

    int numMPCurr = mpKFCurr->getSizeObsMP();
    if (numMPCurr > 30) {
//                DoLocalBA();  // 用局部图优化更新Tcw, 并以此更新Twb
    }

    UpdateCovisKFCurr();

    UpdateLocalMap(1);

    DrawImgCurr();

    DetectIfLost();


    UpdateLocalMap(1);
    MatchLocalMap();    // 添加观测信息
    Mat Tcw = DoPoseGraphOptimization(20);  // 根据观测做位姿图优化

    // 验证优化结果的准确性

    return true;
}

}  // namespace se2lam
