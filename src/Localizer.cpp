/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include "converter.h"
#include "Localizer.h"
#include "ORBmatcher.h"
#include "cvutil.h"
#include "optimizer.h"
#include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/flann.hpp>

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

    thMaxDistance = Config::MaxLinearSpeed / Config::FPS;
    thMaxAngular = (float)g2o::deg2rad(Config::MaxAngularSpeed / Config::FPS);
}

Localizer::~Localizer()
{
    delete mpORBextractor;
    mpORBextractor = nullptr;
}

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

    computeBowVecAll();

    timer.stop();
    fprintf(stderr, "[Localizer] Compute Bow Vectors all cost time: %.2fms\n", timer.time);
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
        double imgTime(0.);
        Se2 odo;
        Point3f odo_3f;
        bool sensorUpdated = mpSensors->update();
        if (!sensorUpdated)
            continue;

        mpSensors->readData(odo, img);
        readFrameInfo(img, imgTime, odo);  // 每一帧都是KF，mpKFRef数据赋值

        //! 定位成功后会更新Tcw, 并由Tcw更新Twb
        if (getTrackingState() == cvu::FIRST_FRAME) {
            bool bIfRelocalized = relocalization();
            if (bIfRelocalized) {
                setTrackingState(cvu::OK);
                Se2 Twb = mpKFCurr->getTwb();
                fprintf(stderr, "[Localizer] #%ld relocalization successed! Set pose to: [%.4f, %.4f]\n",
                       mpKFCurr->mIdKF, Twb.x / 1000, Twb.y / 1000);
            }
            continue;
        }

        //! TODO 丢失后,先通过LocalMap计算位姿, 如果位姿不对, 再用回环
        //! TODO 回环验证需要更严格的条件
        if (getTrackingState() == cvu::LOST) {
            bool bIfRelocalized = false;
//            bool bIfLocalMatched = false;

//            bIfLocalMatched = TrackLocalMap();
//            if (bIfLocalMatched) {
//                setTrackingState(cvu::OK);
//                fprintf(stderr, "[Localizer] #%ld Track LocalMap successed! Set pose to: [%.4f, %.4f]\n",
//                       mpKFCurr->mIdKF, mpKFCurr->Twb.x / 1000, mpKFCurr->Twb.y / 1000);
//                continue;
//            }


            bIfRelocalized = relocalization();
            if (bIfRelocalized) {
                setTrackingState(cvu::OK);
                nLostFrames = 0;
                Se2 Twb = mpKFCurr->getTwb();
                fprintf(stderr, "[Localizer] #%ld Relocalization successed! Set pose to: [%.4f, %.4f]\n",
                       mpKFCurr->mIdKF, Twb.x / 1000, Twb.y / 1000);
            } else {
                nLostFrames++;
                updatePoseCurr(); // 重定位失败,暂时用odom信息更新位姿
                Se2 Twb = mpKFCurr->getTwb();
                fprintf(stderr, "[Localizer] #%ld Relocalization failed! Set pose to: [%.4f, %.4f]\n",
                       mpKFCurr->mIdKF, Twb.x / 1000, Twb.y / 1000);
            }
            continue;
        }

        if (getTrackingState() == cvu::OK) {
            // 根据mpKFRef的odom信息更新Tcw, 作为初始估计
            updatePoseCurr();

            matchLocalMap();

            size_t numMPCurr = mpKFCurr->countObservations();
            printf("[Local] #%ld(KF%ld) have %ld MPs observation.\n", mpKFCurr->id, mpKFCurr->mIdKF, numMPCurr);
            if (numMPCurr > 30) {
                doLocalBA();  // 用局部图优化更新Tcw, 并以此更新Twb
            }

            updateCovisKFCurr();

            updateLocalMap(1);

//            drawImgCurr();

            detectIfLost();
        }

        mpKFCurrRefined = mpKFCurr;
//        WriteTrajFile(fileOutTraj);

        timer.stop();
        Se2 Twb = mpKFCurr->getTwb();
        printf("[Localizer] #%ld Localization tracking time: %.2fms, Pose: [%.4f, %.4f]\n", mpKFCurr->mIdKF,
               timer.time, Twb.x / 1000, Twb.y / 1000);

        if (checkFinish()) {
            break;
        }

        rate.sleep();
    }

    cerr << "[Localizer] Exiting locaizer .." << endl;

    setFinish();
    ros::shutdown();
}

void Localizer::writeTrajFile(ofstream& file)
{
    if (mpKFCurrRefined == nullptr || mpKFCurrRefined->isNull()) {
        return;
    }

    Mat wTb = cvu::inv(se2lam::Config::Tbc * mpKFCurrRefined->getPose());
    Mat wRb = wTb.rowRange(0, 3).colRange(0, 3);
    g2o::Vector3D euler = g2o::internal::toEuler(se2lam::toMatrix3d(wRb));

    file << mpKFCurrRefined->id << "," << wTb.at<float>(0, 3) << "," << wTb.at<float>(1, 3) << ","
         << euler(2) << endl;
}

void Localizer::readFrameInfo(const Mat& img, float imgTime, const Se2& odo)
{
    locker lock(mMutexKFLocal);

    mFrameRef = mFrameCurr;
    mpKFRef = mpKFCurr;

    mFrameCurr = Frame(img, odo, mpORBextractor);
    Se2 Twb = mFrameRef.getTwb() + (odo - mFrameRef.odom);
    mFrameCurr.setPose(Twb);

    mpKFCurr = make_shared<KeyFrame>(mFrameCurr);
    mpKFCurr->computeBoW(mpORBVoc);
}

void Localizer::matchLocalMap()
{
    //! Match in local map
    vector<PtrMapPoint> vpMPLocal = getLocalMPs();
    vector<int> vIdxMPMatched;
    ORBmatcher matcher;
    int numMPMatched = matcher.SearchByProjection(mpKFCurr, vpMPLocal, vIdxMPMatched, 15, 2);

    if (numMPMatched > 0) {
        //! Renew KF observation
        for (int idxKPCurr = 0, idend = vIdxMPMatched.size(); idxKPCurr < idend; idxKPCurr++) {
            int idxMPLocal = vIdxMPMatched[idxKPCurr];

            if (idxMPLocal == -1)
                continue;

            PtrMapPoint pMP = vpMPLocal[idxMPLocal];
            mpKFCurr->setObservation(pMP, idxKPCurr);
        }
    }
}

void Localizer::doLocalBA()
{
    locker lock(mutex mMutexKFLocal);  //!@Vance: 20190729新增,解决MapPub闪烁问题

    SlamOptimizer optimizer;
    SlamLinearSolverCholmod* linearSolver = new SlamLinearSolverCholmod();
    SlamBlockSolver* blockSolver = new SlamBlockSolver(linearSolver);
    SlamAlgorithmLM* solver = new SlamAlgorithmLM(blockSolver);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    int camParaId = 0;
    addCamPara(optimizer, Config::Kcam, camParaId);

    // Add KFCurr
    addVertexSE3Expmap(optimizer, toSE3Quat(mpKFCurr->getPose()), mpKFCurr->mIdKF, false);
    addEdgeSE3ExpmapPlaneConstraint(optimizer, toSE3Quat(mpKFCurr->getPose()), mpKFCurr->mIdKF, Config::Tbc);
    int maxKFid = mpKFCurr->mIdKF + 1;

    // Add MPs in local map as fixed
    const float delta = Config::ThHuber;
    vector<PtrMapPoint> setMPs = mpKFCurr->getObservations(true, false);

    // Add Edges
    for (auto iter = setMPs.begin(); iter != setMPs.end(); iter++) {
        PtrMapPoint pMP = *iter;
        if (pMP->isNull() || !pMP->isGoodPrl())
            continue;

        bool marginal = false;
        bool fixed = true;
        addVertexSBAXYZ(optimizer, toVector3d(pMP->getPos()), maxKFid + pMP->mId, marginal, fixed);

        int ftrIdx = /*Observations[pMP]*/0;
        int octave = pMP->getOctave(mpKFCurr); //! TODO 可能返回负数
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
    mpKFCurr->setPose(Tcw);  // 更新Tcw

    Se2 Twb = mpKFCurr->getTwb();
    printf("[Localizer] #%ld localBA Time = %.2fms, set pose to [%.4f, %.4f]\n", mpKFCurr->mIdKF,
           timer.time, Twb.x / 1000, Twb.y / 1000);
}


cv::Mat Localizer::doPoseGraphOptimization(int iterNum)
{
    locker lock(mutex mMutexKFLocal);  // 加锁, 解决MapPub闪烁问题

    SlamOptimizer optimizer;
    SlamLinearSolverCholmod* linearSolver = new SlamLinearSolverCholmod();
    SlamBlockSolver* blockSolver = new SlamBlockSolver(linearSolver);
    SlamAlgorithmLM* solver = new SlamAlgorithmLM(blockSolver);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    int camParaId = 0;
    addCamPara(optimizer, Config::Kcam, camParaId);

    // Add KFCurr
    addVertexSE3Expmap(optimizer, toSE3Quat(mpKFCurr->getPose()), 0, false); // mpKFCurr->mIdKF
    addEdgeSE3ExpmapPlaneConstraint(optimizer, toSE3Quat(mpKFCurr->getPose()), 0, Config::Tbc);

    // Add MPs in local map as fixed
    const float delta = Config::ThHuber;
    vector<PtrMapPoint> setMPs = mpKFCurr->getObservations(true, false);

    // Add MP Vertex and Edges
    for (auto iter = setMPs.begin(); iter != setMPs.end(); iter++) {
        PtrMapPoint pMP = *iter;
        if (pMP->isNull() || !pMP->isGoodPrl())
            continue;

        bool marginal = false;
        bool fixed = true;
        addVertexSBAXYZ(optimizer, toVector3d(pMP->getPos()), 1 + pMP->mId, marginal, fixed);

        int ftrIdx = 0/*Observations[pMP]*/; // 对应的KP索引
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
    Twb.fromCvSE3(cvu::inv(Tcw) * Config::Tcb);
    printf("[Localizer] #%ld localBA Time = %.2fms, pose after optimization is: [%.4f, %.4f]\n",
           mpKFCurr->mIdKF, timer.time, Twb.x / 1000, Twb.y / 1000);

    return Tcw;
}

void Localizer::detectIfLost()
{
    locker lock(mMutexState);

    bool haveKFLocal = getLocalKFs().size() > 0;
    if (!haveKFLocal) {
        fprintf(stderr, "[Localizer] #%ld Lost because with no local KFs!", mpKFCurr->mIdKF);
        mState = cvu::LOST;
        return;
    }

    Se2 dvo = mpKFCurr->getTwb() - mpKFRef->getTwb();
    Se2 dodom = mpKFCurr->odom - mpKFRef->odom;
    float distVO = cv::norm(cv::Point2f(dvo.x, dvo.y));
    float distOdom = cv::norm(cv::Point2f(dodom.x, dodom.y));

    bool distanceOK = distVO <= (1.5 * distOdom + 50);
    bool angleOK = abs(normalizeAngle(dvo.theta)) < (1.5 * abs(normalizeAngle(dodom.theta)) + 0.05);  // rad
    if (!distanceOK) {
        fprintf(stderr, "[Localizer] #%ld Lost because too large distance: %f compared to odom: %f\n",
                mpKFCurr->mIdKF, distVO, distOdom);
        mState = cvu::LOST;
        return;
    }
    else if (!angleOK) {
        fprintf(stderr, "[Localizer] #%ld Lost because too large angle: %f degree compared to odom: %f\n",
                mpKFCurr->mIdKF, g2o::rad2deg(abs(normalizeAngle(dvo.theta))), g2o::rad2deg(abs(normalizeAngle(dvo.theta))));
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

void Localizer::computeBowVecAll()
{
    // Compute BowVector for all KFs, when BowVec does not exist
    vector<PtrKeyFrame> vpKFs;
    vpKFs = mpMap->getAllKFs();
    int numKFs = vpKFs.size();
    for (int i = 0; i < numKFs; ++i) {
        PtrKeyFrame pKF = vpKFs[i];
        if (pKF->mbBowVecExist) {
            continue;
        }
        pKF->computeBoW(mpORBVoc);
    }
}

bool Localizer::detectLoopClose()
{
    mvScores.clear();
    mvLocalScores.clear();

    // Loop closure detection with ORB-BOW method
    bool bDetected = false;
    double minScoreBest = 0.05;

    PtrKeyFrame pKFCurr = mpKFCurr;
    if (pKFCurr == nullptr) {
        return false;
    }

    DBoW2::BowVector BowVecCurr = pKFCurr->mBowVec;

    vector<PtrKeyFrame> vpKFsAll = mpMap->getAllKFs();
    int numKFs = vpKFsAll.size();
    PtrKeyFrame pKFBest;
    double scoreBest = 0;
    int bestIndex = -1;

    for (int i = 0; i < numKFs; ++i) {
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
    if (pKFBest != nullptr && scoreBest > minScoreBest) {
        mpKFLoop = pKFBest;
        bDetected = true;
    } else {
        mpKFLoop.reset();
    }

    //! DEBUG: Print loop closing info
    if (bDetected && Config::LocalPrint) {
        fprintf(stderr, "[Localizer] #%ld Detect a loop close to #%ld, bestScore = %f\n",
               pKFCurr->mIdKF, mpKFLoop->mIdKF, scoreBest);
    } /*else {
        printf("[Localizer] #%ld NO good loop close detected!\n", pKFCurr->mIdKF);
    }*/

    return bDetected;
}

bool Localizer::verifyLoopClose(map<int, int>& mapMatchMP, map<int, int>& mapMatchGood,
                                map<int, int>& mapMatchRaw)
{
    if (mpKFCurr == nullptr || mpKFLoop == nullptr) {
        return false;
    }

    mapMatchMP.clear();
    mapMatchGood.clear();
    mapMatchRaw.clear();
    map<int, int> mapMatch;

    bool bVerified = false;
    int numMinMatch = 45; // Config::MinKPMatchNum

    //! Match ORB KPs
    ORBmatcher matcher;
    bool bIfMatchMPOnly = false;
    matcher.SearchByBoW(static_cast<Frame*>(mpKFCurr.get()), static_cast<Frame*>(mpKFLoop.get()), mapMatch, bIfMatchMPOnly);
    mapMatchRaw = mapMatch;

    //! Remove Outliers: by RANSAC of Fundamental
    removeMatchOutlierRansac(mpKFCurr, mpKFLoop, mapMatch);
    mapMatchGood = mapMatch;
    int numGoodMatch = mapMatch.size();

    if (numGoodMatch >= numMinMatch) {
        fprintf(stderr, "[Localizer] #%ld Loop close verification PASSED! numGoodMatch = %d >= %d\n",
               mpKFCurr->mIdKF, numGoodMatch, numMinMatch);
        bVerified = true;
    } else {
        fprintf(stderr, "[Localizer] #%ld Loop close verification FAILED! numGoodMatch = %d < %d\n",
                mpKFCurr->mIdKF, numGoodMatch, numMinMatch);
    }
    return bVerified;
}

vector<PtrKeyFrame> Localizer::getLocalKFs()
{
    locker lock(mutex mMutexKFLocal);
    return vector<PtrKeyFrame>(mspKFLocal.begin(), mspKFLocal.end());
}

vector<PtrMapPoint> Localizer::getLocalMPs()
{
    locker lock(mutex mMutexMPLocal);
    return vector<PtrMapPoint>(mspMPLocal.begin(), mspMPLocal.end());
}

//void Localizer::drawImgCurr()
//{
//    locker lockImg(mMutexImg);

//    if (mpKFCurr == nullptr)
//        return;

//    mpKFCurr->copyImgTo(mImgCurr);
//    if (mImgCurr.channels() == 1)
//        cvtColor(mImgCurr, mImgCurr, CV_GRAY2BGR);

//    for (int i = 0, iend = mpKFCurr->mvKeyPoints.size(); i < iend; ++i) {
//        KeyPoint kpCurr = mpKFCurr->mvKeyPoints[i];
//        Point2f ptCurr = kpCurr.pt;

//        bool ifMPCurr = mpKFCurr->hasObservationByIndex(i);
//        Scalar colorCurr;
//        if (ifMPCurr) {
//            colorCurr = Scalar(0, 255, 0);
//        } else {
//            colorCurr = Scalar(255, 0, 0);
//        }

//        circle(mImgCurr, ptCurr, 3, colorCurr, 1);
//    }
//}

//void Localizer::drawImgMatch(const map<int, int>& mapMatch)
//{
//    locker lockImg(mMutexImg);

//    //! Renew images
//    if (mpKFCurr == nullptr || mpKFLoop == nullptr) {
//        return;
//    }
//    cv::Mat curr = mImgCurr.clone();
//    cv::Mat loop = mImgLoop.clone();
//    if (mpKFLoop != nullptr) {
//        mpKFLoop->copyImgTo(mImgLoop);
//    } else {
//        mImgCurr.copyTo(mImgLoop);
//        mImgLoop.setTo(cv::Scalar(0));
//    }

//    if (mImgCurr.channels() == 1)
//        cvtColor(mImgCurr, mImgCurr, CV_GRAY2BGR);
//    if (mImgLoop.channels() == 1)
//        cvtColor(mImgLoop, mImgLoop, CV_GRAY2BGR);
//    if (mImgMatch.channels() == 1)
//        cvtColor(mImgMatch, mImgMatch, CV_GRAY2BGR);
//    vconcat(mImgCurr, mImgLoop, mImgMatch);  // 垂直拼接

//    //! Draw Features
//    for (int i = 0, iend = mpKFCurr->mvKeyPoints.size(); i < iend; ++i) {
//        KeyPoint kpCurr = mpKFCurr->mvKeyPoints[i];
//        Point2f ptCurr = kpCurr.pt;
//        bool ifMPCurr = mpKFCurr->hasObservationByIndex(i);
//        Scalar colorCurr;
//        if (ifMPCurr) {
//            colorCurr = Scalar(0, 255, 0);
//        } else {
//            colorCurr = Scalar(255, 0, 0);
//        }
//        circle(mImgMatch, ptCurr, 3, colorCurr, 1);
//    }
//    for (int i = 0, iend = mpKFLoop->mvKeyPoints.size(); i < iend; ++i) {
//        KeyPoint kpLoop = mpKFLoop->mvKeyPoints[i];
//        Point2f ptLoop = kpLoop.pt;
//        Point2f ptLoopMatch = ptLoop;
//        ptLoopMatch.y += mImgCurr.rows;

//        bool ifMPLoop = mpKFLoop->hasObservationByIndex(i);
//        Scalar colorLoop;
//        if (ifMPLoop) {
//            colorLoop = Scalar(0, 255, 0);
//        } else {
//            colorLoop = Scalar(255, 0, 0);
//        }
//        circle(mImgMatch, ptLoopMatch, 3, colorLoop, 1);
//    }

//    //! Draw Matches
//    for (auto iter = mapMatch.begin(); iter != mapMatch.end(); iter++) {
//        int idxCurr = iter->first;
//        KeyPoint kpCurr = mpKFCurr->mvKeyPoints[idxCurr];
//        Point2f ptCurr = kpCurr.pt;

//        int idxLoop = iter->second;
//        KeyPoint kpLoop = mpKFLoop->mvKeyPoints[idxLoop];
//        Point2f ptLoop = kpLoop.pt;
//        Point2f ptLoopMatch = ptLoop;
//        ptLoopMatch.y += mImgCurr.rows;

//        bool ifMPCurr = mpKFCurr->hasObservationByIndex(idxCurr);
//        bool ifMPLoop = mpKFLoop->hasObservationByIndex(idxLoop);

//        Scalar colorCurr, colorLoop;
//        if (ifMPCurr) {
//            colorCurr = Scalar(0, 255, 0);
//        } else {
//            colorCurr = Scalar(255, 0, 0);
//        }
//        if (ifMPLoop) {
//            colorLoop = Scalar(0, 255, 0);
//        } else {
//            colorLoop = Scalar(255, 0, 0);
//        }

//        circle(mImgMatch, ptCurr, 3, colorCurr, 1);
//        circle(mImgMatch, ptLoopMatch, 3, colorLoop, 1);
//        if (ifMPCurr && ifMPLoop) {
//            line(mImgMatch, ptCurr, ptLoopMatch, Scalar(0, 97, 255), 2);
//        } else {
//            line(mImgMatch, ptCurr, ptLoopMatch, colorLoop, 1);
//        }
//    }

//    //! text frame id
//    string idCurr = to_string(mpKFCurr->mIdKF), idLoop = to_string(mpKFLoop->mIdKF);
//    string score = to_string(mvScores[0]*100), nMatches = to_string(mapMatch.size());
//    putText(mImgMatch, idCurr, Point(20, 15), 1, 1.1, Scalar(0, 255, 0), 2);
//    putText(mImgMatch, idLoop, Point(20, mImgMatch.rows - 15), 1, 1.1, Scalar(0, 255, 0), 2);
//    putText(mImgMatch, nMatches, Point(mImgMatch.cols - 60, 15), 1, 1.1, Scalar(0, 255, 0), 2);
//    putText(mImgMatch, score, Point(mImgMatch.cols - 60, mImgMatch.rows - 15), 1, 1.1, Scalar(0, 255, 0), 2);

//    if (Config::SaveMatchImage) {
//        string fileName = Config::MatchImageStorePath + "../loop/" + to_string(mpKFCurr->mIdKF) + ".bmp";
//        imwrite(fileName, mImgMatch);
//        fprintf(stderr, "[Localizer] #%ld Save image to %s\n", mpKFCurr->mIdKF, fileName.c_str());
//    }
//}

void Localizer::removeMatchOutlierRansac(PtrKeyFrame _pKFCurr, PtrKeyFrame _pKFLoop,
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
    // findFundamentalMat(vPtCurr, vPtLoop, FM_RANSAC, 3.0, 0.99, vInlier);
    findHomography(vPtCurr, vPtLoop, FM_RANSAC, 3.0, vInlier);
    for (unsigned int i = 0; i < vInlier.size(); ++i) {
        int idxCurr = vIdxCurr[i];
        int idxLoop = vIdxLoop[i];
        if (vInlier[i] == true) {
            mapMatchGood[idxCurr] = idxLoop;
        }
    }

    // Return good Matches
    mapMatch = mapMatchGood;
}

void Localizer::updatePoseCurr()
{
    locker lock(mMutexKFLocal);

    Se2 dOdo = mpKFRef->odom - mpKFCurr->odom;

//    mpKFCurr->Tcr = Config::Tcb * Se2(dOdo.x, dOdo.y, dOdo.theta).toCvSE3() * Config::Tbc;
//    mpKFCurr->Tcw = mpKFCurr->Tcr * mpKFRef->Tcw;
//    mpKFCurr->Twb.fromCvSE3(cvu::inv(mpKFCurr->Tcw) * Config::Tcb);  // 更新Twb
    mpKFCurr->setTcr(Config::Tcb * Se2(dOdo.x, dOdo.y, dOdo.theta).toCvSE3() * Config::Tbc);
    mpKFCurr->setPose(mpKFCurr->getTcr() * mpKFRef->getPose());
}

void Localizer::resetLocalMap()
{
    mspKFLocal.clear();
    mspMPLocal.clear();
}

void Localizer::updateLocalMap(int searchLevel)
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
            vector<PtrKeyFrame> spKF = pKF->getAllCovisibleKFs();
            mspKFLocal.insert(spKF.begin(), spKF.end());
        }
        searchLevel--;
    }
//    std::cout << "searchLevel size: " << mspKFLocal.size() - n << std::endl;
//    std::cout << "Tatal local KF size: " << mspKFLocal.size() << std::endl;

    for (auto iter = mspKFLocal.begin(), iend = mspKFLocal.end(); iter != iend; iter++) {
        PtrKeyFrame pKF = *iter;
        vector<PtrMapPoint> spMP = pKF->getObservations(true, true);    // MP要有良好视差
        mspMPLocal.insert(spMP.begin(), spMP.end());
    }
//    std::cout << "get MPs size: " << mspMPLocal.size() << std::endl;
}


void Localizer::matchLoopClose(map<int, int> mapMatchGood)
{
    // mapMatchGood: KP index map from KFCurr to KFLoop

    //! Set MP observation in KFCurr
    for (auto iter = mapMatchGood.begin(); iter != mapMatchGood.end(); iter++) {
        int idxCurr = iter->first;
        int idxLoop = iter->second;
        bool isMPLoop = mpKFLoop->hasObservationByIndex(idxLoop);

        if (isMPLoop) {
            PtrMapPoint pMP = mpKFLoop->getObservation(idxLoop);
            mpKFCurr->setObservation(pMP, idxCurr);
        }
    }
}

void Localizer::updateCovisKFCurr()
{
    for (auto iter = mspKFLocal.begin(); iter != mspKFLocal.end(); iter++) {
        set<PtrMapPoint> spMPs;
        PtrKeyFrame pKF = *iter;

        findCommonMPs(mpKFCurr, pKF, spMPs);

        if (spMPs.size() > 0.1 * mpKFCurr->countObservations())
            mpKFCurr->addCovisibleKF(pKF);
    }
}

int Localizer::findCommonMPs(const PtrKeyFrame pKF1, const PtrKeyFrame pKF2,
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

Se2 Localizer::getCurrentFrameOdom()
{
    locker lock(mMutexKFLocal);
    return mpKFCurr->odom;
}

Se2 Localizer::getCurrKFPose()
{
    locker lock(mMutexKFLocal);
    return mpKFCurr->getTwb();
}

Se2 Localizer::getRefKFPose()
{
    locker lock(mMutexKFLocal);
    return mpKFRef->getTwb();
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

    bIfLoopCloseDetected = detectLoopClose();

    if (bIfLoopCloseDetected) {
        map<int, int> mapMatchMP, mapMatchGood, mapMatchRaw;
        bIfLoopCloseVerified = verifyLoopClose(mapMatchMP, mapMatchGood, mapMatchRaw);

        if (bIfLoopCloseVerified) {
            locker lock(mMutexKFLocal);

            mpKFCurr->setPose(mpKFLoop->getPose());
            mpKFCurr->addCovisibleKF(mpKFLoop);

            // Update local map from KFLoop. 更新LocalMap里的KF和MP
            lock.unlock();
            updateLocalMap(2);
            lock.lock();

            // Set MP observation of KFCurr from KFLoop. KFLoop里的MP关联到当前KF观测中
            matchLoopClose(mapMatchGood);

            // Do Local BA and Do outlier rejection. 根据观测的MP做位姿图优化
            doLocalBA();

            // Set MP observation of KFCurr from local map. LocalMap里的MP添加到当前KF观测中
            matchLocalMap();

            // Do local BA again and do outlier rejection
            doLocalBA();

            // output scores
            string file = Config::MatchImageStorePath + "../loop/scores.txt";
            ofstream ofs(file, ios::app | ios::out);
            if (Config::GlobalPrint) {
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
//            drawImgCurr();
//            drawImgMatch(mapMatchGood);
        } else {
//            drawImgCurr();
            resetLocalMap();
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
    vector<PtrKeyFrame> vKFsAll = mpMap->getAllKFs();
    vector<Point3f> vKFPoses(vKFsAll.size());
    for (size_t i = 0, iend = vKFsAll.size(); i != iend; ++i) {
        Mat Twb = cvu::inv(vKFsAll[i]->getPose()) * Config::Tcb;
        Point3f pose(Twb.at<float>(0, 3) * 0.001f, Twb.at<float>(1, 3) * 0.001f,
                     Twb.at<float>(2, 3) * 0.001f);
        vKFPoses[i] = pose;
    }

    cv::flann::KDTreeIndexParams kdtreeParams;
    cv::flann::Index kdtree(Mat(vKFPoses).reshape(1), kdtreeParams);

    Se2 pose = getCurrKFPose();
    std::vector<float> query = {pose.x * 0.001f, pose.y * 0.001f, 0.f};
    std::vector<int> indices;
    std::vector<float> dists;
    kdtree.radiusSearch(query, indices, dists, Config::LocalFrameSearchRadius,
                        Config::Config::MaxLocalFrameNum * 0.5, cv::flann::SearchParams());
    for (size_t i = 0, iend = indices.size(); i != iend; ++i) {
        if (indices[i] > 0 && vKFsAll[indices[i]])
            setLocalKFs.insert(vKFsAll[indices[i]]);
    }
}

bool Localizer::trackLocalMap()
{
    updatePoseCurr();

    matchLocalMap();

    int numMPCurr = mpKFCurr->countObservations();
    if (numMPCurr > 30) {
        doLocalBA();  // 用局部图优化更新Tcw, 并以此更新Twb
    }

    updateCovisKFCurr();

    updateLocalMap(1);

//    drawImgCurr();

    detectIfLost();


    updateLocalMap(1);
    matchLocalMap();    // 添加观测信息
    Mat Tcw = doPoseGraphOptimization(20);  // 根据观测做位姿图优化

    // 验证优化结果的准确性

    return true;
}

}  // namespace se2lam
