/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include "LocalMapper.h"
#include "GlobalMapper.h"
#include "ORBmatcher.h"
#include "Track.h"
#include "converter.h"
#include "cvutil.h"
#include "optimizer.h"
#include <condition_variable>
#include <fstream>
#include <ros/ros.h>

namespace se2lam
{

using namespace std;
using namespace cv;
using namespace g2o;


#ifdef TIME_TO_LOG_LOCAL_BA
std::ofstream local_ba_time_log;
#endif
typedef lock_guard<mutex> locker;

LocalMapper::LocalMapper()
{
    mbUpdated = false;
    mbAbortBA = false;
    mbAcceptNewKF = true;
    mbGlobalBABegin = false;
    mbPrintDebugInfo = false;

    mbFinished = false;
    mbFinishRequested = false;
}

void LocalMapper::setMap(Map* pMap)
{
    mpMap = pMap;
    mpMap->setLocalMapper(this);
}

void LocalMapper::setGlobalMapper(GlobalMapper* pGlobalMapper)
{
    mpGlobalMapper = pGlobalMapper;
}


void LocalMapper::addNewKF(PtrKeyFrame& pKF, const vector<Point3f>& localMPs,
                           const vector<int>& vMatched12, const vector<bool>& vbGoodPrl)
{

    mpNewKF = pKF;

    findCorrespd(vMatched12, localMPs, vbGoodPrl);

    mpMap->updateCovisibility(mpNewKF);

    {
        PtrKeyFrame pKF0 = mpMap->getCurrentKF();

        // Add KeyFrame-KeyFrame relation
        {
            // There must be covisibility between NewKF and PrevKF
            pKF->addCovisibleKF(pKF0);
            pKF0->addCovisibleKF(pKF);
            Mat measure;
            g2o::Matrix6d info;
            Track::calcOdoConstraintCam(pKF->odom - pKF0->odom, measure, info);

            pKF0->setOdoMeasureFrom(pKF, measure, toCvMat6f(info));
            pKF->setOdoMeasureTo(pKF0, measure, toCvMat6f(info));
        }

        mpMap->insertKF(pKF);
        mbUpdated = true;
    }

    mbAbortBA = false;
    mbAcceptNewKF = false;
}

void LocalMapper::findCorrespd(const vector<int>& vMatched12, const vector<Point3f>& localMPs,
                               const vector<bool>& vbGoodPrl)
{

    bool bNoMP = (mpMap->countMPs() == 0);

    // Identify tracked map points
    PtrKeyFrame pPrefKF = mpMap->getCurrentKF();
    if (!bNoMP) {

        for (int i = 0, iend = pPrefKF->N; i < iend; i++) {
            if (pPrefKF->hasObservation(i) && vMatched12[i] >= 0) {
                PtrMapPoint pMP = pPrefKF->getObservation(i);
                if (!pMP) {
                    printf("This is NULL. /in LM\n");
                }
                Eigen::Matrix3d xyzinfo, xyzinfo0;
                Track::calcSE3toXYZInfo(pPrefKF->mViewMPs[i], cv::Mat::eye(4, 4, CV_32FC1),
                                        mpNewKF->Tcr, xyzinfo0, xyzinfo);
                mpNewKF->setViewMP(cvu::se3map(mpNewKF->Tcr, pPrefKF->mViewMPs[i]), vMatched12[i], xyzinfo);
                mpNewKF->addObservation(pMP, vMatched12[i]);
                pMP->addObservation(mpNewKF, vMatched12[i]);
            }
        }
    }


    // Match features of MapPoints with those in NewKF
    if (!bNoMP) {

        // vector<PtrMapPoint> vLocalMPs(mLocalGraphMPs.begin(), mLocalGraphMPs.end());
        vector<PtrMapPoint> vLocalMPs = mpMap->getLocalMPs();
        vector<int> vMatchedIdxMPs;
        ORBmatcher matcher;
        matcher.MatchByProjection(mpNewKF, vLocalMPs, 15, 2, vMatchedIdxMPs);
        for (int i = 0; i < mpNewKF->N; i++) {
            if (vMatchedIdxMPs[i] < 0)
                continue;
            PtrMapPoint pMP = vLocalMPs[vMatchedIdxMPs[i]];

            // We do triangulation here because we need to produce constraint of
            // mNewKF to the matched old MapPoint.
            Point3f x3d = cvu::triangulate(pMP->getMainMeasure(), mpNewKF->keyPointsUn[i].pt,
                                           Config::Kcam * pMP->mMainKF->Tcw.rowRange(0, 3),
                                           Config::Kcam * mpNewKF->Tcw.rowRange(0, 3));
            Point3f posNewKF = cvu::se3map(mpNewKF->Tcw, x3d);
            if (!pMP->acceptNewObserve(posNewKF, mpNewKF->mvKeyPoints[i])) {
                continue;
            }
            if (posNewKF.z > Config::UpperDepth || posNewKF.z < Config::LowerDepth)
                continue;
            Eigen::Matrix3d infoNew, infoOld;
            Track::calcSE3toXYZInfo(posNewKF, mpNewKF->Tcw, pMP->mMainKF->Tcw, infoNew, infoOld);
            mpNewKF->setViewMP(posNewKF, i, infoNew);
            mpNewKF->addObservation(pMP, i);
            pMP->addObservation(mpNewKF, i);
        }
    }

    // Add new points from mNewKF to the map
    for (int i = 0, iend = pPrefKF->N; i < iend; i++) {
        if (!pPrefKF->hasObservation(i) && vMatched12[i] >= 0) {
            if (mpNewKF->hasObservation(vMatched12[i]))
                continue;

            Point3f posW = cvu::se3map(cvu::inv(pPrefKF->Tcw), localMPs[i]);
            Point3f posKF = cvu::se3map(mpNewKF->Tcr, localMPs[i]);
            Eigen::Matrix3d xyzinfo, xyzinfo0;
            Track::calcSE3toXYZInfo(localMPs[i], pPrefKF->Tcw, mpNewKF->Tcw, xyzinfo0, xyzinfo);

            mpNewKF->setViewMP(posKF, vMatched12[i], xyzinfo);
            pPrefKF->setViewMP(localMPs[i], i, xyzinfo0);
            // PtrMapPoint pMP = make_shared<MapPoint>(mNewKF, vMatched12[i], posW, vbGoodPrl[i]);
            PtrMapPoint pMP = make_shared<MapPoint>(posW, vbGoodPrl[i]);

            pMP->addObservation(pPrefKF, i);
            pMP->addObservation(mpNewKF, vMatched12[i]);
            pPrefKF->addObservation(pMP, i);
            mpNewKF->addObservation(pMP, vMatched12[i]);

            mpMap->insertMP(pMP);
        }
    }
}

void LocalMapper::removeOutlierChi2()
{
    std::unique_lock<mutex> lockmapper(mutexMapper);

    SlamOptimizer optimizer;
    initOptimizer(optimizer);

    vector<vector<EdgeProjectXYZ2UV*>> vpEdgesAll;
    vector<vector<int>> vnAllIdx;
    mpMap->loadLocalGraph(optimizer, vpEdgesAll, vnAllIdx);

    WorkTimer timer;
    timer.start();

    const float chi2 = 25;

    optimizer.initializeOptimization(0);
    optimizer.optimize(10);

    const int nAllMP = vpEdgesAll.size();
    int nBadMP = 0;
    vector<vector<int>> vnOutlierIdxAll;

    for (int i = 0; i < nAllMP; i++) {

        vector<int> vnOutlierIdx;
        for (int j = 0, jend = vpEdgesAll[i].size(); j < jend; j++) {

            EdgeProjectXYZ2UV* eij = vpEdgesAll[i][j];

            if (eij->level() > 0)
                continue;

            eij->computeError();
            bool chi2Bad = eij->chi2() > chi2;

            int idKF = vnAllIdx[i][j];

            if (chi2Bad) {
                eij->setLevel(1);
                vnOutlierIdx.push_back(idKF);
            }
        }

        vnOutlierIdxAll.push_back(vnOutlierIdx);
    }

    timer.stop();

    nBadMP = mpMap->removeLocalOutlierMP(vnOutlierIdxAll);

    vpEdgesAll.clear();
    vnAllIdx.clear();

    if (mbPrintDebugInfo) {
        printf("-- DEBUG LM: Remove Outlier Time %f\n", timer.time);
        printf("-- DEBUG LM: Outliers: %d; totally %d\n", nBadMP, nAllMP);
    }
}

void LocalMapper::localBA()
{

    if (mbGlobalBABegin)
        return;

    std::unique_lock<mutex> lockmapper(mutexMapper);

    SlamOptimizer optimizer;
    SlamLinearSolver* linearSolver = new SlamLinearSolver();
    SlamBlockSolver* blockSolver = new SlamBlockSolver(linearSolver);
    SlamAlgorithm* solver = new SlamAlgorithm(blockSolver);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(Config::LocalVerbose);
#ifndef TIME_TO_LOG_LOCAL_BA
    optimizer.setForceStopFlag(&mbAbortBA);
#endif
    mpMap->loadLocalGraph(optimizer);

    WorkTimer timer;
#ifdef TIME_TO_LOG_LOCAL_BA
    int numKf = mpMap->countLocalKFs();
    int numMp = mpMap->countLocalMPs();

    timer.start();
#endif
    // optimizer.verifyInformationMatrices(true);
    // assert(optimizer.verifyInformationMatrices(true));

    optimizer.initializeOptimization(0);
    optimizer.optimize(Config::LocalIterNum);

#ifdef TIME_TO_LOG_LOCAL_BA
    timer.stop();

    local_ba_time_log << numKf << " " << numMp << " " << timer.time;

    optimizer.clear();
    optimizer.clearParameters();

    vector<vector<EdgeProjectXYZ2UV*>> vpEdgesAll;
    vector<vector<int>> vnAllIdx;
    mpMap->loadLocalGraphOnlyBa(optimizer, vpEdgesAll, vnAllIdx);
    timer.start();
    optimizer.initializeOptimization(0);
    optimizer.optimize(Config::LocalIterNum);
    timer.stop();
    local_ba_time_log << " " << timer.time << std::endl;
#endif

    cout << "[Local][Info ] #" << mpNewKF->id << "(KF#" << mpNewKF->mIdKF
         << ") L6.局部BA涉及的KF/RefKF/MP数量为: " << mpMap->countLocalKFs() << "/"
         << mpMap->countLocalRefKFs() << "/" << mpMap->countLocalMPs() << endl;


#ifdef REJECT_IF_LARGE_LAMBDA
    if (solver->currentLambda() > 100.0) {
        cerr << "-- DEBUG LM: current lambda too large " << solver->currentLambda()
             << " , reject optimized result" << endl;
        return;
    }
#endif

    if (mbGlobalBABegin) {
        return;
    }

#ifndef TIME_TO_LOG_LOCAL_BA
    mpMap->optimizeLocalGraph(optimizer);
#endif
}

void LocalMapper::run()
{

    if (Config::LocalizationOnly)
        return;

    mbPrintDebugInfo = Config::LocalPrint;

    WorkTimer timer;


#ifdef TIME_TO_LOG_LOCAL_BA
    local_ba_time_log.open("/home/vance/output/se2lam_lobal_time.txt");
#endif

    ros::Rate rate(Config::FPS * 10);
    while (ros::ok()) {
        if (mbUpdated) {

            timer.start();
            updateLocalGraphInMap();
            double t1 = timer.count();
            cout << "[Local][Timer] #" << mpNewKF->id << "(KF#" << mpNewKF->mIdKF
                 << ") L4.更新局部地图, 耗时: " << t1 << "ms" << endl;

            timer.start();
            pruneRedundantKfInMap();
            double t2 = timer.count();
            cout << "[Local][Timer] #" << mpNewKF->id << "(KF#" << mpNewKF->mIdKF
                 << ") L5.修剪冗余KF, 耗时: " << t2 << "ms" << endl;

            // removeOutlierChi2();

            // updateLocalGraphInMap();

            timer.start();
            localBA();
            double t3 = timer.count();
            cout << "[Local][Timer] #" << mpNewKF->id << "(KF#" << mpNewKF->mIdKF
                 << ") L6.局部BA, 耗时: " << t3 << "ms" << endl;

            timer.start();
            mbUpdated = false;
            mpGlobalMapper->waitIfBusy();

            updateLocalGraphInMap();

            cout << "[Local][Timer] #" << mpNewKF->id << "(KF#" << mpNewKF->mIdKF
                 << ") L7.局部线程当前帧处理总耗时: " << t1 + t2 + t3 + timer.count() << "ms" << endl;
        }

        mbAcceptNewKF = true;

        if (checkFinish())
            break;

        rate.sleep();
    }

#ifdef TIME_TO_LOG_LOCAL_BA
    local_ba_time_log.close();
#endif

    cerr << "[Local][Info ] Exiting localmapper .." << endl;

    setFinish();
}

void LocalMapper::setAbortBA()
{
    mbAbortBA = true;
}

bool LocalMapper::acceptNewKF()
{
    return mbAcceptNewKF;
}

void LocalMapper::printOptInfo(const SlamOptimizer& _optimizer)
{

    // for odometry edges
    for (auto it = _optimizer.edges().begin(); it != _optimizer.edges().end(); it++) {
        g2o::EdgeSE3Expmap* pEdge = static_cast<g2o::EdgeSE3Expmap*>(*it);
        vector<g2o::HyperGraph::Vertex*> vVertices = pEdge->vertices();
        if (vVertices.size() == 2) {
            int id0 = vVertices[0]->id();
            int id1 = vVertices[1]->id();
            if (max(id0, id1) > (mpNewKF->mIdKF)) {
                // Not odometry edge
                continue;
            }
            cerr << "odometry edge: ";
            cerr << "id0 = " << id0 << "; ";
            cerr << "id1 = " << id1 << "; ";
            cerr << "chi2 = " << pEdge->chi2() << "; ";
            cerr << "err = ";
            for (int i = 0; i < 6; i++) {
                cerr << pEdge->error()(i) << "; ";
            }
            cerr << endl;
        }
    }

    // for plane motion edges
    for (auto it = _optimizer.edges().begin(); it != _optimizer.edges().end(); it++) {
        g2o::EdgeSE3Expmap* pEdge = static_cast<g2o::EdgeSE3Expmap*>(*it);
        vector<g2o::HyperGraph::Vertex*> vVertices = pEdge->vertices();
        if (vVertices.size() == 1) {

            int id0 = vVertices[0]->id();

            cerr << "plane motion edge: ";
            cerr << "id0 = " << id0 << "; ";
            cerr << "chi2 = " << pEdge->chi2() << "; ";
            cerr << "err = ";
            for (int i = 0; i < 6; i++) {
                cerr << pEdge->error()(i) << "; ";
            }
            cerr << endl;
        }
    }

    // for XYZ2UV edges
    for (auto it = _optimizer.edges().begin(); it != _optimizer.edges().end(); it++) {
        g2o::EdgeProjectXYZ2UV* pEdge = static_cast<g2o::EdgeProjectXYZ2UV*>(*it);
        vector<g2o::HyperGraph::Vertex*> vVertices = pEdge->vertices();
        if (vVertices.size() == 2) {
            int id0 = vVertices[0]->id();
            int id1 = vVertices[1]->id();
            if (max(id0, id1) > (mpNewKF->mIdKF)) {
                if (pEdge->chi2() < 10)
                    continue;
                cerr << "XYZ2UV edge: ";
                cerr << "id0 = " << id0 << "; ";
                cerr << "id1 = " << id1 << "; ";
                cerr << "chi2 = " << pEdge->chi2() << "; ";
                cerr << "err = ";
                for (int i = 0; i < 2; i++) {
                    cerr << pEdge->error()(i) << "; ";
                }
                cerr << endl;
            }
        }
    }
}

void LocalMapper::updateLocalGraphInMap()
{
    unique_lock<mutex> lock(mutexMapper);
    mpMap->updateLocalGraph();
}

void LocalMapper::pruneRedundantKfInMap()
{
    std::unique_lock<mutex> lockmapper(mutexMapper);
    bool bPruned = false;
    int countPrune = 0;
    do {
        bPruned = mpMap->pruneRedundantKF();
        countPrune++;
    } while (bPruned && countPrune < 5);
}

void LocalMapper::setGlobalBABegin(bool value)
{
    locker lock(mMutexLocalGraph);
    mbGlobalBABegin = value;
    if (value)
        mbAbortBA = true;
}

void LocalMapper::requestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapper::checkFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

bool LocalMapper::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void LocalMapper::setFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

}  // namespace se2lam
