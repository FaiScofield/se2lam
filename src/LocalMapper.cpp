/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG
* (github.com/hbtang)
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

typedef unique_lock<mutex> locker;

LocalMapper::LocalMapper()
{
    mbAcceptNewKF = true;

    mbUpdated = false;
    mbAbortBA = false;
    mbGlobalBABegin = false;
    mbPrintDebugInfo = true;

    mbFinished = false;
    mbFinishRequested = false;
}

void LocalMapper::setMap(Map *pMap)
{
    mpMap = pMap;
    mpMap->setLocalMapper(this);
}

void LocalMapper::setGlobalMapper(GlobalMapper *pGlobalMapper)
{
    mpGlobalMapper = pGlobalMapper;
}

/**
 * @brief LocalMapper::addNewKF 添加KF,更新共视关系,然后插入到Map里
 * @param pKF           待添加的KF
 * @param localMPs      在Tracker里计算出来的MP候选，根据参考帧的观测得到的
 * @param vMatched12    参考帧里KP匹配上当前帧KP的索引
 * @param vbGoodPrl     参考帧里KP匹配上但没有MP对应的点对里，视差比较好的flag
 */
void LocalMapper::addNewKF(PtrKeyFrame &pKF, const vector<Point3f> &localMPs,
                           const vector<int> &vMatched12, const vector<bool> &vbGoodPrl)
{
    {
        locker lock(mMutexNewKFs);
        mpNewKF = pKF;
    }

    // TODO 用BOW加速匹配
//    mNewKF->ComputeBoW(mpORBVoc);

    //! 1.跟新局部地图匹配和共视关系，添加新的MP
    findCorrespd(vMatched12, localMPs, vbGoodPrl);

    //! 2.更新Local Map里的共视关系，MP共同观测超过自身的30%则添加共视关系
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

        // 将KF插入地图
        mpMap->insertKF(pKF);
        mbUpdated = true;   // 这里LocalMapper的主线程会开始工作,优化位姿
    }

    mbAbortBA = false;
    mbAcceptNewKF = false;
}

/**
 * @brief LocalMapper::findCorrespd 根据和参考帧和局部地图的匹配关系关联MP，或生成新的MP并插入到Map里
 * 能关联上的MP都关联上，存在当前帧的mViewMPs变量里。不能关联上MP但和参考帧有匹配的KP，则三角化生成新的MP
 * @param vMatched12    参考帧到当前帧的KP匹配情况
 * @param localMPs      当前帧的MP粗观测, 和参考帧匹配点三角化生成
 * @param vbGoodPrl     参考帧与当前帧KP匹配中没有MP但视差好的标志，生成新MP时要对它的视差好坏进行标记
 */
void LocalMapper::findCorrespd(const vector<int> &vMatched12, const vector<Point3f> &localMPs,
                               const vector<bool> &vbGoodPrl)
{
    bool bNoMP = (mpMap->countMPs() == 0);

    // Identify tracked map points
    PtrKeyFrame pPrefKF = mpMap->getCurrentKF(); // 这是上一参考帧KF

    // 如果参考帧的第i个特征点有对应的MP，且和当前帧KP有对应的匹配，就给当前帧对应的KP关联上MP
    if (!bNoMP) {
        for (int i = 0, iend = pPrefKF->N; i < iend; i++) {

            if (pPrefKF->hasObservation(i) && vMatched12[i] >= 0) {
                PtrMapPoint pMP = pPrefKF->getObservation(i);
                if (!pMP) {
                    cerr << "This is NULL. 这不应该发生啊！！！" << endl;
                    continue;
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
    // 和局部地图匹配，关联局部地图里的MP
    if (!bNoMP) {
        // vector<PtrMapPoint> vLocalMPs(mLocalGraphMPs.begin(),
        // mLocalGraphMPs.end());
        vector<PtrMapPoint> vLocalMPs = mpMap->getLocalMPs();
        vector<int> vMatchedIdxMPs;
        ORBmatcher matcher;
        matcher.MatchByProjection(mpNewKF, vLocalMPs, 20, 2, vMatchedIdxMPs);   // 15
        for (int i = 0; i < mpNewKF->N; i++) {
            if (vMatchedIdxMPs[i] < 0)
                continue;
            PtrMapPoint pMP = vLocalMPs[vMatchedIdxMPs[i]];

            // We do triangulation here because we need to produce constraint of
            // mNewKF to the matched old MapPoint.
            Point3f x3d = cvu::triangulate(pMP->getMainMeasure(), mpNewKF->mvKeyPoints[i].pt,
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

    // Add new points from mNewKF to the map. 给新的KF添加MP
    // 首帧没有处理到这，第二帧进来有了参考帧，但还没有MPS，就会直接执行到这，生成MPs，所以第二帧才有MP
    for (int i = 0, iend = pPrefKF->N; i < iend; i++) {
        // 上一参考帧的特征点i没有对应的MP，且与当前帧KP存在匹配(也没有对应的MP)，则给他们创造MP
        if (!pPrefKF->hasObservation(i) && vMatched12[i] >= 0) {
            if (mpNewKF->hasObservation(vMatched12[i]))
                continue;

            //! TODO 这里对localMPs的坐标系要再确认一下！
            Point3f posW = cvu::se3map(cvu::inv(pPrefKF->Tcw), localMPs[i]);
            Point3f posKF = cvu::se3map(mpNewKF->Tcr, localMPs[i]);
            Eigen::Matrix3d xyzinfo, xyzinfo0;
            Track::calcSE3toXYZInfo(localMPs[i], pPrefKF->Tcw, mpNewKF->Tcw, xyzinfo0, xyzinfo);

            mpNewKF->setViewMP(posKF, vMatched12[i], xyzinfo);
            pPrefKF->setViewMP(localMPs[i], i, xyzinfo0);
            // PtrMapPoint pMP = make_shared<MapPoint>(mNewKF, vMatched12[i],
            // posW, vbGoodPrl[i]);
            PtrMapPoint pMP = make_shared<MapPoint>(posW, vbGoodPrl[i]);

            pMP->addObservation(pPrefKF, i);
            pMP->addObservation(mpNewKF, vMatched12[i]);
            pPrefKF->addObservation(pMP, i);
            mpNewKF->addObservation(pMP, vMatched12[i]);

            mpMap->insertMP(pMP);
        }
    }
}

/**
 * @brief LocalMapper::removeOutlierChi2
 * 把Local Map中的KFs和MPs添加到图中进行优化，然后解除离群MPs的联接关系.
 * 离群MPs观测数小于2时会被删除
 */
void LocalMapper::removeOutlierChi2()
{
    locker lockmapper(mutexMapper);

    SlamOptimizer optimizer;
    initOptimizer(optimizer);

    // 通过Map给优化器添加节点和边，添加的节点和边还会存在下面两个变量里
    vector<vector<EdgeProjectXYZ2UV *>> vpEdgesAll;
    vector<vector<int>> vnAllIdx;
    mpMap->loadLocalGraph(optimizer, vpEdgesAll, vnAllIdx);

    WorkTimer timer;
    timer.start();

    const float chi2 = 25;

    optimizer.initializeOptimization(0);
    optimizer.optimize(10);

    // 优化后去除离群MP
    const int nAllMP = vpEdgesAll.size();
    int nBadMP = 0;
    vector<vector<int>> vnOutlierIdxAll;

    for (int i = 0; i < nAllMP; i++) {
        vector<int> vnOutlierIdx;
        for (int j = 0, jend = vpEdgesAll[i].size(); j < jend; j++) {
            EdgeProjectXYZ2UV *eij = vpEdgesAll[i][j];

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

    printf("[Local] #%d(KF#%d) Remove removeOutlierChi2 Time %fms\n",
            mpNewKF->id, mpNewKF->mIdKF, timer.time);
    printf("[Local] #%d(KF#%d) Outlier MP: %d; total MP: %d\n",
            mpNewKF->id, mpNewKF->mIdKF, nBadMP, nAllMP);
}

/**
 * @brief LocalMapper::localBA 局部图优化
 */
void LocalMapper::localBA()
{
    // 如果这时候全局优化在执行则不会做局部优化
    if (mbGlobalBABegin)
        return;

    locker lockmapper(mutexMapper);

    SlamOptimizer optimizer;
    SlamLinearSolver *linearSolver = new SlamLinearSolver();
    SlamBlockSolver *blockSolver = new SlamBlockSolver(linearSolver);
    SlamAlgorithm *solver = new SlamAlgorithm(blockSolver);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(Config::LocalVerbose);

    optimizer.setForceStopFlag(&mbAbortBA);

    mpMap->loadLocalGraph(optimizer);

    WorkTimer timer;

    // assert(optimizer.verifyInformationMatrices(true));

    optimizer.initializeOptimization(0);
    optimizer.optimize(Config::LocalIterNum);

//    if (mbPrintDebugInfo) {
//        cerr << "[Local] LocalBA cost time " << timer.time << ", number of KFs: "
//             << mpMap->getLocalKFs().size()
////                 << ", number of MP " << vpEdgesAll.size()
//             << endl;
//    }

//    if (solver->currentLambda() > 100.0) {
//        cerr << "-- DEBUG LM: current lambda too large " << solver->currentLambda()
//             << " , reject optimized result" << endl;
//        return;
//    }

    if (mbGlobalBABegin) {
        return;
    }

    mpMap->optimizeLocalGraph(optimizer);   // 用优化后的结果更新KFs和MPs

}

void LocalMapper::run()
{
    if (Config::LocalizationOnly)
        return;

    mbPrintDebugInfo = Config::LocalPrint;

    ros::Rate rate(Config::FPS * 10);
    while (ros::ok()) {
        //! 在处理完addNewKF()函数后(关联并添加MP和连接关系，KF加入LocalMap)，mbUpdated才为true.
        if (mbUpdated) {
            setAcceptNewKF(false);      // 干活了，这单先处理，现在不接单了

            WorkTimer timer;
            timer.start();

            //! 更新了Map里面mLocalGraphKFs,mRefKFs和mLocalGraphMPs三个成员变量
            updateLocalGraphInMap();    // 加了新的KF进来，要更新一下Local Map

            //! 去除冗余的KF和MP，共视关系会被取消，mLocalGraphKFs和mLocalGraphMPs会更新
            pruneRedundantKFinMap();

            //! NOTE 原作者把这个步骤给注释掉了.
            removeOutlierChi2();        // 这里做了一次LocalBA,并对离群MPs取消联接关系,但没有更新位姿

            //! 再次更新LocalMap，由于冗余的KF和MP共视关系已经被取消，所以不必但心它们被添加回来
            updateLocalGraphInMap();

            //! LocalMap优化，并更新Local KFs和MPs的位姿
            localBA();                  // 这里又做了一次LocalBA，有更新位姿

            timer.stop();
            fprintf(stderr, "[Local] #%d(KF#%d) Time cost for LocalMapper's process: %fms.\n",
                    mpNewKF->id, mpNewKF->mIdKF, timer.time);

            //! 标志位置为false防止多次处理，直到加入新的KF才会再次启动
            mbUpdated = false;

            //! 看全局地图有没有在执行Global BA，如果在执行会等它先执行完毕
            mpGlobalMapper->waitIfBusy();

            //! 位姿优化后, 第三次更新LocalMap！
            updateLocalGraphInMap();
        }

        setAcceptNewKF(true);   // 干完了上一单才能告诉Tracker准备好干下一单了

        if (checkFinish())
            break;

        rate.sleep();
    }
    cout << "[Local] Exiting localmapper .." << endl;
    setFinish();
}

void LocalMapper::setAbortBA()
{
    mbAbortBA = true;
}

bool LocalMapper::acceptNewKF()
{
    locker lock(mMutexAccept);
    return mbAcceptNewKF;
}

void LocalMapper::setAcceptNewKF(bool flag)
{
    locker lock(mMutexAccept);
    mbAcceptNewKF = flag;
}

void LocalMapper::printOptInfo(const SlamOptimizer &_optimizer)
{
    // for odometry edges
    for (auto it = _optimizer.edges().begin(); it != _optimizer.edges().end(); it++) {
        g2o::EdgeSE3Expmap *pEdge = static_cast<g2o::EdgeSE3Expmap *>(*it);
        vector<g2o::HyperGraph::Vertex *> vVertices = pEdge->vertices();
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
        g2o::EdgeSE3Expmap *pEdge = static_cast<g2o::EdgeSE3Expmap *>(*it);
        vector<g2o::HyperGraph::Vertex *> vVertices = pEdge->vertices();
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
        g2o::EdgeProjectXYZ2UV *pEdge = static_cast<g2o::EdgeProjectXYZ2UV *>(*it);
        vector<g2o::HyperGraph::Vertex *> vVertices = pEdge->vertices();
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
    locker lock(mutexMapper);
    mpMap->updateLocalGraph();
}

//! 去除冗余关键帧
//! 只要有冗余就一直去除冗余的KF，直到去不了为止; 但也不能去太狠，最多去5帧。
void LocalMapper::pruneRedundantKFinMap()
{
    locker lock(mutexMapper);
    bool bPruned = false;
    int countPrune = 0;
    do {
        bPruned = mpMap->pruneRedundantKF();
        countPrune++;
    } while (bPruned && countPrune < 5);

    printf("[Local] #%d(KF#%d) Prune %d Redundant Local KFs\n",
           mpNewKF->id, mpNewKF->mIdKF, countPrune);
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
    locker lock(mMutexFinish);
    mbFinishRequested = true;
}

//! checkFinish()成功后break跳出主循环,然后就会调用setFinish()结束线程
bool LocalMapper::checkFinish()
{
    locker lock(mMutexFinish);
    return mbFinishRequested;
}

bool LocalMapper::isFinished()
{
    locker lock(mMutexFinish);
    return mbFinished;
}

void LocalMapper::setFinish()
{
    locker lock(mMutexFinish);
    mbFinished = true;
}

//void LocalMapper::getNumFKsInQueue(){
//    locker lock(mMutexNewKFs);
//    return mlNewKeyFrames.size();
//}

}  // namespace se2lam
