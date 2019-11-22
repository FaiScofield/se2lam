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

typedef std::unique_lock<std::mutex> locker;

LocalMapper::LocalMapper()
    : mbFinishRequested(false), mbFinished(false), mpMap(nullptr), mpGlobalMapper(nullptr),
      mpNewKF(nullptr), mbAcceptNewKF(true), mbUpdated(false), mbAbortBA(false), mbGlobalBABegin(false)
{
    mbPrintDebugInfo = Config::LocalPrint;
    mnMaxLocalFrames = Config::MaxLocalFrameNum;
    mnSearchLevel = Config::LocalFrameSearchLevel;
    mfSearchRadius = Config::LocalFrameSearchRadius;
}

void LocalMapper::addNewKF(const PtrKeyFrame& pKF)
{
    locker lock(mMutexNewKFs);
    mlNewKFs.push_back(pKF);
    mbAbortBA = true;   // 来不急的话要放弃优化, 但可以保证处理了共视关系
}

bool LocalMapper::checkNewKF()
{
    locker lock(mMutexNewKFs);
    return (!mlNewKFs.empty());
}

/**
 * @brief 添加KF,更新共视关系,然后插入到Map里
 * @param pKF         待添加的KF
 * @param localMPs    在Tracker里计算出来的MP候选，根据参考帧的观测得到的
 * @param vMatched12  参考帧里KP匹配上当前帧KP的索引
 * @param vbGoodPrl   参考帧里KP匹配上但没有MP对应的点对里，视差比较好的flag
 */
void LocalMapper::processNewKF()
{
    WorkTimer timer;

    {
        locker lock(mMutexNewKFs);
        mpNewKF = mlNewKFs.front();
        mlNewKFs.pop_front();
    }

    //! 1.和上一个KF(即参考帧)以及局部地图关联MP，并添加新的MP, 关键函数!!!
    findCorresponds(vMatched12, vMPCandidates);
    double t1 = timer.count();
    printf("[Local][Timer] #%ld(KF#%ld) L1.1.关联地图点总耗时: %.2fms\n", mpNewKF->id, mpNewKF->mIdKF, t1);

    //! 2.更新局部地图里的共视关系，MP共同观测超过自身的30%则添加共视关系, 更新 mspCovisibleKFs
    timer.start();
    mpNewKF->updateCovisibleKFs();
    double t2 = timer.count();
    printf("[Local][Timer] #%ld(KF#%ld) L1.2.更新共视关系耗时: %.2fms, 共获得%ld个共视KF\n",
           mpNewKF->id, mpNewKF->mIdKF, t2, mpNewKF->countCovisibleKFs());

    timer.start();

    // Map的CurrentKF还是上一时刻的KF, 当前KF处理完后才加入
    PtrKeyFrame pKFLast = mpMap->getCurrentKF();
//    pKFNew->addCovisibleKF(pKFLast);
//    pKFLast->addCovisibleKF(pKFNew);

    // Add KeyFrame-KeyFrame relation. 添加前后KF的约束
    Mat measure;
    g2o::Matrix6d info;
    Track::calcOdoConstraintCam(mpNewKF->odom - pKFLast->odom, measure, info);
    pKFLast->setOdoMeasureFrom(mpNewKF, measure, toCvMat6f(info));
    mpNewKF->setOdoMeasureTo(pKFLast, measure, toCvMat6f(info));

    // 将KF插入地图
    mpMap->insertKF(mpNewKF);
    mbUpdated = true;  //! 这里LocalMapper的主线程会开始工作, 优化新KF的位姿

    mbAbortBA = false;
    mbAcceptNewKF = false;

    double t3 = timer.count();
    printf("[Local][Timer] #%ld(KF#%ld) L1.3. LocalMap的预处理总耗时: %.2fms\n", mpNewKF->id,
           mpNewKF->mIdKF, t1 + t2 + t3);
}


/**
 * @brief 根据和参考帧和局部地图的匹配关系关联MP，会生成新的MP并插入到Map里
 * 能关联上的MP都关联上，不能关联上MP但和参考帧有匹配的KP，则三角化生成新的MP,最后都会更新两KF的mViewMPs。
 * @param vMatched12    参考帧到当前帧的KP匹配情况
 * @param localMPs      参考帧对应观测MP的坐标值, 即Pc1. 理论上比mvViewMPs的有效点更多
 * @param vbGoodPrl 参考帧与当前帧KP匹配中没有MP但视差好的标志，生成新MP时会对它的视差好坏进行标记
 */
void LocalMapper::findCorresponds(const vector<int>& vMatched12, const map<size_t, Point3f>& vMPCandidates)
{
    const bool bNoMP = (mpMap->countMPs() == 0);

    // Identify tracked map points
    PtrKeyFrame pRefKF = mpMap->getCurrentKF();
    const size_t nMPs = mpMap->countMPs();
    const size_t nObs = pRefKF->countObservations();

    if (!bNoMP) {
        int nCros = 0, nProj = 0;

        //! 1.如果参考帧的第i个特征点有对应的MP，且和当前帧KP有对应的匹配，就给当前帧对应的KP关联上MP
        for (int i = 0, iend = pRefKF->N; i < iend; ++i) {
            if (vMatched12[i] >= 0 && pRefKF->hasObservation(i)) {
                PtrMapPoint pMP = pRefKF->getObservation(i);
                if (!pMP || pMP->isNull())
                    continue;
                Eigen::Matrix3d xyzinfo1, xyzinfo2;
                Mat Tcr = mpNewKF->getTcr();
                Track::calcSE3toXYZInfo(pRefKF->mvpMapPoints[i], cv::Mat::eye(4, 4, CV_32FC1), Tcr,
                                        xyzinfo1, xyzinfo2);
                mpNewKF->setObsAndInfo(cvu::se3map(Tcr, pRefKF->mvpMapPoints[i]), vMatched12[i], xyzinfo2);
                mpNewKF->addObservation(pMP, vMatched12[i]);
                pMP->addObservation(mpNewKF, vMatched12[i]);
                nCros++;
            }
        }

        //! 2.和局部地图匹配，关联局部地图里的MP,其中已经和参考KF有关联的MP不会被匹配,故不会重复关联
        vector<PtrMapPoint> vLocalMPs = mpMap->getLocalMPs();
        vector<int> vMatchedIdxMPs;
        ORBmatcher matcher;
        int m = matcher.SearchByProjection(mpNewKF, vLocalMPs, 20, 2, vMatchedIdxMPs);
        for (int i = 0, iend = mpNewKF->N; i < iend; ++i) {
            if (vMatchedIdxMPs[i] < 0)  // vMatchedIdxMPs.size() = mpNewKF->N
                continue;

            PtrMapPoint& pMP = vLocalMPs[vMatchedIdxMPs[i]];

            // We do triangulation here because we need to produce constraint of
            // mNewKF to the matched old MapPoint.
            Mat Tcw = mpNewKF->getPose();
            Point3f Pw = cvu::triangulate(pMP->getMainMeasureProjection(), mpNewKF->mvKeyPoints[i].pt,
                                          Config::Kcam * pMP->getMainKF()->getPose().rowRange(0, 3),
                                          Config::Kcam * Tcw.rowRange(0, 3));
            Point3f Pc = cvu::se3map(Tcw, Pw);
            if (!Config::acceptDepth(Pc.z))
                continue;
            if (!pMP->acceptNewObserve(Pc, mpNewKF->mvKeyPoints[i]))
                continue;

            Eigen::Matrix3d infoNew, infoOld;
            Track::calcSE3toXYZInfo(Pc, Tcw, pMP->getMainKF()->getPose(), infoNew, infoOld);
            mpNewKF->setObsAndInfo(Pc, i, infoNew);
            mpNewKF->addObservation(pMP, i);
            pMP->addObservation(mpNewKF, i);
            nProj++;
        }

        printf("[Local][Info ] #%ld(KF#%ld) 关联地图点1/3, 关联参考帧MP数/参考帧MP总数: %d/%ld\n",
               mpNewKF->id, mpNewKF->mIdKF, nCros, nObs);
        printf("[Local][Info ] #%ld(KF#%ld) 关联地图点2/3, 关联的MP数/投影匹配数/当前MP总数: "
               "%d/%d/%ld\n",
               mpNewKF->id, mpNewKF->mIdKF, nProj, m, nMPs);
    }


    //! 把所有的可见MP变成观测
    int nAddNewMP = 0;
    const vector<PtrMapPoint> vViewMPs = mpNewKF->getObservations();
    for (size_t i = 0, iend = mpNewKF->N; i != iend; ++i) {
        const PtrMapPoint& pMP = vViewMPs[i];
        if (pMP == nullptr || pMP->isNull())
            continue;
        if (mpNewKF->hasObservation(pMP)) //TODO 要先做局部地图投影
            continue;

        Point3f Pcr = cvu::se3map(pRefKF->getPose(), pMP->getPos());
        Eigen::Matrix3d xyzinfo1, xyzinfo2;
        Track::calcSE3toXYZInfo(Pcr, pRefKF->getPose(), pMP->getMainKF()->getPose(), xyzinfo1, xyzinfo2);

        pMP->addObservation(mpNewKF, i);

        // 参考帧的特征点i没有对应的MP，且与当前帧KP存在匹配
        if (vMatched12[i] >= 0 && !pRefKF->hasObservation(i)) {
            // 情况1, 当前帧KP存在MP观测(可能是通过局部地图投影匹配得到的), 应与参考帧关联. (新增)
            if (mpNewKF->hasObservation(vMatched12[i])) {
                PtrMapPoint pMP = mpNewKF->getObservation(vMatched12[i]);

                Point3f Pcr = cvu::se3map(pRefKF->getPose(), pMP->getPos());
                Eigen::Matrix3d xyzinfo1, xyzinfo2;
                Track::calcSE3toXYZInfo(Pcr, pRefKF->getPose(), pMP->getMainKF()->getPose(), xyzinfo1, xyzinfo2);
                pRefKF->setObsAndInfo(Pcr, i, xyzinfo1);
                pRefKF->addObservation(pMP, i);
                pMP->addObservation(pRefKF, i);
                continue;
            }

            //! TODO to delete, for debug.
            //! 这个应该会出现. 内点数不多的时候没有三角化, 则虽有匹配, 但mvViewMPs没有更新,
            //! 故这里不能生成MP! 照理说 localMPs[i] 有一个正常的值的话, 那么就应该有观测出现啊???
            if (localMPs[i].z < 0) {
                fprintf(stderr, "[Local][Warni] KF#%ld的mvViewMPs[%ld].z < 0: [%.1f, %.1f, %.1f]\n",
                        pRefKF->mIdKF, i, localMPs[i].x, localMPs[i].y, localMPs[i].z);
                continue;
            }

            // 情况2, 当前帧KP也没有对应的MP，这时就三角化为它们创造MP
            Point3f posW = cvu::se3map(cvu::inv(pRefKF->getPose()), localMPs[i]);
            Point3f Pc2 = cvu::se3map(mpNewKF->getTcr(), localMPs[i]);
            Eigen::Matrix3d xyzinfo1, xyzinfo2;
            Track::calcSE3toXYZInfo(localMPs[i], pRefKF->getPose(), mpNewKF->getPose(), xyzinfo1, xyzinfo2);

            PtrMapPoint pNewMP = std::make_shared<MapPoint>(posW, vbGoodPrl[i]);
            pRefKF->setObsAndInfo(localMPs[i], i, xyzinfo1);
            pRefKF->addObservation(pNewMP, i);
            pNewMP->addObservation(pRefKF, i);
            mpNewKF->setObsAndInfo(Pc2, vMatched12[i], xyzinfo2);
            mpNewKF->addObservation(pNewMP, vMatched12[i]);
            pNewMP->addObservation(mpNewKF, vMatched12[i]);

            mpMap->insertMP(pNewMP);
            nAddNewMP++;
        }
    }
    printf("[Local][Info ] #%ld(KF#%ld) 关联地图点3/3, 共添加了%d个新MP, 目前MP总数为%ld个\n",
           mpNewKF->id, mpNewKF->mIdKF, nAddNewMP, mpMap->countMPs());
}

/**
 * @brief LocalMapper::removeOutlierChi2
 * 把Local Map中的KFs和MPs添加到图中进行优化，然后解除离群MPs的联接关系.
 * 离群MPs观测数小于2时会被删除
 */
void LocalMapper::removeOutlierChi2()
{
    WorkTimer timer;

    locker lockmapper(mutexMapper);

    SlamOptimizer optimizer;
    initOptimizer(optimizer);

    // 通过Map给优化器添加节点和边，添加的节点和边还会存在下面两个变量里
    vector<vector<EdgeProjectXYZ2UV*>> vpEdgesAll;
    vector<vector<int>> vnAllIdx;
    mpMap->loadLocalGraph(optimizer, vpEdgesAll, vnAllIdx);

    const float chi2 = 25;

    optimizer.initializeOptimization(0);
    optimizer.optimize(10);
    printf("[Local][Info ] #%ld(KF#%ld) 移除离群点前优化成功! 耗时%.2fms. 正在提取离群MP...\n",
           mpNewKF->id, mpNewKF->mIdKF, timer.count());
    timer.start();

    // 优化后去除离群MP
    const size_t nAllMP = vpEdgesAll.size();
    int nBadMP = 0;
    vector<vector<int>> vnOutlierIdxAll;

    for (size_t i = 0; i != nAllMP; ++i) {
        vector<int> vnOutlierIdx;
        for (size_t j = 0, jend = vpEdgesAll[i].size(); j < jend; ++j) {
            EdgeProjectXYZ2UV*& eij = vpEdgesAll[i][j];

            if (eij->level() > 0)
                continue;

            eij->computeError();
            bool chi2Bad = eij->chi2() > chi2;

            int& idKF = vnAllIdx[i][j];

            if (chi2Bad) {
                eij->setLevel(1);
                vnOutlierIdx.push_back(idKF);
            }
        }

        vnOutlierIdxAll.push_back(vnOutlierIdx);
    }

    nBadMP = mpMap->removeLocalOutlierMP(vnOutlierIdxAll);
    printf("[Local][Timer] #%ld(KF#%ld) 提取+移除离群MP耗时: %.2fms\n", mpNewKF->id, mpNewKF->mIdKF,
           timer.count());

    vpEdgesAll.clear();
    vnAllIdx.clear();
    printf("[Local][Info ] #%ld(KF#%ld) 移除离群点共移除了%d个MP, 当前MP总数为%ld个\n", mpNewKF->id,
           mpNewKF->mIdKF, nBadMP, nAllMP);
}

/**
 * @brief LocalMapper::localBA 局部图优化
 */
void LocalMapper::localBA()
{
    // 如果这时候全局优化在执行则不会做局部优化
    if (mbGlobalBABegin)
        return;

    printf("[Local][Info ] #%ld(KF#%ld) 正在执行localBA()...\n", mpNewKF->id, mpNewKF->mIdKF);
    WorkTimer timer;
    timer.start();

    locker lockMapper(mutexMapper);

    SlamOptimizer optimizer;
    SlamLinearSolver* linearSolver = new SlamLinearSolver();
    SlamBlockSolver* blockSolver = new SlamBlockSolver(linearSolver);
    SlamAlgorithm* solver = new SlamAlgorithm(blockSolver);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(Config::LocalVerbose);
    optimizer.setForceStopFlag(&mbAbortBA);

    mpMap->loadLocalGraph(optimizer);
    if (optimizer.edges().empty()) {
        fprintf(stderr, "[Local][Error] #%ld(KF#%ld) No MPs in graph, leave localBA().\n",
                mpNewKF->id, mpNewKF->mIdKF);
        return;
    }

    // optimizer.verifyInformationMatrices(true);
    // assert(optimizer.verifyInformationMatrices(true));

    optimizer.initializeOptimization(0);
    optimizer.optimize(Config::LocalIterNum);

    double t1 = timer.count();
    printf("[Local][Timer] #%ld(KF#%ld) L2.localBA()耗时: %.2fms\n", mpNewKF->id, mpNewKF->mIdKF, t1);

    if (solver->currentLambda() > 100.0) {
        cerr << "[Local][Error] current lambda too large " << solver->currentLambda()
             << " , reject optimized result!" << endl;
        return;
    }

    if (mbGlobalBABegin) {
        return;
    }

    timer.start();
    mpMap->optimizeLocalGraph(optimizer);  // 用优化后的结果更新KFs和MPs
    // double t2 = timer.count();
    // rintf("[Local][Timer] #%ld(KF#%ld) localBA()优化结果更新耗时: %.2fms\n", mpNewKF->id,
    // mpNewKF->mIdKF, t2);
}

void LocalMapper::run()
{
    if (Config::LocalizationOnly)
        return;

    WorkTimer timer;
    ros::Rate rate(Config::FPS * 5);
    while (ros::ok()) {
        //! 在处理完addNewKF()函数后(关联并添加MP和连接关系，KF加入LocalMap)，mbUpdated才为true.
        if (checkNewKF()) {
            timer.start();

            setAcceptNewKF(false);  // 干活了，这单先处理，现在不接单了

            processNewKF();

            //! 更新了Map里面mLocalGraphKFs,mRefKFs和mLocalGraphMPs三个成员变量
            updateLocalGraphInMap();  // 加了新的KF进来，要更新一下Map里的Local Map 和 RefKFs.

            //! 去除冗余的KF和MP，共视关系会被取消，mLocalGraphKFs和mLocalGraphMPs会更新
            pruneRedundantKFinMap();

            //! NOTE 原作者把这个步骤给注释掉了.
            // removeOutlierChi2();  // 这里做了一次LocalBA,并对离群MPs取消联接关系,但没有更新KF位姿

            //! 再次更新LocalMap，由于冗余的KF和MP共视关系已经被取消，所以不必但心它们被添加回来
            updateLocalGraphInMap();

            //! LocalMap优化，并更新Local KFs和MPs的位姿
            localBA();  // 这里又做了一次LocalBA，更新了KF和MP的位姿

            //! 标志位置为false防止多次处理，直到加入新的KF才会再次启动
            mbUpdated = false;

            //! 看全局地图有没有在执行Global BA，如果在执行会等它先执行完毕
            mpGlobalMapper->waitIfBusy();

            //! 位姿优化后, 第三次更新LocalMap! (没有必要?)
            // updateLocalGraphInMap();

            fprintf(stdout, "[Local][Timer] #%ld(KF#%ld) L3.LocalMap线程本次运行总耗时: %.2fms\n",
                    mpNewKF->id, mpNewKF->mIdKF, timer.count());

            setAcceptNewKF(true);
        }

        if (checkFinish())
            break;

        rate.sleep();
    }

    cerr << "[Local][Info ] Exiting Localmapper..." << endl;

    setFinish();
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

void LocalMapper::printOptInfo(const SlamOptimizer& optimizer)
{
    // for odometry edges
    for (auto it = optimizer.edges().begin(), itend = optimizer.edges().end(); it != itend; ++it) {
        g2o::EdgeSE3Expmap* pEdge = static_cast<g2o::EdgeSE3Expmap*>(*it);
        vector<g2o::HyperGraph::Vertex*> vVertices = pEdge->vertices();
        if (vVertices.size() == 2) {
            int id0 = vVertices[0]->id();
            int id1 = vVertices[1]->id();
            if (max(id0, id1) > static_cast<int>(mpNewKF->mIdKF)) {
                // Not odometry edge
                continue;
            }
            cerr << "odometry edge: ";
            cerr << "id0 = " << id0 << "; ";
            cerr << "id1 = " << id1 << "; ";
            cerr << "chi2 = " << pEdge->chi2() << "; ";
            cerr << "err = ";
            for (int i = 0; i != 6; ++i) {
                cerr << pEdge->error()(i) << "; ";
            }
            cerr << endl;
        }
    }

    // for plane motion edges
    for (auto it = optimizer.edges().begin(), itend = optimizer.edges().end(); it != itend; ++it) {
        g2o::EdgeSE3Expmap* pEdge = static_cast<g2o::EdgeSE3Expmap*>(*it);
        vector<g2o::HyperGraph::Vertex*> vVertices = pEdge->vertices();
        if (vVertices.size() == 1) {

            int id0 = vVertices[0]->id();

            cerr << "plane motion edge: ";
            cerr << "id0 = " << id0 << "; ";
            cerr << "chi2 = " << pEdge->chi2() << "; ";
            cerr << "err = ";
            for (int i = 0; i != 6; ++i) {
                cerr << pEdge->error()(i) << "; ";
            }
            cerr << endl;
        }
    }

    // for XYZ2UV edges
    for (auto it = optimizer.edges().begin(), itend = optimizer.edges().end(); it != itend; ++it) {
        g2o::EdgeProjectXYZ2UV* pEdge = static_cast<g2o::EdgeProjectXYZ2UV*>(*it);
        vector<g2o::HyperGraph::Vertex*> vVertices = pEdge->vertices();
        if (vVertices.size() == 2) {
            int id0 = vVertices[0]->id();
            int id1 = vVertices[1]->id();
            if (max(id0, id1) > static_cast<int>(mpNewKF->mIdKF)) {
                if (pEdge->chi2() < 10)
                    continue;
                cerr << "XYZ2UV edge: ";
                cerr << "id0 = " << id0 << "; ";
                cerr << "id1 = " << id1 << "; ";
                cerr << "chi2 = " << pEdge->chi2() << "; ";
                cerr << "err = ";
                for (int i = 0; i != 2; ++i) {
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
    mpMap->updateLocalGraph(mnSearchLevel, mnMaxLocalFrames, mfSearchRadius);
}

//! 去除冗余关键帧
void LocalMapper::pruneRedundantKFinMap()
{
    printf("[Local][Info ] #%ld(KF#%ld) 正在修剪冗余KF...\n", mpNewKF->id, mpNewKF->mIdKF);
    locker lock(mutexMapper);
    mpMap->pruneRedundantKF();
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
    mbFinished = true;
}

// void LocalMapper::getNumFKsInQueue(){
//    locker lock(mMutexNewKFs);
//    return mlNewKeyFrames.size();
//}

}  // namespace se2lam
