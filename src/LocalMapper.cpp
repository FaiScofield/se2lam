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
 * @brief 添加KF,更新共视关系,然后插入到Map里
 * @param pKF         待添加的KF
 * @param localMPs    在Tracker里计算出来的MP候选，根据参考帧的观测得到的
 * @param vMatched12  参考帧里KP匹配上当前帧KP的索引
 * @param vbGoodPrl   参考帧里KP匹配上但没有MP对应的点对里，视差比较好的flag
 */
void LocalMapper::addNewKF(PtrKeyFrame &pKF, const vector<Point3f> &localMPs,
                           const vector<int> &vMatched12, const vector<bool> &vbGoodPrl)
{
    {
        locker lock(mMutexNewKFs);
        mpNewKF = pKF;
    }

    WorkTimer timer;
    timer.start();

    //! 1.和参考帧以及局部地图关联MP，并添加新的MP, 关键函数!!!
    //! 更新 KF.mViewMPs, KF.mObservations 和 MP.mObservations
    findCorrespd(vMatched12, localMPs, vbGoodPrl);
    timer.stop();
    printf("[Local] #%ld(#KF%ld) findCorrespd() cost time: %fms\n",
           mpNewKF->id, mpNewKF->mIdKF, timer.time);

    timer.start();
    //! 2.更新局部地图里的共视关系，MP共同观测超过自身的30%则添加共视关系, 更新 mspCovisibleKFs
    mpMap->updateCovisibility(mpNewKF);
    timer.stop();
    printf("[Local] #%ld(#KF%ld) updateCovisibility() cost time: %fms\n",
           mpNewKF->id, mpNewKF->mIdKF, timer.time);

    timer.start();
    {
        PtrKeyFrame pKF0 = mpMap->getCurrentKF();  // Map的CurrentKF还是上一时刻的KF, 当前KF处理完后才加入

        // Add KeyFrame-KeyFrame relation. 添加前后KF的约束
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
        mbUpdated = true;   //! 这里LocalMapper的主线程会开始工作, 优化新KF的位姿
    }

    mbAbortBA = false;
    mbAcceptNewKF = false;

    timer.stop();
    printf("[Local] #%ld(#KF%ld) addKFConstraint cost time: %fms\n",
           mpNewKF->id, mpNewKF->mIdKF, timer.time);
}

/**
 * @brief 根据和参考帧和局部地图的匹配关系关联MP，会生成新的MP并插入到Map里
 * 能关联上的MP都关联上，不能关联上MP但和参考帧有匹配的KP，则三角化生成新的MP, 最后都会更新两KF的mViewMPs。
 * @param vMatched12    参考帧到当前帧的KP匹配情况
 * @param localMPs      参考帧对应观测MP的坐标值, 即Pc1
 * @param vbGoodPrl     参考帧与当前帧KP匹配中没有MP但视差好的标志，生成新MP时会对它的视差好坏进行标记
 */
void LocalMapper::findCorrespd(const vector<int> &vMatched12, const vector<Point3f> &localMPs,
                               const vector<bool> &vbGoodPrl)
{
    bool bNoMP = (mpMap->countMPs() == 0);

    // Identify tracked map points
    PtrKeyFrame pPrefKF = mpMap->getCurrentKF(); // 这是上一参考帧KF

    //! TODO to delete, for debug.
    printf("[Local] #%ld(#KF%ld) findCorrespd() Count MPs: %ld\n",
           mpNewKF->id, mpNewKF->mIdKF, mpMap->countMPs());
    printf("[Local] #%ld(#KF%ld) findCorrespd() Count observations of last KF: %ld\n",
           mpNewKF->id, mpNewKF->mIdKF, pPrefKF->countObservation());

    //! 1.如果参考帧的第i个特征点有对应的MP，且和当前帧KP有对应的匹配，就给当前帧对应的KP关联上MP
    if (!bNoMP) {
        for (int i = 0, iend = pPrefKF->N; i != iend; ++i) {
            if (vMatched12[i] >= 0 && pPrefKF->hasObservation(i)) {
                PtrMapPoint pMP = pPrefKF->getObservation(i);
                if (!pMP) {
                    //! TODO to delete, for debug.
                    cerr << "[LocalMap] This is NULL. 这不应该发生啊！！！" << endl;
                    continue;
                }
                Eigen::Matrix3d xyzinfo1, xyzinfo2;
                Mat Tcr = mpNewKF->getTcr();
                Track::calcSE3toXYZInfo(pPrefKF->mvViewMPs[i], cv::Mat::eye(4, 4, CV_32FC1),
                                        Tcr, xyzinfo1, xyzinfo2);
                mpNewKF->setViewMP(cvu::se3map(Tcr, pPrefKF->mvViewMPs[i]), vMatched12[i], xyzinfo2);
                mpNewKF->addObservation(pMP, vMatched12[i]);
                pMP->addObservation(mpNewKF, vMatched12[i]);
            }
        }
    }

    //! 2.和局部地图匹配，关联局部地图里的MP, 其中已经和参考KF有关联的MP不会被匹配, 故不会重复关联
    if (!bNoMP) {
        vector<PtrMapPoint> vLocalMPs = mpMap->getLocalMPs();
        vector<int> vMatchedIdxMPs;
        ORBmatcher matcher;
        matcher.MatchByProjection(mpNewKF, vLocalMPs, 20, 2, vMatchedIdxMPs);   // 15
        for (int i = 0, iend = mpNewKF->N; i != iend; ++i) {
            if (vMatchedIdxMPs[i] < 0)
                continue;

            PtrMapPoint pMP = vLocalMPs[vMatchedIdxMPs[i]];

            // We do triangulation here because we need to produce constraint of
            // mNewKF to the matched old MapPoint.
            Mat Tcw = mpNewKF->getPose();
            Point3f x3d = cvu::triangulate(pMP->getMainMeasure(), mpNewKF->mvKeyPoints[i].pt,
                                           Config::Kcam * pMP->getMainKF()->getPose().rowRange(0, 3),
                                           Config::Kcam * Tcw.rowRange(0, 3));
            Point3f posNewKF = cvu::se3map(Tcw, x3d);
            if (posNewKF.z > Config::UpperDepth || posNewKF.z < Config::LowerDepth)
                continue;
            if (!pMP->acceptNewObserve(posNewKF, mpNewKF->mvKeyPoints[i]))
                continue;
            Eigen::Matrix3d infoNew, infoOld;
            Track::calcSE3toXYZInfo(posNewKF, Tcw, pMP->getMainKF()->getPose(), infoNew, infoOld);
            mpNewKF->setViewMP(posNewKF, i, infoNew);
            mpNewKF->addObservation(pMP, i);
            pMP->addObservation(mpNewKF, i);
        }
    }

    //! 3.根据匹配情况给新的KF添加MP
    //! 首帧没有处理到这，第二帧进来有了参考帧，但还没有MPS，就会直接执行到这，生成MPs，所以第二帧才有MP
    int nAddNewMP = 0;
    assert(pPrefKF->N == localMPs.size());
    for (size_t i = 0, iend = localMPs.size(); i != iend; ++i) {
        // 参考帧的特征点i没有对应的MP，且与当前帧KP存在匹配(也没有对应的MP)，则给他们创造MP
        if (vMatched12[i] >= 0 && !pPrefKF->hasObservation(i)) {
            if (mpNewKF->hasObservation(vMatched12[i])) {
                //! TODO to delete, for debug.
                //! 这个应该很可能会出现, 局部MPs投影到当前KF上可能会关联上.
                //! 如果出现了这种情况, 应该要给参考帧也关联上此MP. 目前在等待这种情况出现
                PtrMapPoint pMP = mpNewKF->getObservation(vMatched12[i]);
                fprintf(stderr, "[LocalMap] 这个可能会出现, 局部MPs投影到当前#KF%ld上可能会关联上.! 如果出现了这种情况, 应该要给参考帧也关联上此MP%ld.\n", mpNewKF->mIdKF, pMP->mId);

                pMP->addObservation(pPrefKF, i);
                pPrefKF->addObservation(pMP, i);
                continue;
            }

            Point3f posW = cvu::se3map(cvu::inv(pPrefKF->getPose()), localMPs[i]);

            //! TODO to delete, for debug.
            //! 这个应该会出现. 内点数不多的时候没有三角化, 则虽有匹配, 但mvViewMPs没有更新, 故这里不能生成MP!
            //! 照理说 localMPs[i] 有一个正常的值的话, 那么就应该有观测出现啊???
            if (posW.z < 0.f) {
                fprintf(stderr, "[LocalMap] #KF%ld的mvViewMPs[%d].z < 0. \n", pPrefKF->mIdKF, i);
                cerr << "[LocalMap] 此点在成为MP之前的坐标Pc是: " << localMPs[i] << endl;
                cerr << "[LocalMap] 此点在成为MP之后的坐标Pw是: " << posW << endl;
                continue;
            }

            Point3f Pc2 = cvu::se3map(mpNewKF->getTcr(), localMPs[i]);
            Eigen::Matrix3d xyzinfo1, xyzinfo2;
            Track::calcSE3toXYZInfo(localMPs[i], pPrefKF->getPose(), mpNewKF->getPose(), xyzinfo1, xyzinfo2);

            pPrefKF->setViewMP(localMPs[i], i, xyzinfo1);
            mpNewKF->setViewMP(Pc2, vMatched12[i], xyzinfo2);
            PtrMapPoint pMP = std::make_shared<MapPoint>(posW, vbGoodPrl[i]);

            pMP->addObservation(pPrefKF, i);
            pMP->addObservation(mpNewKF, vMatched12[i]);
            pPrefKF->addObservation(pMP, i);
            mpNewKF->addObservation(pMP, vMatched12[i]);

            mpMap->insertMP(pMP);
            nAddNewMP++;
        }
    }
    printf("[Local] #%ld(#KF%ld) findCorrespd() Add new MPs: %d, now tatal MPs = %ld\n",
           mpNewKF->id, mpNewKF->mIdKF, nAddNewMP, mpMap->countMPs());
}

/**
 * @brief LocalMapper::removeOutlierChi2
 * 把Local Map中的KFs和MPs添加到图中进行优化，然后解除离群MPs的联接关系.
 * 离群MPs观测数小于2时会被删除
 */
void LocalMapper::removeOutlierChi2()
{
    locker lockmapper(mutexMapper);

    WorkTimer timer;
    timer.start();

    SlamOptimizer optimizer;
    initOptimizer(optimizer);

    // 通过Map给优化器添加节点和边，添加的节点和边还会存在下面两个变量里
    vector<vector<EdgeProjectXYZ2UV *>> vpEdgesAll;
    vector<vector<int>> vnAllIdx;
    mpMap->loadLocalGraph(optimizer, vpEdgesAll, vnAllIdx);

    const float chi2 = 25;

    optimizer.initializeOptimization(0);
    optimizer.optimize(10);

    // 优化后去除离群MP
    const size_t nAllMP = vpEdgesAll.size();
    int nBadMP = 0;
    vector<vector<int>> vnOutlierIdxAll;

    for (size_t i = 0; i != nAllMP; ++i) {
        vector<int> vnOutlierIdx;
        for (size_t j = 0, jend = vpEdgesAll[i].size(); j < jend; ++j) {
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

    nBadMP = mpMap->removeLocalOutlierMP(vnOutlierIdxAll);

    vpEdgesAll.clear();
    vnAllIdx.clear();

    timer.stop();
    printf("[Local] #%ld(KF#%ld) removeOutlierChi2 cost time = %fms\n",
            mpNewKF->id, mpNewKF->mIdKF, timer.time);
    printf("[Local] #%ld(KF#%ld) removeOutlierChi2 Outlier MP: %d; total MP: %ld\n",
            mpNewKF->id, mpNewKF->mIdKF, nBadMP, nAllMP);
}

/**
 * @brief LocalMapper::localBA 局部图优化
 */
void LocalMapper::localBA()
{
    printf("[Local] #%ld(KF#%ld) Doing localBA()...\n", mpNewKF->id, mpNewKF->mIdKF);

    // 如果这时候全局优化在执行则不会做局部优化
    if (mbGlobalBABegin)
        return;

    locker lockmapper(mutexMapper);

    WorkTimer timer;
    timer.start();

    SlamOptimizer optimizer;
    SlamLinearSolver *linearSolver = new SlamLinearSolver();
    SlamBlockSolver *blockSolver = new SlamBlockSolver(linearSolver);
    SlamAlgorithm *solver = new SlamAlgorithm(blockSolver);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(Config::LocalVerbose);
    optimizer.setForceStopFlag(&mbAbortBA);

    mpMap->loadLocalGraph(optimizer);
    if (optimizer.edges().size() == 0) {
        fprintf(stderr, "#%ld(KF#%ld) No MPs in graph, leave localBA().\n",
                mpNewKF->id, mpNewKF->mIdKF);
        return;
    }

//    optimizer.verifyInformationMatrices(true);
//    assert(optimizer.verifyInformationMatrices(true));

    optimizer.initializeOptimization(0);
    optimizer.optimize(Config::LocalIterNum);

    timer.stop();
    if (mbPrintDebugInfo) {
        fprintf(stderr, "[Local] LocalBA cost time = %fms, number of KFs = %ld, number of MPs =  %ld\n", timer.time, mpMap->countLocalKFs(), mpMap->countLocalMPs());
    }

    if (solver->currentLambda() > 100.0) {
        cerr << "[Local] current lambda too large " << solver->currentLambda()
             << " , reject optimized result" << endl;
        return;
    }

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
            updateLocalGraphInMap();    // 加了新的KF进来，要更新一下Map里的Local Map 和 RefKFs.

            //! 去除冗余的KF和MP，共视关系会被取消，mLocalGraphKFs和mLocalGraphMPs会更新
            pruneRedundantKFinMap();

            //! NOTE 原作者把这个步骤给注释掉了.
            removeOutlierChi2();        // 这里做了一次LocalBA,并对离群MPs取消联接关系,但没有更新位姿

            //! 再次更新LocalMap，由于冗余的KF和MP共视关系已经被取消，所以不必但心它们被添加回来
            updateLocalGraphInMap();

            //! LocalMap优化，并更新Local KFs和MPs的位姿
            localBA();                  // 这里又做了一次LocalBA，有更新位姿

            //! 标志位置为false防止多次处理，直到加入新的KF才会再次启动
            mbUpdated = false;

            //! 看全局地图有没有在执行Global BA，如果在执行会等它先执行完毕
            mpGlobalMapper->waitIfBusy();

            //! 位姿优化后, 第三次更新LocalMap！
            updateLocalGraphInMap();

            timer.stop();
            fprintf(stderr, "[Local] #%ld(KF#%ld) Time cost for LocalMapper's process: %fms.\n",
                    mpNewKF->id, mpNewKF->mIdKF, timer.time);
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

void LocalMapper::printOptInfo(const SlamOptimizer &optimizer)
{
    // for odometry edges
    for (auto it = optimizer.edges().begin(), itend = optimizer.edges().end(); it != itend; ++it) {
        g2o::EdgeSE3Expmap *pEdge = static_cast<g2o::EdgeSE3Expmap *>(*it);
        vector<g2o::HyperGraph::Vertex *> vVertices = pEdge->vertices();
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
        g2o::EdgeSE3Expmap *pEdge = static_cast<g2o::EdgeSE3Expmap *>(*it);
        vector<g2o::HyperGraph::Vertex *> vVertices = pEdge->vertices();
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
        g2o::EdgeProjectXYZ2UV *pEdge = static_cast<g2o::EdgeProjectXYZ2UV *>(*it);
        vector<g2o::HyperGraph::Vertex *> vVertices = pEdge->vertices();
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

    printf("[Local] #%ld(KF#%ld) Prune %d Redundant Local KFs\n",
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
