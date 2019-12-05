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
      mpNewKF(nullptr), mbAcceptNewKF(true), mbAbortBA(false), mbGlobalBABegin(false)
{
    mbPrintDebugInfo = Config::LocalPrint;
    mnMaxLocalFrames = Config::MaxLocalFrameNum;
    mnSearchLevel = Config::LocalFrameSearchLevel;
    mfSearchRadius = Config::LocalFrameSearchRadius;
}

void LocalMapper::addNewKF(const PtrKeyFrame& pKF, const map<size_t, MPCandidate>& MPCandidates)
{
    locker lock(mMutexNewKFs);
    mlNewKFs.push_back(pKF);
    mMPCandidates = MPCandidates;
    mbAbortBA = true;  // 来不急的话要放弃优化, 但可以保证处理了共视关系
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

    map<size_t, MPCandidate> MPCandidates;
    {
        locker lock(mMutexNewKFs);
        assert(!mlNewKFs.empty());
        mpNewKF = mlNewKFs.front();
        mlNewKFs.pop_front();
        MPCandidates = mMPCandidates;
        mMPCandidates.clear();
    }

    //! 1.根据自己可视MP更新信息矩阵, 局部地图投影关联MP，并由MP候选生成新的MP, 关键函数!!!
    findCorresponds(MPCandidates);
    double t1 = timer.count();
    printf("[Local][Timer] #%ld(KF#%ld) L1.1.关联地图点总耗时: %.2fms\n", mpNewKF->id, mpNewKF->mIdKF, t1);

    //! 2.更新局部地图里的共视关系，MP共同观测超过自身的30%则添加共视关系, 更新 mspCovisibleKFs
    timer.start();

    mpNewKF->updateCovisibleGraph();
    double t2 = timer.count();
    printf("[Local][Timer] #%ld(KF#%ld) L1.2.更新共视关系耗时: %.2fms, 共获得%ld个共视KF, MP观测数: %ld\n",
           mpNewKF->id, mpNewKF->mIdKF, t2, mpNewKF->countCovisibleKFs(), mpNewKF->countObservations());
    timer.start();

    //! 3.Add KeyFrame-KeyFrame relation. 添加前后KF的约束
    PtrKeyFrame pKFLast = mpMap->getCurrentKF();
    Mat measure;
    g2o::Matrix6d info;
    calcOdoConstraintCam(mpNewKF->odom - pKFLast->odom, measure, info);
    pKFLast->setOdoMeasureFrom(mpNewKF, measure, toCvMat6f(info));
    mpNewKF->setOdoMeasureTo(pKFLast, measure, toCvMat6f(info));

    //! 4.将KF插入地图
    mpMap->insertKF(mpNewKF);
    mbAbortBA = false;
    mbAcceptNewKF = false;

    double t3 = timer.count();
    printf("[Local][Timer] #%ld(KF#%ld) L1.3.LocalMap的预处理总耗时: %.2fms\n", mpNewKF->id,
           mpNewKF->mIdKF, t1 + t2 + t3);
}


/**
 * @brief   给newKF添加观测. (这里不需要再处理和refKF的newKF的MP关联了)
 * 1. 将newKF的可视MP(目前视差均为好)添加信息矩阵
 * 2. 将局部地图投影到newKF上进行关联. 可能会覆盖候选MP对应的KP， 这时候要丢弃候选.
 * 3. 将MP候选生成真正的MP, 为新的KF添加MP观测;
 *  经过Track里三角化函数处理, refKF和newKF只关联视差好的MP. 视差不好的MP不是refKF和newKF生成的,
 * 不能关联.
 *  候选MP里面(全是视差不好的)在这里会生成真正的MP并和refKF相互添加观测,
 * 和newKF三角化产生的新MP才给newKF添加观测.
 *
 *
 * @param MPCandidates  MP候选
 */
void LocalMapper::findCorresponds(const map<size_t, MPCandidate>& MPCandidates)
{
    WorkTimer timer;

    PtrKeyFrame pRefKF = nullptr;
    if (MPCandidates.empty())
        pRefKF = mpMap->getCurrentKF();
    else
        pRefKF = (*MPCandidates.begin()).second.pKF;  // 回环成功refKF为LoopKF
    assert(pRefKF != nullptr);
    assert(mpNewKF->id > pRefKF->id);

    const Mat Tc1w = pRefKF->getPose();
    const Mat Tc2w = mpNewKF->getPose();

    // 1.为newKF在Track线程中添加的可视MP添加info. 可以保证视差都是好的
    int nAddInfo = 0;
    const int nObs = mpNewKF->countObservations();
    for (size_t i = 0, iend = mpNewKF->N; i < iend; ++i) {
        PtrMapPoint pMP = mpNewKF->getObservation(i);
        if (pMP) {
            assert(pMP->isGoodPrl());
            Point3f Pc1 = cvu::se3map(Tc1w, pMP->getPos());
            Eigen::Matrix3d xyzinfo1, xyzinfo2;
            calcSE3toXYZInfo(Pc1, Tc1w, Tc2w, xyzinfo1, xyzinfo2);
            mpNewKF->setObsAndInfo(pMP, i, xyzinfo2);
            pMP->addObservation(mpNewKF, i);
            nAddInfo++;
        }
    }
    double t1 = timer.count();
    printf("[Local][Info ] #%ld(KF#%ld) L1.1.关联地图点1/3, 可视MP添加信息矩阵数/KF可视MP数: %d/%d, 耗时: %.2fms\n",
           mpNewKF->id, mpNewKF->mIdKF, nAddInfo, nObs, t1);
    assert(nObs == mpNewKF->countObservations());

    // 2.局部地图中非newKF的MPs投影到newKF, 视差不好的不投. (新投影的MP可能会把MP候选的坑占了)
    timer.start();
    const vector<PtrMapPoint> vLocalMPs = mpMap->getLocalMPs();  //! 目前不应该有视差不好的
    const int nLocalMPs = vLocalMPs.size();
    int nProjLocalMPs = 0;
    if (nLocalMPs > 0) {
        vector<int> vMatchedIdxMPs;
        ORBmatcher matcher;
        int m = matcher.SearchByProjection(mpNewKF, vLocalMPs, vMatchedIdxMPs, 20, 1);
        if (m > 0) {
            for (int i = 0, iend = mpNewKF->N; i < iend; ++i) {
                if (vMatchedIdxMPs[i] < 0)  // vMatchedIdxMPs.size() = mpNewKF->N
                    continue;

                const PtrMapPoint& pMP = vLocalMPs[vMatchedIdxMPs[i]];
                if (!pMP || pMP->isNull())
                    continue;
                assert(pMP->isGoodPrl());  // 视差好的才投

                // 通过三角化验证一下投影匹配对不对
                Point3f Pw =
                    cvu::triangulate(pMP->getMainMeasureProjection(), mpNewKF->mvKeyPoints[i].pt,
                                     Config::Kcam * pMP->getMainKF()->getPose().rowRange(0, 3),
                                     Config::Kcam * Tc2w.rowRange(0, 3));
                Point3f Pc2 = cvu::se3map(Tc2w, Pw);
                if (!Config::acceptDepth(Pc2.z))
                    continue;
                if (!pMP->acceptNewObserve(Pc2, mpNewKF->mvKeyPoints[i]))
                    continue;

                // 验证通过给newKF关联此MP.
                Eigen::Matrix3d infoOld, infoNew;
                calcSE3toXYZInfo(Pc2, Tc2w, pMP->getMainKF()->getPose(), infoNew, infoOld);
                mpNewKF->setObsAndInfo(pMP, i, infoNew);
                pMP->addObservation(mpNewKF, i);
                nProjLocalMPs++;
            }
            double t2 = timer.count();
            printf("[Local][Info ] #%ld(KF#%ld) L1.1.关联地图点2/3, "
                   "KF关联MP数/有效投影MP(视差好)数/局部MP总数: %d/%d/%d, 耗时: %.2fms\n",
                   mpNewKF->id, mpNewKF->mIdKF, nProjLocalMPs, m, nLocalMPs, t2);
        } else {
            printf("[Local][Info ] #%ld(KF#%ld) L1.1.关联地图点2/3, "
                   "有效投影MP(视差好)数为0, 局部MP(视差好)总数为: %d\n",
                   mpNewKF->id, mpNewKF->mIdKF, nLocalMPs);
        }
    }

    // 3.处理所有的候选MP.(候选观测完全是新的MP)
    timer.start();
    int nAddNewMP = 0, nReplacedCands = 0;
    for (const auto& cand : MPCandidates) {
        const size_t idx1 = cand.first;
        const size_t idx2 = cand.second.kpIdx2;
        const unsigned long frameId = cand.second.id2;

        assert(pRefKF == cand.second.pKF);
        assert(!pRefKF->hasObservationByIndex(idx1));

        // 局部地图投影到newKF中的MP, 如果把候选的坑占了, 则取消此候选.
        if (frameId == mpNewKF->id && mpNewKF->hasObservationByIndex(idx2)) {  // 只有可能是局部地图投影下来的
            PtrMapPoint pMP = mpNewKF->getObservation(idx2);
            Point3f Pc1 = cvu::se3map(Tc1w, pMP->getPos());
            Eigen::Matrix3d xyzinfo1, xyzinfo2;
            calcSE3toXYZInfo(Pc1, Tc1w, Tc2w, xyzinfo1, xyzinfo2);
            pRefKF->setObsAndInfo(pMP, idx1, xyzinfo1);
            pMP->addObservation(pRefKF, idx1);
            nReplacedCands++;
            continue;
        }

        Eigen::Matrix3d xyzinfo1, xyzinfo2;
        calcSE3toXYZInfo(cand.second.Pc1, Tc1w, cand.second.Tc2w, xyzinfo1, xyzinfo2);
        Point3f Pw = cvu::se3map(Tc1w, cand.second.Pc1);
        PtrMapPoint pNewMP = make_shared<MapPoint>(Pw, false);  // 候选的视差都是不好的

        pRefKF->setObsAndInfo(pNewMP, idx1, xyzinfo1);
        pNewMP->addObservation(pRefKF, idx1);
        if (frameId == mpNewKF->id) {
            assert(!mpNewKF->hasObservationByIndex(idx2));
            mpNewKF->setObsAndInfo(pNewMP, idx2, xyzinfo2);
            pNewMP->addObservation(mpNewKF, idx2);
        }
        mpMap->insertMP(pNewMP);
        nAddNewMP++;
    }
    double t3 = timer.count();
    printf("[Local][Info ] #%ld(KF#%ld) L1.1.关联地图点3/3, MP候选数/新增数/替换数/当前MP总数: "
           "%ld/%d/%d/%ld, 耗时: %.2fms\n",
           mpNewKF->id, mpNewKF->mIdKF, MPCandidates.size(), nAddNewMP, nReplacedCands, mpMap->countMPs(), t3);
    assert(nReplacedCands <= nProjLocalMPs);
    assert(nAddNewMP + nReplacedCands == MPCandidates.size());
}

/**
 * @brief   通过一次localBA计算MP内点, 剔除MP外点
 * 把Local Map中的KFs和MPs添加到图中进行优化，然后解除离群MPs的联接关系.
 */
void LocalMapper::removeOutlierChi2()
{
/*
    WorkTimer timer;

    locker lockmapper(mutexMapper);

    if (mpMap->countMPs() == 0)
        return;

    SlamOptimizer optimizer;
    initOptimizer(optimizer);

    // 通过Map给优化器添加节点和边，添加的节点和边还会存在下面两个变量里
    vector<vector<EdgeProjectXYZ2UV*>> vpEdgesAll;
    vector<vector<int>> vnAllIdx;
    mpMap->loadLocalGraph(optimizer, vpEdgesAll, vnAllIdx);

    const float chi2 = 25;

    optimizer.initializeOptimization(0);
    optimizer.optimize(Config::LocalIterNum);
    printf("[Local][Info ] #%ld(KF#%ld) 移除离群点前优化成功! 耗时%.2fms. 正在对%ld个局部MP计算内点...\n",
           mpNewKF->id, mpNewKF->mIdKF, timer.count(), vpEdgesAll.size());
    timer.start();

    // 优化后去除离群MP
    const size_t nAllMP = vpEdgesAll.size();
    int nBadMP = 0;
    vector<vector<int>> vnOutlierIdxAll;

    int nToRemoved = 0;
    for (size_t i = 0; i != nAllMP; ++i) {
        vector<int> vnOutlierIdx;
        for (size_t j = 0, jend = vpEdgesAll[i].size(); j < jend; ++j) {
            EdgeProjectXYZ2UV*& eij = vpEdgesAll[i][j];

            if (eij->level() > 0)
                continue;

            eij->computeError();
            const bool chi2Bad = eij->chi2() > chi2;

            const int& idKF = vnAllIdx[i][j];

            if (chi2Bad) {
                eij->setLevel(1);
                vnOutlierIdx.push_back(idKF);
                nToRemoved++;
            }
        }

        vnOutlierIdxAll.push_back(vnOutlierIdx);
    }

    nBadMP = mpMap->removeLocalOutlierMP(vnOutlierIdxAll);
    vpEdgesAll.clear();
    vnAllIdx.clear();

    printf("[Local][Info ] #%ld(KF#%ld) MP外点提取/移除数为: %d/%d个, 耗时: %.2fms\n",
           mpNewKF->id, mpNewKF->mIdKF, nToRemoved, nBadMP, timer.count());
*/
    WorkTimer timer;

    if (mpMap->countMPs() == 0)
        return;

    SlamOptimizer optimizer;
    initOptimizer(optimizer);

    int nBadMP = mpMap->removeLocalOutlierMP(optimizer);

    printf("[INFO] #%ld(KF#%ld) 移除离群点%d成功! 耗时: %.2fms\n", mpNewKF->id, mpNewKF->mIdKF,
           nBadMP, timer.count());
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
    SlamLinearSolverCholmod* linearSolver = new SlamLinearSolverCholmod();
    SlamBlockSolver* blockSolver = new SlamBlockSolver(linearSolver);
    SlamAlgorithmLM* solver = new SlamAlgorithmLM(blockSolver);
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
        if (checkNewKF()) {
            timer.start();

            setAcceptNewKF(false);  // 干活了，这单先处理，现在不接单了

            //!FIXME MP的关联可能要放到前端去处理, 否则局部线程太慢会影响前端的匹配效果.
            processNewKF();  // 更新MP观测和共视图, 加入到Map中

            updateLocalGraphInMap();  // 加了新的KF进来，要更新一下Map里的Local Map 和 RefKFs.

            if (mpNewKF->mIdKF < 2) {
                setAcceptNewKF(true);
                continue;
            }

            //! 去除冗余的KF和MP，共视关系会被取消，mLocalGraphKFs和mLocalGraphMPs会更新
            //pruneRedundantKFinMap();

            //! NOTE 原作者把这个步骤给注释掉了. // FIXME 好像有死锁造成本线程无法继续工作 20191202
            // removeOutlierChi2();  // 这里做了一次LocalBA,并对离群MPs取消联接关系,但没有更新KF位姿

            //! 再次更新LocalMap，由于冗余的KF和MP共视关系已经被取消，所以不必但心它们被添加回来
            // updateLocalGraphInMap();

            //! LocalMap优化，并更新Local KFs和MPs的位姿
            localBA();  // 这里又做了一次LocalBA，更新了KF和MP的位姿

            //! 看全局地图有没有在执行Global BA，如果在执行会等它先执行完毕
            // mpGlobalMapper->waitIfBusy();

            //! 位姿优化后, 第三次更新LocalMap!
            // updateLocalGraphInMap();  // 好像没必要, local里的变量存的都是指针.

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

bool LocalMapper::checkIfAcceptNewKF()
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
//    mpMap->updateLocalGraph(mnSearchLevel, mnMaxLocalFrames, mfSearchRadius);
    mpMap->updateLocalGraph_new();
}

//! 去除冗余关键帧
void LocalMapper::pruneRedundantKFinMap()
{
    WorkTimer timer;
    locker lock(mutexMapper);

    int nLocalKFs = mpMap->countLocalKFs();
    int nLocalMPs = mpMap->countLocalMPs();
    int nLocalRefs = mpMap->countRefKFs();

    int nPruned = mpMap->pruneRedundantKF();

    printf("[Local][Info ] #%ld(KF#%ld) 修剪冗余KF前后, 局部KF/MP/RefKF的数量为: %d/%d/%d, %ld/%ld/%ld\n",
           mpNewKF->id, mpNewKF->mIdKF, nLocalKFs, nLocalMPs, nLocalRefs,
           mpMap->countLocalKFs(), mpMap->countLocalMPs(), mpMap->countRefKFs());
    printf("[Local][Info ] #%ld(KF#%ld) 修剪冗余KF, 共修剪了%d帧KF, 耗时:%.2fms.\n",
           mpNewKF->id, mpNewKF->mIdKF, nPruned, timer.count());
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
