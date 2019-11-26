/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include "GlobalMapper.h"
#include "LocalMapper.h"
#include "Map.h"
#include "cvutil.h"
#include "sparsifier.h"
#include <ros/ros.h>

#include <g2o/core/optimizable_graph.h>
#include <opencv2/calib3d/calib3d.hpp>

namespace se2lam
{
using namespace cv;
using namespace std;
using namespace g2o;

typedef std::unique_lock<std::mutex> locker;

GlobalMapper::GlobalMapper()
    : mpMap(nullptr), mpLocalMapper(nullptr), mpKFCurr(nullptr), mpKFLoop(nullptr),
      mbUpdated(false), mbNewKF(false), mbGlobalBALastLoop(false), mbIsBusy(false),
      mbFinished(false), mbFinishRequested(false)
{
    mdeqPairKFs.clear();
}

void GlobalMapper::setUpdated(bool val)
{
    mbUpdated = val;
    mpKFCurr = mpMap->getCurrentKF();
}

//! 检查Map里是不是来了最新的帧
bool GlobalMapper::checkGMReady()
{
    if (mpMap->empty())
        return false;

    // 新的KF会先经过LocalMapper处理添加共视关系,然后交给Map. LocalMapper又会对其进行局部优化.
    // GlobalMapper的当前帧从Map里提取,所以 mpKFCurr != mpMap->getCurrentKF()
    PtrKeyFrame pKFCurr = mpMap->getCurrentKF();
    if (pKFCurr != mpKFCurr && pKFCurr != nullptr && !pKFCurr->isNull()) {
        mbNewKF = true;
        mpKFCurr = pKFCurr;
        return true;
    } else {
        return false;
    }
}

void GlobalMapper::run()
{
    if (Config::LocalizationOnly)
        return;

    ros::Rate rate(Config::FPS * 5);
    while (ros::ok() && !mbFinished) {

        if (checkFinish())
            break;

        //! Check if everything is ready for global mapping
        if (!checkGMReady()) {
            rate.sleep();
            continue;
        }

        bool bIfFeatGraphRenewed = false;
        bool bIfLoopCloseDetected = false;
        bool bIfLoopCloseVerified = false;

        WorkTimer timer;
        timer.start();

        //! 全局优化开始, 局部优化会停掉.
        setBusy(true);

        //! Update FeatGraph with Covisibility-Graph. 根据共视图更新特征图
        bIfFeatGraphRenewed = mpMap->updateFeatGraph(mpKFCurr);

        /* // 原本这里就是注释掉的
        vector<pair<PtrKeyFrame, PtrKeyFrame>> vKFPairs = SelectKFPairFeat(mpKFCurr);
        if (!vKFPairs.empty()) {
            UpdataFeatGraph(vKFPairs);
            bIfFeatGraphRenewed = true;
        }
        */

        timer.stop();
        double t1 = timer.time;
        printf("[Globa][Timer] #%ld(KF#%ld) G1.更新特征图耗时: %.2fms\n", mpKFCurr->id, mpKFCurr->mIdKF, t1);
        timer.start();

        //! Refresh BowVec for all KFs
        computeBowVecAll();

        timer.stop();
        double t2 = timer.time;
        printf("[Globa][Timer] #%ld(KF#%ld) G2.计算词向量耗时: %.2fms\n", mpKFCurr->id, mpKFCurr->mIdKF, t2);
        timer.start();

        //! Detect loop close
        bIfLoopCloseDetected = detectLoopClose();

        timer.stop();
        double t3 = timer.time;
        printf("[Globa][Timer] #%ld(KF#%ld) G3.回环检测耗时: %.2fms\n", mpKFCurr->id, mpKFCurr->mIdKF, t3);
        timer.start();

        //! Verify loop close
        map<int, int> mapMatchMP, mapMatchGood, mapMatchRaw;
        if (bIfLoopCloseDetected) {
            locker lock(mpLocalMapper->mutexMapper);
            // 验证回环,如果通过了会对其进行Merge
            bIfLoopCloseVerified = verifyLoopClose(mapMatchMP, mapMatchGood, mapMatchRaw);
        }

        //! Create feature edge from loop close
        // TODO ...

        //! Draw Matches
//        drawMatch(mapMatchGood);

        timer.stop();
        double t4 = timer.time;
        if (bIfLoopCloseVerified)
            printf("[Globa][Timer] #%ld(KF#%ld) G4.回环验证通过, 耗时: %.2fms\n", mpKFCurr->id, mpKFCurr->mIdKF, t4);

        timer.start();

        //! Do Global Correction if Needed
        if (!mbGlobalBALastLoop && (bIfLoopCloseVerified || bIfFeatGraphRenewed)) {
            locker lock(mpLocalMapper->mutexMapper);

            WorkTimer wt;
            globalBA();  // 更新KF和MP

            mbGlobalBALastLoop = true;
            printf("[Globa][Timer] #%ld(KF#%ld) G5.全局优化耗时: %.2fms, 总KF数: %ld, 总MP数: %ld\n",
                   mpKFCurr->id, mpKFCurr->mIdKF, wt.count(), mpMap->countKFs(), mpMap->countMPs());
        } else {
            mbGlobalBALastLoop = false;
        }

        timer.stop();
        double t5 = t1 + t2 + t3 + t4 + timer.time;

        printf("[Globa][Timer] #%ld(KF#%ld) G6.GlobalMap线程本次运行总耗时: %.2fms, 总KF数: %ld, 总MP数: %ld\n",
                mpKFCurr->id, mpKFCurr->mIdKF, t5, mpMap->countKFs(), mpMap->countMPs());

        mbNewKF = false;

        //! 全局优化结束,条件变量会告知其他在等待的线程
        setBusy(false);

        rate.sleep();
    }
    cerr << "[Globa][Info ] Exiting globalmapper..." << endl;

    setFinish();
}

/**
 * @brief GlobalMapper::UpdataFeatGraph 更新特征图
 * @param _vKFPairs 输入KF匹配对
 */
void GlobalMapper::updataFeatGraph(vector<pair<PtrKeyFrame, PtrKeyFrame>>& _vKFPairs)
{
    cout << "[ Map ] 正在更新特征图.... " << endl;
    int numPairKFs = _vKFPairs.size();
    for (int i = 0; i < numPairKFs; ++i) {
        pair<PtrKeyFrame, PtrKeyFrame> pairKF = _vKFPairs[i];
        PtrKeyFrame ptKFFrom = pairKF.first;
        PtrKeyFrame ptKFTo = pairKF.second;
        SE3Constraint ftrCnstr;

        if (createFeatEdge(ptKFFrom, ptKFTo, ftrCnstr) == 0) {
            ptKFFrom->addFtrMeasureFrom(ptKFTo, ftrCnstr.measure, ftrCnstr.info);
            ptKFTo->addFtrMeasureTo(ptKFFrom, ftrCnstr.measure, ftrCnstr.info);
            if (Config::GlobalPrint) {
                cerr << "## DEBUG GM: add feature constraint from " << ptKFFrom->id << " to "
                     << ptKFTo->id << endl;
            }
        } else {
            if (Config::GlobalPrint)
                cerr << "## DEBUG GM: add feature constraint failed" << endl;
        }
    }
}

/**
 * @brief GlobalMapper::DetectLoopClose 回环检测
 * 间隔必须25帧以上; 场景相似度最大得分要大于0.005;
 * 选出符合条件中场景相似度最大的一帧作为回环候选, 给下一步做回环验证.
 * @return  返回是否找到候选回环帧的标志
 */
bool GlobalMapper::detectLoopClose()
{
    // Loop closure detection with ORB-BOW method

    bool bDetected = false;
    int minKFIdOffset = Config::MinKFidOffset;   // 25
    double minScoreBest = Config::MinScoreBest;  // 0.005

    PtrKeyFrame pKFCurr = mpMap->getCurrentKF();
    if (pKFCurr == nullptr) {
        return bDetected;
    }
    if (mpLastKFLoopDetect == pKFCurr) {
        return bDetected;
    }

    DBoW2::BowVector& BowVecCurr = pKFCurr->mBowVec;
    int idKFCurr = pKFCurr->mIdKF;

    vector<PtrKeyFrame> vpKFsAll = mpMap->getAllKFs();
    int numKFs = vpKFsAll.size();
    PtrKeyFrame pKFBest;
    double scoreBest = 0;

    for (int i = 0; i < numKFs; ++i) {
        PtrKeyFrame pKF = vpKFsAll[i];
        DBoW2::BowVector& BowVec = pKF->mBowVec;

        // Omit neigbor KFs
        int idKF = pKF->mIdKF;
        if (abs(idKF - idKFCurr) < minKFIdOffset) {
            continue;
        }

        double score = mpORBVoc->score(BowVecCurr, BowVec);
        if (score > scoreBest) {
            scoreBest = score;
            pKFBest = pKF;
        }
    }

    // Loop CLosing Threshold ...
    if (pKFBest != nullptr && scoreBest > minScoreBest) {
        mpKFLoop = pKFBest;
        bDetected = true;
    } else {
        mpKFLoop.reset();
    }

    return bDetected;
}

/**
 * @brief   回环验证
 * @param _mapMatchMP
 * @param _mapMatchGood 良好匹配点对
 * @param _mapMatchRaw  原始匹配点对
 * @return
 */
bool GlobalMapper::verifyLoopClose(map<int, int>& _mapMatchMP, map<int, int>& _mapMatchGood,
                                   map<int, int>& _mapMatchRaw)
{
    assert(mpKFCurr != nullptr && mpKFLoop != nullptr);

    _mapMatchMP.clear();
    _mapMatchGood.clear();
    _mapMatchRaw.clear();
    map<int, int> mapMatch;

    bool bVerified = false;
    const int numMinMatchMP = Config::MinMPMatchNum;         // 15, MP最少匹配数
    const int numMinMatchKP = Config::MinKPMatchNum;         // 30, KP最少匹配数
    const double ratioMinMatchMP = Config::MinMPMatchRatio;  // 0.05

    //! Match ORB KPs
    ORBmatcher matcher;
    bool bIfMatchMPOnly = false;
    matcher.SearchByBoW(mpKFCurr, mpKFLoop, mapMatch, bIfMatchMPOnly);
    _mapMatchRaw = mapMatch;

    //! Remove Outliers: by RANSAC of Fundamental
    removeMatchOutlierRansac(mpKFCurr, mpKFLoop, mapMatch);
    _mapMatchGood = mapMatch;
    int nGoodKFMatch = mapMatch.size();  // KP匹配数,包含了MP匹配数

    //! Remove all KPs matches
    removeKPMatch(mpKFCurr, mpKFLoop, mapMatch);
    _mapMatchMP = mapMatch;
    int nGoodMPMatch = mapMatch.size();  // MP匹配数

    //! Create New Feature based Constraint. 匹配数达到阈值要求,构建特征图和约束
    int nMPsCurrent = mpKFCurr->countObservations();  // 当前KF的有效MP观测数
    double ratioMPMatched = nGoodMPMatch * 1.0 / nMPsCurrent;
    if (nGoodMPMatch >= numMinMatchMP && nGoodKFMatch >= numMinMatchKP &&
        ratioMPMatched >= ratioMinMatchMP) {
        // Generate feature based constraint
        SE3Constraint Se3_Curr_Loop;
        bool bFtrCnstrErr = createFeatEdge(mpKFCurr, mpKFLoop, _mapMatchMP, Se3_Curr_Loop);
        if (!bFtrCnstrErr) {
            mpKFCurr->addFtrMeasureFrom(mpKFLoop, Se3_Curr_Loop.measure, Se3_Curr_Loop.info);
            mpKFLoop->addFtrMeasureTo(mpKFCurr, Se3_Curr_Loop.measure, Se3_Curr_Loop.info);
            bVerified = true;
        }

        if (Config::GlobalPrint) {
            fprintf(stderr, "[Globa] #%ld(KF#%ld) 回环验证通过! 和KF#%ld添加了特征约束, KP匹配数/MP匹配数/匹配率 = %d/%d/%.2f\n",
                    mpKFCurr->id, mpKFCurr->mIdKF, mpKFLoop->mIdKF, nGoodKFMatch, nGoodMPMatch, ratioMPMatched);
        }
    } else {
        if (Config::GlobalPrint) {
            fprintf(stderr, "[Globa] #%ld(KF#%ld) 回环验证失败! MP匹配点数不足, KP匹配数/MP匹配数/匹配率 = %d/%d/%.2f\n",
                    mpKFCurr->id, mpKFCurr->mIdKF, nGoodKFMatch, nGoodMPMatch, ratioMPMatched);
        }
    }

    //! Renew Co-Visibility Graph and Merge MPs
    if (bVerified) {
        mpMap->mergeLoopClose(_mapMatchMP, mpKFCurr, mpKFLoop);
    }

    return bVerified;
}

void GlobalMapper::globalBA()
{
    cout << "[ Map ] 正在进行GlobalBA().... " << endl;
    mpLocalMapper->setGlobalBABegin(true);

#ifdef PRE_REJECT_FTR_OUTLIER
    double threshFeatEdgeChi2 = 30.0;
    double threshFeatEdgeChi2Pre = 1000.0;
#endif

    vector<PtrKeyFrame> vecKFs = mpMap->getAllKFs();

    SlamOptimizer optimizer;
    SlamLinearSolver* linearSolver = new SlamLinearSolver();
    SlamBlockSolver* blockSolver = new SlamBlockSolver(linearSolver);
    SlamAlgorithm* solver = new SlamAlgorithm(blockSolver);
    optimizer.setAlgorithm(solver);

    int SE3OffsetParaId = 0;
    addParaSE3Offset(optimizer, g2o::Isometry3D::Identity(), SE3OffsetParaId);

    unsigned long maxKFid = 0;

    // Add all KFs
    map<unsigned long, PtrKeyFrame> mapId2pKF;
    vector<g2o::EdgeSE3Prior*> vpEdgePlane;

    for (auto it = vecKFs.begin(); it != vecKFs.end(); ++it) {
        PtrKeyFrame pKF = (*it);

        if (pKF->isNull())
            continue;

        Mat Twc = cvu::inv(pKF->getPose());
        bool bIfFix = (pKF->mIdKF == 0);

//        addVertexSE3(optimizer, toIsometry3D(T_w_c), pKF->mIdKF, bIfFix);
        g2o::EdgeSE3Prior* pEdge = addVertexSE3PlaneMotion(optimizer, toIsometry3D(Twc), pKF->mIdKF,
                                                           Config::Tbc, SE3OffsetParaId, bIfFix);
        vpEdgePlane.push_back(pEdge);
//        pEdge->setLevel(1);

        mapId2pKF[pKF->mIdKF] = pKF;

        if (pKF->mIdKF > maxKFid)
            maxKFid = pKF->mIdKF;
    }

    // Add odometry based constraints
    int numOdoCnstr = 0;
    vector<g2o::EdgeSE3*> vpEdgeOdo;
    for (auto it = vecKFs.begin(); it != vecKFs.end(); ++it) {
        PtrKeyFrame pKF = (*it);
        if (pKF->isNull())
            continue;
        if (pKF->mOdoMeasureFrom.first == nullptr)
            continue;

        g2o::Matrix6d info = toMatrix6d(pKF->mOdoMeasureFrom.second.info);

        g2o::EdgeSE3* pEdgeOdoTmp =
            addEdgeSE3(optimizer, toIsometry3D(pKF->mOdoMeasureFrom.second.measure), pKF->mIdKF,
                       pKF->mOdoMeasureFrom.first->mIdKF, info);
        vpEdgeOdo.push_back(pEdgeOdoTmp);

        numOdoCnstr++;
    }

    // Add feature based constraints
    int numFtrCnstr = 0;
    vector<g2o::EdgeSE3*> vpEdgeFeat;
    for (auto it = vecKFs.begin(); it != vecKFs.end(); ++it) {
        PtrKeyFrame ptrKFFrom = (*it);
        if (ptrKFFrom->isNull())
            continue;

        for (auto it2 = ptrKFFrom->mFtrMeasureFrom.begin(); it2 != ptrKFFrom->mFtrMeasureFrom.end();
             it2++) {

            PtrKeyFrame ptrKFTo = (*it2).first;
            if (std::find(vecKFs.begin(), vecKFs.end(), ptrKFTo) == vecKFs.end())
                continue;

            Mat meas = (*it2).second.measure;
            g2o::Matrix6d info = toMatrix6d((*it2).second.info);

            g2o::EdgeSE3* pEdgeFeatTmp =
                addEdgeSE3(optimizer, toIsometry3D(meas), ptrKFFrom->mIdKF, ptrKFTo->mIdKF, info);
            vpEdgeFeat.push_back(pEdgeFeatTmp);

            numFtrCnstr++;
        }
    }

#ifdef PRE_REJECT_FTR_OUTLIER
    // Pre-reject outliers in feature edges
    {
        vector<g2o::EdgeSE3*> vpEdgeFeatGood;
        for (auto iter = vpEdgeFeat.begin(); iter != vpEdgeFeat.end(); iter++) {
            g2o::EdgeSE3* pEdge = *iter;
            pEdge->computeError();
            double chi2 = pEdge->chi2();
            if (chi2 > threshFeatEdgeChi2Pre) {
                int id0 = pEdge->vertex(0)->id();
                int id1 = pEdge->vertex(1)->id();
                PtrKeyFrame pKF0 = mapId2pKF[id0];
                PtrKeyFrame pKF1 = mapId2pKF[id1];
                pKF0->eraseFtrMeasureFrom(pKF1);
                pKF0->eraseCovisibleKF(pKF1);
                pKF1->eraseFtrMeasureTo(pKF0);
                pKF1->eraseCovisibleKF(pKF0);
                pEdge->setLevel(1);
            } else {
                vpEdgeFeatGood.push_back(pEdge);
            }
        }
        vpEdgeFeat.swap(vpEdgeFeatGood);
    }
#endif

    optimizer.setVerbose(Config::GlobalVerbose);
    optimizer.initializeOptimization();
    optimizer.optimize(Config::GlobalIterNum);

#ifdef PRE_REJECT_FTR_OUTLIER

    if (Config::GLOBAL_VERBOSE) {
        PrintOptInfo(vpEdgeOdo, vpEdgeFeat, vpEdgePlane, 1.0, false);
    }
    OptKFPair
        // Reject outliers in feature edges
        bool bFindOutlier = false;
    vector<g2o::EdgeSE3*> vpEdgeFeatGood;
    for (auto iter = vpEdgeFeat.begin(); iter != vpEdgeFeat.end(); iter++) {
        g2o::EdgeSE3* pEdge = *iter;
        double chi2 = pEdge->chi2();
        if (chi2 > threshFeatEdgeChi2) {
            int id0 = pEdge->vertex(0)->id();
            int id1 = pEdge->vertex(1)->id();
            PtrKeyFrame pKF0 = mapId2pKF[id0];
            PtrKeyFrame pKF1 = mapId2pKF[id1];
            pKF0->eraseFtrMeasureFrom(pKF1);
            pKF0->eraseCovisibleKF(pKF1);
            pKF1->eraseFtrMeasureTo(pKF0);
            pKF1->eraseCovisibleKF(pKF0);
            pEdge->setLevel(1);
            bFindOutlier = true;
        } else {
            vpEdgeFeatGood.push_back(pEdge);
        }
    }
    vpEdgeFeat.swap(vpEdgeFeatGood);

    if (!bFindOutlier) {
        OptKFPair break;
    }

#endif

#ifdef REJECT_IF_LARGE_LAMBDA
    if (solver->currentLambda() > 100) {
        mpLocalMapper->setGlobalBABegin(false);
        return;
    }
#endif

    // Update local graph KeyFrame poses
    for (auto it = vecKFs.begin(), iend = vecKFs.end(); it != iend; ++it) {
        PtrKeyFrame pKF = (*it);
        if (pKF->isNull()) {
            continue;
        }
        Mat Twc = toCvMat(estimateVertexSE3(optimizer, pKF->mIdKF));
        pKF->setPose(cvu::inv(Twc));
    }

    // Update local graph MapPoint positions
    vector<PtrMapPoint> vMPsAll = mpMap->getAllMPs();
    for (auto it = vMPsAll.begin(); it != vMPsAll.end(); ++it) {
        PtrMapPoint pMP = (*it);

        if (pMP->isNull()) {
            continue;
        }

        PtrKeyFrame pKF = pMP->getMainKF();
        Mat Twc = pKF->getPose().inv();
        Mat Rwc = Twc.rowRange(0, 3).colRange(0, 3);
        Mat twc = Twc.rowRange(0, 3).colRange(3, 4);

        if (!pKF->hasObservationByPointer(pMP)) {
            continue;
        }

        int idx = pKF->getFeatureIndex(pMP);

        Point3f Pt3_MP_KF = pKF->getMPPoseInCamareFrame(idx);
        Mat t3_MP_KF = (Mat_<float>(3, 1) << Pt3_MP_KF.x, Pt3_MP_KF.y, Pt3_MP_KF.z);
        Mat t3_MP_w = Rwc * t3_MP_KF + twc;
        Point3f Pt3_MP_w(t3_MP_w);
        pMP->setPos(Pt3_MP_w);
    }

    mpLocalMapper->setGlobalBABegin(false);
}

void GlobalMapper::printOptInfo(const vector<g2o::EdgeSE3*>& vpEdgeOdo,
                                const vector<g2o::EdgeSE3*>& vpEdgeFeat,
                                const vector<g2o::EdgeSE3Prior*>& vpEdgePlane, double threshChi2,
                                bool bPrintMatInfo)
{


    cerr << "Edges with large chi2: " << endl;
    // print odometry edges
    for (auto it = vpEdgeOdo.begin(); it != vpEdgeOdo.end(); ++it) {
        g2o::EdgeSE3* pEdge = *it;
        vector<g2o::HyperGraph::Vertex*> vVertices = pEdge->vertices();

        int id0 = vVertices[0]->id();
        int id1 = vVertices[1]->id();

        if (pEdge->chi2() > threshChi2 || pEdge->chi2() < 0) {
            cerr << "odometry edge: ";
            cerr << "id0 = " << id0 << "; ";
            cerr << "id1 = " << id1 << "; ";
            cerr << "chi2 = " << pEdge->chi2() << "; ";

            Matrix6d info = pEdge->information();
            Matrix3D infoTrans = info.block<3, 3>(0, 0);
            Matrix3D infoRot = info.block<3, 3>(3, 3);
            Vector6d err = pEdge->error();
            Vector3D errTrans = err.block<3, 1>(0, 0);
            Vector3D errRot = err.block<3, 1>(3, 0);

            cerr << "chi2Trans = " << errTrans.dot(infoTrans * errTrans) << "; ";
            cerr << "chi2Rot = " << errRot.dot(infoRot * errRot) << "; ";

            cerr << "err = ";
            for (int i = 0; i < 6; ++i) {
                cerr << pEdge->error()(i) << "; ";
            }

            if (bPrintMatInfo) {
                cerr << endl;
                cerr << "info = " << endl << pEdge->information();
            }
            cerr << endl;
        }
    }

    // print loop closing edge
    for (auto it = vpEdgeFeat.begin(); it != vpEdgeFeat.end(); ++it) {
        g2o::EdgeSE3* pEdge = *it;
        vector<g2o::HyperGraph::Vertex*> vVertices = pEdge->vertices();

        int id0 = vVertices[0]->id();
        int id1 = vVertices[1]->id();

        if (pEdge->chi2() > threshChi2 || pEdge->chi2() < 0) {
            cerr << "feature edge: ";
            cerr << "id0 = " << id0 << "; ";
            cerr << "id1 = " << id1 << "; ";
            cerr << "chi2 = " << pEdge->chi2() << "; ";

            Matrix6d info = pEdge->information();
            Matrix3D infoTrans = info.block<3, 3>(0, 0);
            Matrix3D infoRot = info.block<3, 3>(3, 3);
            Vector6d err = pEdge->error();
            Vector3D errTrans = err.block<3, 1>(0, 0);
            Vector3D errRot = err.block<3, 1>(3, 0);

            cerr << "chi2Trans = " << errTrans.dot(infoTrans * errTrans) << "; ";
            cerr << "chi2Rot = " << errRot.dot(infoRot * errRot) << "; ";

            cerr << "err = ";
            for (int i = 0; i < 6; ++i) {
                cerr << pEdge->error()(i) << "; ";
            }

            if (bPrintMatInfo) {
                cerr << endl;
                cerr << "info = " << endl << pEdge->information();
            }
            cerr << endl;
        }
    }

    // print plane motion edge
    for (auto it = vpEdgePlane.begin(); it != vpEdgePlane.end(); ++it) {
        g2o::EdgeSE3Prior* pEdge = *it;
        vector<g2o::HyperGraph::Vertex*> vVertices = pEdge->vertices();

        int id0 = vVertices[0]->id();

        pEdge->computeError();

        if (pEdge->chi2() > threshChi2 || pEdge->chi2() < 0) {
            cerr << "plane edge: ";
            cerr << "id0 = " << id0 << "; ";
            cerr << "chi2 = " << pEdge->chi2() << "; ";

            Matrix6d info = pEdge->information();
            Matrix3D infoTrans = info.block<3, 3>(0, 0);
            Matrix3D infoRot = info.block<3, 3>(3, 3);
            Vector6d err = pEdge->error();
            Vector3D errTrans = err.block<3, 1>(0, 0);
            Vector3D errRot = err.block<3, 1>(3, 0);

            cerr << "chi2Trans = " << errTrans.dot(infoTrans * errTrans) << "; ";
            cerr << "chi2Rot = " << errRot.dot(infoRot * errRot) << "; ";

            cerr << "err = ";
            for (int i = 0; i < 6; ++i) {
                cerr << pEdge->error()(i) << "; ";
            }

            if (bPrintMatInfo) {
                cerr << endl;
                cerr << "info = " << endl << pEdge->information();
            }
            cerr << endl;
        }
    }
}

void GlobalMapper::printOptInfo(const SlamOptimizer& _optimizer)
{

    double threshChi2 = 5.0;

    // print odometry edges
    for (auto it = _optimizer.edges().begin(); it != _optimizer.edges().end(); ++it) {
        g2o::EdgeSE3* pEdge = static_cast<g2o::EdgeSE3*>(*it);
        vector<g2o::HyperGraph::Vertex*> vVertices = pEdge->vertices();
        if (vVertices.size() == 2) {
            int id0 = vVertices[0]->id();
            int id1 = vVertices[1]->id();
            if (abs(id1 - id0) < 5) {
                if (pEdge->chi2() > threshChi2) {
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
        }
    }

    // print loop closing edge
    for (auto it = _optimizer.edges().begin(); it != _optimizer.edges().end(); ++it) {
        g2o::EdgeSE3* pEdge = static_cast<g2o::EdgeSE3*>(*it);
        vector<g2o::HyperGraph::Vertex*> vVertices = pEdge->vertices();
        if (vVertices.size() == 2) {
            int id0 = vVertices[0]->id();
            int id1 = vVertices[1]->id();
            if (abs(id1 - id0) >= 5) {
                if (pEdge->chi2() > threshChi2) {
                    cerr << "loop close edge: ";
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
        }
    }

    // print plane motion edge
    for (auto it = _optimizer.edges().begin(); it != _optimizer.edges().end(); ++it) {
        g2o::EdgeSE3* pEdge = static_cast<g2o::EdgeSE3*>(*it);
        vector<g2o::HyperGraph::Vertex*> vVertices = pEdge->vertices();
        if (vVertices.size() == 1) {
            if (pEdge->chi2() > threshChi2) {
                int id0 = vVertices[0]->id();
                cerr << "plane motion edge: ";
                cerr << "id0 = " << id0 << "; ";
                cerr << "chi2 = " << pEdge->chi2() << "; ";
                cerr << "err = ";
                for (int i = 0; i < 6; ++i) {
                    cerr << pEdge->error()(i) << "; ";
                }
                cerr << endl;
            }
        }
    }
}

// Interface function:
// Called by LocalMapper when feature constraint generation is needed
//! 并没有用上
void GlobalMapper::setKFPairFeatEdge(PtrKeyFrame _pKFFrom, PtrKeyFrame _pKFTo)
{

    std::pair<PtrKeyFrame, PtrKeyFrame> pairKF;
    pairKF.first = _pKFFrom;
    pairKF.second = _pKFTo;
    mdeqPairKFs.push_back(pairKF);
}

// Generate feature constraint between 2 KFs
int GlobalMapper::createFeatEdge(PtrKeyFrame _pKFFrom, PtrKeyFrame _pKFTo,
                                 SE3Constraint& SE3CnstrOutput)
{
    // Find co-observed MPs
    std::set<PtrMapPoint> spMPs;
    Map::compareViewMPs(_pKFFrom, _pKFTo, spMPs);

    // FindCoObsMPs(_pKFFrom, _pKFTo, setMPs);

    // Return when lack of co-observed MPs
    unsigned int numMinMPs = 10;
    if (spMPs.size() < numMinMPs) {
        return 1;
    }

    // Local BA with only 2 KFs and the co-observed MPs
    vector<PtrKeyFrame> vPtrKFs;
    vPtrKFs.push_back(_pKFFrom);
    vPtrKFs.push_back(_pKFTo);

    vector<PtrMapPoint> vPtrMPs;
    for (auto iter = spMPs.begin(); iter != spMPs.end(); iter++) {
        vPtrMPs.push_back(*iter);
    }

    vector<g2o::SE3Quat, Eigen::aligned_allocator<g2o::SE3Quat>> vSe3KFs;
    vector<g2o::Vector3D, Eigen::aligned_allocator<g2o::Vector3D>> vPt3MPs;
    optKFPair(vPtrKFs, vPtrMPs, vSe3KFs, vPt3MPs);

    // Generate feature based constraint between 2 KFs
    vector<MeasSE3XYZ, Eigen::aligned_allocator<MeasSE3XYZ>> vMeasSE3XYZ;
    g2o::SE3Quat meas_out;
    g2o::Matrix6d info_out;

    createVecMeasSE3XYZ(vPtrKFs, vPtrMPs, vMeasSE3XYZ);
    Sparsifier::DoMarginalizeSE3XYZ(vSe3KFs, vPt3MPs, vMeasSE3XYZ, meas_out, info_out);

    // Return
    SE3CnstrOutput.measure = toCvMat(meas_out);
    SE3CnstrOutput.info = toCvMat6f(info_out);
    return 0;
}

int GlobalMapper::createFeatEdge(PtrKeyFrame _pKFFrom, PtrKeyFrame _pKFTo, map<int, int>& mapMatch,
                                 SE3Constraint& SE3CnstrOutput)
{
    vector<g2o::SE3Quat, Eigen::aligned_allocator<g2o::SE3Quat>> vSe3KFs;
    vector<g2o::Vector3D, Eigen::aligned_allocator<g2o::Vector3D>> vPt3MPs;

    //! Optimize Local Graph with Only 2 KFs and MPs matched, and remove outlier by 3D measurements
    optKFPairMatch(_pKFFrom, _pKFTo, mapMatch, vSe3KFs, vPt3MPs);
    if (mapMatch.size() < 3) {
        return 1;
    }

    // Generate feature based constraint between 2 KFs
    vector<MeasSE3XYZ, Eigen::aligned_allocator<MeasSE3XYZ>> vMeasSE3XYZ;

    int count = 0;
    for (auto iter = mapMatch.begin(); iter != mapMatch.end(); iter++) {

        int idxMPin1 = iter->first;
        //        PtrMapPoint pMPin1 = _pKFFrom->mDualObservations[idxMPin1];
        MeasSE3XYZ Meas1;
        Meas1.idKF = 0;
        Meas1.idMP = count;
        Meas1.z = toVector3d(_pKFFrom->getMPPoseInCamareFrame(idxMPin1));
        Meas1.info = _pKFFrom->mvViewMPsInfo[idxMPin1];

        int idxMPin2 = iter->second;
        //        PtrMapPoint pMPin2 = _pKFTo->mDualObservations[idxMPin2];
        MeasSE3XYZ Meas2;
        Meas2.idKF = 1;
        Meas2.idMP = count;
        Meas2.z = toVector3d(_pKFTo->getMPPoseInCamareFrame(idxMPin2));
        Meas2.info = _pKFTo->mvViewMPsInfo[idxMPin2];


        //! TODO to delete. DEBUG ON NAN
        double d = Meas1.info(0, 0);
        if (std::isnan(d)) {
            cerr << "ERROR!!!" << endl;
        }
        d = Meas2.info(0, 0);
        if (std::isnan(d)) {
            cerr << "ERROR!!!" << endl;
        }

        vMeasSE3XYZ.push_back(Meas1);
        vMeasSE3XYZ.push_back(Meas2);

        count++;
    }

    g2o::SE3Quat meas_out;
    g2o::Matrix6d info_out;
    Sparsifier::DoMarginalizeSE3XYZ(vSe3KFs, vPt3MPs, vMeasSE3XYZ, meas_out, info_out);

    // Return
    SE3CnstrOutput.measure = toCvMat(meas_out);
    SE3CnstrOutput.info = toCvMat6f(info_out);
    return 0;
}


// Do localBA
void GlobalMapper::optKFPair(
    const vector<PtrKeyFrame>& _vPtrKFs, const vector<PtrMapPoint>& _vPtrMPs,
    vector<g2o::SE3Quat, Eigen::aligned_allocator<g2o::SE3Quat>>& _vSe3KFs,
    vector<g2o::Vector3D, Eigen::aligned_allocator<g2o::Vector3D>>& _vPt3MPs)
{
    // Init optimizer
    SlamOptimizer optimizer;
    initOptimizer(optimizer, true);
    addParaSE3Offset(optimizer, g2o::Isometry3D::Identity(), 0);

    // Init KF vertex
    int vertexId = 0;
    int numKFs = _vPtrKFs.size();
    for (int i = 0; i < numKFs; ++i) {
        PtrKeyFrame PtrKFi = _vPtrKFs[i];
        Mat T3_kf_w = PtrKFi->getPose();
        g2o::Isometry3D Iso3_w_kf = toIsometry3D(T3_kf_w.inv());

        if (i == 0) {
            addVertexSE3PlaneMotion(optimizer, Iso3_w_kf, vertexId, Config::Tbc, 0, true);
        } else {
            addVertexSE3PlaneMotion(optimizer, Iso3_w_kf, vertexId, Config::Tbc, 0, false);
        }

        vertexId++;
    }

    // Init MP vertex
    int numMPs = _vPtrMPs.size();
    for (int i = 0; i != numMPs; ++i) {
        PtrMapPoint PtrMPi = _vPtrMPs[i];
        g2o::Vector3D Pt3MPi = toVector3d(PtrMPi->getPos());

        addVertexXYZ(optimizer, Pt3MPi, vertexId, true);
        vertexId++;
    }

    // Set SE3XYZ edges
    for (int i = 0; i < numKFs; ++i) {
        int vertexIdKF = i;
        PtrKeyFrame PtrKFi = _vPtrKFs[i];

        for (int j = 0; j < numMPs; ++j) {
            int vertexIdMP = j + numKFs;
            PtrMapPoint PtrMPj = _vPtrMPs[j];

            if (!PtrKFi->hasObservationByPointer(PtrMPj)) {
                continue;
            }

            int idx = PtrKFi->getFeatureIndex(PtrMPj);

            g2o::Vector3D meas = toVector3d(PtrKFi->getMPPoseInCamareFrame(idx));
            g2o::Matrix3D info = PtrKFi->mvViewMPsInfo[idx];

            addEdgeSE3XYZ(optimizer, meas, vertexIdKF, vertexIdMP, 0, info, 5.99);
        }
    }

    // Do optimize with g2o
    WorkTimer timer;
    timer.start();

    optimizer.initializeOptimization();
    optimizer.setVerbose(false);
    optimizer.optimize(15);

    timer.stop();

    // Return optimize results
    _vSe3KFs.clear();
    for (int i = 0; i < numKFs; ++i) {
        g2o::SE3Quat Se3KFi = toSE3Quat(estimateVertexSE3(optimizer, i));
        _vSe3KFs.push_back(Se3KFi);
    }

    _vPt3MPs.clear();
    for (int j = 0; j < numMPs; ++j) {
        g2o::Vector3D Pt3MPj = estimateVertexXYZ(optimizer, j + numKFs);
        _vPt3MPs.push_back(Pt3MPj);
    }
}

void GlobalMapper::optKFPairMatch(
    PtrKeyFrame _pKF1, PtrKeyFrame _pKF2, map<int, int>& mapMatch,
    vector<g2o::SE3Quat, Eigen::aligned_allocator<g2o::SE3Quat>>& _vSe3KFs,
    vector<g2o::Vector3D, Eigen::aligned_allocator<g2o::Vector3D>>& _vPt3MPs)
{
    // Init optimizer
    SlamOptimizer optimizer;
    initOptimizer(optimizer, true);
    addParaSE3Offset(optimizer, g2o::Isometry3D::Identity(), 0);

    // Set vertex KF1
    Mat T3_kf1_w = _pKF1->getPose();
    g2o::Isometry3D Iso3_w_kf1 = toIsometry3D(T3_kf1_w.inv());
    addVertexSE3PlaneMotion(optimizer, Iso3_w_kf1, 0, Config::Tbc, 0, false);

    // Set vertex KF2
    Mat T3_kf2_w = _pKF2->getPose();
    g2o::Isometry3D Iso3_w_kf2 = toIsometry3D(T3_kf2_w.inv());
    addVertexSE3PlaneMotion(optimizer, Iso3_w_kf2, 1, Config::Tbc, 0, false);

    int vertexId = 2;

    // Init MP vertex from KF1 and Set edge
    int numMatch = mapMatch.size();
    vector<int> vIdMPin1;
    for (auto iter = mapMatch.begin(); iter != mapMatch.end(); iter++) {

        int idMPin1 = iter->first;
        vIdMPin1.push_back(idMPin1);
        int idMPin2 = iter->second;

        PtrMapPoint pMPin1 = _pKF1->getObservation(idMPin1);
        if (!pMPin1)
            cerr << "This is NULL /in GM::OptKFPairMatch 1\n";
        if (!pMPin1)
            continue;
        PtrMapPoint pMPin2 = _pKF2->getObservation(idMPin2);
        if (!pMPin2)
            cerr << "This is NULL /in GM::OptKFPairMatch 2\n";
        if (!pMPin2)
            continue;

        g2o::Vector3D Pt3MP = toVector3d(pMPin1->getPos());
        addVertexXYZ(optimizer, Pt3MP, vertexId, true);

        g2o::Vector3D meas1 = toVector3d(_pKF1->getMPPoseInCamareFrame(idMPin1));
        g2o::Matrix3D info1 = _pKF1->mvViewMPsInfo[idMPin1];
        addEdgeSE3XYZ(optimizer, meas1, 0, vertexId, 0, info1, 5.99);

        g2o::Vector3D meas2 = toVector3d(_pKF2->getMPPoseInCamareFrame(idMPin2));
        g2o::Matrix3D info2 = _pKF2->mvViewMPsInfo[idMPin2];
        addEdgeSE3XYZ(optimizer, meas2, 1, vertexId, 0, info2, 5.99);

        vertexId++;
    }

    // Do optimize with g2o
    optimizer.initializeOptimization();
    optimizer.setVerbose(false);
    optimizer.optimize(30);


    // Remove outliers by checking 3D measurement error
    double dThreshChi2 = 5.0;
    set<int> sIdMPin1Outlier;

    for (auto it = optimizer.edges().begin(); it != optimizer.edges().end(); ++it) {
        g2o::EdgeSE3PointXYZ* pEdge = static_cast<g2o::EdgeSE3PointXYZ*>(*it);
        vector<g2o::HyperGraph::Vertex*> vVertices = pEdge->vertices();

        if (vVertices.size() == 2) {
            int id1 = vVertices[1]->id();
            double chi2 = pEdge->chi2();
            if (chi2 > dThreshChi2) {
                sIdMPin1Outlier.insert(vIdMPin1[id1 - 2]);
            }
        }
    }

    if (Config::GlobalPrint) {
        cerr << "## DEBUG GM: "
             << "Find " << sIdMPin1Outlier.size() << " outliers by 3D MP to KF measurements."
             << endl;
    }

    for (auto it = sIdMPin1Outlier.begin(); it != sIdMPin1Outlier.end(); ++it) {
        int idMatch = *it;
        mapMatch.erase(idMatch);
    }
    numMatch = mapMatch.size();

    // Return optimize results
    _vSe3KFs.clear();
    for (int i = 0; i < 2; ++i) {
        g2o::SE3Quat Se3KFi = toSE3Quat(estimateVertexSE3(optimizer, i));
        _vSe3KFs.push_back(Se3KFi);
    }

    _vPt3MPs.clear();
    for (int j = 0; j < numMatch; ++j) {
        g2o::Vector3D Pt3MPj = estimateVertexXYZ(optimizer, j + 2);
        _vPt3MPs.push_back(Pt3MPj);
    }
}

void GlobalMapper::printSE3(const g2o::SE3Quat se3)
{
    Eigen::Vector3d _t = se3.translation();
    Eigen::Quaterniond _r = se3.rotation();

    cerr << "t = ";
    cerr << _t[0] << " ";
    cerr << _t[1] << " ";
    cerr << _t[2] << "; ";
    cerr << "q = ";
    cerr << _r.w() << " ";
    cerr << _r.x() << " ";
    cerr << _r.y() << " ";
    cerr << _r.z() << "; ";
    cerr << endl;
}

void GlobalMapper::createVecMeasSE3XYZ(
    const vector<PtrKeyFrame>& _vpKFs, const vector<PtrMapPoint>& _vpMPs,
    vector<MeasSE3XYZ, Eigen::aligned_allocator<MeasSE3XYZ>>& vMeas)
{
    int numKFs = _vpKFs.size();
    int numMPs = _vpMPs.size();
    vMeas.clear();

    for (int i = 0; i < numKFs; ++i) {
        PtrKeyFrame PtrKFi = _vpKFs[i];
        for (int j = 0; j < numMPs; ++j) {
            PtrMapPoint PtrMPj = _vpMPs[j];

            MeasSE3XYZ Meas_ij;
            Meas_ij.idKF = i;
            Meas_ij.idMP = j;

            if (!PtrKFi->hasObservationByPointer(PtrMPj)) {
                continue;
            }

            int idxMPinKF = PtrKFi->getFeatureIndex(PtrMPj);

            Meas_ij.z = toVector3d(PtrKFi->getMPPoseInCamareFrame(idxMPinKF));
            Meas_ij.info = PtrKFi->mvViewMPsInfo[idxMPinKF];

            vMeas.push_back(Meas_ij);
        }
    }
}

void GlobalMapper::computeBowVecAll()
{
    // Compute BowVector for all KFs, when BowVec does not exist
    vector<PtrKeyFrame> vpKFs = mpMap->getAllKFs();
    for (size_t i = 0, numKFs = vpKFs.size(); i < numKFs; ++i) {
        PtrKeyFrame pKF = vpKFs[i];
        if (pKF->mbBowVecExist)
            continue;
        pKF->computeBoW(mpORBVoc);
    }
}

//void GlobalMapper::drawMatch(const map<int, int>& mapMatch)
//{
//    if (!Config::NeedVisualization)
//        return;

//    //! Renew images
//    if (mpKFCurr == nullptr || mpKFCurr->isNull()) {
//        return;
//    }

//    mpKFCurr->copyImgTo(mImgCurr);

//    if (mpKFLoop == nullptr || mpKFLoop->isNull()) {
//        mImgLoop.setTo(cv::Scalar(0));
//        return;
//    } else {
//        mpKFLoop->copyImgTo(mImgLoop);
//    }

//    //! 把图像转为彩色
//    if (mImgCurr.channels() == 1) {
//        Mat imgTemp = mImgCurr.clone();
//        cvtColor(mImgCurr, imgTemp, CV_GRAY2BGR);
//        imgTemp.copyTo(mImgCurr);
//    }
//    if (mImgLoop.channels() == 1) {
//        Mat imgTemp = mImgLoop.clone();
//        cvtColor(mImgLoop, imgTemp, CV_GRAY2BGR);
//        imgTemp.copyTo(mImgLoop);
//    }
//    Size sizeImgCurr = mImgCurr.size();
//    Size sizeImgLoop = mImgLoop.size();

//    Mat imgMatch(sizeImgCurr.height * 2, sizeImgCurr.width, mImgCurr.type());
//    mImgCurr.copyTo(imgMatch(cv::Rect(0, 0, sizeImgCurr.width, sizeImgCurr.height)));
//    mImgLoop.copyTo(
//        imgMatch(cv::Rect(0, sizeImgCurr.height, sizeImgLoop.width, sizeImgLoop.height)));
//    imgMatch.copyTo(mImgMatch);

//    //! Draw Features
//    for (int i = 0, iend = mpKFCurr->mvKeyPoints.size(); i < iend; ++i) {
//        KeyPoint kpCurr = mpKFCurr->mvKeyPoints[i];
//        Point2f ptCurr = kpCurr.pt;
//        bool ifMPCurr = bool(mpKFCurr->hasObservation(i));
//        Scalar colorCurr;
//        if (ifMPCurr) {
//            colorCurr = Scalar(0, 255, 0);  // 绿色为可观测到的地图点
//        } else {
//            colorCurr = Scalar(255, 0, 0);  // 蓝色为非地图点
//        }
//        circle(mImgMatch, ptCurr, 4, colorCurr, 1);
//    }

//    for (int i = 0, iend = mpKFLoop->mvKeyPoints.size(); i < iend; ++i) {
//        KeyPoint kpLoop = mpKFLoop->mvKeyPoints[i];
//        Point2f ptLoop = kpLoop.pt;
//        Point2f ptLoopMatch = ptLoop;
//        ptLoopMatch.y += 480;

//        bool ifMPLoop = bool(mpKFLoop->hasObservation(i));
//        Scalar colorLoop;
//        if (ifMPLoop) {
//            colorLoop = Scalar(0, 255, 0);
//        } else {
//            colorLoop = Scalar(255, 0, 0);
//        }
//        circle(mImgMatch, ptLoopMatch, 4, colorLoop, 1);
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
//        ptLoopMatch.y += 480;

//        bool ifMPCurr = bool(mpKFCurr->hasObservation(idxCurr));
//        bool ifMPLoop = bool(mpKFLoop->hasObservation(idxLoop));

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

//        circle(mImgMatch, ptCurr, 4, colorCurr, 1);
//        circle(mImgMatch, ptLoopMatch, 4, colorLoop, 1);
//        if (ifMPCurr && ifMPLoop) {
//            line(mImgMatch, ptCurr, ptLoopMatch, Scalar(0, 97, 255), 2);
//        } else {
//            line(mImgMatch, ptCurr, ptLoopMatch, colorCurr, 1);
//        }
//    }
//}

/**
 * @brief GlobalMapper::RemoveMatchOutlierRansac
 * @param _pKFCurr
 * @param _pKFLoop
 * @param mapMatch  返回良好匹配点对
 */
void GlobalMapper::removeMatchOutlierRansac(const PtrKeyFrame& _pKFCurr, const PtrKeyFrame& _pKFLoop,
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
    //findFundamentalMat(vPtCurr, vPtLoop, FM_RANSAC, 3.0, 0.99, vInlier);
    findHomography(vPtCurr, vPtLoop, FM_RANSAC, 3.0, vInlier); // 11.18改
    for (size_t i = 0, iend = vInlier.size(); i < iend; ++i) {
        int idxCurr = vIdxCurr[i];
        int idxLoop = vIdxLoop[i];
        if (vInlier[i] == true) {
            mapMatchGood[idxCurr] = idxLoop;
        }
    }

    // Return good Matches
    mapMatch = mapMatchGood;
}

// Remove match pair with KP. 去掉只有KP匹配但是没有对应MP的匹配, 回环验证时调用
void GlobalMapper::removeKPMatch(const PtrKeyFrame& _pKFCurr, const PtrKeyFrame& _pKFLoop,
                                 map<int, int>& mapMatch)
{
    vector<int> vIdxToErase;

    for (auto iter = mapMatch.begin(); iter != mapMatch.end(); iter++) {

        int idxCurr = iter->first;
        int idxLoop = iter->second;

        bool ifMPCurr = _pKFCurr->hasObservationByIndex(idxCurr);
        bool ifMPLoop = _pKFLoop->hasObservationByIndex(idxLoop);

        if (ifMPCurr && ifMPLoop) {
            continue;
        } else {
            vIdxToErase.push_back(idxCurr);
        }
    }

    size_t numToErase = vIdxToErase.size();
    for (size_t i = 0; i < numToErase; ++i) {
        mapMatch.erase(vIdxToErase[i]);
    }
}

/**
 * @brief 获得和KF特征图上相连的所有KFs
 * @param _pKF          输入KF,搜索起点
 * @param _sKFSelected  已经被选上的KFs集合
 * @return
 */
set<PtrKeyFrame> GlobalMapper::getAllConnectedKFs(const PtrKeyFrame& _pKF,
                                                  const set<PtrKeyFrame>& _sKFSelected)
{
    set<PtrKeyFrame> sKFConnected;

    // 加入与之相连的后一帧
    PtrKeyFrame pKFOdoChild = _pKF->mOdoMeasureFrom.first;
    if (pKFOdoChild != nullptr) {
        sKFConnected.insert(pKFOdoChild);
    }
    // 加入与之相连的前一帧
    PtrKeyFrame pKFOdoParent = _pKF->mOdoMeasureTo.first;
    if (pKFOdoParent != nullptr) {
        sKFConnected.insert(pKFOdoParent);
    }

    // 加入所有特征图中连在它后面的帧
    for (auto iter = _pKF->mFtrMeasureFrom.begin(); iter != _pKF->mFtrMeasureFrom.end(); iter++) {
        sKFConnected.insert(iter->first);
    }
    // 加入所有特征图中连在它前面的帧
    for (auto iter = _pKF->mFtrMeasureTo.begin(); iter != _pKF->mFtrMeasureTo.end(); iter++) {
        sKFConnected.insert(iter->first);
    }
    // 已经添加的候选也加入进来
    for (auto iter = _sKFSelected.begin(); iter != _sKFSelected.end(); iter++) {
        sKFConnected.insert(*iter);
    }

    return sKFConnected;
}

/**
 * @brief 获得和KF特征图上相连的所有KFs(多层)
 * @param _pKF          当前帧KF
 * @param numLayers     层数,本程序里为5
 * @param _sKFSelected  已经被选上的KFs集合
 * @return
 */
set<PtrKeyFrame> GlobalMapper::getAllConnectedKFs_nLayers(const PtrKeyFrame& _pKF, int numLayers,
                                                          const set<PtrKeyFrame>& _sKFSelected)
{
    set<PtrKeyFrame> sKFLocal;   // Set of KFs whose distance from _pKF smaller than maxDist
    set<PtrKeyFrame> sKFActive;  // Set of KFs who are active for next loop;
    sKFActive.insert(_pKF);

    for (int i = 0; i < numLayers; ++i) {
        set<PtrKeyFrame> sKFNew;  // 每层新添的KF

        for (auto iter = sKFActive.begin(); iter != sKFActive.end(); iter++) {
            PtrKeyFrame pKFTmp = *iter;
            set<PtrKeyFrame> sKFAdjTmp = getAllConnectedKFs(pKFTmp, _sKFSelected);
            for (auto iter2 = sKFAdjTmp.begin(); iter2 != sKFAdjTmp.end(); iter2++) {
                if (sKFLocal.count(*iter2) == 0) {
                    sKFNew.insert(*iter2);
                }
            }
        }

        sKFLocal.insert(sKFNew.begin(), sKFNew.end());
        sKFActive.swap(sKFNew);
    }

    return sKFLocal;
}


// Select KF pairs to creat feature constraint between which
vector<pair<PtrKeyFrame, PtrKeyFrame>> GlobalMapper::selectKFPairFeat(const PtrKeyFrame& _pKF)
{
    vector<pair<PtrKeyFrame, PtrKeyFrame>> _vKFPairs;

    // Smallest distance between KFs in covis-graph to create a new feature edge
    int threshCovisGraphDist = 5;

    set<PtrKeyFrame> sKFSelected;
    vector<PtrKeyFrame> sKFCovis = _pKF->getAllCovisibleKFs();
    set<PtrKeyFrame> sKFLocal = getAllConnectedKFs_nLayers(_pKF, threshCovisGraphDist, sKFSelected);

    for (auto iter = sKFCovis.begin(); iter != sKFCovis.end(); iter++) {
        PtrKeyFrame _pKFCand = *iter;
        if (sKFLocal.count(_pKFCand) == 0) {
            sKFSelected.insert(*iter);
            sKFLocal = getAllConnectedKFs_nLayers(_pKF, threshCovisGraphDist, sKFSelected);
        } else {
            continue;
        }
    }

    _vKFPairs.clear();
    for (auto iter = sKFSelected.begin(); iter != sKFSelected.end(); iter++) {
        _vKFPairs.push_back(make_pair(_pKF, *iter));
    }

    return _vKFPairs;
}

void GlobalMapper::setBusy(bool v)
{
    locker lock(mMutexBusy);
    mbIsBusy = v;
    if (!v) {
        mcIsBusy.notify_one();
    }
}

void GlobalMapper::waitIfBusy()
{
    locker lock(mMutexBusy);
    while (mbIsBusy) {
        mcIsBusy.wait(lock);
    }
}

void GlobalMapper::requestFinish()
{
    locker lock(mMutexFinish);
    mbFinishRequested = true;
}

//! checkFinish()成功后break跳出主循环,然后就会调用setFinish()结束线程
bool GlobalMapper::checkFinish()
{
    locker lock(mMutexFinish);
    return mbFinishRequested;
}

bool GlobalMapper::isFinished()
{
    locker lock(mMutexFinish);
    return mbFinished;
}

void GlobalMapper::setFinish()
{
    locker lock(mMutexFinish);
    mbFinished = true;
}


}  // namespace se2lam
