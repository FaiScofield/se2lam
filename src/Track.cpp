/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG
* (github.com/hbtang)
*/

#include "Track.h"
#include "Config.h"
#include "LocalMapper.h"
#include "Map.h"
#include "ORBmatcher.h"
#include "converter.h"
#include "cvutil.h"
#include "utility.h"
#include "optimizer.h"
#include <ros/ros.h>


namespace se2lam
{
using namespace std;
using namespace cv;
using namespace Eigen;

typedef unique_lock<mutex> locker;

bool Track::mbUseOdometry = true;

Track::Track()
{
    mState = cvu::NO_READY_YET;
    mLastState = cvu::NO_READY_YET;

    mLocalMPs = vector<Point3f>(Config::MaxFtrNumber, Point3f(-1, -1, -1)); // 这里是参考帧观测到的MP
    nMinFrames = min(2, cvCeil(0.5 * Config::FPS));
    nMaxFrames = cvFloor(2 * Config::FPS);
    mnGoodPrl = 0;

    mpORBextractor = new ORBextractor(Config::MaxFtrNumber, Config::ScaleFactor, Config::MaxLevel);
    mMatchIdx.clear();
    mvbGoodPrl.clear();

    mbFinished = false;
    mbFinishRequested = false;

    nLostFrames = 0;
    mMatchAveOffset = Point2f(0.f, 0.f);
    H = Mat::eye(3, 3, CV_64F);
}

Track::~Track()
{
    delete mpORBextractor;
    mpORBextractor = nullptr;
}

void Track::setMap(Map *pMap)
{
    mpMap = pMap;
}

void Track::setLocalMapper(LocalMapper *pLocalMapper)
{
    mpLocalMapper = pLocalMapper;
}

void Track::setGlobalMapper(GlobalMapper *pGlobalMapper)
{
    mpGlobalMapper = pGlobalMapper;
}

void Track::setSensors(Sensors *pSensors)
{
    mpSensors = pSensors;
}

void Track::run()
{
    if (Config::LocalizationOnly)
        return;

    if (mState == cvu::NO_READY_YET)
        mState = cvu::FIRST_FRAME;

    cv::Mat img;
    Se2 odo;
    ros::Rate rate(Config::FPS * 5);
    while (ros::ok()) {
        bool sensorUpdated = mpSensors->update();
        if (sensorUpdated) {
            mLastState = mState;

            WorkTimer timer;
            timer.start();

            float timeOdo, timeImg;
            Point3f odo_3f;
            mpSensors->readData(odo_3f, img, timeOdo, timeImg);
            if (mbUseOdometry)
                odo = Se2(odo_3f.x, odo_3f.y, odo_3f.z, timeOdo);
            else
                odo = Se2();    //! TODO 不用odom也要能计算出位姿

            {
                locker lock(mMutexForPub);
                if (mState == cvu::FIRST_FRAME) {
                    mCreateFrame(img, odo);
                } else if (mState == cvu::OK) {
                    mTrack(img, odo);
                } else {
                    relocalization(img, odo);
                }
            }
            mpMap->setCurrentFramePose(mFrame.Tcw);
            mLastOdom = odo;

            timer.stop();
            cout << "[Track] #" << mFrame.id << " Tracking consuming time: " << timer.time
                 << "ms." << endl;
        }

        if (checkFinish())
            break;

        rate.sleep();
    }

    cerr << "[Track] Exiting tracking .." << endl;

    setFinish();
}

// 创建首帧
void Track::mCreateFrame(const Mat &img, const Se2 &odo)
{
    mFrame = Frame(img, odo, mpORBextractor, Config::Kcam, Config::Dcam);

    mFrame.Twb = Se2(0, 0, 0);          // 当前帧World->Body，即Body的世界坐标，首帧为原点
    mFrame.Tcw = Config::Tcb.clone();   // 当前帧Camera->World

    if (mFrame.mvKeyPoints.size() > 100) {
        cout << "========================================================" << endl;
        cout << "[Track] Create first frame with " << mFrame.N << " features. "
             << "And the start odom is: " << odo << endl;
        cout << "========================================================" << endl;
        mpKF = make_shared<KeyFrame>(mFrame);  // 首帧为关键帧
        //! NOTE 首帧的KF直接给Map,没有给LocalMapper，还没有参考KF
        mpMap->insertKF(mpKF);

        resetLocalTrack();  // 数据转至参考帧

        mState = cvu::OK;
    } else {
        cout << "[Track] Failed to create first frame for too less keyPoints: "
             << mFrame.mvKeyPoints.size() << endl;
        Frame::nextId = 0;

        mState = cvu::FIRST_FRAME;
    }
}

//! TODO 加入判断追踪失败的代码，加入时间戳保护机制，转重定位
void Track::mTrack(const Mat &img, const Se2 &odo)
{
    assert(mState == cvu::OK);

    WorkTimer timer;
    timer.start();

    mFrame = Frame(img, odo, mpORBextractor, Config::Kcam, Config::Dcam);

    updateFramePose();

    ORBmatcher matcher(0.75);   // 由于trackWithRefKF,间隔有些帧数，nnratio不应过小

    //! 上一帧的KF附近40*40方形cell内进行搜索获取匹配索引
    //! mPrevMatched为参考帧KP的位置, 这里和当前帧匹配上的KP会更新成当前帧对应KP的位置
    //! mPrevMatched这个变量没啥用, 直接用mRefFrame.keyPoints其实就可以了
//    int nMatchedSum = matcher.MatchByWindow(mRefFrame, mFrame, mPrevMatched, 20, mMatchIdx); // 20

    //! 基于H矩阵的透视变换先验估计投影Cell的位置
    int nMatchedSum = matcher.MatchByWindowWarp(mRefFrame, mFrame, H, mMatchIdx, 20);

    //! 利用基础矩阵F计算匹配内点，内点数大于10才能继续
    //! 已改成利用单应矩阵H计算
    int nMatched = removeOutliers(mRefFrame.mvKeyPoints, mFrame.mvKeyPoints, mMatchIdx);

    // 跟踪成功后则三角化计算MP，获取 匹配上参考帧的MP数量 和 未匹配上MP但有良好视差的点对数
    // Check parallax and do triangulation
    int nTrackedOld = doTriangulate();
    printf("[Track] #%d ORBmatcher get tracked/matched/matchedSum points %d/%d/%d.\n",
           mFrame.id, nTrackedOld, nMatched, nMatchedSum);
    N1 += nTrackedOld; N2 += nMatched; N3 += nMatchedSum;
    printf("[Track] #%d tracked/matched/matchedSum points average: %.2f/%.2f/%.2f .\n",
           mFrame.id, N1/(mFrame.id+1.), N2/(mFrame.id+1.), N3/(mFrame.id+1.));

    // Need new KeyFrame decision
    if (needNewKF(nTrackedOld, nMatched)) {
        // Insert KeyFrame
        PtrKeyFrame pKF = make_shared<KeyFrame>(mFrame);

        assert(mpMap->getCurrentKF()->mIdKF == mpKF->mIdKF);

        // 预计分量，这里暂时没用上
        mpKF->preOdomFromSelf = make_pair(pKF, preSE2);
        pKF->preOdomToSelf = make_pair(mpKF, preSE2);

        // 添加给LocalMapper，LocalMapper会根据mLocalMPs生成真正的MP
        // LocalMapper会在更新完共视关系和MPs后把它交给Map
        mpLocalMapper->addNewKF(pKF, mLocalMPs, mMatchIdx, mvbGoodPrl);

        resetLocalTrack();

        mpKF = pKF;

        fprintf(stderr, "[Track] Add new KF at #%d(KF#%d)\n", mFrame.id, mpKF->mIdKF);
        mMatchAveOffset = Point2f(0.f, 0.f);
    }

    timer.stop();
}

//! mpKF是Last KF，根据Last KF和当前帧的里程计更新先验位姿和变换关系, odom是se2绝对位姿而非增量
//! 注意相机的平移量不等于里程的平移量!!!
//! 由于相机和里程计的坐标系不一致，在没有旋转时这两个平移量会相等，但是一旦有旋转，这两个量就不一致了!
//! 如果当前帧不是KF，则当前帧位姿由Last KF叠加上里程计数据计算得到
//! 这里默认场景是工业级里程计，精度比较准，KF之间里程误差可以忽略
void Track::updateFramePose()
{
    // Tb1b2, 参考帧Body->当前帧Body, se2形式
    mFrame.Trb = mFrame.odom - mpKF->odom;
    // Tb2b1, 当前帧Body->参考帧Body, se2形式
    Se2 dOdo = mpKF->odom - mFrame.odom;
    // Tc2c1, 当前帧Camera->参考帧Camera, Tc2c1 = Tcb * Tb2b1 * Tbc
    mFrame.Tcr = Config::Tcb * dOdo.toCvSE3() * Config::Tbc;
    // Tc2w, 当前帧Camera->World, Tc2w = Tc2c1 * Tc1w
    mFrame.Tcw = mFrame.Tcr * mpKF->Tcw;
    // Twb2, 当前帧World->Body，se2形式，故相加， Twb2 = Twb1 * Tb1b2
    mFrame.Twb = mpKF->Twb + mFrame.Trb;

    // preintegration 预积分
    //! NOTE 这里并没有使用上预积分？都是局部变量，且实际一帧图像仅对应一帧Odom数据
    Eigen::Map<Vector3d> meas(preSE2.meas);
    Se2 odok = mFrame.odom - mLastOdom;
    Vector2d odork(odok.x, odok.y);
    Matrix2d Phi_ik = Rotation2Dd(meas[2]).toRotationMatrix();
    meas.head<2>() += Phi_ik * odork;
    meas[2] += odok.theta;

    Matrix3d Ak = Matrix3d::Identity();
    Matrix3d Bk = Matrix3d::Identity();
    Ak.block<2, 1>(0, 2) = Phi_ik * Vector2d(-odork[1], odork[0]);
    Bk.block<2, 2>(0, 0) = Phi_ik;
    Eigen::Map<Matrix3d, RowMajor> Sigmak(preSE2.cov);
    Matrix3d Sigma_vk = Matrix3d::Identity();
    Sigma_vk(0, 0) = (Config::OdoNoiseX * Config::OdoNoiseX);
    Sigma_vk(1, 1) = (Config::OdoNoiseY * Config::OdoNoiseY);
    Sigma_vk(2, 2) = (Config::OdoNoiseTheta * Config::OdoNoiseTheta);
    Matrix3d Sigma_k_1 = Ak * Sigmak * Ak.transpose() + Bk * Sigma_vk * Bk.transpose();
    Sigmak = Sigma_k_1;
}

//! 当前帧设为KF时才执行，将当前KF变参考帧，将当前帧的KP转到mPrevMatched中.
//! mRefFrame这个变量有点多余
void Track::resetLocalTrack()
{
    // mFrame变参考帧后，这两个值没用了，归零
    mFrame.Tcr = cv::Mat::eye(4, 4, CV_32FC1);
    mFrame.Trb = Se2(0, 0, 0);
    mRefFrame = mFrame;
    KeyPoint::convert(mFrame.mvKeyPoints, mPrevMatched);  // cv::KeyPoint 转 cv::Point2f

    // 更新当前Local MP为参考帧观测到的MP
    mLocalMPs = mpKF->mViewMPs;
    mnGoodPrl = 0;
    mMatchIdx.clear();

    for (int i = 0; i < 3; i++)
        preSE2.meas[i] = 0;
    for (int i = 0; i < 9; i++)
        preSE2.cov[i] = 0;

    H = Mat::eye(3, 3, CV_64F);
}

// 可视化用，数据拷贝
int Track::copyForPub(vector<KeyPoint> &kp1, vector<KeyPoint> &kp2, Mat &img1, Mat &img2,
                      vector<int> &vMatches12)
{

    locker lock(mMutexForPub);
    mRefFrame.copyImgTo(img1);
    mFrame.copyImgTo(img2);

    kp1 = mRefFrame.mvKeyPoints;
    kp2 = mFrame.mvKeyPoints;
    vMatches12 = mMatchIdx;

    return !mMatchIdx.empty();
}

// 后端优化用，信息矩阵
void Track::calcOdoConstraintCam(const Se2 &dOdo, Mat &cTc, g2o::Matrix6d &Info_se3)
{
    const Mat bTc = Config::Tbc;
    const Mat cTb = Config::Tcb;

    const Mat bTb = Se2(dOdo.x, dOdo.y, dOdo.theta).toCvSE3();

    cTc = cTb * bTb * bTc;

    float dx = dOdo.x * Config::OdoUncertainX + Config::OdoNoiseX;
    float dy = dOdo.y * Config::OdoUncertainY + Config::OdoNoiseY;
    float dtheta = dOdo.theta * Config::OdoUncertainTheta + Config::OdoNoiseTheta;

    g2o::Matrix6d Info_se3_bTb = g2o::Matrix6d::Zero();
    //    float data[6] = { 1.f/(dx*dx), 1.f/(dy*dy), 1, 1e4, 1e4,
    //    1.f/(dtheta*dtheta) };
    float data[6] = {1.f / (dx * dx), 1.f / (dy * dy), 1e-4, 1e-4, 1e-4, 1.f / (dtheta * dtheta)};
    for (int i = 0; i < 6; i++)
        Info_se3_bTb(i, i) = data[i];
    Info_se3 = Info_se3_bTb;


    //    g2o::Matrix6d J_bTb_cTc = toSE3Quat(bTc).adj();
    //    J_bTb_cTc.block(0,3,3,3) = J_bTb_cTc.block(3,0,3,3);
    //    J_bTb_cTc.block(3,0,3,3) = g2o::Matrix3D::Zero();

    //    Info_se3 = J_bTb_cTc.transpose() * Info_se3_bTb * J_bTb_cTc;

    //    for (int i = 0; i < 6; i++)
    //        for (int j = 0; j < i; j++)
    //            Info_se3(i,j) = Info_se3(j,i);

    // assert(verifyInfo(Info_se3));
}

// 后端优化用，计算约束(R^t)*∑*R
void Track::calcSE3toXYZInfo(Point3f xyz1, const Mat &Tcw1, const Mat &Tcw2, Eigen::Matrix3d &info1,
                             Eigen::Matrix3d &info2)
{
    Point3f O1 = Point3f(cvu::inv(Tcw1).rowRange(0, 3).col(3));
    Point3f O2 = Point3f(cvu::inv(Tcw2).rowRange(0, 3).col(3));
    Point3f xyz = cvu::se3map(cvu::inv(Tcw1), xyz1);
    Point3f vO1 = xyz - O1;
    Point3f vO2 = xyz - O2;
    float sinParallax = cv::norm(vO1.cross(vO2)) / (cv::norm(vO1) * cv::norm(vO2));

    Point3f xyz2 = cvu::se3map(Tcw2, xyz);
    float length1 = cv::norm(xyz1);
    float length2 = cv::norm(xyz2);
    float dxy1 = 2.f * length1 / Config::fx;
    float dxy2 = 2.f * length2 / Config::fx;
    float dz1 = dxy2 / sinParallax;
    float dz2 = dxy1 / sinParallax;

    Mat info_xyz1 = (Mat_<float>(3, 3) << 1.f / (dxy1 * dxy1), 0, 0, 0, 1.f / (dxy1 * dxy1), 0, 0,
                     0, 1.f / (dz1 * dz1));

    Mat info_xyz2 = (Mat_<float>(3, 3) << 1.f / (dxy2 * dxy2), 0, 0, 0, 1.f / (dxy2 * dxy2), 0, 0,
                     0, 1.f / (dz2 * dz2));

    Point3f z1 = Point3f(0, 0, length1);
    Point3f z2 = Point3f(0, 0, length2);
    Point3f k1 = xyz1.cross(z1);
    float normk1 = cv::norm(k1);
    float sin1 = normk1 / (cv::norm(z1) * cv::norm(xyz1));
    k1 = k1 * (std::asin(sin1) / normk1);
    Point3f k2 = xyz2.cross(z2);
    float normk2 = cv::norm(k2);
    float sin2 = normk2 / (cv::norm(z2) * cv::norm(xyz2));
    k2 = k2 * (std::asin(sin2) / normk2);

    Mat R1, R2;
    Mat k1mat = (Mat_<float>(3, 1) << k1.x, k1.y, k1.z);
    Mat k2mat = (Mat_<float>(3, 1) << k2.x, k2.y, k2.z);
    cv::Rodrigues(k1mat, R1);
    cv::Rodrigues(k2mat, R2);

    info1 = toMatrix3d(R1.t() * info_xyz1 * R1);
    info2 = toMatrix3d(R2.t() * info_xyz2 * R2);
}

/**
 * @brief Track::removeOutliers 根据基础矩阵F剔除外点，利用了RANSAC算法
 * @param kp1       参考帧KPs
 * @param kp2       当前帧KPs
 * @param matches   kp1到kp2的匹配索引
 * @return          返回内点数
 */
int Track::removeOutliers(const vector<KeyPoint> &kp1, const vector<KeyPoint> &kp2,
                          vector<int> &matches)
{
    vector<Point2f> pt1, pt2;
    vector<int> idx;
    pt1.reserve(kp1.size());
    pt2.reserve(kp2.size());
    idx.reserve(kp1.size());

    for (int i = 0, iend = kp1.size(); i < iend; i++) {
        if (matches[i] < 0)
            continue;
        idx.push_back(i);
        pt1.push_back(kp1[i].pt);
        pt2.push_back(kp2[matches[i]].pt);
    }

    vector<unsigned char> mask;
    if (pt1.size() != 0) {
//        findFundamentalMat(pt1, pt2, FM_RANSAC, 3, 0.99, mask);  // 默认RANSAC法计算F矩阵
        H = findHomography(pt1, pt2, RANSAC, 3, mask);
    }

    int nInlier = 0;
    for (int i = 0, iend = mask.size(); i < iend; i++) {
        if (!mask[i])
            matches[idx[i]] = -1;
        else
            nInlier++;
    }

    // If too few match inlier, discard all matches. The enviroment might not be
    // suitable for image tracking.
    if (nInlier < 10) {
        nInlier = 0;
        std::fill(mMatchIdx.begin(), mMatchIdx.end(), -1);
        H = Mat::eye(3, 3, CV_64F);
    }

    return nInlier;
}

bool Track::needNewKF(int nTrackedOldMP, int nMatched)
{
    int nOldKP = mpKF->getSizeObsMP();
    bool c0 = mFrame.id - mpKF->id >= nMinFrames;  // 1.间隔首先要足够大
    bool c1 = (float)nTrackedOldMP <= (float)nOldKP * 0.5f;  // 2.1 和参考帧的匹配上的MP数不能太多(小于50%)
    bool c2 = mnGoodPrl > 20;        // 且没匹配上的KP中拥有良好视差的点数大于40
    bool c3 = mFrame.id - mpKF->id > nMaxFrames;  // 2.2 或间隔达到上限了
    bool c4 = nMatched < 0.1f * Config::MaxFtrNumber;  // 2.3 或匹配对小于1/10最大特征点数
    // 满足条件1，和条件2中的一个(共2个条件)，则可能需要加入关键帧，实际还得看odom那边的条件
    bool bNeedNewKF = c0 && ((c1 && c2) || c3 || c4);

    bool bNeedKFByOdo = true;
    if (mbUseOdometry) {
        Se2 dOdo = mFrame.odom - mpKF->odom;
        bool c5 = abs(dOdo.theta) >= g2o::deg2rad(20.);  // 旋转量超过30°，这里原来少了绝对值符号
        cv::Mat cTc = Config::Tcb * dOdo.toCvSE3() * Config::Tbc;  // NOTE 注意相机的平移量不等于里程的平移量
        cv::Mat xy = cTc.rowRange(0, 2).col(3);
        bool c6 = cv::norm(xy) >= (0.5 * Config::UpperDepth * 0.1);  // 相机的平移量足够大 0.5*

        bNeedKFByOdo = c5 || c6;  // 相机旋转超过2°，或者相机移动距离超过一定距离(取决于深度上限,考虑了不同深度下视野的不同)
    }
    bNeedNewKF = bNeedNewKF || bNeedKFByOdo;  // 加上odom的移动条件, 把与改成了或

    // 最后还要看LocalMapper准备好了没有，LocalMapper正在执行优化的时候是不接收新KF的
    if (mpLocalMapper->acceptNewKF()) {
        return bNeedNewKF;
    } else if (c0 && (c4 || c3) && bNeedKFByOdo) {
        mpLocalMapper->setAbortBA(); // 如果必须要加入关键帧,则终止LocalMap的优化,下一帧进来时就可以变成KF了
    }

    return false;
}


/**
 * @brief Track::doTriangulate 初步跟踪成功后对点进行三角化获取深度
 *  mLocalMPs, mvbGoodPrl, mnGoodPrl 在此更新
 * @return  返回匹配上参考帧的MP的数量
 */
int Track::doTriangulate()
{
    // 帧数没到最小间隔就不做三角化，防止KF和MP太多,同样它也不会成为KF
    if (mFrame.id - mpKF->id < nMinFrames) {
        return 0;
    }

    Mat Trc = cvu::inv(mFrame.Tcr);
    Point3f Ocam1 = Point3f(0, 0, 0);
    Point3f Ocam2 = Point3f(Trc.rowRange(0, 3).col(3));
    int nTrackedOld = 0;
    mvbGoodPrl = vector<bool>(mRefFrame.N, false);
    mnGoodPrl = 0;

    // 相机1和2的投影矩阵
    cv::Mat P_KF = Config::PrjMtrxEye;  // P1 = K * cv::Mat::eye(3, 4, CV_32FC1)
    cv::Mat P = Config::Kcam * mFrame.Tcr.rowRange(0, 3); // P2 = K * T21

    // 1.遍历参考帧的KP
    for (int i = 0; i < mRefFrame.N; i++) {
        // 2.如果参考帧的KP与当前帧的KP有匹配
        if (mMatchIdx[i] < 0)
            continue;

        // 2.且参考帧KP已经有对应的MP观测了，则局部地图点更新为此MP
        if (mpKF->hasObservation(i)) {
            mLocalMPs[i] = mpKF->mViewMPs[i];
            nTrackedOld++;
            continue;
        }

        // 3.如果参考帧KP没有对应的MP，则为此匹配对KP三角化计算深度
        Point2f pt_KF = mpKF->mvKeyPoints[i].pt;
        Point2f pt = mFrame.mvKeyPoints[mMatchIdx[i]].pt;
        Point3f pos = cvu::triangulate(pt_KF, pt, P_KF, P);

        // 3.如果深度计算符合预期，就将有深度的KP更新到LocalMPs里
        if (Config::acceptDepth(pos.z)) {
            mLocalMPs[i] = pos;
            // 检查视差
            if (cvu::checkParallax(Ocam1, Ocam2, pos, 2)) {
                mnGoodPrl++;
                mvbGoodPrl[i] = true;
            }
        } else {  // 3.如果深度计算不符合预期，就剔除此匹配点对
            mMatchIdx[i] = -1;
        }
    }
    printf("[Track] #%d Generate %d points through triangulation.\n", mFrame.id, mnGoodPrl);

    return nTrackedOld; // 匹配点对中参考帧KP有对应MP的数量
}

void Track::requestFinish()
{
    locker lock(mMutexFinish);
    mbFinishRequested = true;
}

bool Track::checkFinish()
{
    locker lock(mMutexFinish);
    return mbFinishRequested;
}

bool Track::isFinished()
{
    locker lock(mMutexFinish);
    return mbFinished;
}

void Track::setFinish()
{
    locker lock(mMutexFinish);
    mbFinished = true;
}

//! TODO 完成重定位功能
void Track::relocalization(const Mat &img, const Se2 &odo)
{
}
/*
// 步骤1：计算当前帧特征点的Bow映射
mFrame.ComputeBoW();

// 步骤2：找到与当前帧相似的候选关键帧
vector<KeyFrame *> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mFrame);

if (vpCandidateKFs.empty())
    return false;

const size_t nKFs = vpCandidateKFs.size();

// We perform first an ORB matching with each candidate
// If enough matches are found we setup a PnP solver
ORBmatcher matcher(0.75, true);

vector<PnPsolver*> vpPnPsolvers;
vpPnPsolvers.resize(nKFs);

vector<vector<MapPoint *>> vvpMapPointMatches;
vvpMapPointMatches.resize(nKFs);

vector<bool> vbDiscarded;
vbDiscarded.resize(nKFs);

int nCandidates = 0;

for (size_t i = 0; i < nKFs; i++) {
    KeyFrame *pKF = vpCandidateKFs[i];
    if (pKF->isBad())
        vbDiscarded[i] = true;
    else {
        // 步骤3：通过BoW进行匹配
        int nmatches =
            matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
        if (nmatches < 15) {
            vbDiscarded[i] = true;
            continue;
        } else {
            // 初始化PnPsolver
            PnPsolver *pSolver =
                new PnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
            pSolver->SetRansacParameters(0.99, 10, 300, 4, 0.5, 5.991f);
            vpPnPsolvers[i] = pSolver;
            nCandidates++;
        }
    }
}

// Alternatively perform some iterations of P4P RANSAC
// Until we found a camera pose supported by enough inliers
bool bMatch = false;
ORBmatcher matcher2(0.9f, true);

while (nCandidates > 0 && !bMatch) {
    for (size_t i = 0; i < nKFs; i++) {
        if (vbDiscarded[i])
            continue;

        // Perform 5 Ransac Iterations
        vector<bool> vbInliers;
        int nInliers;
        bool bNoMore;

        // 步骤4：通过EPnP算法估计姿态
        PnPsolver *pSolver = vpPnPsolvers[i];
        cv::Mat Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

        // If Ransac reachs max. iterations discard keyframe
        if (bNoMore) {
            vbDiscarded[i] = true;
            nCandidates--;
        }

        // If a Camera Pose is computed, optimize
        if (!Tcw.empty()) {
            Tcw.copyTo(mCurrentFrame.mTcw);

            set<MapPoint *> sFound;

            const size_t np = vbInliers.size();

            for (size_t j = 0; j < np; j++) {
                if (vbInliers[j]) {
                    mCurrentFrame.mvpMapPoints[j] =
                        vvpMapPointMatches[i][j];
                    sFound.insert(vvpMapPointMatches[i][j]);
                } else
                    mCurrentFrame.mvpMapPoints[j] = NULL;
            }

            // 步骤5：通过PoseOptimization对姿态进行优化求解
            int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

            if (nGood < 10)
                continue;

            for (int io = 0; io < mCurrentFrame.N; io++)
                if (mCurrentFrame.mvbOutlier[io])
                    mCurrentFrame.mvpMapPoints[io] =
                        static_cast<MapPoint *>(NULL);

            // If few inliers, search by projection in a coarse window and
            // optimize again
            // 步骤6：如果内点较少，则通过投影的方式对之前未匹配的点进行匹配，再进行优化求解
            if (nGood < 50) {
                int nadditional = matcher2.SearchByProjection(
                    mCurrentFrame, vpCandidateKFs[i], sFound, 10, 100);

                if (nadditional + nGood >= 50) {
                    nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                    // If many inliers but still not enough, search by
                    // projection again in a narrower window
                    // the camera has been already optimized with many
                    // points
                    if (nGood > 30 && nGood < 50) {
                        sFound.clear();
                        for (int ip = 0; ip < mCurrentFrame.N; ip++)
                            if (mCurrentFrame.mvpMapPoints[ip])
                                sFound.insert(
                                    mCurrentFrame.mvpMapPoints[ip]);
                        nadditional = matcher2.SearchByProjection(
                            mCurrentFrame, vpCandidateKFs[i], sFound, 3,
                            64);

                        // Final optimization
                        if (nGood + nadditional >= 50) {
                            nGood =
                                Optimizer::PoseOptimization(&mCurrentFrame);

                            for (int io = 0; io < mCurrentFrame.N; io++)
                                if (mCurrentFrame.mvbOutlier[io])
                                    mCurrentFrame.mvpMapPoints[io] = NULL;
                        }
                    }
                }
            }

            // If the pose is supported by enough inliers stop ransacs and
            // continue
            if (nGood >= 50) {
                bMatch = true;
                break;
            }
        }
    }
}

if (!bMatch) {
    return false;
} else {
    mnLastRelocFrameId = mCurrentFrame.mnId;
    return true;
}
}
*/

}  // namespace se2lam
