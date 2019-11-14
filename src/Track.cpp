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
#include "optimizer.h"
#include <ros/ros.h>
#include <opencv2/calib3d/calib3d.hpp>


namespace se2lam
{
using namespace std;
using namespace cv;
using namespace Eigen;

typedef unique_lock<mutex> locker;

bool Track::mbUseOdometry = true;

Track::Track()
    : mState(cvu::NO_READY_YET), mLastState(cvu::NO_READY_YET), mbPrint(true),
      mbNeedVisualization(false), mnGoodPrl(0), mnInliers(0), mnMatchSum(0), mnTrackedOld(0),
      mnLostFrames(0), mbFinishRequested(false), mbFinished(false)
{
    mpORBextractor = new ORBextractor(Config::MaxFtrNumber, Config::ScaleFactor, Config::MaxLevel);
    mpORBmatcher = new ORBmatcher(0.9, true);

    mLocalMPs = vector<Point3f>(Config::MaxFtrNumber, Point3f(-1.f, -1.f, -1.f));

    nMinFrames = min(2, cvCeil(0.25 * Config::FPS));  // 上溢
    nMaxFrames = cvFloor(2 * Config::FPS);            // 下溢
    nMinMatches = std::min(cvFloor(0.1 * Config::MaxFtrNumber), 50);
    mMaxAngle = static_cast<float>(g2o::deg2rad(40.));
    mMaxDistance = 0.2f * Config::UpperDepth;

    mK = Config::Kcam;
    mD = Config::Dcam;
    mAffineMatrix = Mat::eye(2, 3, CV_64FC1);  // double

    mbPrint = Config::GlobalPrint;
    mbNeedVisualization = Config::NeedVisualization;

    fprintf(stderr, "[Track][Info ] 相关参数如下: \n - 最小/最大KF帧数: %d/%d\n"
           " - 最大移动距离/角度: %.0fmm/%.0fdeg\n - 最少匹配数量: %d\n",
           nMinFrames, nMaxFrames, mMaxDistance, g2o::rad2deg(mMaxAngle), nMinMatches);
}


Track::~Track()
{
    delete mpORBextractor;
    delete mpORBmatcher;
}


void Track::run()
{
    if (Config::LocalizationOnly)
        return;

    if (mState == cvu::NO_READY_YET)
        mState = cvu::FIRST_FRAME;

    WorkTimer timer;
    double imgTime = 0.;

    ros::Rate rate(Config::FPS * 5);
    while (ros::ok()) {
        if (checkFinish())
            break;

        bool sensorUpdated = mpSensors->update();
        if (sensorUpdated) {
            timer.start();

            mLastState = mState;

            Mat img;
            Se2 odo;
            mpSensors->readData(odo, img);
            double t1 = timer.count();

            {  // 计算位姿时不做可视化, 防止数据竞争
                // locker lock(mMutexForPub);
                timer.start();
                if (mState == cvu::FIRST_FRAME) {
                    createFirstFrame(img, imgTime, odo);
                } else if (mState == cvu::OK) {
                    trackReferenceKF(img, imgTime, odo);
                } else {
                    relocalization(img, imgTime, odo);
                }
                mpMap->setCurrentFramePose(mCurrentFrame.getPose());
            }
            double t2 = timer.count();
            trackTimeTatal += t2;
            fprintf(stdout, "[Track][Timer] #%ld T3.当前帧前端读取数据/追踪/总耗时为: "
                            "%.2f/%.2f/%.2fmsm, 平均追踪耗时: %.2fms\n",
                    mCurrentFrame.id, t1, t2, t1 + t2, trackTimeTatal / mCurrentFrame.id);

            mLastOdom = odo;
        }

        rate.sleep();
    }

    cerr << "[Track][Info ] Exiting tracking .." << endl;
    setFinish();
}

// 创建首帧
void Track::createFirstFrame(const Mat& img, const double& imgTime, const Se2& odo)
{
    mCurrentFrame = Frame(img, imgTime, odo, mpORBextractor, mK, mD);

    if (mCurrentFrame.mvKeyPoints.size() > 200) {
        cout << "========================================================" << endl;
        cout << "[Track][Info ] Create first frame with " << mCurrentFrame.N << " features. "
                << "And the start odom is: " << odo << endl;
        cout << "========================================================" << endl;

        mCurrentFrame.setPose(Se2(0, 0, 0));
        mpReferenceKF = make_shared<KeyFrame>(mCurrentFrame);  // 首帧为关键帧
        mpMap->insertKF(mpReferenceKF);  // 首帧的KF直接给Map,没有给LocalMapper
        mpMap->updateLocalGraph();       // 首帧添加到LocalMap里

        resetLocalTrack();

        mState = cvu::OK;
    } else {
        cout << "[Track][Warni] Failed to create first frame for too less keyPoints: "
                << mCurrentFrame.mvKeyPoints.size() << endl;
        Frame::nextId = 1;

        mState = cvu::FIRST_FRAME;
    }
}

//! TODO 加入判断追踪失败的代码，加入时间戳保护机制，转重定位
void Track::trackReferenceKF(const Mat& img, const double& imgTime, const Se2& odo)
{
    mCurrentFrame = Frame(img, imgTime, odo, mpORBextractor, mK, mD);
    updateFramePose();

    //! 基于A仿射变换先验估计投影Cell的位置
    mnMatchSum = mpORBmatcher->MatchByWindowWarp(*mpReferenceKF, mCurrentFrame, mAffineMatrix,
                                                 mvMatchIdx, 25);

    //! 利用仿射矩阵A计算匹配内点，内点数大于10才能继续
    mnInliers = removeOutliers();    // A在此更新
    mnTrackedOld = doTriangulate();  // 更新viewMPs

    if (mnInliers >= 10) {
        drawMatchesForPub(true);
        printf("[Track][Info ] #%ld T2.与参考帧匹配获得的点数: trackedOld/inliers/matchedSum = %d/%d/%d.\n",
               mCurrentFrame.id, mnTrackedOld, mnInliers, mnMatchSum);
    } else {
        drawMatchesForPub(false);
        fprintf(stderr,
                "[Track][Warni] #%ld T2.与参考帧匹配内点数少于10! trackedOld/matchedSum = %d/%d.\n",
                mCurrentFrame.id, mnTrackedOld, mnMatchSum);
    }

    N1 += mnTrackedOld;
    N2 += mnInliers;
    N3 += mnMatchSum;
    if (mbPrint && mCurrentFrame.id % 50 == 0) {  // 每隔50帧输出一次平均匹配情况
        float sum = mCurrentFrame.id - 1.0;
        printf("[Track][Info ] #%ld 与参考帧匹配平均点数: tracked/matched/matchedSum = %.2f/%.2f/%.2f\n",
               mCurrentFrame.id, N1 * 1.f / sum, N2 * 1.f / sum, N3 * 1.f / sum);
    }

    // Need new KeyFrame decision
    if (needNewKF()) {
        PtrKeyFrame pKF = make_shared<KeyFrame>(mCurrentFrame);

        //! TODO 预计分量，这里暂时没用上
        mpReferenceKF->preOdomFromSelf = make_pair(pKF, preSE2);
        pKF->preOdomToSelf = make_pair(mpReferenceKF, preSE2);

        // 添加给LocalMapper，LocalMapper会根据mLocalMPs生成真正的MP
        // LocalMapper会在更新完共视关系和MPs后把当前KF交给Map
        mpLocalMapper->addNewKF(pKF, mLocalMPs, mvMatchIdx, mvbGoodPrl);

        mpReferenceKF = pKF;
        resetLocalTrack();
    }
}

//! 根据RefKF和当前帧的里程计更新先验位姿和变换关系, odom是se2绝对位姿而非增量
//! 如果当前帧不是KF，则当前帧位姿由RefK叠加上里程计数据计算得到
//! 这里默认场景是工业级里程计，精度比较准
void Track::updateFramePose()
{
    // Tb1b2, 参考帧Body->当前帧Body, se2形式
    // mCurrentFrame.Trb = mCurrentFrame.odom - mpReferenceKF->odom;
    // Tb2b1, 当前帧Body->参考帧Body, se2形式
    Se2 Tb1b2 = mCurrentFrame.odom - mpReferenceKF->odom;
    Se2 Tb2b1 = mpReferenceKF->odom - mCurrentFrame.odom;
    // Tc2c1, 当前帧Camera->参考帧Camera, Tc2c1 = Tcb * Tb2b1 * Tbc
    // mCurrentFrame.Tcr = Config::Tcb * dOdo.toCvSE3() * Config::Tbc;
    // Tc2w, 当前帧Camera->World, Tc2w = Tc2c1 * Tc1w
    // mCurrentFrame.Tcw = mCurrentFrame.Tcr * mpReferenceKF->Tcw;
    // Twb2, 当前帧World->Body，se2形式，故相加， Twb2 = Twb1 * Tb1b2
    // mCurrentFrame.Twb = mpReferenceKF->Twb + mCurrentFrame.Trb;


    Mat Tc2c1 = Config::Tcb * Tb2b1.toCvSE3() * Config::Tbc;
    Mat Tc1w = mpReferenceKF->getPose();
    mCurrentFrame.setTrb(Tb1b2);
    mCurrentFrame.setTcr(Tc2c1);
    mCurrentFrame.setPose(Tc2c1 * Tc1w);

    // preintegration 预积分
    //! TODO 这里并没有使用上预积分？都是局部变量，且实际一帧图像仅对应一帧Odom数据
/*
    Eigen::Map<Vector3d> meas(preSE2.meas);
    Se2 odok = mCurrentFrame.odom - mLastOdom;
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
*/
}


//! 当前帧设为KF时才执行，将当前KF变参考帧，将当前帧的KP转到mPrevMatched中.
void Track::resetLocalTrack()
{
    KeyPoint::convert(mCurrentFrame.mvKeyPoints, mPrevMatched);  // cv::KeyPoint转cv::Point2f

    // 更新当前Local MP为参考帧观测到的MP
    mLocalMPs = mpReferenceKF->mvViewMPs;
    mnGoodPrl = 0;
    mvMatchIdx.clear();

    for (int i = 0; i < 3; ++i)
        preSE2.meas[i] = 0;
    for (int i = 0; i < 9; ++i)
        preSE2.cov[i] = 0;

    mAffineMatrix = Mat::eye(2, 3, CV_64FC1);
}


//! 可视化用，数据拷贝
size_t Track::copyForPub(Mat& img1, Mat& img2, vector<KeyPoint>& kp1, vector<KeyPoint>& kp2,
                         vector<int>& vMatches12)
{
    locker lock(mMutexForPub);

    if (!mbNeedVisualization)
        return 0;
    if (mvMatchIdx.empty())
        return 0;

    mpReferenceKF->copyImgTo(img1);
    mCurrentFrame.copyImgTo(img2);

    kp1 = mpReferenceKF->mvKeyPoints;
    kp2 = mCurrentFrame.mvKeyPoints;
    vMatches12 = mvMatchIdx;

    return mvMatchIdx.size();
}

void Track::drawMatchesForPub(bool warp)
{
    if (!mbNeedVisualization)
        return;
    if (mCurrentFrame.id == mpReferenceKF->id)
        return;

    Mat imgWarp, A21;
    Mat imgCur = mCurrentFrame.mImage.clone();
    Mat imgRef = mpReferenceKF->mImage.clone();
    if (imgCur.channels() == 1)
        cvtColor(imgCur, imgCur, CV_GRAY2BGR);
    if (imgRef.channels() == 1)
        cvtColor(imgRef, imgRef, CV_GRAY2BGR);

    // 所有KP先上蓝色
    drawKeypoints(imgCur, mCurrentFrame.mvKeyPoints, imgCur, Scalar(255, 0, 0), DrawMatchesFlags::DRAW_OVER_OUTIMG);
    drawKeypoints(imgRef, mpReferenceKF->mvKeyPoints, imgRef, Scalar(255, 0, 0), DrawMatchesFlags::DRAW_OVER_OUTIMG);

    if (warp) {
        // 去掉A12中的尺度变换, 只保留旋转, 并取逆得到A21
        invertAffineTransform(mAffineMatrix, A21);
//        double angle = normalizeAngle(mpReferenceKF->odom.theta - mCurrentFrame.odom.theta);
//        Point2f rotationCenter(160.5827 - 0.01525, 117.7329 - 3.6984); // TODO
//        Mat R = getRotationMatrix2D(rotationCenter, angle * 180 / CV_PI, 1.);
//        A21.colRange(0, 2) = R.colRange(0, 2).clone();
        warpAffine(imgCur, imgWarp, A21, imgCur.size());
        hconcat(imgRef, imgWarp, mImgOutMatch);
    } else {
        hconcat(imgRef, imgCur, mImgOutMatch);
    }

    char strMatches[64];
    std::snprintf(strMatches, 64, "KF: %ld-%ld, M: %d/%d", mpReferenceKF->mIdKF,
                  mCurrentFrame.id - mpReferenceKF->id, mnInliers, mnMatchSum);
    putText(mImgOutMatch, strMatches, Point(240, 15), 1, 1, Scalar(0, 0, 255), 2);

    for (size_t i = 0, iend = mvMatchIdx.size(); i != iend; ++i) {
        if (mvMatchIdx[i] < 0) {
            continue;
        } else {
            Point2f& ptRef = mpReferenceKF->mvKeyPoints[i].pt;
            Point2f& ptCur = mCurrentFrame.mvKeyPoints[mvMatchIdx[i]].pt;
            Point2f ptr;

            if (warp) {
                Mat pt1 = (Mat_<double>(3, 1) << ptCur.x, ptCur.y, 1);
                Mat pt1_warp = A21 * pt1;
                ptr = Point2f(pt1_warp.at<double>(0), pt1_warp.at<double>(1)) + Point2f(imgRef.cols, 0);
            } else {
                ptr = ptCur + Point2f(imgRef.cols, 0);
            }

            // 匹配上KP的为绿色
            circle(mImgOutMatch, ptRef, 3, Scalar(0, 255, 0));
            circle(mImgOutMatch, ptr, 3, Scalar(0, 255, 0));
            line(mImgOutMatch, ptRef, ptr, Scalar(255, 255, 0, 0.5));
        }
    }
}

cv::Mat Track::getImageMatches()
{
    locker lock(mMutexForPub);
    return mImgOutMatch;
}

/**
 * @brief 计算KF之间的残差和信息矩阵, 后端优化用
 * @param dOdo      [Input ]后一帧与前一帧之间的里程计差值
 * @param Tc1c2     [Output]残差
 * @param Info_se3  [Output]信息矩阵
 */
void Track::calcOdoConstraintCam(const Se2& dOdo, Mat& Tc1c2, g2o::Matrix6d& Info_se3)
{
    const Mat Tbc = Config::Tbc;
    const Mat Tcb = Config::Tcb;
    const Mat Tb1b2 = Se2(dOdo.x, dOdo.y, dOdo.theta).toCvSE3();

    Tc1c2 = Tcb * Tb1b2 * Tbc;

    float dx = dOdo.x * Config::OdoUncertainX + Config::OdoNoiseX;
    float dy = dOdo.y * Config::OdoUncertainY + Config::OdoNoiseY;
    float dtheta = dOdo.theta * Config::OdoUncertainTheta + Config::OdoNoiseTheta;


    g2o::Matrix6d Info_se3_bTb = g2o::Matrix6d::Zero();
    float data[6] = {1.f / (dx * dx), 1.f / (dy * dy), 1e-4, 1e-4, 1e-4, 1.f / (dtheta * dtheta)};
    for (int i = 0; i < 6; ++i)
        Info_se3_bTb(i, i) = data[i];
    Info_se3 = Info_se3_bTb;
}

/**
 * @brief 计算KF与MP之间的误差不确定度，计算约束(R^t)*∑*R
 * @param Pc1   [Input ]MP在KF1相机坐标系下的坐标
 * @param Tc1w  [Input ]KF1相机到世界坐标系的变换
 * @param Tc2w  [Input ]KF2相机到世界坐标系的变换
 * @param info1 [Output]MP在KF1中投影误差的信息矩阵
 * @param info2 [Output]MP在KF2中投影误差的信息矩阵
 */
void Track::calcSE3toXYZInfo(const Point3f& Pc1, const Mat& Tc1w, const Mat& Tc2w,
                             Eigen::Matrix3d& info1, Eigen::Matrix3d& info2)
{
    Point3f O1 = Point3f(cvu::inv(Tc1w).rowRange(0, 3).col(3));
    Point3f O2 = Point3f(cvu::inv(Tc2w).rowRange(0, 3).col(3));
    Point3f Pw = cvu::se3map(cvu::inv(Tc1w), Pc1);
    Point3f vO1 = Pw - O1;
    Point3f vO2 = Pw - O2;
    float sinParallax = cv::norm(vO1.cross(vO2)) / (cv::norm(vO1) * cv::norm(vO2));

    Point3f Pc2 = cvu::se3map(Tc2w, Pw);
    float length1 = cv::norm(Pc1);
    float length2 = cv::norm(Pc2);
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
    Point3f k1 = Pc1.cross(z1);
    float normk1 = cv::norm(k1);
    float sin1 = normk1 / (cv::norm(z1) * cv::norm(Pc1));
    k1 = k1 * (std::asin(sin1) / normk1);
    Point3f k2 = Pc2.cross(z2);
    float normk2 = cv::norm(k2);
    float sin2 = normk2 / (cv::norm(z2) * cv::norm(Pc2));
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
 * @brief   根据仿射矩阵A剔除外点，利用了RANSAC算法
 * @return  返回内点数
 */
int Track::removeOutliers()
{
    vector<Point2f> ptRef, ptCur;
    vector<size_t> idxRef;
    idxRef.reserve(mpReferenceKF->N);
    ptRef.reserve(mpReferenceKF->N);
    ptCur.reserve(mCurrentFrame.N);

    for (size_t i = 0, iend = mpReferenceKF->N; i < iend; ++i) {
        if (mvMatchIdx[i] < 0)
            continue;
        idxRef.push_back(i);
        ptRef.push_back(mpReferenceKF->mvKeyPoints[i].pt);
        ptCur.push_back(mCurrentFrame.mvKeyPoints[mvMatchIdx[i]].pt);
    }

    if (ptRef.size() == 0)
        return 0;

    vector<unsigned char> mask;
    mAffineMatrix = estimateAffine2D(ptRef, ptCur, mask, RANSAC, 3.0);
//   Mat H = findHomography(pt1, pt2, RANSAC, 3, mask);  // 朝天花板摄像头应该用H矩阵, F会退化

    int nInlier = 0;
    for (size_t i = 0, iend = mask.size(); i < iend; ++i) {
        if (!mask[i])
            mvMatchIdx[idxRef[i]] = -1;
        else
            nInlier++;
    }

    return nInlier;
}

bool Track::needNewKF()
{
    int nOldKP = mpReferenceKF->countObservations();
    int deltaFrames = static_cast<int>(mCurrentFrame.id - mpReferenceKF->id);
    bool c0 = deltaFrames > nMinFrames;
    bool c1 = mnTrackedOld <= cvFloor(nOldKP * 0.5f);
    bool c2 = mnGoodPrl > cvFloor(mnGoodDepth * 0.6);
    bool c3 = deltaFrames > nMaxFrames;
    bool c4 = mnMatchSum < nMinMatches;
    bool bNeedNewKF = c0 && (c3 || c4 || (c1 && c2));

    bool bNeedKFByOdo = false;
    bool c5 = false, c6 = false;
    if (mbUseOdometry) {
        Se2 dOdo = mCurrentFrame.odom - mpReferenceKF->odom;
        c5 = static_cast<double>(abs(dOdo.theta)) >= mMaxAngle;  // 旋转量超过40°
        cv::Mat cTc = Config::Tcb * dOdo.toCvSE3() * Config::Tbc;
        cv::Mat xy = cTc.rowRange(0, 2).col(3);
        c6 = cv::norm(xy) >= mMaxDistance;  // 相机的平移量足够大

        bNeedKFByOdo = c5 || c6;  // 相机移动取决于深度上限,考虑了不同深度下视野的不同
    }
    bNeedNewKF = bNeedNewKF || bNeedKFByOdo;  // 加上odom的移动条件, 把与改成了或

    if (bNeedNewKF)
        printf("[Track][Info ] #%ld T4.应该成为KF, 其KF条件满足情况: %d/%d/%d/%d/%d/%d/%d\n", mCurrentFrame.id,
               c0, c1, c2, c3, c4, c5, c6);

    // 最后还要看LocalMapper准备好了没有，LocalMapper正在执行优化的时候是不接收新KF的
    if (mpLocalMapper->acceptNewKF()) {
        return bNeedNewKF;
    } else if (c0 && (c4 || c3)/* && bNeedKFByOdo*/) {
        fprintf(stderr, "[Track][Info ] #%ld(#KF%ld) 强制添加KF, 关键帧条件的满足情况: %d/%d/%d/%d/%d/%d/%d\n",
                mCurrentFrame.id, KeyFrame::nextId, c0, c1, c2, c3, c4, c5, c6);
        mpLocalMapper->setAbortBA();  // 如果必须要加入关键帧,则终止LocalMap的优化,下一帧就可以变成KF
        return true;
    }

//    int nMPObs = mpReferenceKF->countObservations();  // 注意初始帧观测数为0
//    bool c1 = (float)mnTrackedOld > nMPObs * 0.5f;   // 2.关联MP数不能太多(要小于50%)
//    if (nMPObs && c1) {
//        printf("[Track][Info ] #%ld 不是KF, 因为关联MP数超过了50%%(%d)\n", mCurrentFrame.id,
//               mnTrackedOld);
//        return false;
//    }

//    bool c4 = mnInliers < 0.05f * Config::MaxFtrNumber;  // 3.3
//    或匹配内点数小于1/10最大特征点数
//    if (c4 && mpLocalMapper->acceptNewKF()) {
//        printf("[Track][Info ] #%ld 成为了KF, 因为匹配内点数小于10%%\n", mCurrentFrame.id);
//        return true;
//    } else if (c4) {
//        printf("[Track][Info ] #%ld 不是KF, 虽然匹配内点数小于10%%, 但局部地图正在工作!\n",
//               mCurrentFrame.id);
//        return false;
//    }

    return false;
}


/**
 * @brief 关键函数, 三角化获取特征点的深度
 *  mLocalMPs, mvbGoodPrl, mnGoodPrl 在此更新
 * @return  返回匹配上参考帧的MP的数量
 */
int Track::doTriangulate()
{
    if (static_cast<int>(mCurrentFrame.id - mpReferenceKF->id) < nMinFrames)
        return 0;

    Mat Tcr = mCurrentFrame.getTcr();
    Mat Tc1c2 = cvu::inv(Tcr);
    Point3f Ocam1 = Point3f(0.f, 0.f, 0.f);
    Point3f Ocam2 = Point3f(Tc1c2.rowRange(0, 3).col(3));
    mvbGoodPrl = vector<bool>(mpReferenceKF->N, false);
    mnGoodPrl = 0;
    mnGoodDepth = 0;

    // 相机1和2的投影矩阵
    cv::Mat Proj1 = Config::PrjMtrxEye;                 // P1 = K * cv::Mat::eye(3, 4, CV_32FC1)
    cv::Mat Proj2 = Config::Kcam * Tcr.rowRange(0, 3);  // P2 = K * Tc2c1(3*4)

    // 1.遍历参考帧的KP
    int nTrackedOld(0), nBadDepth(0);
    for (int i = 0, iend = mpReferenceKF->N; i < iend; ++i) {
        if (mvMatchIdx[i] < 0)
            continue;

        // 2.如果参考帧的KP与当前帧的KP有匹配,且参考帧KP已经有对应的MP观测了，则可见地图点更新为此MP
        if (mpReferenceKF->hasObservation(i)) {
            mLocalMPs[i] = mpReferenceKF->mvViewMPs[i];
            // mLocalMPs[i] = mpReferenceKF->getViewMPPoseInCamareFrame(i);
            nTrackedOld++;
            continue;
        }

        // 3.如果参考帧KP没有对应的MP，则为此匹配对KP三角化计算深度(相对参考帧的坐标)
        // 由于两个投影矩阵是两KF之间的相对投影, 故三角化得到的坐标是相对参考帧的坐标, 即Pc1
        Point2f pt1 = mpReferenceKF->mvKeyPoints[i].pt;
        Point2f pt2 = mCurrentFrame.mvKeyPoints[mvMatchIdx[i]].pt;
        Point3f Pc1 = cvu::triangulate(pt1, pt2, Proj1, Proj2);

        // 3.如果深度计算符合预期，就将有深度的KP更新到LocalMPs里, 其中视差较好的会被标记
        if (Config::acceptDepth(Pc1.z)) {
            mnGoodDepth++;
            mLocalMPs[i] = Pc1;
            // 检查视差
            if (cvu::checkParallax(Ocam1, Ocam2, Pc1, 2)) {
                mnGoodPrl++;
                mvbGoodPrl[i] = true;
            }
        } else {  // 3.如果深度计算不符合预期，就剔除此匹配点对
            nBadDepth++;
            mvMatchIdx[i] = -1;
        }
    }
    printf("[Track][Info ] #%ld T1.三角化, 良好视差点数/生成点数/因深度不符而剔除的匹配点对数: %d/%d/%d\n",
           mCurrentFrame.id, mnGoodPrl, mnGoodDepth, nBadDepth);

    return nTrackedOld;
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
void Track::relocalization(const cv::Mat& img, const double& imgTime, const Se2& odo)
{
    return;
}


}  // namespace se2lam

