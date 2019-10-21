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

    mLocalMPs = vector<Point3f>(Config::MaxFtrNumber, Point3f(-1.f, -1.f, -1.f));

    nMinFrames = min(1, cvCeil(0.25 * Config::FPS));  // 上溢
    nMaxFrames = cvFloor(2 * Config::FPS);            // 下溢
    mMaxAngle = g2o::deg2rad(20.);
    mMaxDistance = 0.5 * Config::UpperDepth * 0.15;

    mK = Config::Kcam;
    mD = Config::Dcam;
    mHomography = Mat::eye(3, 3, CV_64FC1);  // float

    mbPrint = Config::GlobalPrint;
    mbNeedVisualization = Config::NeedVisualization;
}

Track::~Track()
{
    delete mpORBextractor;
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
        bool sensorUpdated = mpSensors->update();
        if (sensorUpdated) {
            timer.start();

            mLastState = mState;

            Mat img;
            Se2 odo;
            mpSensors->readData(odo, img);

            {  // 计算位姿时不做可视化, 防止数据竞争
//                locker lock(mMutexForPub);
                if (mState == cvu::FIRST_FRAME) {
                    createFirstFrame(img, imgTime, odo);
                } else if (mState == cvu::OK) {
                    trackReferenceKF(img, imgTime, odo);
                } else {
                    relocalization(img, imgTime, odo);
                }
            }
            mpMap->setCurrentFramePose(mCurrentFrame.getPose());
            mLastOdom = odo;

            timer.stop();
            fprintf(stdout, "[Track][Timer] #%ld 当前帧前段追踪总耗时: %.2fms\n", mCurrentFrame.id,
                    timer.time);
        }

        if (checkFinish())
            break;

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

    ORBmatcher matcher(0.8);  // 由于trackWithRefKF,间隔有些帧数，nnratio不应过小

    //! 上一帧的KF附近40*40方形cell内进行搜索获取匹配索引
    //! mPrevMatched为参考帧KP的位置, 这里和当前帧匹配上的KP会更新成当前帧对应KP的位置
    //! mPrevMatched这个变量没啥用, 直接用mpReferenceKF->keyPoints其实就可以了
    // mnMatchSum = matcher.MatchByWindow(mRefFrame, mFrame, mPrevMatched, 20, mMatchIdx);

    //! 基于H矩阵的透视变换先验估计投影Cell的位置
    mnMatchSum =
        matcher.MatchByWindowWarp(*mpReferenceKF, mCurrentFrame, mHomography, mvMatchIdx, 25);
    if (mnMatchSum < 0.1 * Config::MaxFtrNumber)
        mnMatchSum =
            matcher.MatchByWindowWarp(*mpReferenceKF, mCurrentFrame, mHomography, mvMatchIdx, 40);

    //! 利用单应矩阵H计算匹配内点，内点数大于10才能继续
    mnInliers = removeOutliers();        // H更新
    if (mnInliers) {                     // 内点数大于10则三角化计算MP
        mnTrackedOld = doTriangulate();  // 更新viewMPs
        drawMatchesForPub(true);
        if (mbPrint)
            printf("[Track][Info ] #%ld 与参考帧匹配获得的点数: trackedOld/inliers/matchedSum = %d/%d/%d.\n",
                   mCurrentFrame.id, mnTrackedOld, mnInliers, mnMatchSum);
    } else {
        drawMatchesForPub(false);
        if (mbPrint)
            fprintf(stderr, "[Track][Warni] #%ld 与参考帧匹配内点数少于10! trackedOld/matchedSum = %d/%d.\n",
                    mCurrentFrame.id, mnTrackedOld, mnMatchSum);
    }

    N1 += mnTrackedOld;
    N2 += mnInliers;
    N3 += mnMatchSum;
    if (mbPrint && mCurrentFrame.id % 5 == 0) {  // 每隔50帧输出一次平均匹配情况
        float sum = mCurrentFrame.id - 1.0;
        printf("[Track][Info ] #%ld 与参考帧匹配平均点数: tracked/matched/matchedSum = "
               "%.2f/%.2f/%.2f .\n",
               mCurrentFrame.id, N1 * 1.f / sum, N2 * 1.f / sum, N3 * 1.f / sum);
    }

    // Need new KeyFrame decision
    if (needNewKF()) {
        PtrKeyFrame pKF = make_shared<KeyFrame>(mCurrentFrame);

        assert(mpMap->getCurrentKF()->mIdKF == mpReferenceKF->mIdKF);

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

//! mpKF是Last KF，根据Last KF和当前帧的里程计更新先验位姿和变换关系, odom是se2绝对位姿而非增量
//! 注意相机的平移量不等于里程的平移量!!!
//! 由于相机和里程计的坐标系不一致，在没有旋转时这两个平移量会相等，但是一旦有旋转，这两个量就不一致了!
//! 如果当前帧不是KF，则当前帧位姿由Last KF叠加上里程计数据计算得到
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
    //    Eigen::Map<Vector3d> meas(preSE2.meas);
    //    Se2 odok = mCurFrame.odom - mLastOdom;
    //    Vector2d odork(odok.x, odok.y);
    //    Matrix2d Phi_ik = Rotation2Dd(meas[2]).toRotationMatrix();
    //    meas.head<2>() += Phi_ik * odork;
    //    meas[2] += odok.theta;

    //    Matrix3d Ak = Matrix3d::Identity();
    //    Matrix3d Bk = Matrix3d::Identity();
    //    Ak.block<2, 1>(0, 2) = Phi_ik * Vector2d(-odork[1], odork[0]);
    //    Bk.block<2, 2>(0, 0) = Phi_ik;
    //    Eigen::Map<Matrix3d, RowMajor> Sigmak(preSE2.cov);
    //    Matrix3d Sigma_vk = Matrix3d::Identity();
    //    Sigma_vk(0, 0) = (Config::OdoNoiseX * Config::OdoNoiseX);
    //    Sigma_vk(1, 1) = (Config::OdoNoiseY * Config::OdoNoiseY);
    //    Sigma_vk(2, 2) = (Config::OdoNoiseTheta * Config::OdoNoiseTheta);
    //    Matrix3d Sigma_k_1 = Ak * Sigmak * Ak.transpose() + Bk * Sigma_vk * Bk.transpose();
    //    Sigmak = Sigma_k_1;
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

    mHomography = Mat::eye(3, 3, CV_64FC1);
}

//! 可视化用，数据拷贝
size_t Track::copyForPub(Mat& img1, Mat& img2, vector<KeyPoint>& kp1, vector<KeyPoint>& kp2,
                         vector<int>& vMatches12)
{
    if (!mbNeedVisualization)
        return 0;
    if (mvMatchIdx.empty())
        return 0;

    locker lock(mMutexForPub);

    mpReferenceKF->copyImgTo(img1);
    mCurrentFrame.copyImgTo(img2);

    kp1 = mpReferenceKF->mvKeyPoints;
    kp2 = mCurrentFrame.mvKeyPoints;
    vMatches12 = mvMatchIdx;

    return mvMatchIdx.size();
}

void Track::drawFrameForPub(Mat& imgLeft)
{
    if (!mbNeedVisualization)
        return;

    locker lock(mMutexForPub);

    //! 画左侧两幅图
    Mat imgUp = mCurrentFrame.mImage.clone();
    Mat imgDown = mpReferenceKF->mImage.clone();
    if (imgUp.channels() == 1)
        cvtColor(imgUp, imgUp, CV_GRAY2BGR);
    if (imgDown.channels() == 1)
        cvtColor(imgDown, imgDown, CV_GRAY2BGR);

    for (size_t i = 0, iend = mvMatchIdx.size(); i != iend; ++i) {
        Point2f ptRef = mpReferenceKF->mvKeyPoints[i].pt;
        if (mvMatchIdx[i] < 0) {
            circle(imgDown, ptRef, 3, Scalar(255, 0, 0), 1);  // 未匹配上的为蓝色
            continue;
        } else {
            circle(imgDown, ptRef, 3, Scalar(0, 255, 0), 1);  // 匹配上的为绿色

            Point2f ptCurr = mCurrentFrame.mvKeyPoints[mvMatchIdx[i]].pt;
            circle(imgUp, ptCurr, 3, Scalar(0, 255, 0), 1);  // 当前KP为绿色
            circle(imgUp, ptRef, 3, Scalar(0, 0, 255), 1);   // 参考KP为红色
            line(imgUp, ptRef, ptCurr, Scalar(0, 255, 0));
        }
    }
    vconcat(imgUp, imgDown, imgLeft);

    char strMatches[64];
    std::snprintf(strMatches, 64, "F: %ld-%ld, M: %d/%d", mpReferenceKF->id, mCurrentFrame.id,
                  mnInliers, mnMatchSum);
    putText(imgLeft, strMatches, Point(50, 15), 1, 1, Scalar(0, 0, 255), 2);
}

void Track::drawMatchesForPub(bool warp)
{
    if (!mbNeedVisualization)
        return;
    if (mCurrentFrame.id == mpReferenceKF->id)
        return;

    Mat imgWarp, H21;
    Mat imgCur = mCurrentFrame.mImage.clone();
    Mat imgRef = mpReferenceKF->mImage.clone();
    if (imgCur.channels() == 1)
        cvtColor(imgCur, imgCur, CV_GRAY2BGR);
    if (imgRef.channels() == 1)
        cvtColor(imgRef, imgRef, CV_GRAY2BGR);

    // 所有KP先上蓝色
    drawKeypoints(imgCur, mCurrentFrame.mvKeyPoints, imgCur, Scalar(255, 0, 0),
                  DrawMatchesFlags::DRAW_OVER_OUTIMG);
    drawKeypoints(imgRef, mpReferenceKF->mvKeyPoints, imgRef, Scalar(255, 0, 0),
                  DrawMatchesFlags::DRAW_OVER_OUTIMG);

    if (warp) {
        H21 = mHomography.inv(DECOMP_SVD);
        warpPerspective(imgCur, imgWarp, H21, imgRef.size());
        hconcat(imgWarp, imgRef, mImgOutMatch);
    } else {
        hconcat(imgCur, imgRef, mImgOutMatch);
    }

    char strMatches[64];
    std::snprintf(strMatches, 64, "KF: %ld-%ld, M: %d/%d", mpReferenceKF->mIdKF,
                  mCurrentFrame.id - mpReferenceKF->id, mnInliers, mnMatchSum);
    putText(mImgOutMatch, strMatches, Point(240, 15), 1, 1, Scalar(0, 0, 255), 2);

    for (size_t i = 0, iend = mvMatchIdx.size(); i != iend; ++i) {
        if (mvMatchIdx[i] < 0) {
            continue;
        } else {
            Point2f ptRef = mpReferenceKF->mvKeyPoints[i].pt;
            Point2f ptCur = mCurrentFrame.mvKeyPoints[mvMatchIdx[i]].pt;

            Point2f ptl, ptr;
            if (warp) {
                Mat pt1 = (Mat_<double>(3, 1) << ptCur.x, ptCur.y, 1);
                Mat pt2 = H21 * pt1;
                pt2 /= pt2.at<double>(2);
                ptl = Point2f(pt2.at<double>(0), pt2.at<double>(1));
            } else {
                ptl = ptCur;
            }
            ptr = ptRef + Point2f(imgCur.cols, 0);

            // 匹配上KP的为绿色
            circle(mImgOutMatch, ptl, 3, Scalar(0, 255, 0));
            circle(mImgOutMatch, ptr, 3, Scalar(0, 255, 0));
            line(mImgOutMatch, ptl, ptr, Scalar(255, 255, 0, 0.6));
        }
    }
}

cv::Mat Track::getImageMatches()
{
    locker lock(mMutexForPub);
    return mImgOutMatch;
}

//! 后端优化用，信息矩阵
//! TODO 公式待推导
void Track::calcOdoConstraintCam(const Se2& dOdo, Mat& cTc, g2o::Matrix6d& Info_se3)
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
    for (int i = 0; i < 6; ++i)
        Info_se3_bTb(i, i) = data[i];
    Info_se3 = Info_se3_bTb;


    //    g2o::Matrix6d J_bTb_cTc = toSE3Quat(bTc).adj();
    //    J_bTb_cTc.block(0,3,3,3) = J_bTb_cTc.block(3,0,3,3);
    //    J_bTb_cTc.block(3,0,3,3) = g2o::Matrix3D::Zero();

    //    Info_se3 = J_bTb_cTc.transpose() * Info_se3_bTb * J_bTb_cTc;

    //    for (int i = 0; i < 6; ++i)
    //        for (int j = 0; j < i; ++j)
    //            Info_se3(i,j) = Info_se3(j,i);

    // assert(verifyInfo(Info_se3));
}

//! 后端优化用，计算约束(R^t)*∑*R
//! TODO 公式待推导
void Track::calcSE3toXYZInfo(Point3f Pc1, const Mat& Tc1w, const Mat& Tc2w, Eigen::Matrix3d& info1,
                             Eigen::Matrix3d& info2)
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
 * @brief   根据本质矩阵H剔除外点，利用了RANSAC算法
 * @return  返回内点数
 */
int Track::removeOutliers()
{
    vector<Point2f> pt1, pt2;
    vector<size_t> idx;
    pt1.reserve(mpReferenceKF->N);
    pt2.reserve(mCurrentFrame.N);
    idx.reserve(mpReferenceKF->N);

    for (size_t i = 0, iend = mpReferenceKF->N; i < iend; ++i) {
        if (mvMatchIdx[i] < 0)
            continue;
        idx.push_back(i);
        pt1.push_back(mpReferenceKF->mvKeyPoints[i].pt);
        pt2.push_back(mCurrentFrame.mvKeyPoints[mvMatchIdx[i]].pt);
    }

    if (pt1.size() == 0)
        return 0;

    vector<unsigned char> mask;
//   Mat F = findFundamentalMat(pt1, pt2, FM_RANSAC, 3, 0.99, mask);
    mHomography = findHomography(pt1, pt2, RANSAC, 3, mask);  // 朝天花板摄像头应该用H矩阵, F会退化

    int nInlier = 0;
    for (size_t i = 0, iend = mask.size(); i < iend; ++i) {
        if (!mask[i])
            mvMatchIdx[idx[i]] = -1;
        else
            nInlier++;
    }

    // If too few match inlier, discard all matches. The enviroment might not be
    // suitable for image tracking.
    if (nInlier < 10) {
        nInlier = 0;
        std::fill(mvMatchIdx.begin(), mvMatchIdx.end(), -1);
    }

    return nInlier;
}

bool Track::needNewKF()
{
    int nOldKP = mpReferenceKF->countObservations();
    bool c0 = mCurrentFrame.id - mpReferenceKF->id > nMinFrames;
    bool c1 = static_cast<float>(mnTrackedOld) <= nOldKP * 0.5f;
    bool c2 = mnGoodPrl > 40;
    bool c3 = mCurrentFrame.id - mpReferenceKF->id > nMaxFrames;
    bool c4 = mnMatchSum < 0.1f * Config::MaxFtrNumber || mnMatchSum < 20;
    bool bNeedNewKF = c0 && ( (c1 && c2) || c3 || c4 );

    bool bNeedKFByOdo = false;
    if (mbUseOdometry) {
        Se2 dOdo = mCurrentFrame.odom - mpReferenceKF->odom;
        bool c5 = static_cast<double>(abs(dOdo.theta)) >= mMaxAngle;  // 旋转量超过20°
        cv::Mat cTc = Config::Tcb * dOdo.toCvSE3() * Config::Tbc;
        cv::Mat xy = cTc.rowRange(0, 2).col(3);
        bool c6 = cv::norm(xy) >= mMaxDistance;  // 相机的平移量足够大

        bNeedKFByOdo = c5 || c6;  // 相机移动取决于深度上限,考虑了不同深度下视野的不同
    }
    bNeedNewKF = bNeedNewKF && bNeedKFByOdo;  // 加上odom的移动条件, 把与改成了或

    // 最后还要看LocalMapper准备好了没有，LocalMapper正在执行优化的时候是不接收新KF的
    if (mpLocalMapper->acceptNewKF()) {
        return bNeedNewKF;
    } else if (c0 && (c4 || c3) && bNeedKFByOdo) {
        printf("[Track][Info ] #%ld 强制添加KF\n", mCurrentFrame.id, mnTrackedOld);
        mpLocalMapper->setAbortBA();  // 如果必须要加入关键帧,则终止LocalMap的优化,下一帧进来时就可以变成KF了
        return bNeedNewKF;
    }

/*
    int nMPObs = mpReferenceKF->countObservation();  // 注意初始帧观测数为0
    bool c1 = (float)mnTrackedOld > nMPObs * 0.5f;   // 2.关联MP数不能太多(要小于50%)
    if (nMPObs && c1) {
        printf("[Track][Info ] #%ld 不是KF, 因为关联MP数超过了50%%(%d)\n", mCurrentFrame.id,
               mnTrackedOld);
        return false;
    }

    bool c4 = mnInliers < 0.1f * Config::MaxFtrNumber;  // 3.3 或匹配内点数小于1/10最大特征点数
    if (c4 && mpLocalMapper->acceptNewKF()) {
        printf("[Track][Info ] #%ld 成为了KF, 因为匹配内点数小于10%%\n", mCurrentFrame.id);
        return true;
    } else if (c4) {
        printf("[Track][Info ] #%ld 不是KF, 虽然匹配内点数小于10%%, 但局部地图正在工作!\n",
               mCurrentFrame.id);
        return false;
    }
*/

    return false;
}


/**
 * @brief 关键函数, 与参考帧匹配内点数大于10则进行三角化获取深度
 *  mLocalMPs, mvbGoodPrl, mnGoodPrl 在此更新
 * @return  返回匹配上参考帧的MP的数量
 */
int Track::doTriangulate()
{
    //    if (mCurrentFrame.id - mpReferenceKF->id < nMinFrames)
    //        return 0;

    Mat Tcr = mCurrentFrame.getTcr();
    Mat Tc1c2 = cvu::inv(Tcr);
    Point3f Ocam1 = Point3f(0.f, 0.f, 0.f);
    Point3f Ocam2 = Point3f(Tc1c2.rowRange(0, 3).col(3));
    mvbGoodPrl = vector<bool>(mpReferenceKF->N, false);
    mnGoodPrl = 0;

    // 相机1和2的投影矩阵
    cv::Mat Proj1 = Config::PrjMtrxEye;                 // P1 = K * cv::Mat::eye(3, 4, CV_32FC1)
    cv::Mat Proj2 = Config::Kcam * Tcr.rowRange(0, 3);  // P2 = K * Tc2c1(3*4)

    // 1.遍历参考帧的KP
    int nTrackedOld(0), nGoodDepth(0), nBadDepth(0);
    for (int i = 0, iend = mpReferenceKF->N; i < iend; ++i) {
        if (mvMatchIdx[i] < 0)
            continue;

        // 2.如果参考帧的KP与当前帧的KP有匹配,且参考帧KP已经有对应的MP观测了，则可见地图点更新为此MP
        if (mpReferenceKF->hasObservation(i)) {
            mLocalMPs[i] = mpReferenceKF->mvViewMPs[i];
            //            mLocalMPs[i] = mpReferenceKF->getViewMPPoseInCamareFrame(i);
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
            nGoodDepth++;
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
    printf("[Track][Info ] #%ld 1.三角化生成了%d个点且%d个具有良好视差, "
           "因深度不符合预期而剔除的匹配点对有%d个\n",
           mCurrentFrame.id, nGoodDepth, mnGoodPrl, nBadDepth);

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

Se2 Track::dataAlignment(std::vector<Se2>& dataOdoSeq, double& timeImg)
{
    Se2 res;
    size_t n = dataOdoSeq.size();
    if (n < 2) {
        cerr << "[Track][Warni] Less odom sequence input!" << endl;
        return res;
    }

    //! 计算单帧图像时间内的平均速度
    Se2 tranSum = dataOdoSeq[n - 1] - dataOdoSeq[0];
    double dt = dataOdoSeq[n - 1].timeStamp - dataOdoSeq[0].timeStamp;
    double r = (timeImg - dataOdoSeq[n - 1].timeStamp) / dt;

    assert(r >= 0.f);

    Se2 transDelta(tranSum.x * r, tranSum.y * r, tranSum.theta * r);
    res = dataOdoSeq[n - 1] + transDelta;

    return res;
}

//! TODO 完成重定位功能
void Track::relocalization(const cv::Mat& img, const double& imgTime, const Se2& odo)
{
    return;
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

for (size_t i = 0; i < nKFs; ++i) {
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
    for (size_t i = 0; i < nKFs; ++i) {
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

            for (size_t j = 0; j < np; ++j) {
                if (vbInliers[j]) {
                    mCurrentFrame.mvpMapPoints[j] =
                        vvpMapPointMatches[i][j];
                    sFound.insert(vvpMapPointMatches[i][j]);
                } else
                    mCurrentFrame.mvpMapPoints[j] = nullptr;
            }

            // 步骤5：通过PoseOptimization对姿态进行优化求解
            int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

            if (nGood < 10)
                continue;

            for (int io = 0; io < mCurrentFrame.N; io++)
                if (mCurrentFrame.mvbOutlier[io])
                    mCurrentFrame.mvpMapPoints[io] =
                        static_cast<MapPoint *>(nullptr);

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
                                    mCurrentFrame.mvpMapPoints[io] = nullptr;
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
