/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include "MapPoint.h"
#include "Config.h"
#include "KeyFrame.h"
#include "ORBmatcher.h"
#include "Track.h"
#include "cvutil.h"

namespace se2lam
{
using namespace cv;
using namespace std;
typedef unique_lock<mutex> locker;

int MapPoint::mNextId = 0;

MapPoint::MapPoint()
{
    mbNull = false;
    mbGoodParallax = false;
    mNormalVector = Point3f(0, 0, 0);
    mMainKF = nullptr;
    mObservations.clear();

//    mId = -1;
//    mMainOctave = 0;
//    mLevelScaleFactor = Config::ScaleFactor;
//    mPos = Point3f(0, 0, 0);
//    mMinDist = 0.f;
//    mMaxDist = 0.f;
}

MapPoint::MapPoint(Point3f pos, bool goodPrl)
{
    mObservations.clear();
    mNormalVector = Point3f(0, 0, 0);
    mMainKF = nullptr;

    mbNull = false;
    mPos = pos;
    mbGoodParallax = goodPrl;

    mId = mNextId;
    mNextId++;

//    mMainOctave = 0;
//    mLevelScaleFactor = Config::ScaleFactor;
//    mMinDist = 0.f;
//    mMaxDist = 0.f;
}

MapPoint::~MapPoint()
{
}

bool MapPoint::isNull()
{
    return mbNull;
}

bool MapPoint::isGoodPrl()
{
    return mbGoodParallax;
}

void MapPoint::setNull(const shared_ptr<MapPoint>& pThis)
{
    locker lock(mMutexObs);
    mbNull = true;
    mbGoodParallax = false;
    for (auto it = mObservations.begin(), iend = mObservations.end(); it != iend; it++) {
        PtrKeyFrame pKF = it->first;
        if (pKF->hasObservation(pThis))
            pKF->eraseObservation(pThis);
    }
    mObservations.clear();
    mMainDescriptor.release();
}

// Abandon a MP as an outlier, only for internal use
void MapPoint::setNull()
{
    mbNull = true;
    mbGoodParallax = false;
    for (auto it = mObservations.begin(), iend = mObservations.end(); it != iend; it++) {
        PtrKeyFrame pKF = it->first;
        if (pKF->hasObservation(it->second))
            pKF->eraseObservation(it->second);
    }
    mObservations.clear();
    mMainDescriptor.release();
}


Point3f MapPoint::getPos()
{
    locker lock(mMutexPos);
    return mPos;
}

/**
 * @brief MapPoint::eraseObservation 取消对KF的关联,在删减冗余KF时会用到
 * @param pKF   取消关联的KF对象
 */
void MapPoint::eraseObservation(const PtrKeyFrame& pKF)
{
    locker lock(mMutexObs);

    Point3f viewPos = pKF->mViewMPs[mObservations[pKF]];
    Point3f normPos = viewPos * (1.f / cv::norm(viewPos));

    mObservations.erase(pKF);
    if (!mbNull && mObservations.size() == 0) {
        setNull();
    } else {
        updateMainKFandDescriptor();

        int size = mObservations.size();
        mNormalVector = (mNormalVector * (float)(size + 1) - normPos) * (1.f / (float)size);
    }
}

// Do pKF.setViewMP() before using this
void MapPoint::addObservation(const PtrKeyFrame& pKF, int idx)
{
    locker lock(mMutexObs);

    int oldObsSize = mObservations.size();
    mObservations.insert(make_pair(pKF, idx));
    assert(idx < pKF->mDescriptors.rows);

    updateMainKFandDescriptor();

    updateParallax(pKF);

    Point3f newObs = pKF->mViewMPs[idx];
    Point3f newNorm = newObs * (1.f / cv::norm(newObs));
    mNormalVector = (mNormalVector * (float)oldObsSize + newNorm) * (1.f / (float)(oldObsSize + 1));

    if (mbNull) {
        mbNull = false;
    }
}

/**
 * @brief MapPoint::updateParallax 增加新的观测后会更新视差
 * 如果连续6帧KF后视差仍不好, 会抛弃该MP
 * @param pKF   能观测到此MP的KF指针
 */
void MapPoint::updateParallax(const PtrKeyFrame& pKF)
{
    if (mbGoodParallax || mObservations.size() <= 2)
        return;

    // Get the oldest KF in the last 6 KFs. 由从此KF往前的第6帧KF和此KF做三角化更新MP坐标
    PtrKeyFrame pKF0 = mObservations.begin()->first;
    for (auto it = mObservations.begin(), iend = mObservations.end(); it != iend; it++) {
        PtrKeyFrame pKF_ = it->first;
        if (!pKF_)
            continue;
        if (pKF->mIdKF - pKF_->mIdKF > 6)
            continue;
//        if (pKF_->mIdKF < pKF_->mIdKF) {  //! NOTE 这里应该是写错了!
        if (pKF_->mIdKF < pKF0->mIdKF) {    //! @Vance: 20190918改
            pKF0 = pKF_;
        }
    }

    // Do triangulation
    cv::Mat P0 = Config::Kcam * pKF0->Tcw.rowRange(0, 3);
    cv::Mat P1 = Config::Kcam * pKF->Tcw.rowRange(0, 3);
    Point2f pt0 = pKF0->mvKeyPoints[mObservations[pKF0]].pt;
    Point2f pt1 = pKF->mvKeyPoints[mObservations[pKF]].pt;
    Point3f posW = cvu::triangulate(pt0, pt1, P0, P1);

    Point3f pos0 = cvu::se3map(pKF0->Tcw, posW);
    Point3f pos1 = cvu::se3map(pKF->Tcw, posW);
    if (Config::acceptDepth(pos0.z) && Config::acceptDepth(pos1.z)) {
        Point3f Ocam0 = Point3f(cvu::inv(pKF0->Tcw).rowRange(0, 3).col(3));
        Point3f Ocam1 = Point3f(cvu::inv(pKF->Tcw).rowRange(0, 3).col(3));
        //! 视差良好, 则更新此点的三维坐标
        if (cvu::checkParallax(Ocam0, Ocam1, posW, 2)) {
            mPos = posW;
            mbGoodParallax = true;

            // Update measurements in KFs. 更新约束和信息矩阵
            Eigen::Matrix3d xyzinfo0, xyzinfo1;
            Track::calcSE3toXYZInfo(pos0, pKF0->Tcw, pKF->Tcw, xyzinfo0, xyzinfo1);
            pKF0->setViewMP(pos0, mObservations[pKF0], xyzinfo0);
            pKF->setViewMP(pos1, mObservations[pKF], xyzinfo1);

            cv::Mat Rcw0 = pKF0->Tcw.rowRange(0, 3).colRange(0, 3);
            cv::Mat xyzinfoW = Rcw0.t() * toCvMat(xyzinfo0) * Rcw0;

            // Update measurements in other KFs
            for (auto it = mObservations.begin(), iend = mObservations.end(); it != iend; it++) {
                PtrKeyFrame pKF_k = it->first;
                if (!pKF_k)
                    continue;
                if (pKF_k->mIdKF == pKF->mIdKF || pKF_k->mIdKF == pKF0->mIdKF) {
                    continue;
                }
                Point3f posk = cvu::se3map(pKF_k->Tcw, posW);
                cv::Mat Rcwk = pKF_k->Tcw.rowRange(0, 3).colRange(0, 3);
                cv::Mat xyzinfok = Rcwk * xyzinfoW * Rcwk.t();
                pKF_k->setViewMP(posk, mObservations[pKF_k], toMatrix3d(xyzinfok));
            }
        }
    }

    // If no good parallax after more than 6 KFs, abandon this MP
    if (pKF->mIdKF - pKF0->mIdKF >= 6 && !mbGoodParallax) {
        setNull();
    }
}

float MapPoint::getInvLevelSigma2(const PtrKeyFrame& pKF)
{
    int index = mObservations[pKF];
    return pKF->mvInvLevelSigma2[index];
}

int MapPoint::getOctave(const PtrKeyFrame pKF)
{
    int index = mObservations[pKF];
    return pKF->mvKeyPoints[index].octave;
}

void MapPoint::setPos(const cv::Point3f& pt3f)
{
    locker lock(mMutexPos);
    mPos = pt3f;
}

bool MapPoint::acceptNewObserve(Point3f posKF, const KeyPoint kp)
{
    float dist = cv::norm(posKF);
    float cosAngle = cv::norm(posKF.dot(mNormalVector)) / (dist * cv::norm(mNormalVector));
    bool c1 = std::abs(mMainKF->mvKeyPoints[mObservations[mMainKF]].octave - kp.octave) <= 2;
    bool c2 = cosAngle >= 0.866f;  // no larger than 30 degrees
    bool c3 = dist >= mMinDist && dist <= mMaxDist;
    return c1 && c2 && c3;
}

std::set<PtrKeyFrame> MapPoint::getObservations()
{
    locker lock(mMutexObs);
    std::set<PtrKeyFrame> pKFs;
    for (auto i = mObservations.begin(), iend = mObservations.end(); i != iend; i++) {
        PtrKeyFrame pKF = i->first;
        if (!pKF)
            continue;
        if (pKF->isNull())
            continue;
        pKFs.insert(i->first);
    }
    return pKFs;
}

Point2f MapPoint::getMainMeasure()
{
    return mMainKF->mvKeyPoints[mObservations[mMainKF]].pt;
}


void MapPoint::updateMainKFandDescriptor()
{
    if (mbNull || mObservations.empty())
        return;

    vector<Mat> vDes;
    vector<PtrKeyFrame> vKFs;
    vDes.reserve(mObservations.size());
    vKFs.reserve(mObservations.size());
    for (auto i = mObservations.begin(), iend = mObservations.end(); i != iend; i++) {
        PtrKeyFrame pKF = i->first;
        if (!pKF)
            continue;

//        unique_lock<mutex> lck(pKF->mMutexDes);
        if (!pKF->isNull()) {
            vKFs.push_back(pKF);
            vDes.push_back(pKF->mDescriptors.row(i->second));
        }
    }

    if (vDes.empty())
        return;

    // Compute distances between them
    const int N = vDes.size();

    float Distances[N][N];

    for (int i = 0; i < N; i++) {
        Distances[i][i] = 0;
        for (int j = i + 1; j < N; j++) {
            int distij = ORBmatcher::DescriptorDistance(vDes[i], vDes[j]);
            Distances[i][j] = distij;
            Distances[j][i] = distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    int bestMedian = INT_MAX;
    int bestIdx = 0;
    for (int i = 0; i < N; i++) {
        vector<int> vDists(Distances[i], Distances[i] + N);
        std::sort(vDists.begin(), vDists.end());
        int median = vDists[0.5 * (N - 1)];
        if (median < bestMedian) {
            bestMedian = median;
            bestIdx = i;
        }
    }

    mMainDescriptor = vDes[bestIdx].clone();

    if (mMainKF != nullptr && mMainKF->mIdKF == vKFs[bestIdx]->mIdKF)
        return;

    mMainKF = vKFs[bestIdx];
    int idx = mObservations[mMainKF];
    mMainOctave = mMainKF->mvKeyPoints[idx].octave;
    mLevelScaleFactor = mMainKF->mvScaleFactors[mMainOctave];
    float dist = cv::norm(mMainKF->mViewMPs[idx]);
    int nlevels = mMainKF->mnScaleLevels;

    mMaxDist = dist * mLevelScaleFactor;
    mMinDist = mMaxDist / mMainKF->mvScaleFactors[nlevels - 1];
}

void MapPoint::updateMeasureInKFs()
{
    locker lock(mMutexObs);
    for (auto it = mObservations.begin(), iend = mObservations.end(); it != iend; it++) {
        PtrKeyFrame pKF = it->first;
        if (!pKF)
            continue;
        if (pKF->isNull())
            continue;

        Mat Tcw = pKF->getPose();
        Point3f posKF = cvu::se3map(Tcw, mPos);
        pKF->mViewMPs[it->second] = posKF;
    }
}

int MapPoint::countObservation()
{
    locker lock(mMutexObs);
    return mObservations.size();
}


// This MP would be replaced and abandoned later
void MapPoint::mergedInto(const shared_ptr<MapPoint>& pMP)
{
    locker lock(mMutexObs);
    for (auto it = mObservations.begin(), iend = mObservations.end(); it != iend; it++) {
        PtrKeyFrame pKF = it->first;
        if (!pKF)
            continue;
        if (pKF->isNull())
            continue;
        int idx = it->second;
        pKF->setObservation(pMP, idx);
        pMP->addObservation(pKF, idx);
    }
}

int MapPoint::getFtrIdx(PtrKeyFrame pKF)
{
    locker lock(mMutexObs);
    return mObservations[pKF];
}

void MapPoint::setGoodPrl(bool value)
{
    mbGoodParallax = value;
}


bool MapPoint::hasObservation(const PtrKeyFrame& pKF)
{
    locker lock(mMutexObs);
    return (mObservations.find(pKF) != mObservations.end());
}

}  // namespace se2lam
