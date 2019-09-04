/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG
* (github.com/hbtang)
*/

/*** This file is part of ORB-SLAM.
* It is based on the file orb.cpp from the OpenCV library (see BSD license
* below)
*
* Copyright (C) 2014 Raúl Mur-Artal <raulmur at unizar dot es> (University of
* Zaragoza)
* For more information see <http://webdiis.unizar.es/~raulmur/orbslam/>
*
* ORB-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#include "ORBmatcher.h"

#include <limits.h>

//#include <stdint-gcc.h>
#include <stdint.h>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <ros/ros.h>

#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include "cvutil.h"


namespace se2lam
{
using namespace cv;
using namespace std;

const int ORBmatcher::TH_HIGH = 100;
const int ORBmatcher::TH_LOW = 60;  // 75
const int ORBmatcher::HISTO_LENGTH = 30;


//获取匹配线段的起始匹配点
void getMatcheLinesEndPoints(const Frame frame1, const Frame frame2,
                             std::vector<line_s_e>& matchesLine1_S_E, int linelable1,
                             int linelable2, int pl1, int pl2)
{
    double k1 = frame1.lineFeature[linelable1].k;
    Point2f s = frame1.keyPointsUn[pl1].pt;
    Point2f e = frame1.keyPointsUn[pl1].pt;
    Point2f ms = frame2.keyPointsUn[pl2].pt;
    Point2f me = frame2.keyPointsUn[pl2].pt;
    if (abs(k1 < 1)) {
        if (abs(matchesLine1_S_E[linelable1].star_p.x - 0) < 0.01) {
            matchesLine1_S_E[linelable1].star_p = s;
            matchesLine1_S_E[linelable1].end_p = e;
            matchesLine1_S_E[linelable1].match_star = ms;
            matchesLine1_S_E[linelable1].match_end = me;
        } else {
            Point2f ss = matchesLine1_S_E[linelable1].star_p;
            Point2f ee = matchesLine1_S_E[linelable1].end_p;
            if (s.x < ss.x) {
                matchesLine1_S_E[linelable1].star_p = s;
                matchesLine1_S_E[linelable1].match_star = ms;
            } else if (e.x > ee.x) {
                matchesLine1_S_E[linelable1].end_p = e;
                matchesLine1_S_E[linelable1].match_end = me;
            }
        }
    } else {
        if (abs(matchesLine1_S_E[linelable1].star_p.x - 0) < 0.01) {
            matchesLine1_S_E[linelable1].star_p = s;
            matchesLine1_S_E[linelable1].end_p = e;
            matchesLine1_S_E[linelable1].match_star = ms;
            matchesLine1_S_E[linelable1].match_end = me;
        } else {
            Point2f ss = matchesLine1_S_E[linelable1].star_p;
            Point2f ee = matchesLine1_S_E[linelable1].end_p;
            if (s.y < ss.y) {
                matchesLine1_S_E[linelable1].star_p = s;
                matchesLine1_S_E[linelable1].match_star = ms;
            } else if (e.y > ee.y) {
                matchesLine1_S_E[linelable1].end_p = e;
                matchesLine1_S_E[linelable1].match_end = me;
            }
        }
    }
}

void GetRotatePoints(Mat img, cv::Point2f origenPoint, cv::Point2f& rotatePoint, double angle)
{
    float x1 = origenPoint.x;
    float y1 = img.rows - origenPoint.y;
    float x2 = img.cols / 2;
    float y2 = img.rows - img.rows / 2 + 20;
    rotatePoint.x = cvRound((x1 - x2) * cos(angle) - (y1 - y2) * sin(angle) + x2);
    rotatePoint.y = cvRound((x1 - x2) * sin(angle) + (y1 - y2) * cos(angle) + y2);
    rotatePoint.y = img.rows - rotatePoint.y;
    // cout<<angle<<"  "<<rotatePoint<<endl;
}


/**
 * Constructor
 * @param nnratio  ratio of the best and the second score
 * @param checkOri check orientation
 */
ORBmatcher::ORBmatcher(float nnratio, bool checkOri, bool withline)
    : mfNNratio(nnratio), mbCheckOrientation(checkOri), mbWithLineFeature(withline)
{
}

float ORBmatcher::RadiusByViewingCos(const float& viewCos)
{
    if (viewCos > 0.998)
        return 2.5;
    else
        return 4.0;
}

//! 取出直方图中最高的三个index，这里主要是在特征匹配时保证旋转连续性
void ORBmatcher::ComputeThreeMaxima(vector<int>* histo, const int L, int& ind1, int& ind2,
                                    int& ind3)
{
    int max1 = 0;
    int max2 = 0;
    int max3 = 0;

    for (int i = 0; i < L; i++) {
        const int s = histo[i].size();
        if (s > max1) {
            max3 = max2;
            max2 = max1;
            max1 = s;
            ind3 = ind2;
            ind2 = ind1;
            ind1 = i;
        } else if (s > max2) {
            max3 = max2;
            max2 = s;
            ind3 = ind2;
            ind2 = i;
        } else if (s > max3) {
            max3 = s;
            ind3 = i;
        }
    }

    // 次高或者第三高的直方图过低, 说明旋转一致性不能够很好的保持
    if (max2 < 0.1f * (float)max1) {
        ind2 = -1;
        ind3 = -1;
    } else if (max3 < 0.1f * (float)max1) {
        ind3 = -1;
    }
}


// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int ORBmatcher::DescriptorDistance(const cv::Mat& a, const cv::Mat& b)
{
    const int* pa = a.ptr<int32_t>();
    const int* pb = b.ptr<int32_t>();

    int dist = 0;

    for (int i = 0; i < 8; i++, pa++, pb++) {
        unsigned int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

/**
 * @brief 通过词包，对关键帧的特征点进行跟踪，该函数用于闭环检测时两个关键帧间的特征点匹配
 *
 * 通过bow对pKF和F中的特征点进行快速匹配（不属于同一node的特征点直接跳过匹配）\n
 * 对属于同一node的特征点通过描述子距离进行匹配 \n
 * 根据匹配，更新vpMatches12 \n
 * 通过距离阈值、比例阈值和角度投票进行剔除误匹配
 * @param  pKF1         KeyFrame1
 * @param  pKF2         KeyFrame2
 * @param  vpMatches12  pKF2中与pKF1匹配的MapPoint，null表示没有匹配
 * @return              成功匹配的数量
 */
int ORBmatcher::SearchByBoW(PtrKeyFrame pKF1, PtrKeyFrame pKF2, map<int, int>& mapMatches12,
                            bool bIfMPOnly)
{
    mapMatches12.clear();

    if (pKF1 == NULL || pKF1->isNull() || pKF2 == NULL || pKF2->isNull()) {
        return 0;
    }

    vector<cv::KeyPoint> vKeysUn1 = pKF1->keyPointsUn;
    DBoW2::FeatureVector vFeatVec1 = pKF1->GetFeatureVector();
    vector<PtrMapPoint> vpMapPoints1 = pKF1->GetMapPointMatches();
    cv::Mat Descriptors1 = pKF1->descriptors;

    vector<cv::KeyPoint> vKeysUn2 = pKF2->keyPointsUn;
    DBoW2::FeatureVector vFeatVec2 = pKF2->GetFeatureVector();
    vector<PtrMapPoint> vpMapPoints2 = pKF2->GetMapPointMatches();
    cv::Mat Descriptors2 = pKF2->descriptors;


    vector<bool> vbMatched2(vpMapPoints2.size(), false);

    vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f / HISTO_LENGTH;

    int nmatches = 0;

    //! 将属于同一节点(特定层)的ORB特征进行匹配
    DBoW2::FeatureVector::iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::iterator f2end = vFeatVec2.end();

    while (f1it != f1end && f2it != f2end) {
        //! 步骤1：分别取出属于同一node的ORB特征点
        //! 只有属于同一node，才有可能是匹配点
        if (f1it->first == f2it->first) {
            //! 步骤2：遍历KF中属于该node的特征点
            for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++) {
                size_t idx1 = f1it->second[i1];

                PtrMapPoint pMP1 = vpMapPoints1[idx1];

                if (bIfMPOnly) {
                    if (!pMP1)
                        continue;
                    if (pMP1->isNull())
                        continue;
                }

                cv::Mat d1 = Descriptors1.row(idx1);

                int bestDist1 = INT_MAX;
                int bestIdx2 = -1;
                int bestDist2 = INT_MAX;

                //! 步骤3：遍历F中属于该node的特征点，找到了最佳匹配点
                for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2; i2++) {
                    size_t idx2 = f2it->second[i2];

                    PtrMapPoint pMP2 = vpMapPoints2[idx2];

                    if (bIfMPOnly) {
                        if (!pMP2)
                            continue;
                        if (pMP2->isNull())
                            continue;
                    }

                    if (vbMatched2[idx2])
                        continue;

                    cv::Mat d2 = Descriptors2.row(idx2);    // 取出该特征对应的描述子
                    int dist = DescriptorDistance(d1, d2);  // 求描述子的距离

                    if (dist < bestDist1) {
                        bestDist2 = bestDist1;
                        bestDist1 = dist;
                        bestIdx2 = idx2;
                    } else if (dist < bestDist2) {
                        bestDist2 = dist;
                    }
                }

                //! 步骤4：根据阈值和角度投票剔除误匹配
                if (bestDist1 < TH_LOW) {
                    //! trick!
                    //! 最佳匹配比次佳匹配明显要好，那么最佳匹配才真正靠谱
                    if (static_cast<float>(bestDist1) < mfNNratio * static_cast<float>(bestDist2)) {
                        //! 步骤5：更新特征点的MapPoint
                        mapMatches12[idx1] = bestIdx2;
                        vbMatched2[bestIdx2] = true;

                        if (mbCheckOrientation) {
                            //! trick!
                            //! angle：每个特征点在提取描述子时的旋转主方向角度，如果图像旋转了，这个角度将发生改变
                            //! 所有的特征点的角度变化应该是一致的，通过直方图统计得到最准确的角度变化值
                            float rot = vKeysUn1[idx1].angle -
                                        vKeysUn2[bestIdx2].angle;  // 该特征点的角度变化值
                            if (rot < 0.0)
                                rot += 360.0f;
                            int bin = round(rot * factor);  // 将rot分配到bin组
                            if (bin == HISTO_LENGTH)
                                bin = 0;
                            assert(bin >= 0 && bin < HISTO_LENGTH);
                            rotHist[bin].push_back(idx1);
                        }
                        nmatches++;
                    }
                }
            }

            f1it++;
            f2it++;
        } else if (f1it->first < f2it->first) {
            f1it = vFeatVec1.lower_bound(f2it->first);
        } else {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    //! 根据方向剔除误匹配的点
    if (mbCheckOrientation) {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        //! 计算rotHist中最大的三个的index
        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; i++) {
            //! 如果特征点的旋转角度变化量属于这三个组，则保留
            if (i == ind1 || i == ind2 || i == ind3)
                continue;
            //! 将除了ind1 ind2 ind3以外的匹配点去掉
            for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
                mapMatches12.erase(rotHist[i][j]);
                nmatches--;
            }
        }
    }

    return nmatches;
}


/**
 * @brief 根据运动模型投影，对上一帧的特征点进行跟踪,
 *
 * 上一帧中包含了MapPoints，对这些MapPoints进行tracking，由此增加当前帧的MapPoints \n
 * 1. 将上一帧的MapPoints投影到当前帧(根据速度模型可以估计当前帧的Tcw)
 * 2. 在投影点附近根据描述子距离选取匹配，以及最终的方向投票机制进行剔除
 * @param  CurrentFrame 当前帧
 * @param  LastFrame    上一帧
 * @param  th           阈值
 * @return              成功匹配的数量
 * @see SearchByBoW()
 */
int ORBmatcher::SearchByProjection(Frame& CurrentFrame, KeyFrame& LastKF, const float th)
{
    int nmatches = 0;

    // Rotation Histogram (to check rotation consistency) 旋转方向的直方图，用于检查旋转连续性
    vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; i++)
        rotHist[i].reserve(500);
    const float factor = HISTO_LENGTH / 360.0f;

    const cv::Mat Rcw = CurrentFrame.Tcw.rowRange(0, 3).colRange(0, 3);
    const cv::Mat tcw = CurrentFrame.Tcw.rowRange(0, 3).col(3);

    const cv::Mat twc = -Rcw.t() * tcw;  // twc(w)

    const cv::Mat Rlw = LastKF.Tcw.rowRange(0, 3).colRange(0, 3);
    const cv::Mat tlw = LastKF.Tcw.rowRange(0, 3).col(3);  // tlw(l)

    // vector from LastFrame to CurrentFrame expressed in LastFrame
    const cv::Mat tlc = Rlw * twc + tlw;  // Rlw*twc(w) = twc(l), twc(l) + tlw(l) = tlc(l)

    // 判断前进还是后退
    const bool bForward = tlc.at<float>(2) > 0;  // 非单目情况，如果Z大于基线，则表示朝z前进
    const bool bBackward = -tlc.at<float>(2) > 0;  // 非单目情况，如果Z小于基线，则表示朝z前进

    // 对上一帧有效的MapPoints进行跟踪
    for (int i = 0; i < LastKF.N; i++) {
        PtrMapPoint pMP = LastKF.getObservation(i);

        if (pMP && !LastKF.mvbOutlier[i]) {
            // Project
            cv::Mat x3Dw = Mat(pMP->getPos());
            cv::Mat x3Dc = Rcw * x3Dw + tcw;

            const float xc = x3Dc.at<float>(0);
            const float yc = x3Dc.at<float>(1);
            const float invzc = 1.0 / x3Dc.at<float>(2);

            if (invzc < 0)
                continue;

            float u = Config::fxCam * xc * invzc + Config::cxCam;
            float v = Config::fyCam * yc * invzc + Config::cyCam;

            if (u < CurrentFrame.minXUn || u > CurrentFrame.maxXUn)
                continue;
            if (v < CurrentFrame.minYUn || v > CurrentFrame.maxYUn)
                continue;

            int nLastOctave = LastKF.keyPoints[i].octave;

            // Search in a window. Size depends on scale
            float radius = th * CurrentFrame.mvScaleFactors[nLastOctave];  // 尺度越大，搜索范围越大

            vector<size_t> vIndices2;

            // NOTE 尺度越大,图像越小
            // 以下可以这么理解，例如一个有一定面积的圆点，在某个尺度n下它是一个特征点
            // 当前进时，圆点的面积增大，在某个尺度m下它是一个特征点，由于面积增大，则需要在更高的尺度下才能检测出来
            // 因此m>=n，对应前进的情况，nCurOctave>=nLastOctave。后退的情况可以类推
            if (bForward)  // 前进,则上一帧兴趣点在所在的尺度nLastOctave<=nCurOctave
                vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nLastOctave);
            else if (bBackward)  // 后退,则上一帧兴趣点在所在的尺度0<=nCurOctave<=nLastOctave
                vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, 0, nLastOctave);
            else  // 在[nLastOctave-1, nLastOctave+1]中搜索
                vIndices2 =
                    CurrentFrame.GetFeaturesInArea(u, v, radius, nLastOctave - 1, nLastOctave + 1);

            if (vIndices2.empty()) {
                std::cerr << "Empty in GetFeaturesInArea()! " << std::endl;
                continue;
            }

            const cv::Mat dMP = pMP->mMainDescriptor;

            int bestDist = 256;
            int bestIdx2 = -1;

            // 遍历满足条件的特征点
            for (vector<size_t>::const_iterator vit = vIndices2.begin(), vend = vIndices2.end();
                 vit != vend; vit++) {
                // 如果该特征点已经有对应的MapPoint了,则退出该次循环
                const size_t i2 = *vit;
                if (CurrentFrame.mvpMapPoints[i2])
                    if (CurrentFrame.mvpMapPoints[i2]->countObservation() > 0)
                        continue;

                const cv::Mat& d = CurrentFrame.descriptors.row(i2);

                const int dist = DescriptorDistance(dMP, d);

                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdx2 = i2;
                }
            }

            // 详见SearchByBoW(KeyFrame* pKF,Frame &F, vector<MapPoint*>
            // &vpMapPointMatches)函数步骤4
            if (bestDist <= TH_HIGH) {
                CurrentFrame.mvpMapPoints[bestIdx2] = pMP;  // 为当前帧添加MapPoint
                nmatches++;

                if (mbCheckOrientation) {
                    float rot =
                        LastKF.keyPointsUn[i].angle - CurrentFrame.keyPointsUn[bestIdx2].angle;
                    if (rot < 0.0)
                        rot += 360.0f;
                    int bin = round(rot * factor);
                    if (bin == HISTO_LENGTH)
                        bin = 0;
                    assert(bin >= 0 && bin < HISTO_LENGTH);
                    rotHist[bin].push_back(bestIdx2);
                }
            }
        }
    }

    // Apply rotation consistency
    if (mbCheckOrientation) {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; i++) {
            if (i != ind1 && i != ind2 && i != ind3) {
                for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
                    CurrentFrame.mvpMapPoints[rotHist[i][j]] = static_cast<PtrMapPoint>(NULL);
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}


/**
 * @brief ORBmatcher::MatchByWindow
 * 先获得cell里的粗匹配候选，再从候选的KF中根据描述子计算最小和次小距离，剔除错匹配
 *
 * @param frame1        参考帧
 * @param frame2        当前帧
 * @param vbPrevMatched 参考帧特征点[update]
 * @param winSize       cell尺寸
 * @param vnMatches12   匹配情况[output]
 * @param levelOffset   最大金字塔层差
 * @param minLevel      金字塔最小层数
 * @param maxLevel      金字塔最大层数
 * @return              返回匹配点的总数
 */
int ORBmatcher::MatchByWindow(const Frame& frame1, const Frame& frame2,
                              vector<Point2f>& vbPrevMatched, const int winSize,
                              vector<int>& vnMatches12, const int levelOffset, const int minLevel,
                              const int maxLevel)
{
    int nmatches = 0;
    vnMatches12 = vector<int>(frame1.N, -1);

    vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; i++)
        rotHist[i].reserve(500);
    const float factor = 1.f / (float)HISTO_LENGTH;

    vector<int> vMatchesDistance(frame2.N, INT_MAX);
    vector<int> vnMatches21(frame2.N, -1);

    //! 遍历参考帧特征点, 序号i1
    for (int i1 = 0, iend1 = frame1.N; i1 < iend1; i1++) {
        KeyPoint kp1 = frame1.keyPointsUn[i1];
        int level1 = kp1.octave;
        if (level1 > maxLevel || level1 < minLevel)
            continue;
        int minLevel2 = level1 - levelOffset > 0 ? level1 - levelOffset : 0;
        //! 1.对F1中的每个KP先获得F2中一个cell里的粗匹配候选, cell的边长为2*winsize
        vector<size_t> vIndices2 = frame2.GetFeaturesInArea(
            vbPrevMatched[i1].x, vbPrevMatched[i1].y, winSize, minLevel2, level1 + levelOffset);
        if (vIndices2.empty())
            continue;

        cv::Mat d1 = frame1.descriptors.row(i1);

        int bestDist = INT_MAX;
        int bestDist2 = INT_MAX;
        int bestIdx2 = -1;

        //! 2.从F2的KP候选里计算最小和次小汉明距离, 序号i2
        for (auto vit = vIndices2.begin(), vend = vIndices2.end(); vit != vend; vit++) {
            size_t i2 = *vit;

            cv::Mat d2 = frame2.descriptors.row(i2);

            int dist = DescriptorDistance(d1, d2);

            if (vMatchesDistance[i2] <= dist)
                continue;

            if (dist < bestDist) {
                bestDist2 = bestDist;
                bestDist = dist;
                bestIdx2 = i2;
            } else if (dist < bestDist2) {
                bestDist2 = dist;
            }
        }

        //! 3.最小距离小于TH_LOW且小于mfNNratio倍次小距离，则将此KP与F1中对应的KP视为匹配对
        if (bestDist <= TH_LOW) {
            if (bestDist < (float)bestDist2 * mfNNratio) {
                // 如果出现F1中多个点匹配到F2中同一个点的情况, 则取消之前的匹配用新的匹配
                if (vnMatches21[bestIdx2] >= 0) {
                    vnMatches12[vnMatches21[bestIdx2]] = -1;
                    nmatches--;
                }
                vnMatches12[i1] = bestIdx2;
                vnMatches21[bestIdx2] = i1;
                vMatchesDistance[bestIdx2] = bestDist;
                nmatches++;

                //! for orientation check. 4.统计匹配点对的角度差的直方图, 待下一步做旋转检验
                float rot = frame1.keyPointsUn[i1].angle - frame2.keyPointsUn[bestIdx2].angle;
                if (rot < 0.0)
                    rot += 360.f;
                int bin = round(rot * factor);
                if (bin == HISTO_LENGTH)
                    bin = 0;
                rotHist[bin].push_back(i1);
            }
        }
    }

    //! orientation check. 5.进行旋转一致性检验, 匹配点对角度差不在直方图最大的三个方向上, 则视为误匹配剔除
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; i++) {
            if (i == ind1 || i == ind2 || i == ind3)
                continue;
            for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
                int idx1 = rotHist[i][j];
                if (vnMatches12[idx1] >= 0) {
                    vnMatches12[idx1] = -1;
                    nmatches--;
                }
            }
        }
    }

    //! update prev matched
    for (size_t i1 = 0, iend1 = vnMatches12.size(); i1 < iend1; i1++)
        if (vnMatches12[i1] >= 0)
            vbPrevMatched[i1] = frame2.keyPointsUn[vnMatches12[i1]].pt;

    return nmatches;
}

/**
 * @brief ORBmatcher::MatchByWindow 增加运动先验版本
 * 先获得cell里的粗匹配候选，再从候选的KF中根据描述子计算最小和次小距离，剔除错匹配.
 * 只在同一层内搜索
 *
 * @param frame1        参考帧F1
 * @param frame2        当前帧F2
 * @param vbPrevMatched 参考帧F1特征点的位置[update]
 * @param offset        运动先验的偏置
 * @param vnMatches12   匹配情况[output]
 * @param winSize       cell尺寸
 * @return              返回匹配点的总数
 */
int ORBmatcher::MatchByWindow(const Frame& frame1, const Frame& frame2,
                              std::vector<cv::Point2f>& vbPrevMatched, cv::Point2f& offset,
                              std::vector<int>& vnMatches12, const int winSize)
{
    int nmatches = 0;
    vnMatches12 = vector<int>(frame1.N, -1);

    vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; i++)
        rotHist[i].reserve(500);
    const float factor = 1.f / (float)HISTO_LENGTH;

    vector<int> vMatchesDistance(frame2.N, INT_MAX);
    vector<int> vnMatches21(frame2.N, -1);

    //! 遍历参考帧特征点, 序号i1
    for (int i1 = 0, iend1 = frame1.N; i1 < iend1; i1++) {
        KeyPoint kp1 = frame1.keyPointsUn[i1];
        int level = kp1.octave;
        //! 1.对F1中的每个KP先获得F2中一个cell里的粗匹配候选, cell的边长为2*winsize
        vector<size_t> vIndices2 = frame2.GetFeaturesInArea(
            vbPrevMatched[i1].x + offset.x, vbPrevMatched[i1].y + offset.y, winSize, level, level);
        if (vIndices2.empty())
            continue;

        cv::Mat d1 = frame1.descriptors.row(i1);

        int bestDist = INT_MAX;
        int bestDist2 = INT_MAX;
        int bestIdx2 = -1;

        //! 2.从F2的KP候选里计算最小和次小汉明距离, 序号i2
        for (auto vit = vIndices2.begin(), vend = vIndices2.end(); vit != vend; vit++) {
            size_t i2 = *vit;

            cv::Mat d2 = frame2.descriptors.row(i2);

            int dist = DescriptorDistance(d1, d2);

            if (vMatchesDistance[i2] <= dist)
                continue;

            if (dist < bestDist) {
                bestDist2 = bestDist;
                bestDist = dist;
                bestIdx2 = i2;
            } else if (dist < bestDist2) {
                bestDist2 = dist;
            }
        }

        //! 3.最小距离小于TH_LOW且小于mfNNratio倍次小距离，则将此KP与F1中对应的KP视为匹配对
        if (bestDist <= TH_LOW && bestDist < (float)bestDist2 * mfNNratio) {
            // 如果出现F1中多个点匹配到F2中同一个点的情况, 则取消之前的匹配用新的匹配
            //! NOTE 已改成保留汉明距离最小的匹配
            if (vnMatches21[bestIdx2] >= 0) {
                if (bestDist < vMatchesDistance[bestIdx2]) {
                    vnMatches12[vnMatches21[bestIdx2]] = -1;
                    nmatches--;
                } else {
                    vnMatches21[bestIdx2] = -1;
                    continue;
                }
            }
            vnMatches12[i1] = bestIdx2;
            vnMatches21[bestIdx2] = i1;
            vMatchesDistance[bestIdx2] = bestDist;
            nmatches++;

            //! for orientation check. 4.统计匹配点对的角度差的直方图, 待下一步做旋转检验
            float rot = frame1.keyPointsUn[i1].angle - frame2.keyPointsUn[bestIdx2].angle;
            if (rot < 0.0)
                rot += 360.f;
            int bin = round(rot * factor);
            if (bin == HISTO_LENGTH)
                bin = 0;
            rotHist[bin].push_back(i1);
        }
    }

    //! 5.进行旋转一致性检验, 匹配点对角度差不在直方图最大的三个方向上, 则视为误匹配剔除. orientation check
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; i++) {
            if (i == ind1 || i == ind2 || i == ind3)
                continue;
            for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
                int idx1 = rotHist[i][j];
                if (vnMatches12[idx1] >= 0) {
                    vnMatches12[idx1] = -1;
                    nmatches--;
                }
            }
        }
    }

    //! update prev matched. 更新匹配KP的平均偏移offset
    Point2f newOffset(0.f, 0.f);
    for (size_t i1 = 0, iend1 = vnMatches12.size(); i1 < iend1; i1++)
        if (vnMatches12[i1] >= 0) {
            vbPrevMatched[i1] = frame2.keyPointsUn[vnMatches12[i1]].pt;

            Point2f p1 = frame1.keyPointsUn[i1].pt;
            Point2f p2 = frame2.keyPointsUn[vnMatches12[i1]].pt;
            newOffset.x += p2.x - p1.x;
            newOffset.y += p2.y - p1.y;
        }

    offset.x = newOffset.x / nmatches;
    offset.y = newOffset.y / nmatches;

    return nmatches;
}


/**
 * @brief ORBmatcher::MatchByProjection 通过投影，对Local MapPoint进行跟踪
 *   在LocalMapper的addNewKF()里调用
 * @param pNewKF        当前关键帧
 * @param localMPs      局部地图点
 * @param winSize       搜索半径 15
 * @param levelOffset   金字塔层搜索相对范围 2
 * @param vMatchesIdxMP 匹配上的MP索引[output]
 * @return              返回匹配点数
 */
int ORBmatcher::MatchByProjection(PtrKeyFrame& pNewKF, std::vector<PtrMapPoint>& localMPs,
                                  const int winSize, const int levelOffset,
                                  std::vector<int>& vMatchesIdxMP)
{
    int nmatches = 0;

    vMatchesIdxMP = vector<int>(pNewKF->N, -1);
    vector<int> vMatchesDistance(pNewKF->N, INT_MAX);

    for (int i = 0, iend = localMPs.size(); i < iend; i++) {
        PtrMapPoint pMP = localMPs[i];
        if (pMP->isNull() || !pMP->isGoodPrl())
            continue;
        if (pNewKF->hasObservation(pMP))
            continue;
        // Point2f predictUV = scv::prjcPt2Cam(Config::Kcam, pNewKF->Tcw,
        // pMP->getPos());
        Point2f predictUV = cvu::camprjc(Config::Kcam, cvu::se3map(pNewKF->Tcw, pMP->getPos()));
        if (!pNewKF->inImgBound(predictUV))
            continue;
        const int predictLevel = pMP->mMainOctave;
        const int levelWinSize = predictLevel * winSize;
        const int minLevel = predictLevel > levelOffset ? predictLevel - levelOffset : 0;

        // 通过投影点(投影到当前帧,见isInFrustum())以及搜索窗口和预测的尺度进行搜索,找出附近的兴趣点
        vector<size_t> vNearIndices = pNewKF->GetFeaturesInArea(
            predictUV.x, predictUV.y, levelWinSize, minLevel, predictLevel + levelOffset);
        if (vNearIndices.empty())
            continue;

        int bestDist = INT_MAX;
        int bestLevel = -1;
        int bestDist2 = INT_MAX;
        int bestLevel2 = -1;
        int bestIdx = -1;

        // Get best and second matches with near keypoints
        for (auto it = vNearIndices.begin(), iend = vNearIndices.end(); it != iend; it++) {
            int idx = *it;
            if (pNewKF->hasObservation(idx))
                continue;
            cv::Mat d = pNewKF->descriptors.row(idx);

            const int dist = DescriptorDistance(pMP->mMainDescriptor, d);

            if (vMatchesDistance[idx] <= dist)
                continue;

            if (dist < bestDist) {
                bestDist2 = bestDist;
                bestDist = dist;
                bestLevel2 = bestLevel;
                bestLevel = pNewKF->keyPoints[idx].octave;
                bestIdx = idx;
            } else if (dist < bestDist2) {
                bestLevel2 = pNewKF->keyPoints[idx].octave;
                bestDist2 = dist;
            }
        }

        // Apply ratio to second match (only if best and second are in the same
        // scale level)
        if (bestDist <= TH_HIGH) {
            if (bestLevel == bestLevel2 && bestDist > mfNNratio * bestDist2)
                continue;
            if (vMatchesIdxMP[bestIdx] >= 0) {
                vMatchesIdxMP[bestIdx] = -1;
                nmatches--;
            }
            vMatchesIdxMP[bestIdx] = i;
            vMatchesDistance[bestIdx] = bestDist;
            nmatches++;
        }
    }

    return nmatches;
}


int ORBmatcher::MatchByPointAndLine(const Frame& frame1, Frame& frame2,
                                    vector<Point2f>& vbPrevMatched, const int winSize,
                                    vector<int>& vnMatches12, vector<int>& vMatchesDistance,
                                    double angle, const int levelOffset, const int minLevel,
                                    const int maxLevel)
{
    //线匹配容器初始化
    int m = frame1.lineIncludePoints.size();
    int n = frame2.lineIncludePoints.size();
    std::vector<std::vector<int>> lineToLineLable(m);  //点线匹配关系
    for (int i = 0; i < m; i++)
        lineToLineLable[i].resize(n, 0);
    std::vector<int> matcherLineLable(m, -1);  //线线匹配关系

    //匹配线所包含的匹配点的lable
    std::vector<std::vector<int>> matchesLine1IncludePoints(m);  //点线匹配关系
    std::vector<std::vector<int>> matchesLine2IncludePoints(n);  //点线匹配关系

    //匹配线的匹配点的起点和终点
    std::vector<line_s_e> matchesLine1_S_E(m);
    std::vector<line_s_e> matchesLine2_S_E(n);

    int nmatches = 0;
    vnMatches12 = vector<int>(frame1.N, -1);

    vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; i++)
        rotHist[i].reserve(500);
    const float factor = 1.f / (float)HISTO_LENGTH;

    // vector<int> vMatchesDistance(frame2.N, INT_MAX);

    // 7.17日修改

    vector<int> vnMatches21(frame2.N, -1);

    int minDist = INT_MAX;
    for (int i1 = 0, iend1 = frame1.N; i1 < iend1; i1++) {
        KeyPoint kp1 = frame1.keyPointsUn[i1];
        int level1 = kp1.octave;
        if (level1 > maxLevel || level1 < minLevel)
            continue;
        int minLevel2 = level1 - levelOffset > 0 ? level1 - levelOffset : 0;

        cv::Point2f rotatePoint;
        // GetRotatePoints(frame1.img,vbPrevMatched[i1],rotatePoint,-angle);


        vector<size_t> vIndices2 = frame2.GetFeaturesInArea(
            vbPrevMatched[i1].x, vbPrevMatched[i1].y, winSize, minLevel2, level1 + levelOffset);
        if (vIndices2.empty())
            continue;

        cv::Mat d1 = frame1.descriptors.row(i1);


        int bestDist = INT_MAX;
        int bestDist2 = INT_MAX;
        int bestIdx2 = -1;

        for (auto vit = vIndices2.begin(), vend = vIndices2.end(); vit != vend; vit++) {
            size_t i2 = *vit;

            cv::Mat d2 = frame2.descriptors.row(i2);

            int dist = DescriptorDistance(d1, d2);

            if (vMatchesDistance[i2] <= dist)
                continue;

            if (dist < bestDist) {
                bestDist2 = bestDist;
                bestDist = dist;
                bestIdx2 = i2;
            } else if (dist < bestDist2) {
                bestDist2 = dist;
            }
        }

        if (bestDist <= TH_LOW) {
            if (bestDist < (float)bestDist2 * mfNNratio) {  // mfNNratio
                if (vnMatches21[bestIdx2] >= 0) {
                    vnMatches12[vnMatches21[bestIdx2]] = -1;
                    nmatches--;
                }
                vnMatches12[i1] = bestIdx2;
                vnMatches21[bestIdx2] = i1;
                vMatchesDistance[bestIdx2] = bestDist;
                if (bestDist < minDist)
                    minDist = bestDist;
                nmatches++;

                // for orientation check
                float rot = frame1.keyPointsUn[i1].angle - frame2.keyPointsUn[bestIdx2].angle;
                if (rot < 0.0)
                    rot += 360.f;
                int bin = round(rot * factor);
                if (bin == HISTO_LENGTH)
                    bin = 0;
                rotHist[bin].push_back(i1);
            }
        }
    }


    // orientation check
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; i++) {
            if (i == ind1 || i == ind2 || i == ind3)
                continue;
            for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
                int idx1 = rotHist[i][j];
                if (vnMatches12[idx1] >= 0) {
                    vnMatches12[idx1] = -1;
                    nmatches--;
                }
            }
        }
    }


    for (size_t i1 = 0, iend1 = vnMatches12.size(); i1 < iend1; i1++)
        if (vnMatches12[i1] >= 0) {
            //线匹配lable对应关系
            //图1中每条线对应的图2中匹配线（1对多）
            int line1lable, line2lable, linenuber;
            line1lable = frame1.pointAndLineLable[i1].lineLable;
            line2lable = frame2.pointAndLineLable[vnMatches12[i1]].lineLable;
            //不在线上的点不考虑
            if (line1lable == -1 || line2lable == -1)
                continue;
            linenuber = lineToLineLable[line1lable][line2lable];
            lineToLineLable[line1lable][line2lable] = linenuber + 1;

            //匹配线所包含的匹配点的lable
            matchesLine1IncludePoints[line1lable].push_back(i1);
            matchesLine2IncludePoints[line2lable].push_back(vnMatches12[i1]);
        }

    //线与线的匹配关系（一对一）
    for (size_t i = 0; i < lineToLineLable.size(); i++) {
        std::vector<int>::iterator biggest =
            std::max_element(std::begin(lineToLineLable[i]), std::end(lineToLineLable[i]));
        if (*biggest > 2) {
            int position = std::distance(std::begin(lineToLineLable[i]), biggest);
            matcherLineLable[i] = position;

            for (size_t j = 0; j < matchesLine1IncludePoints[i].size(); j++) {
                int pl1 = matchesLine1IncludePoints[i][j];
                int pl2 = vnMatches12[pl1];
                if (frame2.pointAndLineLable[pl2].lineLable == position) {
                    //获取匹配线断的起点和终点
                    // getMatcheLinesEndPoints(frame1,frame2,matchesLine1_S_E,i,position,pl1, pl2);

                } else {
                    vnMatches12[pl1] = -1;
                }
            }

            cv::Point2f start1, end1, start2, end2;
            //绘制线与线的匹配效果
            //               start1 = matchesLine1_S_E[i].star_p; //直线起点
            //               end1 = matchesLine1_S_E[i].end_p;   //直线终点
            //               start2 = matchesLine1_S_E[i].match_star; //直线起点
            //               end2 = matchesLine1_S_E[i].match_end;   //直线终点

            //
            //                start1 = frame1.lineFeature[i].star; //直线起点
            //                end1 = frame1.lineFeature[i].end;   //直线终点
            //                start2 = frame2.lineFeature[position].star; //直线起点
            //                end2 = frame2.lineFeature[position].end;   //直线终点
            //
            //                Mat drawImg1 = frame1.img.clone();
            //                Mat drawImg2 = frame2.img.clone();
            //                cv::line(drawImg1, start1, end1, cv::Scalar(0, 0, 255),2);
            //                cv::line(drawImg2, start2, end2, cv::Scalar(0, 0, 255),2);
            //
            //                imshow("lineImag1", drawImg1);
            //                imshow("lineImag2", drawImg2);
            //                waitKey();

        } else {
            //不在线上的点筛选掉
            //                for(int j = 0; j < matchesLine1IncludePoints[i].size(); j++)
            //                {
            //                    int pl1 = matchesLine1IncludePoints[i][j];
            //                    vnMatches12[pl1] = -1;
            //                }
        }
    }

    // update prev matched
    for (size_t i1 = 0, iend1 = vnMatches12.size(); i1 < iend1; i1++)
        if (vnMatches12[i1] >= 0)
            vbPrevMatched[i1] = frame2.keyPointsUn[vnMatches12[i1]].pt;


    return nmatches;
}

}  // namespace ORB_SLAM
