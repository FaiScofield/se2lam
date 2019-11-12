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
const int ORBmatcher::TH_LOW = 75;
const int ORBmatcher::HISTO_LENGTH = 30;

/**
 * Constructor
 * @param nnratio  ratio of the best and the second score
 * @param checkOri check orientation
 */
ORBmatcher::ORBmatcher(float nnratio, bool checkOri)
    : mfNNratio(nnratio), mbCheckOrientation(checkOri)
{}

float ORBmatcher::RadiusByViewingCos(const float& viewCos)
{
    if (viewCos > 0.998)
        return 2.5;
    else
        return 4.0;
}

//! 取出直方图中最高的三个index，这里主要是在特征匹配时保证旋转连续性
void ORBmatcher::ComputeThreeMaxima(vector<int>* histo, const int L,
                                    int& ind1, int& ind2, int& ind3)
{
    int max1 = 0;
    int max2 = 0;
    int max3 = 0;

    for (int i = 0; i < L; ++i) {
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

    for (int i = 0; i < 8; ++i, pa++, pb++) {
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
 * @param  pKF1         KeyFrame current
 * @param  pKF2         KeyFrame loop
 * @param  vpMatches12  pKF2中与pKF1匹配的MapPoint，null表示没有匹配
 * @PARAM  bIfMPOnly    是否仅在有MP的KP中进行匹配, 默认为true, Localizer下为false
 * @return              成功匹配的数量
 */
int ORBmatcher::SearchByBoW(PtrKeyFrame pKF1, PtrKeyFrame pKF2, map<int, int>& mapMatches12,
                            bool bIfMPOnly)
{
    mapMatches12.clear();

    if (pKF1 == nullptr || pKF1->isNull() || pKF2 == nullptr || pKF2->isNull()) {
        return 0;
    }

    vector<cv::KeyPoint> vKeysUn1 = pKF1->mvKeyPoints;
    DBoW2::FeatureVector vFeatVec1 = pKF1->mFeatVec;
    vector<PtrMapPoint> vpMapPoints1 = pKF1->GetMapPointMatches();  // Localizer下无MP
    cv::Mat Descriptors1 = pKF1->mDescriptors;

    vector<cv::KeyPoint> vKeysUn2 = pKF2->mvKeyPoints;
    DBoW2::FeatureVector vFeatVec2 = pKF2->mFeatVec;
    vector<PtrMapPoint> vpMapPoints2 = pKF2->GetMapPointMatches();  // Localizer下无MP
    cv::Mat Descriptors2 = pKF2->mDescriptors;

    vector<bool> vbMatched2(vpMapPoints2.size(), false);

    vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; ++i)
        rotHist[i].reserve(500);

    const float factor = 1.0f / HISTO_LENGTH;

    int nmatches = 0;

    // 将属于同一节点(特定层)的ORB特征进行匹配
    DBoW2::FeatureVector::iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::iterator f2end = vFeatVec2.end();

    while (f1it != f1end && f2it != f2end) {
        // 步骤1：分别取出属于同一node的ORB特征点
        // 只有属于同一node，才有可能是匹配点
        if (f1it->first == f2it->first) {
            // 步骤2：遍历KF中属于该node的特征点
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
                int bestDist2 = INT_MAX;
                int bestIdx2 = -1;

                // 步骤3：遍历F中属于该node的特征点，找到了最佳匹配点
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
                        continue;  // 表明这个点已经被匹配过了，不再匹配，加快速度

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

                // 步骤4：根据阈值和角度投票剔除误匹配
                if (bestDist1 < TH_LOW) {  // 60
                    // trick!
                    // 最佳匹配比次佳匹配明显要好，那么最佳匹配才真正靠谱
                    if (static_cast<float>(bestDist1) < mfNNratio * static_cast<float>(bestDist2)) {
                        // 步骤5：更新特征点的MapPoint
                        mapMatches12[idx1] = bestIdx2;
                        vbMatched2[bestIdx2] = true;

                        if (mbCheckOrientation) {
                            // trick!
                            // angle：每个特征点在提取描述子时的旋转主方向角度，如果图像旋转了，这个角度将发生改变
                            // 所有的特征点的角度变化应该是一致的，通过直方图统计得到最准确的角度变化值
                            float rot = vKeysUn1[idx1].angle - vKeysUn2[bestIdx2].angle;
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

    // 旋转一致性验证, 根据方向剔除误匹配的点
    if (mbCheckOrientation) {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        //! 计算rotHist中最大的三个的index
        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; ++i) {
            //! 如果特征点的旋转角度变化量属于这三个组，则保留
            if (i == ind1 || i == ind2 || i == ind3)
                continue;
            //! 将除了ind1 ind2 ind3以外的匹配点去掉
            for (size_t j = 0, jend = rotHist[i].size(); j < jend; ++j) {
                mapMatches12.erase(rotHist[i][j]);
                nmatches--;
            }
        }
    }

    return nmatches;
}


/**
 * @brief ORBmatcher::MatchByWindow
 * 先获得cell里的粗匹配候选，再从候选的KF中根据描述子计算最小和次小距离，剔除错匹配
 *
 * @param frame1        参考帧F1
 * @param frame2        当前帧F2
 * @param vbPrevMatched 参考帧F1特征点的位置[update]
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
    for (int i = 0; i < HISTO_LENGTH; ++i)
        rotHist[i].reserve(500);
    const float factor = 1.f / (float)HISTO_LENGTH;

    vector<int> vMatchesDistance(frame2.N, INT_MAX);
    vector<int> vnMatches21(frame2.N, -1);

    //! 遍历参考帧特征点, 序号i1
    for (int i1 = 0, iend1 = frame1.N; i1 < iend1; i1++) {
        KeyPoint kp1 = frame1.mvKeyPoints[i1];
        int level1 = kp1.octave;
        if (level1 > maxLevel || level1 < minLevel)
            continue;
        int minLevel2 = level1 - levelOffset > 0 ? level1 - levelOffset : 0;
        //! 1.对F1中的每个KP先获得F2中一个cell里的粗匹配候选, cell的边长为2*winsize
        vector<size_t> vIndices2 = frame2.getFeaturesInArea(
            vbPrevMatched[i1].x, vbPrevMatched[i1].y, winSize, minLevel2, level1 + levelOffset);
        if (vIndices2.empty())
            continue;

        cv::Mat d1 = frame1.mDescriptors.row(i1);

        int bestDist = INT_MAX;
        int bestDist2 = INT_MAX;
        int bestIdx2 = -1;

        //! 2.从F2的KP候选里计算最小和次小汉明距离, 序号i2
        for (auto vit = vIndices2.begin(), vend = vIndices2.end(); vit != vend; ++vit) {
            size_t i2 = *vit;

            cv::Mat d2 = frame2.mDescriptors.row(i2);

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
                float rot = frame1.mvKeyPoints[i1].angle - frame2.mvKeyPoints[bestIdx2].angle;
                if (rot < 0.0)
                    rot += 360.f;
                int bin = round(rot * factor);
                if (bin == HISTO_LENGTH)
                    bin = 0;
                rotHist[bin].push_back(i1);
            }
        }
    }

    //! orientation check. 5.进行旋转一致性检验, 匹配点对角度差不在直方图最大的三个方向上,
    //! 则视为误匹配剔除
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; ++i) {
            if (i == ind1 || i == ind2 || i == ind3)
                continue;
            for (size_t j = 0, jend = rotHist[i].size(); j < jend; ++j) {
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
            vbPrevMatched[i1] = frame2.mvKeyPoints[vnMatches12[i1]].pt;

    return nmatches;
}


/**
 * @brief 利用仿射变换A(或透视变换H矩阵)增加先验
 * 注意内点数较少时H12不可用, 朝天花板的最好用仿射变换
 * @param frame1        参考帧F1
 * @param frame2        当前帧F2
 * @param H12           F1到F2的单应矩阵
 * @param vnMatches12   匹配情况[output]
 * @param winSize       cell尺寸
 * @return              返回匹配点的总数
 */
int ORBmatcher::MatchByWindowWarp(const Frame& frame1, const Frame& frame2, const cv::Mat& HA12,
                                  std::vector<int>& vnMatches12, const int winSize)
{
    assert(HA12.type() == CV_64FC1);

    Mat H = Mat::eye(3, 3, CV_64FC1);

    if (!HA12.data)
        std::cerr << "[Match][Warni] Input argument error for empty H!" << std::endl;
    if (HA12.rows == 2)
        HA12.copyTo(H.rowRange(0, 2));
    else
        HA12.copyTo(H.rowRange(0, 3));

    int nmatches = 0;
    vnMatches12 = vector<int>(frame1.N, -1);

    vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; ++i)
        rotHist[i].reserve(500);
    const float factor = 1.f / HISTO_LENGTH;

    vector<int> vMatchesDistance(frame2.N, INT_MAX);
    vector<int> vnMatches21(frame2.N, -1);

    //! 遍历参考帧特征点, 序号i1
    for (int i1 = 0, iend1 = frame1.N; i1 < iend1; i1++) {
        KeyPoint kp1 = frame1.mvKeyPoints[i1];
        int level = kp1.octave;
        //! 1.对F1中的每个KP先获得F2中一个cell里的粗匹配候选, cell的边长为2*winsize
        Mat pt1 = (Mat_<double>(3, 1) << kp1.pt.x, kp1.pt.y, 1);
        Mat pt2 = H * pt1;
        pt2 /= pt2.at<double>(2);
        vector<size_t> vIndices2 =
            frame2.getFeaturesInArea(pt2.at<double>(0), pt2.at<double>(1), winSize, level, level);
        if (vIndices2.empty())
            continue;

        cv::Mat d1 = frame1.mDescriptors.row(i1);

        int bestDist = INT_MAX;
        int bestDist2 = INT_MAX;
        int bestIdx2 = -1;

        //! 2.从F2的KP候选里计算最小和次小汉明距离, 序号i2
        for (auto vit = vIndices2.begin(), vend = vIndices2.end(); vit != vend; ++vit) {
            size_t i2 = *vit;

            cv::Mat d2 = frame2.mDescriptors.row(i2);

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
            float rot = frame1.mvKeyPoints[i1].angle - frame2.mvKeyPoints[bestIdx2].angle;
            if (rot < 0.0)
                rot += 360.f;
            int bin = round(rot * factor);
            if (bin == HISTO_LENGTH)
                bin = 0;
            rotHist[bin].push_back(i1);
        }
    }

    //! 5.进行旋转一致性检验, 匹配点对角度差不在直方图最大的三个方向上, 则视为误匹配剔除.
    //! orientation check
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; ++i) {
            if (i == ind1 || i == ind2 || i == ind3)
                continue;
            for (size_t j = 0, jend = rotHist[i].size(); j < jend; ++j) {
                int idx1 = rotHist[i][j];
                if (vnMatches12[idx1] >= 0) {
                    vnMatches12[idx1] = -1;
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}

int ORBmatcher::MatchByWindowWarp(const Frame& frame1, const Frame& frame2, const cv::Mat& HA12,
                                  std::map<int, int>& matches12, const int winSize)
{
    assert(HA12.type() == CV_64FC1);

    Mat H = Mat::eye(3, 3, CV_64FC1);

    if (!HA12.data)
        std::cerr << "[Match][Warni] Input argument error for empty H!" << std::endl;
    if (HA12.rows == 2)
        HA12.copyTo(H.rowRange(0, 2));
    else
        HA12.copyTo(H.rowRange(0, 3));

    int nmatches = 0;

    vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; ++i)
        rotHist[i].reserve(200);
    const float factor = 1.f / (float)HISTO_LENGTH;

    vector<int> vMatchesDistance(frame2.N, INT_MAX);
    vector<int> vnMatches21(frame2.N, -1);

    //! 遍历参考帧特征点, 序号i1
    for (int i1 = 0, iend1 = frame1.N; i1 < iend1; i1++) {
        KeyPoint kp1 = frame1.mvKeyPoints[i1];
        int level = kp1.octave;
        //! 1.对F1中的每个KP先获得F2中一个cell里的粗匹配候选, cell的边长为2*winsize
        Mat pt1 = (Mat_<double>(3, 1) << kp1.pt.x, kp1.pt.y, 1);
        Mat pt2 = HA12 * pt1;
        pt2 /= pt2.at<double>(2);
        vector<size_t> vIndices2 =
            frame2.getFeaturesInArea(pt2.at<double>(0), pt2.at<double>(1), winSize, level, level);
        if (vIndices2.empty())
            continue;

        cv::Mat d1 = frame1.mDescriptors.row(i1);

        int bestDist = INT_MAX;
        int bestDist2 = INT_MAX;
        int bestIdx2 = -1;

        //! 2.从F2的KP候选里计算最小和次小汉明距离, 序号i2
        for (auto vit = vIndices2.begin(), vend = vIndices2.end(); vit != vend; ++vit) {
            size_t i2 = *vit;

            cv::Mat d2 = frame2.mDescriptors.row(i2);

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
//                    vnMatches12[vnMatches21[bestIdx2]] = -1;
                    matches12.erase(vnMatches21[bestIdx2]);
                    nmatches--;
                } else {
                    vnMatches21[bestIdx2] = -1;
                    continue;
                }
            }
//            vnMatches12[i1] = bestIdx2;
            matches12[i1] = bestIdx2;
            vnMatches21[bestIdx2] = i1;
            vMatchesDistance[bestIdx2] = bestDist;
            nmatches++;

            //! for orientation check. 4.统计匹配点对的角度差的直方图, 待下一步做旋转检验
            float rot = frame1.mvKeyPoints[i1].angle - frame2.mvKeyPoints[bestIdx2].angle;
            if (rot < 0.0)
                rot += 360.f;
            int bin = round(rot * factor);
            if (bin == HISTO_LENGTH)
                bin = 0;
            rotHist[bin].push_back(i1);
        }
    }

    //! 5.进行旋转一致性检验, 匹配点对角度差不在直方图最大的三个方向上, 则视为误匹配剔除.
    //! orientation check
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; ++i) {
            if (i == ind1 || i == ind2 || i == ind3)
                continue;
            for (size_t j = 0, jend = rotHist[i].size(); j < jend; ++j) {
                int idx1 = rotHist[i][j];
//                if (vnMatches12[idx1] >= 0) {
//                    vnMatches12[idx1] = -1;
//                    nmatches--;
//                }
                matches12.erase(idx1);
            }
        }
    }

    return nmatches;
}

/**
 * @brief 将localMPs中不是KF观测的MPs(且视差好的)投影到当前KF上进行匹配
 *   在LocalMapper的addNewKF()里调用
 * @param pNewKF        要投影的当前KF
 * @param localMPs      Map里提取出的局部地图点
 * @param winSize       搜索半径 20
 * @param levelOffset   金字塔层搜索相对范围 2
 * @param vMatchesIdxMP 匹配上的MP索引[output]
 * @return              返回匹配成功的点对数
 */
int ORBmatcher::MatchByProjection(PtrKeyFrame& pNewKF, std::vector<PtrMapPoint>& localMPs,
                                  const int winSize, const int levelOffset,
                                  std::vector<int>& vMatchesIdxMP)
{
    int nmatches = 0;

    vMatchesIdxMP = vector<int>(pNewKF->N, -1);
    vector<int> vMatchesDistance(pNewKF->N, INT_MAX);

    for (int i = 0, iend = localMPs.size(); i < iend; ++i) {
        PtrMapPoint pMP = localMPs[i];
        if (pMP->isNull() || !pMP->isGoodPrl()) // NOTE 视差暂时不好的不能投影! 20191022
            continue;
        if (pNewKF->hasObservation(pMP))
            continue;

        Point2f predictUV = cvu::camprjc(Config::Kcam, cvu::se3map(pNewKF->getPose(), pMP->getPos()));
        if (!pNewKF->inImgBound(predictUV))
            continue;
        const int predictLevel = pMP->mMainOctave;  // 都在0层
        const int radio = pNewKF->mvScaleFactors[predictLevel] * winSize;
        const int minLevel = predictLevel > levelOffset ? predictLevel - levelOffset : 0;

        // 通过投影点(投影到当前帧,见isInFrustum())以及搜索窗口和预测的尺度进行搜索,找出附近的兴趣点
        vector<size_t> vNearKPIndices = pNewKF->getFeaturesInArea(
                    predictUV.x, predictUV.y, radio,  minLevel, predictLevel + levelOffset);
        if (vNearKPIndices.empty())
            continue;

        int bestDist = INT_MAX;
        int bestDist2 = INT_MAX;
        int bestLevel = -1;
        int bestLevel2 = -1;
        int bestIdx = -1;

        // Get best and second matches with near keypoints
        for (auto it = vNearKPIndices.begin(), iend = vNearKPIndices.end(); it != iend; ++it) {
            int idx = *it;
            if (pNewKF->hasObservation(idx))
                continue;
            cv::Mat d = pNewKF->mDescriptors.row(idx);

            const int dist = DescriptorDistance(pMP->getDescriptor(), d);

            if (vMatchesDistance[idx] <= dist)
                continue;

            if (dist < bestDist) {
                bestDist2 = bestDist;
                bestDist = dist;
                bestLevel2 = bestLevel;
                bestLevel = pNewKF->mvKeyPoints[idx].octave;
                bestIdx = idx;
            } else if (dist < bestDist2) {
                bestLevel2 = pNewKF->mvKeyPoints[idx].octave;
                bestDist2 = dist;
            }
        }

        // Apply ratio to second match (only if best and second are in the same scale level)
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

}  // namespace ORB_SLAM
