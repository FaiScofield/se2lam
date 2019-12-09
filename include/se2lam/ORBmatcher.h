/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

/**
* This file is part of ORB-SLAM.
*
* Copyright (C) 2014 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
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


#ifndef ORBMATCHER_H
#define ORBMATCHER_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

#include "Frame.h"
#include "KeyFrame.h"
#include "MapPoint.h"

namespace se2lam
{

class ORBmatcher
{
public:
    ORBmatcher(float nnratio = 0.6, bool checkOri = true);

    // Computes the Hamming distance between two ORB descriptors
    static int DescriptorDistance(const cv::Mat& a, const cv::Mat& b);


    // Search matches between MapPoints in a KeyFrame and ORB in a Frame.
    // Brute force constrained to ORB that belong to the same vocabulary node (at a certain level)
    // Used in Relocalisation and Loop Detection
    int SearchByBoW(PtrKeyFrame pKF1, PtrKeyFrame pKF2, std::map<int, int>& mapIdxMatches12, bool bIfMPOnly);
    int SearchByBoW(Frame* F1, Frame* F2, std::map<int, int>& mapIdxMatches12, bool bIfMPOnly);

    // int SearchByProjection(Frame& CurrentFrame, KeyFrame& LastFrame, const float th = 7);
    int SearchByProjection(Frame& frameCurr, Frame& frameLast, int winSize);
    int SearchByProjection(Frame& thisFrame, const std::vector<PtrMapPoint>& localMPs,
                           std::vector<int>& vMatchesIdxMP, int winSize, int levelOffset);
    int SearchByProjection(PtrKeyFrame pKF, const std::vector<PtrMapPoint>& localMPs,
                           std::vector<int>& vMatchesIdxMP, int winSize, int levelOffset);

    //    int SearchByBoW(KeyFrame *pKF, Frame &F, std::vector<MapPoint*> &vpMapPointMatches);
    void ComputeThreeMaxima(std::vector<int>* histo, const int L, int& ind1, int& ind2, int& ind3);


    static const int TH_LOW;
    static const int TH_HIGH;
    static const int HISTO_LENGTH;

    float mfNNratio;
    bool mbCheckOrientation;

    // deprecated
    int MatchByWindow(const Frame& frame1, const Frame& frame2, std::vector<cv::Point2f>& vbPrevMatched,
                      const int winSize, std::vector<int>& vnMatches12, const int levelOffset = 1,
                      const int minLevel = 0, const int maxLevel = 8);

    int MatchByWindowWarp(const Frame& frame1, const Frame& frame2, const cv::Mat& HA12,
                          std::vector<int>& vnMatches12, const int winSize);

    // 暂时没用
    int MatchByWindowWarp(const Frame& frame1, const Frame& frame2, const cv::Mat& HA12,
                          std::map<int, int>& matches12, const int winSize);


    float RadiusByViewingCos(const float viewCos);
};

}  // namespace se2lam


#endif  // ORBMATCHER_H
