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

//struct line_s_e {
//    cv::Point2f star_p;
//    cv::Point2f end_p;
//    cv::Point2f match_star;
//    cv::Point2f match_end;
//};

//void getMatcheLines_Star_Eend(const Frame frame1, const Frame frame2,
//                              std::vector<line_s_e>& matchesLine1_S_E, int linelable1,
//                              int linelable2, int pl1, int pl2);


class ORBmatcher
{
public:
    ORBmatcher(float nnratio = 0.6, bool checkOri = true, bool withline = false);

    // Computes the Hamming distance between two ORB descriptors
    static int DescriptorDistance(const cv::Mat& a, const cv::Mat& b);


    // Search matches between MapPoints in a KeyFrame and ORB in a Frame.
    // Brute force constrained to ORB that belong to the same vocabulary node (at a certain level)
    // Used in Relocalisation and Loop Detection
    int SearchByBoW(PtrKeyFrame pKF1, PtrKeyFrame pKF2, std::map<int, int>& mapIdxMatches12, bool bIfMPOnly);

    int SearchByProjection(Frame& CurrentFrame, KeyFrame& LastFrame, const float th = 7);

//    int SearchByBoW(KeyFrame *pKF, Frame &F, std::vector<MapPoint*> &vpMapPointMatches);
    void ComputeThreeMaxima(std::vector<int>* histo, const int L, int& ind1, int& ind2, int& ind3);


    static const int TH_LOW;
    static const int TH_HIGH;
    static const int HISTO_LENGTH;

    float mfNNratio;
    bool mbCheckOrientation;

    int MatchByWindow(const Frame& frame1, const Frame& frame2, std::vector<cv::Point2f>& vbPrevMatched,
                      const int winSize, std::vector<int>& vnMatches12, const int levelOffset = 1,
                      const int minLevel = 0, const int maxLevel = 8);
    int MatchByWindowWarp(const Frame& frame1, const Frame& frame2, const cv::Mat& H,
                          std::vector<int>& vnMatches12, const int winSize);

    int MatchByProjection(PtrKeyFrame& pNewKF, std::vector<PtrMapPoint>& localMPs,
                          const int winSize, const int levelOffset,
                          std::vector<int>& vMatchesIdxMP);


    float RadiusByViewingCos(const float& viewCos);

//    int MatchByPointAndLine(const Frame& frame1, Frame& frame2,
//                            std::vector<cv::Point2f>& vbPrevMatched, const int winSize,
//                            std::vector<int>& vnMatches12, vector<int>& vMatchesDistance,
//                            double angle, const int levelOffset = 1, const int minLevel = 0,
//                            const int maxLevel = 8);  // 7.18修改maxlevel=8

private:
    bool mbWithLineFeature;
};

}  // namespace se2lam


#endif  // ORBMATCHER_H
