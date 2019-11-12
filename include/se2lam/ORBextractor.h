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

#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <list>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>


namespace se2lam
{

class ORBextractor
{
public:
    enum { HARRIS_SCORE = 0, FAST_SCORE = 1 };

    //! 超天花板的摄像头没有z方向位移, 金字塔层数可以设为1
    ORBextractor(int nMaxFeatures = 1000, float scaleFactor = 1.2f, int nLevels = 6,
                 int scoreType = FAST_SCORE, int fastTh = 15);  // 20

    ~ORBextractor() {}

    // Compute the ORB features and descriptors on an image
    void operator()(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints,
                    cv::OutputArray descriptors);

    inline int   getMaxFeaturesNum() { return nMaxFeatures; }
    inline int   getLevels() { return nLevels; }
    inline float getScaleFactor() { return scaleFactor; }

    // 获取给定特征点的描述子
    void getDescriptors(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints,
                       cv::Mat& descriptors);

protected:
    void computePyramid(cv::Mat image, cv::Mat Mask = cv::Mat());
    void computeKeyPoints(std::vector<std::vector<cv::KeyPoint>>& allKeypoints);

    std::vector<cv::Point> mvPattern;

    int nMaxFeatures;
    double scaleFactor;
    int nLevels;
    int scoreType;
    int fastTh;

    std::vector<int> mvFeaturesPerLevel;

    std::vector<int> umax;

    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;

    std::vector<cv::Mat> mvImagePyramid;
    std::vector<cv::Mat> mvMaskPyramid;
};

}  // namespace se2lam

#endif
