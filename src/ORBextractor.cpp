/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

/**
* This file is part of ORB-SLAM.
* It is based on the file orb.cpp from the OpenCV library (see BSD license below).
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

/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*********************************************************************/


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <vector>

#include "ORBextractor.h"

#include <ros/ros.h>

using namespace cv;
using namespace std;

namespace se2lam
{

const float HARRIS_K = 0.04f;

const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD = 16;
const float factorPI = (float)(CV_PI / 180.f);

static void HarrisResponses(const Mat& img, vector<KeyPoint>& pts, int blockSize, float harris_k)
{
    CV_Assert(img.type() == CV_8UC1 && blockSize * blockSize <= 2048);

    size_t ptidx, ptsize = pts.size();

    const uchar* ptr00 = img.ptr<uchar>();
    int step = (int)(img.step / img.elemSize1());
    int r = blockSize / 2;

    float scale = (1 << 2) * blockSize * 255.0f;
    scale = 1.0f / scale;
    float scale_sq_sq = scale * scale * scale * scale;

    AutoBuffer<int> ofsbuf(blockSize * blockSize);
    int* ofs = ofsbuf;
    for (int i = 0; i < blockSize; i++)
        for (int j = 0; j < blockSize; j++)
            ofs[i * blockSize + j] = (int)(i * step + j);

    for (ptidx = 0; ptidx < ptsize; ptidx++) {
        int x0 = cvRound(pts[ptidx].pt.x - r);
        int y0 = cvRound(pts[ptidx].pt.y - r);

        const uchar* ptr0 = ptr00 + y0 * step + x0;
        int a = 0, b = 0, c = 0;

        for (int k = 0; k < blockSize * blockSize; k++) {
            const uchar* ptr = ptr0 + ofs[k];
            int Ix = (ptr[1] - ptr[-1]) * 2 + (ptr[-step + 1] - ptr[-step - 1]) +
                     (ptr[step + 1] - ptr[step - 1]);
            int Iy = (ptr[step] - ptr[-step]) * 2 + (ptr[step - 1] - ptr[-step - 1]) +
                     (ptr[step + 1] - ptr[-step + 1]);
            a += Ix * Ix;
            b += Iy * Iy;
            c += Ix * Iy;
        }
        pts[ptidx].response =
            ((float)a * b - (float)c * c - harris_k * ((float)a + b) * ((float)a + b)) * scale_sq_sq;
    }
}

static float IC_Angle(const Mat& image, Point2f pt, const vector<int>& u_max)
{
    int m_01 = 0, m_10 = 0;

    const uchar* center = &image.at<uchar>(cvRound(pt.y), cvRound(pt.x));

    // Treat the center line differently, v=0
    for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
        m_10 += u * center[u];

    // Go line by line in the circuI853lar patch
    int step = (int)image.step1();
    for (int v = 1; v <= HALF_PATCH_SIZE; ++v) {
        // Proceed over the two lines
        int v_sum = 0;
        int d = u_max[v];
        for (int u = -d; u <= d; ++u) {
            int val_plus = center[u + v * step], val_minus = center[u - v * step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }

    return fastAtan2((float)m_01, (float)m_10);
}

/* static */ void computeOrbDescriptor(const KeyPoint& kpt, const Mat& img, const Point* pattern, uchar* desc)
{
    float angle = (float)kpt.angle * factorPI;
    float a = (float)cos(angle), b = (float)sin(angle);

    const uchar* center = &img.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
    const int step = (int)img.step;

#define GET_VALUE(idx)                                               \
    center[cvRound(pattern[idx].x * b + pattern[idx].y * a) * step + \
           cvRound(pattern[idx].x * a - pattern[idx].y * b)]


    for (int i = 0; i < 32; ++i, pattern += 16) {
        int t0, t1, val;
        t0 = GET_VALUE(0);
        t1 = GET_VALUE(1);
        val = t0 < t1;
        t0 = GET_VALUE(2);
        t1 = GET_VALUE(3);
        val |= (t0 < t1) << 1;
        t0 = GET_VALUE(4);
        t1 = GET_VALUE(5);
        val |= (t0 < t1) << 2;
        t0 = GET_VALUE(6);
        t1 = GET_VALUE(7);
        val |= (t0 < t1) << 3;
        t0 = GET_VALUE(8);
        t1 = GET_VALUE(9);
        val |= (t0 < t1) << 4;
        t0 = GET_VALUE(10);
        t1 = GET_VALUE(11);
        val |= (t0 < t1) << 5;
        t0 = GET_VALUE(12);
        t1 = GET_VALUE(13);
        val |= (t0 < t1) << 6;
        t0 = GET_VALUE(14);
        t1 = GET_VALUE(15);
        val |= (t0 < t1) << 7;

        desc[i] = (uchar)val;
    }

#undef GET_VALUE
}

/* static */ void computeDescriptors(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors,
                               const vector<Point>& pattern)
{
    descriptors = Mat::zeros((int)keypoints.size(), 32, CV_8UC1);

    for (size_t i = 0; i < keypoints.size(); i++)
        computeOrbDescriptor(keypoints[i], image, &pattern[0], descriptors.ptr((int)i));
}

static void computeOrientation(const Mat& image, vector<KeyPoint>& keypoints, const vector<int>& umax)
{
    for (vector<KeyPoint>::iterator keypoint = keypoints.begin(), keypointEnd = keypoints.end();
         keypoint != keypointEnd; ++keypoint) {
        keypoint->angle = IC_Angle(image, keypoint->pt, umax);
    }
}

void ExtractorNode::DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4)
{
    const int halfX = ceil(static_cast<float>(UR.x-UL.x)/2);
    const int halfY = ceil(static_cast<float>(BR.y-UL.y)/2);

    //Define boundaries of childs
    n1.UL = UL;
    n1.UR = cv::Point2i(UL.x+halfX,UL.y);
    n1.BL = cv::Point2i(UL.x,UL.y+halfY);
    n1.BR = cv::Point2i(UL.x+halfX,UL.y+halfY);
    n1.vKeys.reserve(vKeys.size());

    n2.UL = n1.UR;
    n2.UR = UR;
    n2.BL = n1.BR;
    n2.BR = cv::Point2i(UR.x,UL.y+halfY);
    n2.vKeys.reserve(vKeys.size());

    n3.UL = n1.BL;
    n3.UR = n1.BR;
    n3.BL = BL;
    n3.BR = cv::Point2i(n1.BR.x,BL.y);
    n3.vKeys.reserve(vKeys.size());

    n4.UL = n3.UR;
    n4.UR = n2.BR;
    n4.BL = n3.BR;
    n4.BR = BR;
    n4.vKeys.reserve(vKeys.size());

    // Associate points to childs
    for (size_t i = 0; i < vKeys.size(); i++) {
        const cv::KeyPoint& kp = vKeys[i];
        if (kp.pt.x < n1.UR.x) {
            if (kp.pt.y < n1.BR.y)
                n1.vKeys.push_back(kp);
            else
                n3.vKeys.push_back(kp);
        } else if (kp.pt.y < n1.BR.y)
            n2.vKeys.push_back(kp);
        else
            n4.vKeys.push_back(kp);
    }

    if (n1.vKeys.size() == 1)
        n1.bNoMore = true;
    if (n2.vKeys.size() == 1)
        n2.bNoMore = true;
    if (n3.vKeys.size() == 1)
        n3.bNoMore = true;
    if (n4.vKeys.size() == 1)
        n4.bNoMore = true;
}

ORBextractor::ORBextractor(int _nfeatures, float _scaleFactor, int _nlevels, int _scoreType,
                           int _iniThFAST, int _minThFAST) :
        nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels), scoreType(_scoreType),
        iniThFAST(_iniThFAST), minThFAST(_minThFAST)
{

    mvScaleFactor.resize(nlevels);
    mvLevelSigma2.resize(nlevels);
    mvScaleFactor[0] = 1;
    mvLevelSigma2[0]=1.0f;
    for (int i = 1; i < nlevels; i++) {
        mvScaleFactor[i] = mvScaleFactor[i - 1] * scaleFactor;
        mvLevelSigma2[i]=mvScaleFactor[i]*mvScaleFactor[i];
    }

    mvInvScaleFactor.resize(nlevels);
    mvInvLevelSigma2.resize(nlevels);
    for (int i = 0; i < nlevels; i++) {
        mvInvScaleFactor[i] = 1.0f / mvScaleFactor[i];
        mvInvLevelSigma2[i] = 1.0f / mvLevelSigma2[i];
    }

    mvImagePyramid.resize(nlevels);
    mvMaskPyramid.resize(nlevels);

    mnFeaturesPerLevel.resize(nlevels);
    float factor = (float)(1.0 / scaleFactor);
    float nDesiredFeaturesPerScale =
        nfeatures * (1 - factor) / (1 - (float)pow((double)factor, (double)nlevels));

    int sumFeatures = 0;
    for (int level = 0; level < nlevels - 1; level++) {
        mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
        sumFeatures += mnFeaturesPerLevel[level];
        nDesiredFeaturesPerScale *= factor;
    }
    mnFeaturesPerLevel[nlevels - 1] = std::max(nfeatures - sumFeatures, 0);

    const int npoints = 512;
    const Point* pattern0 = (const Point*)bit_pattern_31_;
    pattern.reserve(npoints);
    std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));

    // This is for orientation
    // pre-compute the end of a row in a circular patch
    umax.resize(HALF_PATCH_SIZE + 1);


    int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
    const double hp2 = HALF_PATCH_SIZE * HALF_PATCH_SIZE;
    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(sqrt(hp2 - v * v));

    // Make sure we are symmetric
    for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v) {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }
}

void ORBextractor::operator()(InputArray _image, InputArray _mask, vector<KeyPoint>& _keypoints, OutputArray _descriptors)
{
    if (_image.empty())
        return;

    Mat image = _image.getMat(), mask = _mask.getMat();
    assert(image.type() == CV_8UC1);

    // Pre-compute the scale pyramids
    ComputePyramid(image, mask);

    vector<vector<KeyPoint>> allKeypoints;
    // ComputeKeyPointsOld(allKeypoints);
    // compute KP accordding to mFeatureType
    switch (mFeatureType)
    {
    case GFTT:
        ComputeKeyPointsGFTT(allKeypoints);
        break;
    case SURF:
        ComputeKeyPointsSURF(allKeypoints);
        break;
    case ORB:
    default:
        ComputeKeyPointsOctTree(allKeypoints);
        break;
    }

    Mat descriptors;

    int nkeypoints = 0;
    for (int level = 0; level < nlevels; ++level)
        nkeypoints += (int)allKeypoints[level].size();
    if (nkeypoints == 0)
        _descriptors.release();
    else {
        _descriptors.create(nkeypoints, 32, CV_8U);
        descriptors = _descriptors.getMat();
    }

    _keypoints.clear();
    _keypoints.reserve(nkeypoints);

    int offset = 0;
    for (int level = 0; level < nlevels; ++level) {
        vector<KeyPoint>& keypoints = allKeypoints[level];
        int nkeypointsLevel = (int)keypoints.size();

        if (nkeypointsLevel == 0)
            continue;

        // preprocess the resized image
        Mat& workingMat = mvImagePyramid[level];
        GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);

        // Compute the descriptors
        Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);
        computeDescriptors(workingMat, keypoints, desc, pattern);

        offset += nkeypointsLevel;

        // Scale keypoint coordinates
        if (level != 0) {
            float scale = mvScaleFactor[level];  // getScale(level, firstLevel, scaleFactor);
            for (vector<KeyPoint>::iterator keypoint = keypoints.begin(), keypointEnd = keypoints.end();
                 keypoint != keypointEnd; ++keypoint)
                keypoint->pt *= scale;
        }
        // And add the keypoints to the output
        _keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
    }
}


void ORBextractor::ComputePyramid(const cv::Mat& image, const cv::Mat& Mask)
{
    for (int level = 0; level < nlevels; ++level) {
        float scale = mvInvScaleFactor[level];
        Size sz(cvRound((float)image.cols * scale), cvRound((float)image.rows * scale));
        assert(sz.width > 0 && sz.height > 0);
        Size wholeSize(sz.width + EDGE_THRESHOLD * 2, sz.height + EDGE_THRESHOLD * 2);
        Mat temp(wholeSize, image.type()), masktemp;
        mvImagePyramid[level] = temp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

        if (!Mask.empty()) {
            masktemp = Mat(wholeSize, Mask.type());
            mvMaskPyramid[level] = masktemp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));
        }

        // Compute the resized image
        if (level != 0) {
            resize(mvImagePyramid[level - 1], mvImagePyramid[level], sz, 0, 0, INTER_LINEAR);
            if (!Mask.empty()) {
                resize(mvMaskPyramid[level - 1], mvMaskPyramid[level], sz, 0, 0, INTER_NEAREST);
            }

            copyMakeBorder(mvImagePyramid[level], temp, EDGE_THRESHOLD, EDGE_THRESHOLD,
                           EDGE_THRESHOLD, EDGE_THRESHOLD, BORDER_REFLECT_101 + BORDER_ISOLATED);
            if (!Mask.empty())
                copyMakeBorder(mvMaskPyramid[level], masktemp, EDGE_THRESHOLD, EDGE_THRESHOLD,
                               EDGE_THRESHOLD, EDGE_THRESHOLD, BORDER_CONSTANT + BORDER_ISOLATED);
        } else {
            copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                           EDGE_THRESHOLD, BORDER_REFLECT_101);
            if (!Mask.empty())
                copyMakeBorder(Mask, masktemp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                               EDGE_THRESHOLD, BORDER_CONSTANT + BORDER_ISOLATED);
        }
    }
}

void ORBextractor::ComputeKeyPointsOld(vector<vector<KeyPoint>>& allKeypoints)
{

    allKeypoints.resize(nlevels);

    float imageRatio = (float)mvImagePyramid[0].cols / mvImagePyramid[0].rows;

    for (int level = 0; level < nlevels; ++level) {
        const int nDesiredFeatures = mnFeaturesPerLevel[level];

        const int levelCols = sqrt((float)nDesiredFeatures / (5 * imageRatio));
        const int levelRows = imageRatio * levelCols;

        const int minBorderX = EDGE_THRESHOLD;
        const int minBorderY = minBorderX;
        const int maxBorderX = mvImagePyramid[level].cols - EDGE_THRESHOLD;
        const int maxBorderY = mvImagePyramid[level].rows - EDGE_THRESHOLD;

        const int W = maxBorderX - minBorderX;
        const int H = maxBorderY - minBorderY;
        const int cellW = ceil((float)W / levelCols);
        const int cellH = ceil((float)H / levelRows);

        const int nCells = levelRows * levelCols;
        const int nfeaturesCell = ceil((float)nDesiredFeatures / nCells);

        vector<vector<vector<KeyPoint>>> cellKeyPoints(levelRows, vector<vector<KeyPoint>>(levelCols));

        vector<vector<int>> nToRetain(levelRows, vector<int>(levelCols));
        vector<vector<int>> nTotal(levelRows, vector<int>(levelCols));
        vector<vector<bool>> bNoMore(levelRows, vector<bool>(levelCols, false));
        vector<int> iniXCol(levelCols);
        vector<int> iniYRow(levelRows);
        int nNoMore = 0;
        int nToDistribute = 0;


        float hY = cellH + 6;

        for (int i = 0; i < levelRows; i++) {
            const float iniY = minBorderY + i * cellH - 3;
            iniYRow[i] = iniY;

            if (i == levelRows - 1) {
                hY = maxBorderY + 3 - iniY;
                if (hY <= 0)
                    continue;
            }

            float hX = cellW + 6;

            for (int j = 0; j < levelCols; j++) {
                float iniX;

                if (i == 0) {
                    iniX = minBorderX + j * cellW - 3;
                    iniXCol[j] = iniX;
                } else {
                    iniX = iniXCol[j];
                }


                if (j == levelCols - 1) {
                    hX = maxBorderX + 3 - iniX;
                    if (hX <= 0)
                        continue;
                }


                Mat cellImage = mvImagePyramid[level].rowRange(iniY, iniY + hY).colRange(iniX, iniX + hX);

                Mat cellMask;
                if (!mvMaskPyramid[level].empty())
                    cellMask = cv::Mat(mvMaskPyramid[level], Rect(iniX, iniY, hX, hY));

                cellKeyPoints[i][j].reserve(nfeaturesCell * 5);

                FAST(cellImage, cellKeyPoints[i][j], iniThFAST, true);

                if (cellKeyPoints[i][j].size() <= 3) {
                    cellKeyPoints[i][j].clear();

                    FAST(cellImage, cellKeyPoints[i][j], 7, true);
                }

                if (scoreType == ORB::HARRIS_SCORE) {
                    // Compute the Harris cornerness
                    HarrisResponses(cellImage, cellKeyPoints[i][j], 7, HARRIS_K);
                }

                const int nKeys = cellKeyPoints[i][j].size();
                nTotal[i][j] = nKeys;

                if (nKeys > nfeaturesCell) {
                    nToRetain[i][j] = nfeaturesCell;
                    bNoMore[i][j] = false;
                } else {
                    nToRetain[i][j] = nKeys;
                    nToDistribute += nfeaturesCell - nKeys;
                    bNoMore[i][j] = true;
                    nNoMore++;
                }
            }
        }


        // Retain by score
        while (nToDistribute > 0 && nNoMore < nCells) {
            int nNewFeaturesCell = nfeaturesCell + ceil((float)nToDistribute / (nCells - nNoMore));
            nToDistribute = 0;

            for (int i = 0; i < levelRows; i++) {
                for (int j = 0; j < levelCols; j++) {
                    if (!bNoMore[i][j]) {
                        if (nTotal[i][j] > nNewFeaturesCell) {
                            nToRetain[i][j] = nNewFeaturesCell;
                            bNoMore[i][j] = false;
                        } else {
                            nToRetain[i][j] = nTotal[i][j];
                            nToDistribute += nNewFeaturesCell - nTotal[i][j];
                            bNoMore[i][j] = true;
                            nNoMore++;
                        }
                    }
                }
            }
        }

        vector<KeyPoint>& keypoints = allKeypoints[level];
        keypoints.reserve(nDesiredFeatures * 2);

        const int scaledPatchSize = PATCH_SIZE * mvScaleFactor[level];

        // Retain by score and transform coordinates
        for (int i = 0; i < levelRows; i++) {
            for (int j = 0; j < levelCols; j++) {
                vector<KeyPoint>& keysCell = cellKeyPoints[i][j];
                KeyPointsFilter::retainBest(keysCell, nToRetain[i][j]);
                if ((int)keysCell.size() > nToRetain[i][j])
                    keysCell.resize(nToRetain[i][j]);

                for (size_t k = 0, kend = keysCell.size(); k < kend; k++) {
                    keysCell[k].pt.x += iniXCol[j];
                    keysCell[k].pt.y += iniYRow[i];
                    keysCell[k].octave = level;
                    keysCell[k].size = scaledPatchSize;
                    keypoints.push_back(keysCell[k]);
                }
            }
        }
        if ((int)keypoints.size() > nDesiredFeatures) {
            KeyPointsFilter::retainBest(keypoints, nDesiredFeatures);
            keypoints.resize(nDesiredFeatures);
        }
    }

    // and compute orientations
    for (int level = 0; level < nlevels; ++level)
        computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);
}

void ORBextractor::ComputeKeyPointsOctTree(vector<vector<KeyPoint> >& allKeypoints)
{
    allKeypoints.resize(nlevels);

    const float W = 30;

    for (int level = 0; level < nlevels; ++level) {
        const int minBorderX = EDGE_THRESHOLD - 3;
        const int minBorderY = minBorderX;
        const int maxBorderX = mvImagePyramid[level].cols - EDGE_THRESHOLD + 3;
        const int maxBorderY = mvImagePyramid[level].rows - EDGE_THRESHOLD + 3;

        vector<cv::KeyPoint> vToDistributeKeys;
        vToDistributeKeys.reserve(nfeatures * 10);

        const float width = (maxBorderX - minBorderX);
        const float height = (maxBorderY - minBorderY);

        const int nCols = width / W;
        const int nRows = height / W;
        const int wCell = ceil(width / nCols);
        const int hCell = ceil(height / nRows);

        for (int i = 0; i < nRows; i++) {
            const float iniY = minBorderY + i * hCell;
            float maxY = iniY + hCell + 6;

            if (iniY >= maxBorderY - 3)
                continue;
            if (maxY > maxBorderY)
                maxY = maxBorderY;

            for (int j = 0; j < nCols; j++) {
                const float iniX = minBorderX + j * wCell;
                float maxX = iniX + wCell + 6;
                if (iniX >= maxBorderX - 6)
                    continue;
                if (maxX > maxBorderX)
                    maxX = maxBorderX;

                vector<cv::KeyPoint> vKeysCell;
                FAST(mvImagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX), vKeysCell, iniThFAST, true);

                if (vKeysCell.empty()) {
                    FAST(mvImagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX), vKeysCell,
                         minThFAST, true);
                }

                if (!vKeysCell.empty()) {
                    for (vector<cv::KeyPoint>::iterator vit = vKeysCell.begin(); vit != vKeysCell.end(); vit++) {
                        (*vit).pt.x += j * wCell;
                        (*vit).pt.y += i * hCell;
                        vToDistributeKeys.push_back(*vit);
                    }
                }
            }
        }

        vector<KeyPoint>& keypoints = allKeypoints[level];
        keypoints.reserve(nfeatures);

        keypoints = DistributeOctTree(vToDistributeKeys, minBorderX, maxBorderX, minBorderY,
                                      maxBorderY, mnFeaturesPerLevel[level], level);

        const int scaledPatchSize = PATCH_SIZE * mvScaleFactor[level];

        // Add border to coordinates and scale information
        const int nkps = keypoints.size();
        for (int i = 0; i < nkps; i++) {
            keypoints[i].pt.x += minBorderX;
            keypoints[i].pt.y += minBorderY;
            keypoints[i].octave = level;
            keypoints[i].size = scaledPatchSize;
        }
    }

    // compute orientations
    for (int level = 0; level < nlevels; ++level)
        computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);
}

void ORBextractor::ComputeKeyPointsGFTT(vector<vector<KeyPoint>>& allKeypoints)
{
    allKeypoints.resize(nlevels);

    for (int level = 0; level < nlevels; ++level) {
        const int minBorderX = EDGE_THRESHOLD-3;
        const int minBorderY = minBorderX;
        const int maxBorderX = mvImagePyramid[level].cols-EDGE_THRESHOLD+3;
        const int maxBorderY = mvImagePyramid[level].rows-EDGE_THRESHOLD+3;

        vector<Point2f> vPoints;
        goodFeaturesToTrack(mvImagePyramid[level], vPoints, mnFeaturesPerLevel[level], 0.005, 10/(level+1));

        vector<cv::KeyPoint> vToDistributeKeys;
        vToDistributeKeys.reserve(vPoints.size());

        for (auto iter = vPoints.begin(); iter != vPoints.end(); ++iter) {
            if (iter->x >= minBorderX && iter->x < maxBorderX && 
                iter->y >= minBorderY && iter->y < maxBorderY) {
                    vToDistributeKeys.emplace_back(iter->x - minBorderX, iter->y - minBorderY, 1, -1.f, 0.f, level);
                }
        }
        vector<KeyPoint> & keypoints = allKeypoints[level];
        keypoints.reserve(vToDistributeKeys.size());

        keypoints = DistributeOctTree(vToDistributeKeys, minBorderX, maxBorderX,
                                      minBorderY, maxBorderY, mnFeaturesPerLevel[level], level);


        // Add border to coordinates and scale information
        const int scaledPatchSize = PATCH_SIZE * mvScaleFactor[level];
        const int nkps = keypoints.size();
        for(int i=0; i<nkps ; i++)
        {
            keypoints[i].pt.x += minBorderX;
            keypoints[i].pt.y += minBorderY;
            keypoints[i].octave = level;
            keypoints[i].size = scaledPatchSize;
        }
    }

    // compute orientations
    for (int level = 0; level < nlevels; ++level) {
        computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);
    }
}

void ORBextractor::ComputeKeyPointsSURF(vector<vector<KeyPoint>>& allKeypoints)
{
    allKeypoints.resize(nlevels);

    for (int level = 0; level < nlevels; ++level) {
        const int minBorderX = EDGE_THRESHOLD-3;
        const int minBorderY = minBorderX;
        const int maxBorderX = mvImagePyramid[level].cols-EDGE_THRESHOLD+3;
        const int maxBorderY = mvImagePyramid[level].rows-EDGE_THRESHOLD+3;

        vector<KeyPoint> vPoints;
        Ptr<xfeatures2d::SURF> pSURFDetector = xfeatures2d::SURF::create(300, 1, 1, false, true);
        pSURFDetector->detect(mvImagePyramid[level], vPoints);
        // printf("get %ld KPs in level %d.\n", vPoints.size(), level);

        vector<cv::KeyPoint> vToDistributeKeys;
        vToDistributeKeys.reserve(vPoints.size());

        for (auto iter = vPoints.begin(); iter != vPoints.end(); ++iter) {
            Point2f& kp = iter->pt;
            if (kp.x >= minBorderX && kp.x < maxBorderX && 
                kp.y >= minBorderY && kp.y < maxBorderY) {
                    vToDistributeKeys.emplace_back(kp.x - minBorderX, kp.y - minBorderY, 1, -1.f, 0.f, level);
                }
        }

        vector<KeyPoint> & keypoints = allKeypoints[level];
        keypoints.reserve(vToDistributeKeys.size());

        keypoints = DistributeOctTree(vToDistributeKeys, minBorderX, maxBorderX,
                                      minBorderY, maxBorderY, mnFeaturesPerLevel[level], level);

        // Add border to coordinates and scale information
        const int scaledPatchSize = PATCH_SIZE * mvScaleFactor[level];
        const int nkps = keypoints.size();
        for(int i=0; i<nkps ; i++)
        {
            keypoints[i].pt.x += minBorderX;
            keypoints[i].pt.y += minBorderY;
            keypoints[i].octave = level;
            keypoints[i].size = scaledPatchSize;
        }
    }

    // compute orientations
    for (int level = 0; level < nlevels; ++level) {
        computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);
    }
}

vector<cv::KeyPoint> ORBextractor::DistributeOctTree(const vector<cv::KeyPoint>& vToDistributeKeys,
                                                     const int& minX, const int& maxX, const int& minY,
                                                     const int& maxY, const int& N, const int& level)
{
    // Compute how many initial nodes
    const int nIni = round(static_cast<float>(maxX - minX) / (maxY - minY));

    const float hX = static_cast<float>(maxX - minX) / nIni;

    list<ExtractorNode> lNodes;

    vector<ExtractorNode*> vpIniNodes;
    vpIniNodes.resize(nIni);

    for (int i = 0; i < nIni; i++) {
        ExtractorNode ni;
        ni.UL = cv::Point2i(hX * static_cast<float>(i), 0);
        ni.UR = cv::Point2i(hX * static_cast<float>(i + 1), 0);
        ni.BL = cv::Point2i(ni.UL.x, maxY - minY);
        ni.BR = cv::Point2i(ni.UR.x, maxY - minY);
        ni.vKeys.reserve(vToDistributeKeys.size());

        lNodes.push_back(ni);
        vpIniNodes[i] = &lNodes.back();
    }

    // Associate points to childs
    for (size_t i = 0; i < vToDistributeKeys.size(); i++) {
        const cv::KeyPoint& kp = vToDistributeKeys[i];
        vpIniNodes[kp.pt.x / hX]->vKeys.push_back(kp);
    }

    list<ExtractorNode>::iterator lit = lNodes.begin();

    while (lit != lNodes.end()) {
        if (lit->vKeys.size() == 1) {
            lit->bNoMore = true;
            lit++;
        } else if (lit->vKeys.empty())
            lit = lNodes.erase(lit);
        else
            lit++;
    }

    bool bFinish = false;

    int iteration = 0;

    vector<pair<int, ExtractorNode*>> vSizeAndPointerToNode;
    vSizeAndPointerToNode.reserve(lNodes.size() * 4);

    while (!bFinish) {
        iteration++;

        int prevSize = lNodes.size();

        lit = lNodes.begin();

        int nToExpand = 0;

        vSizeAndPointerToNode.clear();

        while (lit != lNodes.end()) {
            if (lit->bNoMore) {
                // If node only contains one point do not subdivide and continue
                lit++;
                continue;
            } else {
                // If more than one point, subdivide
                ExtractorNode n1, n2, n3, n4;
                lit->DivideNode(n1, n2, n3, n4);

                // Add childs if they contain points
                if (n1.vKeys.size() > 0) {
                    lNodes.push_front(n1);
                    if (n1.vKeys.size() > 1) {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if (n2.vKeys.size() > 0) {
                    lNodes.push_front(n2);
                    if (n2.vKeys.size() > 1) {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if (n3.vKeys.size() > 0) {
                    lNodes.push_front(n3);
                    if (n3.vKeys.size() > 1) {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if (n4.vKeys.size() > 0) {
                    lNodes.push_front(n4);
                    if (n4.vKeys.size() > 1) {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }

                lit = lNodes.erase(lit);
                continue;
            }
        }

        // Finish if there are more nodes than required features
        // or all nodes contain just one point
        if ((int)lNodes.size() >= N || (int)lNodes.size() == prevSize) {
            bFinish = true;
        } else if (((int)lNodes.size() + nToExpand * 3) > N) {

            while (!bFinish) {

                prevSize = lNodes.size();

                vector<pair<int, ExtractorNode*>> vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                vSizeAndPointerToNode.clear();

                sort(vPrevSizeAndPointerToNode.begin(), vPrevSizeAndPointerToNode.end());
                for (int j = vPrevSizeAndPointerToNode.size() - 1; j >= 0; j--) {
                    ExtractorNode n1, n2, n3, n4;
                    vPrevSizeAndPointerToNode[j].second->DivideNode(n1, n2, n3, n4);

                    // Add childs if they contain points
                    if (n1.vKeys.size() > 0) {
                        lNodes.push_front(n1);
                        if (n1.vKeys.size() > 1) {
                            vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n2.vKeys.size() > 0) {
                        lNodes.push_front(n2);
                        if (n2.vKeys.size() > 1) {
                            vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n3.vKeys.size() > 0) {
                        lNodes.push_front(n3);
                        if (n3.vKeys.size() > 1) {
                            vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if (n4.vKeys.size() > 0) {
                        lNodes.push_front(n4);
                        if (n4.vKeys.size() > 1) {
                            vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(), &lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

                    if ((int)lNodes.size() >= N)
                        break;
                }

                if ((int)lNodes.size() >= N || (int)lNodes.size() == prevSize)
                    bFinish = true;
            }
        }
    }

    // Retain the best point in each node
    vector<cv::KeyPoint> vResultKeys;
    vResultKeys.reserve(nfeatures);
    for (list<ExtractorNode>::iterator lit = lNodes.begin(); lit != lNodes.end(); lit++) {
        vector<cv::KeyPoint>& vNodeKeys = lit->vKeys;
        cv::KeyPoint* pKP = &vNodeKeys[0];
        float maxResponse = pKP->response;

        for (size_t k = 1; k < vNodeKeys.size(); k++) {
            if (vNodeKeys[k].response > maxResponse) {
                pKP = &vNodeKeys[k];
                maxResponse = vNodeKeys[k].response;
            }
        }

        vResultKeys.push_back(*pKP);
    }

    return vResultKeys;
}

}  // namespace se2lam
