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

#include "ORBextractor.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

namespace se2lam
{

using namespace cv;
using namespace std;

const float HARRIS_K = 0.04f;

const int PATCH_SIZE = 23;       // 31, 23
const int HALF_PATCH_SIZE = 11;  // 15, 11
const int EDGE_THRESHOLD = 12;   // 16, 12


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
    for (int i = 0; i < blockSize; ++i)
        for (int j = 0; j < blockSize; ++j)
            ofs[i * blockSize + j] = (int)(i * step + j);

    for (ptidx = 0; ptidx < ptsize; ptidx++) {
        int x0 = cvRound(pts[ptidx].pt.x - r);
        int y0 = cvRound(pts[ptidx].pt.y - r);

        const uchar* ptr0 = ptr00 + y0 * step + x0;
        int a = 0, b = 0, c = 0;

        for (int k = 0; k < blockSize * blockSize; ++k) {
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
            ((float)a * b - (float)c * c - harris_k * ((float)a + b) * ((float)a + b)) *
            scale_sq_sq;
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
        // Proceed over the two lines 每次处理对称两行
        int v_sum = 0;
        int d = u_max[v];
        for (int u = -d; u <= d; ++u) {
            int val_plus = center[u + v * step];
            int val_minus = center[u - v * step];
            v_sum += (val_plus - val_minus);  // 因为val_minus对应的是-v,为了统一符号，用减法
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }

    return fastAtan2((float)m_01, (float)m_10);
}

static void computeOrbDescriptor(const KeyPoint& kpt, const Mat& img, const Point* pattern,
                                 uchar* desc)
{
    float angle = (float)kpt.angle * (float)(CV_PI / 180.f);
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

static void computeDescriptors(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors,
                               const vector<Point>& pattern)
{
    descriptors = Mat::zeros((int)keypoints.size(), 32, CV_8UC1);

    for (size_t i = 0; i < keypoints.size(); ++i)
        computeOrbDescriptor(keypoints[i], image, &pattern[0], descriptors.ptr((int)i));
}

static void computeOrientation(const Mat& image, vector<KeyPoint>& keypoints,
                               const vector<int>& umax)
{
    for (auto iter = keypoints.begin(), iend = keypoints.end(); iter != iend; ++iter)
        iter->angle = IC_Angle(image, iter->pt, umax);
}


ORBextractor::ORBextractor(int _nfeatures, float _scaleFactor, int _nlevels, int _scoreType,
                           int _fastTh)
    : nMaxFeatures(_nfeatures), scaleFactor(_scaleFactor), nLevels(_nlevels), scoreType(_scoreType),
      fastTh(_fastTh)
{
    // 计算每一层相对于原始图片的放大倍数
    mvScaleFactor.resize(nLevels);
    mvScaleFactor[0] = 1;
    for (int i = 1; i < nLevels; ++i)
        mvScaleFactor[i] = mvScaleFactor[i - 1] * scaleFactor;

    // 计算每一层想对于原始图片放大倍数的逆
    float invScaleFactor = 1.0f / scaleFactor;
    mvInvScaleFactor.resize(nLevels);
    mvInvScaleFactor[0] = 1;
    for (int i = 1; i < nLevels; ++i)
        mvInvScaleFactor[i] = mvInvScaleFactor[i - 1] * invScaleFactor;

    mvImagePyramid.resize(nLevels);
    mvMaskPyramid.resize(nLevels);

    // 前nlevels层的和为总的特征点数量nfeatures（等比数列的前n项和）
    // 主要是将每层的特征点数量进行均匀控制
    mvFeaturesPerLevel.resize(nLevels);
    float factor = (float)(1.0 / scaleFactor);
    float nDesiredFeaturesPerScale =
        nMaxFeatures * (1 - factor) / (1 - (float)pow((double)factor, (double)nLevels));

    int sumFeatures = 0;
    for (int level = 0; level < nLevels - 1; level++) {
        mvFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
        sumFeatures += mvFeaturesPerLevel[level];
        nDesiredFeaturesPerScale *= factor;
    }
    mvFeaturesPerLevel[nLevels - 1] = max(nMaxFeatures - sumFeatures, 0);

    // 复制训练的模板
    const int npoints = 512;
    const Point* pattern0 = (const Point*)bit_pattern_31_;
    mvPattern.reserve(npoints);
    copy(pattern0, pattern0 + npoints, back_inserter(mvPattern));

    // This is for orientation pre-compute the end of a row in a circular patch
    // 用于计算特征方向时，每个v坐标对应最大的u坐标
    umax.resize(HALF_PATCH_SIZE + 1);

    // 将v坐标划分为两部分进行计算，主要为了确保计算特征主方向的时候，x,y方向对称
    int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
    // 利用勾股定理计算
    const double hp2 = HALF_PATCH_SIZE * HALF_PATCH_SIZE;
    // V坐标的第一部分
    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(sqrt(hp2 - v * v));

    // Make sure we are symmetric
    // V坐标的第二部分,确保对称，即保证是一个圆
    for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v) {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }
}

void ORBextractor::computeKeyPoints(vector<vector<KeyPoint>>& allKeypoints)
{
    allKeypoints.resize(nLevels);

    float imageRatio = (float)mvImagePyramid[0].cols / mvImagePyramid[0].rows;

    for (int level = 0; level < nLevels; ++level) {
        const int nDesiredFeatures = mvFeaturesPerLevel[level];

        const int levelCols = sqrt((float)nDesiredFeatures / (5 * imageRatio));
        const int levelRows = imageRatio * levelCols;

        // 计算边界
        const int minBorderX = EDGE_THRESHOLD;
        const int minBorderY = minBorderX;
        const int maxBorderX = mvImagePyramid[level].cols - EDGE_THRESHOLD;
        const int maxBorderY = mvImagePyramid[level].rows - EDGE_THRESHOLD;

        // 计算窗口的大小
        const int W = maxBorderX - minBorderX;
        const int H = maxBorderY - minBorderY;
        const int cellW = ceil((float)W / levelCols);
        const int cellH = ceil((float)H / levelRows);

        const int nCells = levelRows * levelCols;
        const int nfeaturesCell = ceil((float)nDesiredFeatures / nCells);

        vector<vector<vector<KeyPoint>>> cellKeyPoints(levelRows,
                                                       vector<vector<KeyPoint>>(levelCols));

        vector<vector<int>> nToRetain(levelRows, vector<int>(levelCols));
        vector<vector<int>> nTotal(levelRows, vector<int>(levelCols));
        vector<vector<bool>> bNoMore(levelRows, vector<bool>(levelCols, false));
        vector<int> iniXCol(levelCols);
        vector<int> iniYRow(levelRows);
        int nNoMore = 0;
        int nToDistribute = 0;

        float hY = cellH + 6;

        for (int i = 0; i < levelRows; ++i) {
            const float iniY = minBorderY + i * cellH - 3;
            iniYRow[i] = iniY;

            if (i == levelRows - 1) {
                hY = maxBorderY + 3 - iniY;
                if (hY <= 0)
                    continue;
            }

            float hX = cellW + 6;

            for (int j = 0; j < levelCols; ++j) {
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

                Mat cellImage =
                    mvImagePyramid[level].rowRange(iniY, iniY + hY).colRange(iniX, iniX + hX);

                Mat cellMask;
                if (!mvMaskPyramid[level].empty())
                    cellMask = Mat(mvMaskPyramid[level], Rect(iniX, iniY, hX, hY));

                cellKeyPoints[i][j].reserve(nfeaturesCell * 5);

                FAST(cellImage, cellKeyPoints[i][j], fastTh, true);

                if (cellKeyPoints[i][j].size() <= 3) {  // 3
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

            for (int i = 0; i < levelRows; ++i) {
                for (int j = 0; j < levelCols; ++j) {
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
        for (int i = 0; i < levelRows; ++i) {
            for (int j = 0; j < levelCols; ++j) {
                vector<KeyPoint>& keysCell = cellKeyPoints[i][j];
                KeyPointsFilter::retainBest(keysCell, nToRetain[i][j]);
                if ((int)keysCell.size() > nToRetain[i][j])
                    keysCell.resize(nToRetain[i][j]);

                for (size_t k = 0, kend = keysCell.size(); k < kend; ++k) {
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
    for (int level = 0; level < nLevels; ++level)
        computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);
}

void ORBextractor::operator()(InputArray _image, InputArray _mask, vector<KeyPoint>& _keypoints,
                              OutputArray _descriptors)
{
    if (_image.empty())
        return;

    Mat image = _image.getMat(), mask = _mask.getMat();
    assert(image.type() == CV_8UC1);

    // Pre-compute the scale pyramids 构建高斯金字塔
    computePyramid(image, mask);

    vector<vector<KeyPoint>> allKeypoints;
    computeKeyPoints(allKeypoints);  //! NOTE 这里没有用八叉树存储

    Mat descriptors;

    int nkeypoints = 0;
    for (int level = 0; level < nLevels; ++level)
        nkeypoints += (int)allKeypoints[level].size();
    if (nkeypoints == 0)
        _descriptors.release();
    else {
        _descriptors.create(nkeypoints, 32, CV_8U);
        descriptors = _descriptors.getMat();
    }

    _keypoints.clear();
    _keypoints.reserve(nkeypoints);

    // 计算每个特征点对应的描述子
    int offset = 0;
    for (int level = 0; level < nLevels; ++level) {
        vector<KeyPoint>& keypoints = allKeypoints[level];
        int nkeypointsLevel = (int)keypoints.size();

        if (nkeypointsLevel == 0)
            continue;

        // preprocess the resized image 高斯模糊
        Mat& workingMat = mvImagePyramid[level];
        GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);

        // Compute the descriptors 计算描述子
        Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);
        computeDescriptors(workingMat, keypoints, desc, mvPattern);

        offset += nkeypointsLevel;

        // Scale keypoint coordinates
        if (level != 0) {
            float scale = mvScaleFactor[level];  // getScale(level, firstLevel, scaleFactor);
            for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                                            keypointEnd = keypoints.end();
                 keypoint != keypointEnd; ++keypoint)
                keypoint->pt *= scale;
        }
        // And add the keypoints to the output
        _keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
    }
}

void ORBextractor::computePyramid(Mat image, Mat Mask)
{
    for (int level = 0; level < nLevels; ++level) {
        float scale = mvInvScaleFactor[level];
        Size sz(cvRound((float)image.cols * scale), cvRound((float)image.rows * scale));
        Size wholeSize(sz.width + EDGE_THRESHOLD * 2, sz.height + EDGE_THRESHOLD * 2);
        Mat temp(wholeSize, image.type()), masktemp;
        mvImagePyramid[level] = temp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

        if (!Mask.empty()) {
            masktemp = Mat(wholeSize, Mask.type());
            mvMaskPyramid[level] =
                masktemp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));
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

/**
 * @brief 计算光流特征的描述子
 * @param image     输入图像
 * @param keypoints 输入的特征点
 * @param descs     输出特征点对应的描述子
 * @author Maple.Liu
 * @date 2019.10.23
 */
void ORBextractor::getDescriptors(const Mat& image, vector<KeyPoint>& keypoints, Mat& descs)
{
    computeOrientation(image, keypoints, umax);
    computeDescriptors(image, keypoints, descs, mvPattern);
}

}  // namespace se2lam
