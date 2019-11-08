#include "LineDetector.h"
#include "test_functions.hpp"
#include <opencv2/video/tracking.hpp>

string g_configPath = "/home/vance/dataset/rk/dibeaDataSet/se2_config/";
size_t g_matchToRefSum = 0;

struct PairMask {
    cv::Point2f ptInForw;  // 点在当前帧下的像素坐标
    cv::Point2f ptInCurr;  // 对应上一帧的点的像素坐标
    cv::Point2f ptInPrev;
    unsigned long firstAdd;  // 所在图像的id
    size_t idxToAdd;         // 与生成帧的匹配点索引
    int cellLabel;           // 特征点所在块的lable
    int trackCount;          // 被追踪上的次数
};

struct KLT_KP {
    KLT_KP(const cv::Point2f& pt, unsigned long id, int idx, int label)
        : ptInCurr(pt), firstAdd(id), idxToRef(idx), cellLabel(label)
    {
    }

    void addTrackCount() { trackCount++; }
    void setCorInLast(const cv::Point2f& pt) { ptInLast = pt; }

    cv::Point2f ptInCurr;        // 在当前帧下的像素坐标
    cv::Point2f ptInLast;        // 在上一帧下的像素坐标
    cv::Point2f ptInPred;
    unsigned long firstAdd = 0;  // 所诞生的帧号
    int idxToRef = -1;           // 与参考KF的匹配点索引
    int cellLabel = -1;          // 所处的cell编号
    int trackCount = 0;          // 被追踪上的总次数
};

class KLT
{
public:
    bool firstFrame = true;

    Frame frameCurr, frameLast;
    PtrKeyFrame KFRef;
    Mat AffineMatrix = Mat::eye(2, 3, CV_64FC1);

    // klt variable
    // 64, 48;  80, 60
    int cellWidth = 64, cellHeight = 48;  // 最好整除
    int cellsX = 320 / cellWidth, cellsY = 240 / cellHeight;
    int cells = cellsX * cellsY;
    int maxPtsInCell = Config::MaxFtrNumber / cells;
    int maskRadius = 3;

    Mat mask;
    Mat imgPrev, imgCurr, imgForw;
    vector<Point2f> ptsPrev, ptsCurr, ptsForw, ptsNew;

    vector<unsigned long> idFirstAdd;  // 在哪一帧生成
    vector<size_t> idxToFirstAdd;      // 相对生成帧的KP索引
    vector<int> trackCount;            // 被追踪的次数
    vector<int> cellLable;         // 都在哪个cell里
    vector<int> numInCell;             // 当前帧每个cell里的点数

    vector<int> matchIdx;
    map<int, int> matchIdxWithRefKF;

    ORBextractor* pExtractor;
    ORBmatcher* pMatcher;

    unsigned long idCurr = 0;

    vector<KLT_KP> allKltKPs;

    KLT()
    {
        numInCell.resize(cells, 0);
        pExtractor = new ORBextractor(500, 1.2, 1);
        pMatcher = new ORBmatcher(0.9);
    }

    ~KLT()
    {
        delete pExtractor;
        delete pMatcher;
    }

    void track(const Mat& imgGray, const Se2& odo)
    {
        idCurr++;
        imgForw = imgGray;
        ptsForw.clear();

        if (firstFrame) {
            vector<Keyline> lines;
            Mat lineMask = getLineMask(imgForw, lines, false);
            detectFeaturePointsWithCell(imgForw, lineMask);  // 更新ptsNew, vnPtsInCell, vCellLable
            addNewPoints();  // 更新ptsForw, vIdAddInFrame, vTrackCount, vIdxTrackToFirstAdd

            if (ptsForw.size() < 100) {
                cerr << "Too less points in first frame!" << endl;
                return;
            }

            firstFrame = false;

            // 创建当前帧
            vector<KeyPoint> vKPs;
            vKPs.resize(ptsForw.size());
            for (size_t i = 0, iend = ptsForw.size(); i < iend; ++i) {
                vKPs[i].pt = ptsForw[i];
                vKPs[i].octave = 0;
            }
            frameCurr = Frame(imgForw, odo, vKPs, pExtractor);
            frameCurr.setPose(Se2(0.f, 0.f, 0.f));

            KFRef = make_shared<KeyFrame>(frameCurr);
            resetKltData();

            return;
        }

        size_t n1 = 0, n2 = 0, n3 = 0, n4 = 0;
        Mat show1, show2, show3, show4;

        //! 1.预测上一帧img和KP旋转后的值
        double imuTheta = normalizeAngle(odo.theta - frameLast.odom.theta);
        predictPointsAndImage(imuTheta);  // 由 ptsLast 更新 ptsPred (warped)

        //! 2.光流追踪上一帧的KP, 通过旋转的ptsPrev
        if (!ptsCurr.empty()) {
            vector<uchar> status;
            vector<float> err;
            calcOpticalFlowPyrLK(imgPrev, imgForw, ptsPrev, ptsForw, status, err, Size(21, 21), 0);
            n1 = ptsForw.size();

            for (size_t i = 0, iend = ptsForw.size(); i < iend; i++) {
                if (status[i] && !inBorder(ptsForw[i]))
                    status[i] = 0;
            }
            reduceVector(status);
            n2 = ptsForw.size();
            show1 = drawMachesPointsToLastFrame("Match Predict");

            rejectWithRansac(false);
            n3 = ptsForw.size();
            show2 = drawMachesPointsToLastFrame("Match Ransac ");

            for (auto& n : trackCount)
                n++;
        }

        //! 3.有必要的话提取新的特征点
        setMask(maskRadius);
        n4 = ptsForw.size();
        show3 = drawMachesPointsToLastFrame("Masked & Added");
        printf("#%ld 光流追踪, 去密后剩点数/内点数/追踪上的点数/总点数 = %ld/%ld/%ld/%ld\n", idCurr,
               n4, n3, n2, n1);

        int needPoints = Config::MaxFtrNumber - static_cast<int>(ptsForw.size());
        if (needPoints > 50) {
            vector<Keyline> klTmp;
            Mat maskline = getLineMask(imgForw, klTmp, false);
            mask = mask.mul(maskline);
            cv::goodFeaturesToTrack(imgForw, ptsNew, needPoints, 0.01, 1);
        } else {
            ptsNew.clear();
        }
        imshow("mask", mask);
        printf("#%ld 光流追踪, 新增点数为: %ld, ", idCurr, ptsNew.size());
        drawNewAddPointsInMatch(show3);
        addNewPoints();
        printf("目前共有点数为: %ld\n", ptsForw.size());

        //! 4.更新当前帧与参考关键帧的匹配关系
        matchIdxWithRefKF.clear();
        matchIdx.clear();
        matchIdx.resize(KFRef->N, -1);
        for (size_t i = 0, iend = ptsForw.size(); i < iend; ++i) {
            if (idFirstAdd[i] == KFRef->id) {
                matchIdxWithRefKF.emplace(i, idxToFirstAdd[i]);
                matchIdx[idxToFirstAdd[i]] = i;
            }
        }
        g_matchToRefSum += matchIdxWithRefKF.size();
        show4 = drawMachesPointsToRefFrame("Match Ref KF");
        printf("#%ld 光流追踪, 从参考帧追踪上的点数/平均追踪点数 = %ld/%.2f\n", idCurr,
               matchIdxWithRefKF.size(), g_matchToRefSum / (idCurr - 1.0));

        Mat showMatchs;
        vconcat(show1, show2, showMatchs);
        vconcat(showMatchs, show3, showMatchs);
        vconcat(showMatchs, show4, showMatchs);
        //string fileName = "/home/vance/output/rk_se2lam/klt-match/" + to_string(idCurr) + ".bmp";
        //imwrite(fileName, showMatchs);
        imshow("KLT Matches To Last & Ransac & Ref", showMatchs);

        //! 5.更新Frame
        vector<KeyPoint> vKPsCurFrame(ptsForw.size());
        cv::KeyPoint::convert(ptsForw, vKPsCurFrame);
        frameCurr = Frame(imgForw, odo, vKPsCurFrame, pExtractor);
        assert(idCurr == frameCurr.id);

        //! 6. KF判断
        if (needNewKF(matchIdxWithRefKF.size())) {
            KFRef = make_shared<KeyFrame>(frameCurr);

            // 重置klt相关变量
            size_t n = ptsForw.size();
            vector<int> trackCountTmp(n, 1);
            vector<unsigned long> idFirstAddTmp(n, KFRef->id);
            trackCount.swap(trackCountTmp);
            idFirstAdd.swap(idFirstAddTmp);
            for (size_t i = 0; i < n; ++i)
                idxToFirstAdd[i] = i;
        }

        waitKey(50);

        resetKltData();
    }

    void trackCell(const Mat& imgGray, const Se2& odo)
    {
        idCurr++;
        imgForw = imgGray;
        ptsForw.clear();

        if (firstFrame) {
            vector<Keyline> lines;
            Mat lineMask = getLineMask(imgForw, lines, false);
            detectFeaturePointsWithCell(imgForw, lineMask);  // 更新ptsNew, vNumInCell, vCellLable
            addNewPoints();  // 更新ptsForw, vIdAddInFrame, vTrackCount, vIdxTrackToFirstAdd

            if (ptsForw.size() < 100) {
                cerr << "Too less points in first frame!" << endl;
                return;
            }

            firstFrame = false;

            // 创建当前帧
            vector<KeyPoint> vKPsCurFrame(ptsForw.size());
            cv::KeyPoint::convert(ptsForw, vKPsCurFrame);
            frameCurr = Frame(imgForw, odo, vKPsCurFrame, pExtractor);
            frameCurr.setPose(Se2(0.f, 0.f, 0.f));

            KFRef = make_shared<KeyFrame>(frameCurr);
            resetKltData();

            return;
        }

        size_t n1 = 0, n2 = 0, n3 = 0, n4 = 0;
        Mat show1, show2, show3, show4;

        //! 1.预测上一帧img和KP旋转后的值
        double imuTheta = normalizeAngle(odo.theta - frameLast.odom.theta);
        predictPointsAndImage(imuTheta);  // 由 ptsCurr 更新 ptsPrev (warped)

        //! 2.光流追踪上一帧的KP, 通过旋转的ptsPrev
        if (!ptsCurr.empty()) {
            vector<uchar> status;
            vector<float> err;
            calcOpticalFlowPyrLK(imgPrev, imgForw, ptsPrev, ptsForw, status, err, Size(21, 21), 0);
            n1 = ptsForw.size();

            for (size_t i = 0, iend = ptsForw.size(); i < iend; i++) {
                if (status[i] && !inBorder(ptsForw[i]))
                    status[i] = 0;
            }
            reduceVectorCell(status);  // 把所有跟踪失败和图像外的点剔除
            n2 = ptsForw.size();
            show1 = drawMachesPointsToLastFrame("Match Predict");

            rejectWithRansac(true);  // Ransac匹配剔除outliers, 通过未旋转的ptsCurr
            n3 = ptsForw.size();
            show2 = drawMachesPointsToLastFrame("Match Ransac ");

            for (auto& n : trackCount)  // 光流追踪成功, 特征点被成功跟中的次数就加一
                n++;
        }

        //! 3.有必要的话提取新的特征点
        setMaskCell(maskRadius);  // 设置mask, 去除密集点
        n4 = ptsForw.size();
        show3 = drawMachesPointsToLastFrame("Masked & Added");
        printf("#%ld 光流追踪, 去密后剩点数/内点数/追踪上的点数/总点数 = %ld/%ld/%ld/%ld\n", idCurr,
               n4, n3, n2, n1);

        detectFeaturePointsWithCell(imgForw, mask);
        drawNewAddPointsInMatch(show3);
        imshow("mask", mask);
        printf("#%ld 光流追踪, 新增点数为: %ld, ", idCurr, ptsNew.size());
        addNewPoints();
        printf("目前共有点数为: %ld\n", ptsForw.size());

        //! 4.更新当前帧与参考关键帧的匹配关系
        matchIdxWithRefKF.clear();
        matchIdx.clear();
        matchIdx.resize(KFRef->N, -1);
        for (size_t i = 0, iend = ptsForw.size(); i < iend; ++i) {
            if (idFirstAdd[i] == KFRef->id) {
                matchIdxWithRefKF.emplace(i, idxToFirstAdd[i]);
                matchIdx[idxToFirstAdd[i]] = i;
            }
        }
        g_matchToRefSum += matchIdxWithRefKF.size();
        show4 = drawMachesPointsToRefFrame("Match Ref KF");
        printf("#%ld 光流追踪, 从参考帧追踪上的点数/平均追踪点数 = %ld/%.2f\n", idCurr,
               matchIdxWithRefKF.size(), g_matchToRefSum / (idCurr - 1.0));

        // 可视化
        Mat showMatchs;
        vconcat(show1, show2, showMatchs);
        vconcat(showMatchs, show3, showMatchs);
        vconcat(showMatchs, show4, showMatchs);
        //string fileName = "/home/vance/output/rk_se2lam/klt-match-cell/" + to_string(idCurr) + ".bmp";
        //imwrite(fileName, showMatchs);
        imshow("KLT Matches To Last & Ransac & Ref", showMatchs);

        //! 5.更新Frame
        vector<KeyPoint> vKPsCurFrame(ptsForw.size());
        cv::KeyPoint::convert(ptsForw, vKPsCurFrame);
        frameCurr = Frame(imgForw, odo, vKPsCurFrame, pExtractor);
        assert(idCurr == frameCurr.id);

        //! 6. KF判断
        if (needNewKF(matchIdxWithRefKF.size())) {

            KFRef = make_shared<KeyFrame>(frameCurr);

            // 重置klt相关变量
            size_t n = ptsForw.size();
            vector<int> trackCountTmp(n, 1);
            vector<unsigned long> idFirstAddTmp(n, KFRef->id);
            trackCount.swap(trackCountTmp);
            idFirstAdd.swap(idFirstAddTmp);
            for (size_t i = 0; i < n; ++i)
                idxToFirstAdd[i] = i;
        }

        waitKey(50);

        resetKltData();
    }

    void trackCellToRef(const Mat& imgGray, const Se2& odo) {}

    void predictPointsAndImage(double angle)
    {
        ptsPrev.clear();
        if (abs(angle) > 0.01) {  // 0.001
            Point rotationCenter;
            rotationCenter.x = 160.5827 - 0.01525;  //! TODO
            rotationCenter.y = 117.7329 - 3.6984;

            getRotatedPoints(ptsCurr, ptsPrev, rotationCenter, angle);
            Mat rotationMatrix = getRotationMatrix2D(rotationCenter, angle * 180 / CV_PI, 1.);
            warpAffine(imgCurr, imgPrev, rotationMatrix, imgCurr.size());
        } else {
            imgPrev = imgCurr.clone();
            ptsPrev = ptsCurr;
        }
    }

    void getRotatedPoints(const vector<Point2f>& srcPoints, vector<Point2f>& dstPoints,
                          const Point& center, double angle)
    {
        dstPoints.resize(srcPoints.size());

        double row = static_cast<double>(240);
        for (size_t i = 0, iend = srcPoints.size(); i < iend; i++) {
            double x1 = srcPoints[i].x;
            double y1 = row - srcPoints[i].y;
            double x2 = center.x;
            double y2 = row - center.y;
            double x = cvRound((x1 - x2) * cos(angle) - (y1 - y2) * sin(angle) + x2);
            double y = cvRound((x1 - x2) * sin(angle) + (y1 - y2) * cos(angle) + y2);
            y = row - y;

            dstPoints[i] = Point2f(x, y);
        }
    }

    bool inBorder(const Point2f& pt)
    {
        return Frame::minXUn <= pt.x && pt.x < Frame::maxXUn && Frame::minYUn <= pt.y &&
               pt.y < Frame::maxYUn;
    }

    void reduceVector(const vector<uchar>& status)
    {
        assert(ptsCurr.size() == status.size());
        assert(ptsPrev.size() == status.size());
        assert(ptsForw.size() == status.size());

        size_t j = 0;
        for (size_t i = 0, iend = status.size(); i < iend; i++) {
            if (status[i]) {
                ptsPrev[j] = ptsPrev[i];
                ptsCurr[j] = ptsCurr[i];
                ptsForw[j] = ptsForw[i];
                idFirstAdd[j] = idFirstAdd[i];
                idxToFirstAdd[j] = idxToFirstAdd[i];
                trackCount[j] = trackCount[i];
                j++;
            }
        }
        ptsPrev.resize(j);
        ptsCurr.resize(j);
        ptsForw.resize(j);
        idFirstAdd.resize(j);
        idxToFirstAdd.resize(j);
        trackCount.resize(j);
    }

    void reduceVectorCell(const vector<uchar>& status)
    {
        assert(ptsCurr.size() == status.size());
        assert(ptsPrev.size() == status.size());
        assert(ptsForw.size() == status.size());

        size_t j = 0;
        for (size_t i = 0, iend = status.size(); i < iend; i++) {
            if (status[i]) {
                ptsPrev[j] = ptsPrev[i];
                ptsCurr[j] = ptsCurr[i];
                ptsForw[j] = ptsForw[i];
                idFirstAdd[j] = idFirstAdd[i];
                idxToFirstAdd[j] = idxToFirstAdd[i];
                trackCount[j] = trackCount[i];
                cellLable[j] = cellLable[i];
                j++;
            } else {
                numInCell[cellLable[i]]--;
            }
        }
        ptsPrev.resize(j);
        ptsCurr.resize(j);
        ptsForw.resize(j);
        idFirstAdd.resize(j);
        idxToFirstAdd.resize(j);
        trackCount.resize(j);
        cellLable.resize(j);
    }

    void rejectWithRansac(bool cell)
    {
        if (ptsForw.size() >= 8) {
            vector<unsigned char> inliersMask(ptsForw.size());
            findHomography(ptsCurr, ptsForw, RANSAC, 3.0, inliersMask);
            // AffineMatrix = estimateAffine2D(ptsCurr, ptsForw, inliersMask, RANSAC, 2.0);
            if (cell)
                reduceVectorCell(inliersMask);
            else
                reduceVector(inliersMask);
        }
    }

    void segImageToCells(const Mat& image, vector<Mat>& cellImgs)
    {
        Mat imageCut;
        int m = image.cols / cellWidth;
        int n = image.rows / cellHeight;
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                Rect rect(i * cellWidth, j * cellHeight, cellWidth, cellHeight);
                imageCut = Mat(image, rect);
                cellImgs.push_back(imageCut.clone());
            }
        }
    }

    void detectFeaturePointsWithCell(const Mat& image, const Mat& mask)
    {
        vector<Mat> cellImgs, cellMasks;
        segImageToCells(image, cellImgs);
        segImageToCells(mask, cellMasks);

        ptsNew.clear();
        ptsNew.reserve(maxPtsInCell * cells);
        int th = maxPtsInCell * 0.2;
        for (int i = 0; i < cells; ++i) {
            assert(numInCell[i] >= 0);
            int newPtsToAdd = maxPtsInCell - numInCell[i];
            //newPtsToAdd = newPtsToAdd > maxPtsInCell ? maxPtsInCell : newPtsToAdd;
            if (newPtsToAdd > th) {
                vector<Point2f> ptsInThisCell;
                ptsInThisCell.reserve(newPtsToAdd);
                goodFeaturesToTrack(cellImgs[i], ptsInThisCell, newPtsToAdd, 0.05, maskRadius,
                                    cellMasks[i], 3);
                numInCell[i] += static_cast<int>(ptsInThisCell.size());

                // 获得特征点在图像上的实际坐标
                for (size_t j = 0, jend = ptsInThisCell.size(); j < jend; j++) {
                    int cellIndexX = i % cellsX;
                    int cellIndexY = i / cellsX;

                    Point2f& thisPoint = ptsInThisCell[j];
                    thisPoint.x += cellWidth * cellIndexX;
                    thisPoint.y += cellHeight * cellIndexY;

                    ptsNew.push_back(thisPoint);
                    cellLable.push_back(i);
                }
            }
        }
    }

    void addNewPoints()
    {
        size_t offset = ptsForw.size();
        for (size_t i = 0, iend = ptsNew.size(); i < iend; ++i) {
            ptsForw.push_back(ptsNew[i]);
            trackCount.push_back(1);
            idFirstAdd.push_back(Frame::nextId);  // Frame还未生成
            idxToFirstAdd.push_back(i + offset);
        }
        ptsNew.clear();
        size_t s = ptsForw.size();
        assert(s == idFirstAdd.size());
        assert(s == trackCount.size());
        assert(s == idxToFirstAdd.size());
    }

    void setMask(int maskRadius = 0)
    {
        mask = Mat(imgCurr.rows, imgCurr.cols, CV_8UC1, Scalar(255));
        PairMask pm;
        vector<pair<int, PairMask>> vCountPtsId;
        for (size_t i = 0, iend = ptsForw.size(); i < iend; ++i) {
            pm.firstAdd = idFirstAdd[i];
            pm.idxToAdd = idxToFirstAdd[i];
            pm.ptInForw = ptsForw[i];
            pm.ptInCurr = ptsCurr[i];
            pm.ptInPrev = ptsPrev[i];
            vCountPtsId.emplace_back(trackCount[i], pm);
        }
        sort(vCountPtsId.begin(), vCountPtsId.end(),
             [](const pair<int, PairMask>& a, const pair<int, PairMask>& b) {
                 return a.first > b.first;
             });

        ptsCurr.clear();
        ptsForw.clear();
        ptsPrev.clear();
        trackCount.clear();
        idFirstAdd.clear();
        idxToFirstAdd.clear();
        for (const auto& it : vCountPtsId) {
            if (mask.at<uchar>(it.second.ptInForw) == 255) {
                trackCount.push_back(it.first);
                ptsForw.push_back(it.second.ptInForw);
                ptsCurr.push_back(it.second.ptInCurr);
                ptsPrev.push_back(it.second.ptInPrev);
                idFirstAdd.push_back(it.second.firstAdd);
                idxToFirstAdd.push_back(it.second.idxToAdd);
                cv::circle(mask, it.second.ptInForw, maskRadius, Scalar(0), -1);  // 标记掩模
            }
        }
    }

    void setMaskCell(int maskRadius = 0)
    {
        mask = Mat(imgCurr.rows, imgCurr.cols, CV_8UC1, Scalar(255));
        PairMask pm;
        vector<pair<int, PairMask>> vCountPtsId;
        for (size_t i = 0, iend = ptsForw.size(); i < iend; ++i) {
            pm.firstAdd = idFirstAdd[i];
            pm.idxToAdd = idxToFirstAdd[i];
            pm.ptInForw = ptsForw[i];
            pm.ptInCurr = ptsCurr[i];
            pm.ptInPrev = ptsPrev[i];
            pm.cellLabel = cellLable[i];
            vCountPtsId.push_back(make_pair(trackCount[i], pm));
        }
        sort(vCountPtsId.begin(), vCountPtsId.end(),
             [](const pair<int, PairMask>& a, const pair<int, PairMask>& b) {
                 return a.first > b.first;
             });

        ptsCurr.clear();
        ptsForw.clear();
        ptsPrev.clear();
        trackCount.clear();
        idFirstAdd.clear();
        idxToFirstAdd.clear();
        cellLable.clear();
        for (const auto& it : vCountPtsId) {
            if (it.second.firstAdd == KFRef->id) {
                trackCount.push_back(it.first);
                ptsForw.push_back(it.second.ptInForw);
                ptsCurr.push_back(it.second.ptInCurr);
                ptsPrev.push_back(it.second.ptInPrev);
                idFirstAdd.push_back(it.second.firstAdd);
                idxToFirstAdd.push_back(it.second.idxToAdd);
                cellLable.push_back(it.second.cellLabel);
                cv::circle(mask, it.second.ptInForw, 0, Scalar(0), -1);
                continue;
            }
            if (mask.at<uchar>(it.second.ptInForw) == 255) {
                trackCount.push_back(it.first);
                ptsForw.push_back(it.second.ptInForw);
                ptsCurr.push_back(it.second.ptInCurr);
                ptsPrev.push_back(it.second.ptInPrev);
                idFirstAdd.push_back(it.second.firstAdd);
                idxToFirstAdd.push_back(it.second.idxToAdd);
                cellLable.push_back(it.second.cellLabel);
                cv::circle(mask, it.second.ptInForw, maskRadius, Scalar(0), -1);  // 标记掩模
            } else {
                numInCell[it.second.cellLabel]--;
            }
        }
    }

    void drawNewAddPointsInMatch(Mat& image)
    {
        Point2f offset(imgPrev.cols, 0);
        for (size_t i = 0, iend = ptsNew.size(); i < iend; ++i)
            circle(image, ptsNew[i] + offset, 3, Scalar(255, 0, 255)); // 新点紫色
    }

    Mat drawMachesPointsToLastFrame(const string& title = "")
    {
        Mat imgCur, imgPre, imgMatchLast;
        cvtColor(imgPrev, imgPre, CV_GRAY2BGR);
        cvtColor(imgForw, imgCur, CV_GRAY2BGR);
        hconcat(imgPre, imgCur, imgMatchLast);

        Point2f offset(imgPrev.cols, 0);
        size_t N = ptsForw.size();
        for (size_t i = 0; i < N; ++i) {
            if (idFirstAdd[i] == KFRef->id) { // 从参考帧追下来的点标记黄色实心
                circle(imgMatchLast, ptsPrev[i], 3, Scalar(0, 255, 255), -1);
                circle(imgMatchLast, ptsForw[i] + offset, 3, Scalar(0, 255, 255), -1);
                line(imgMatchLast, ptsPrev[i], ptsForw[i] + offset, Scalar(220, 248, 255));
            } else { // 和上一帧的匹配点标记绿色
                circle(imgMatchLast, ptsPrev[i], 3, Scalar(0, 255, 0));
                circle(imgMatchLast, ptsForw[i] + offset, 3, Scalar(0, 255, 0));
                line(imgMatchLast, ptsPrev[i], ptsForw[i] + offset, Scalar(170, 150, 50));
            }
        }

        string str = title + ": F-F: " + to_string(frameLast.id) + "-" + to_string(idCurr) +
                     ", M:  " + to_string(N);
        putText(imgMatchLast, str, Point(15, 20), 1, 1, Scalar(0, 0, 255), 2);

        return imgMatchLast.clone();
    }

    Mat drawMachesPointsToRefFrame(const string& title = "")
    {
        Mat imgRef, imgCur, imgMatchRef;
        cvtColor(KFRef->mImage, imgRef, CV_GRAY2BGR);
        cvtColor(imgForw, imgCur, CV_GRAY2BGR);
        hconcat(imgRef, imgCur, imgMatchRef);

        Point2f offset(imgPrev.cols, 0);
//        for (const auto& m : matchIdxWithRefKF) {
//            Point2f p1 = ptsForw[m.first] + offset;
//            Point2f p2 = KFRef->mvKeyPoints[m.second].pt;
//            circle(imgMatchRef, p1, 3, Scalar(0, 255, 255));
//            circle(imgMatchRef, p2, 3, Scalar(0, 255, 255));
//            line(imgMatchRef, p1, p2, Scalar(220, 248, 255));
//        }
        for (size_t i = 0; i < KFRef->N; ++i) {
            if (matchIdx[i] < 0)
                continue;
            Point2f p1 = ptsForw[matchIdx[i]] + offset;
            Point2f p2 = KFRef->mvKeyPoints[i].pt;
            circle(imgMatchRef, p1, 3, Scalar(0, 255, 255));
            circle(imgMatchRef, p2, 3, Scalar(0, 255, 255));
            line(imgMatchRef, p1, p2, Scalar(220, 248, 255));
        }


        string str = title + ": " + to_string(KFRef->id) + "-" + to_string(idCurr) + "(" +
                to_string(idCurr - KFRef->id) + "), M: " + to_string(matchIdxWithRefKF.size());
        putText(imgMatchRef, str, Point(15, 20), 1, 1, Scalar(0, 0, 255), 2);

        return imgMatchRef.clone();
    }

    bool needNewKF(int matchedPtsWithRef)
    {
        //! 内点数太少要生成新的参考帧
//        if (matchedPtsWithRef <= 0.1 * Config::MaxFtrNumber)
//            return true;
        if (idCurr % 30 == 0)
            return true;
        return false;
    }

    void resetKltData()
    {
        // 当下一帧图像到来时，当前帧数据就成为了上一帧发布的数据
        imgCurr = imgForw.clone();  // 需要深拷贝
        ptsCurr = ptsForw;
        frameLast = frameCurr;
    }

    size_t countCellPts()
    {
        size_t num = 0;
        for (int i = 0; i < cells; ++i)
            num += numInCell[i];

        return num;
    }
};


int main(int argc, char* argv[])
{
    //! check input
    if (argc < 2) {
        fprintf(stderr, "Usage: test_kltTracking <dataPath> [number_frames_to_process]");
        exit(-1);
    }
    int num = INT_MAX;
    if (argc == 3) {
        num = atoi(argv[2]);
        cout << " - set number_frames_to_process = " << num << endl << endl;
    }

    //! initialization
    Config::readConfig(g_configPath);
    Mat K = Config::Kcam, D = Config::Dcam;

    string dataFolder = string(argv[1]) + "slamimg";
    vector<RK_IMAGE> imgFiles;
    readImagesRK(dataFolder, imgFiles);

    string odomRawFile = string(argv[1]) + "odo_raw.txt";  // [mm]
    ifstream rec(odomRawFile);
    if (!rec.is_open()) {
        cerr << "[Main ][Error] Please check if the file exists! " << odomRawFile << endl;
        rec.close();
        ros::shutdown();
        exit(-1);
    }
    float x, y, theta;
    string line;

    //! main loop
    KLT klt;
    fprintf(stderr, "\n - KLT Parameters:\n - Cell Size: %d x %d\n - Cells: %d x %d = %d\n - Max"
                    "points in cell: %d\n",
            klt.cellWidth, klt.cellHeight, klt.cellsX, klt.cellsY, klt.cells, klt.maxPtsInCell);
    Mat imgGray, imgUn, imgClahe;
    Ptr<CLAHE> clahe = createCLAHE(3.0, cv::Size(8, 8));
    num = std::min(num, static_cast<int>(imgFiles.size()));
    int skipFrames = 30;
    WorkTimer timer;
    for (int i = 0; i < num; ++i) {
        if (i < skipFrames) {
            std::getline(rec, line);
            continue;
        }

        std::getline(rec, line);
        istringstream iss(line);
        iss >> x >> y >> theta;
        Se2 odo(x, y, normalizeAngle(theta));

        imgGray = imread(imgFiles[i].fileName, CV_LOAD_IMAGE_GRAYSCALE);
        if (imgGray.data == nullptr)
            continue;
        clahe->apply(imgGray, imgClahe);
        cv::undistort(imgClahe, imgUn, K, D);

        timer.start();
        klt.track(imgUn, odo);
        //klt.trackCell(imgUn, odo);
        printf("#%ld 当前帧处理耗时: %.2fms\n", klt.idCurr, timer.count());
    }

    return 0;
}
