/*
 * @Description: LineDetection class
 * @Author: Maple.Liu
 * @Date: 2019-07-27
 * @LastEditTime: 2019-11-04 11:18:51
 * @LastEditors: Vance.Wu
 */

#include "LineDetection.h"

namespace se2lam
{

using namespace std;
using namespace cv;

LineDetection::LineDetection() : shiftX(2), shiftY(2), endX(238), endY(318)
{
    roiX = Range(shiftX, endX);
    roiY = Range(shiftY, endY);
    edlinesDetect = line_descriptor::BinaryDescriptor::createBinaryDescriptor();
}


LineDetection::~LineDetection()
{
    edlinesDetect.release();
}

/**
 * @brief: 获取直线掩模
 * @param {image}   输入图像
 * @param {lines}   检测到的线特征
 * @param {extend}  是否将线段扩展为直线
 * @return:         返回直线掩模
 *
 * @author: Maple.Liu
 */
Mat getLineMask(const Mat& image, vector<lineSort_S>& lines, bool extend)
{
    LineDetection linedetect;
    struct img_line imageline;
    Mat img = image.clone();
    Mat roiImage = img(linedetect.roiX, linedetect.roiY);
    img.copyTo(imageline.imgGray);
    //提取线特征
    linedetect.LineDetect(roiImage, 30.0, 40.0, imageline.linesTh1, imageline.linesTh2, lines);
    //获取mask
    Mat mask;
    Mat imgOut = Mat::zeros(image.rows, image.cols, CV_8UC1);
    Point2f star, end;

    for (int i = 0; i < imageline.linesTh1.size() / 4; ++i) {
        star.x = imageline.linesTh1(0, i);
        star.y = imageline.linesTh1(1, i);
        end.x = imageline.linesTh1(2, i);
        end.y = imageline.linesTh1(3, i);
        double k = double(end.y - star.y) / double(end.x - star.x);
        // cout<<"k="<<k<<endl;
        double b = star.y - k * star.x;
        //将线段扩展为直线
        if (extend) {
            if (lines[i].length > 80) {
                if (k < 1) {
                    star.x = 0;
                    star.y = int(b);
                    end.x = int(image.cols - 1);
                    end.y = int(k * end.x + b);
                } else {
                    star.x = int(-b / k);
                    star.y = 0;
                    end.x = int((image.rows - b - 1) / k);
                    end.y = image.rows - 1;
                }
                lines[i].star = star;
                lines[i].end = end;
            }
        }
        line(imgOut, star, end, Scalar(255, 255, 255), 8, CV_AA);
    }
    mask = imgOut.clone();
    return mask;
}

/**
 * @brief: 获取直线掩模
 * @param {image}   输入图像
 * @param {extend}  是否将线段扩展为直线
 * @return:         返回直线掩模
 *
 * @author: Maple.Liu
 */
Mat getLineMask(const Mat& image, bool extend)
{
    vector<lineSort_S> linefeatures;
    LineDetection linedetect;
    struct img_line imageline;
    Mat img = image.clone();
    Mat roiImage = img(linedetect.roiX, linedetect.roiY);
    img.copyTo(imageline.imgGray);
    //提取线特征
    linedetect.LineDetect(roiImage, 20.0, 40.0, imageline.linesTh1, imageline.linesTh2,
                          linefeatures);
    //获取mask
    Mat mask;
    Mat imgOut = Mat::zeros(image.rows, image.cols, CV_8UC1);
    Point2f star, end;

    for (int i = 0; i < imageline.linesTh1.size() / 4; i++) {
        star.x = imageline.linesTh1(0, i);
        star.y = imageline.linesTh1(1, i);
        end.x = imageline.linesTh1(2, i);
        end.y = imageline.linesTh1(3, i);
        double k = double(end.y - star.y) / double(end.x - star.x);
        // cout<<"k="<<k<<endl;
        double b = star.y - k * star.x;
        //将线段扩展为直线
        if (extend) {
            if (linefeatures[i].length > 80) {
                if (k < 1) {
                    star.x = 0;
                    star.y = int(b);
                    end.x = int(image.cols - 1);
                    end.y = int(k * end.x + b);
                } else {
                    star.x = int(-b / k);
                    star.y = 0;
                    end.x = int((image.rows - b - 1) / k);
                    end.y = image.rows - 1;
                }
                linefeatures[i].star = star;
                linefeatures[i].end = end;
            }
        }
        line(imgOut, star, end, Scalar(255, 255, 255), 4, CV_AA);
    }
    mask = imgOut.clone();
    return mask;
}

/***
 * @brief: 利用霍夫变换检测直线
 * @param {img}     输入图像
 * @param {extend}  是否将线段扩展为线
 * @return:         获得直线的掩模

 * @author: Maple.Liu
 */
Mat LineDetection::getLineMaskByHf(const Mat& img, bool entend)
{
    Mat gaus, edges, lineImage;
    vector<Vec4i> lines;
    //高斯模糊
    GaussianBlur(img, gaus, Size(3, 3), 0);
    // canny边缘检测
    Canny(gaus, edges, 40, 80, 3);
    //检测直线
    int minLineLength = 50;
    int maxLineGap = 10;
    HoughLinesP(edges, lines, 1, CV_PI / 180, 120, minLineLength, maxLineGap);
    Mat mask;
    Mat imgOut = Mat::zeros(img.rows, img.cols, CV_8UC1);
    for (size_t i = 0; i < lines.size(); ++i) {
        Vec4i L = lines[i];
        double k = double(L[3] - L[1]) / double(L[2] - L[0]);
        double b = L[1] - k * L[0];
        if (entend) {
            Point star, end;
            if (k < 1) {
                star.x = 0;
                star.y = int(b);
                end.x = int(img.cols - 1);
                end.y = int(k * end.x + b);
            } else {
                star.x = int(-b / k);
                star.y = 0;
                end.x = int((img.rows - b - 1) / k);
                end.y = img.rows - 1;
            }
            line(imgOut, star, end, Scalar(255, 255, 255), 3, CV_AA);
        } else
            line(imgOut, Point(L[0], L[1]), Point(L[2], L[3]), Scalar(255, 255, 255), 5, CV_AA);
    }
    mask = imgOut.clone();
    return mask;
}

//统计最多打四组线
void LineDetection::ComputeThreeMaxima(const vector<vector<int>>& rotHist, int lenth, int& ind1, int& ind2,
                                       int& ind3, int& ind4)
{
    for (int i = 0; i < lenth; ++i) {
        int n = rotHist[i].size();
        if (n > ind1) {
            ind4 = ind3;
            ind3 = ind2;
            ind2 = ind1;
            ind1 = i;
        } else if (n > ind2) {
            ind4 = ind3;
            ind2 = ind3;
            ind2 = n;
        } else if (n > ind3) {
            ind4 = ind3;
            ind3 = i;
        } else if (n > ind4) {
            ind4 = i;
        }
    }
}
//将线分类统计
void LineDetection::lineStatistics(double theta, int label, vector<vector<int>>& rotHist, int lenth,
                                   float factor)
{
    if (theta < 0.0)
        theta += 180.0f;
    int bin = round(theta * factor);
    if (bin == lenth)
        bin = 0;
    assert(bin >= 0 && bin < lenth);
    rotHist[bin].push_back(label);
}

//获取直线方程打斜率k和常数b
void LineDetection::getLineKandB(const Point& starPoint, const Point& endPoint, double& k, double& b)
{
    k = double(endPoint.y - starPoint.y) / double(endPoint.x - starPoint.x);
    b = starPoint.y - k * starPoint.x;
}

/*
 * 检测线段，计算线特征
 * image:输入图像
 * thLength1:线段长度阈值1
 * thLength2:线段长度阈值2
 * linesTh1:阈值1下检测打线段集合
 * linesTh2:阈值2下检测的线段集合
 * linefeatures:线段特征信息
 */
void LineDetection::LineDetect(const Mat& image, double thLength1, double thLength2,
                               Eigen::Matrix<double, 4, Eigen::Dynamic>& linesTh1,
                               Eigen::Matrix<double, 4, Eigen::Dynamic>& linesTh2,
                               vector<lineSort_S>& linefeatures)
{
    if (thLength1 > thLength2)
        cout << "Warning in LineDetect: thLength2 should be bigger than thLength1!" << endl
             << "Otherwise, linesTh2 will be empty!" << endl;

    vector<vector<double>> linesOut1;
    vector<vector<double>> linesOut2;
    vector<double> lineTemp(4);
    vector<lineSort_S> lineSort;
    lineSort_S lineSortTemp;
    int number;

#ifdef USE_EDLINE
    /* extract edlines */
    edlines.clear();

    edlinesDetect->detect(image, edlines);
    // edlinesDetect.release();
    number = edlines.size();
    lineSort.reserve(number);
    // cout<<number<<endl;
    for (int i = 0; i < number; ++i) {
        lineSortTemp.star.x = edlines[i].startPointX + shiftX;
        lineSortTemp.star.y = edlines[i].startPointY + shiftY;
        lineSortTemp.end.x = edlines[i].endPointX + shiftX;
        lineSortTemp.end.y = edlines[i].endPointY + shiftY;
        // lineSortTemp.length = sqrt((lineSortTemp.x1 - lineSortTemp.x2) * (lineSortTemp.x1 -
        // lineSortTemp.x2) + (lineSortTemp.y1 - lineSortTemp.y2) * (lineSortTemp.y1 -
        // lineSortTemp.y2));
        lineSortTemp.length = edlines[i].lineLength;
        getLineKandB(lineSortTemp.star, lineSortTemp.end, lineSortTemp.k, lineSortTemp.b);
        lineSort.push_back(lineSortTemp);
    }
    sort(lineSort.begin(), lineSort.end(), lineSort_compareClass());  //从大到小排序
    vector<vector<double>> linesOut1o;
    linesOut1o.reserve(number);
    for (int i = 0; i < number; ++i) {
        double x1 = lineSort[i].star.x;
        double y1 = lineSort[i].star.y;
        double x2 = lineSort[i].end.x;
        double y2 = lineSort[i].end.y;
        double l = lineSort[i].length;
        if (l > thLength1) {
            // cout<<"lineSort[i].length"<<lineSort[i].length<<endl;
            lineTemp[0] = x1;
            lineTemp[1] = y1;
            lineTemp[2] = x2;
            lineTemp[3] = y2;
            linesOut1.push_back(lineTemp);
            linefeatures.push_back(lineSort[i]);
            // cout<<lineTemp[0]<<endl;
            //计算线的斜率k和常数b
            if (l > thLength2)
                linesOut2.push_back(lineTemp);
        }
    }

#else
    // use lsd
    Mat src_gray;
    double* imageLSD;
    double* linesLSD;
    image.convertTo(src_gray, CV_64FC1);
    imageLSD = src_gray.ptr<double>(0);

    int xsize = image.cols;
    int ysize = image.rows;

    /* LSD call */
    linesLSD = lsd(&number, imageLSD, xsize, ysize);
    lineSort.reserve(number);
    for (int i = 0; i < number; ++i) {
        lineSortTemp.x1 = linesLSD[7 * i + 0];
        lineSortTemp.y1 = linesLSD[7 * i + 1];
        lineSortTemp.x2 = linesLSD[7 * i + 2];
        lineSortTemp.y2 = linesLSD[7 * i + 3];
        lineSortTemp.length =
            sqrt((lineSortTemp.x1 - lineSortTemp.x2) * (lineSortTemp.x1 - lineSortTemp.x2) +
                 (lineSortTemp.y1 - lineSortTemp.y2) * (lineSortTemp.y1 - lineSortTemp.y2));
        lineSort.push_back(lineSortTemp);
    }
    sort(lineSort.begin(), lineSort.end(), lineSort_compareClass());  // ��������
    linesOut1.reserve(number);
    linesOut2.reserve(number);
    for (int i = 0; i < number; ++i) {
        double x1 = lineSort[i].x1;
        double y1 = lineSort[i].y1;
        double x2 = lineSort[i].x2;
        double y2 = lineSort[i].y2;
        double l = lineSort[i].length;
        if (l > thLength1) {
            lineTemp[0] = x1 + shiftX;
            lineTemp[1] = y1 + shiftY;
            lineTemp[2] = x2 + shiftX;
            lineTemp[3] = y2 + shiftY;
            linesOut1.push_back(lineTemp);
            if (l > thLength2) {
                linesOut2.push_back(lineTemp);
            }
        }
    }
    free(linesLSD);
#endif

    int lineNum1 = linesOut1.size();
    int lineNum2 = linesOut2.size();
#ifdef SHOW_LOG
    printf("%d line1 segments found.\n", lineNum1);
    printf("%d line2 segments found.\n", lineNum2);
#endif

    linesTh1.resize(4, lineNum1);
    for (int i = 0; i < lineNum1; ++i) {
        linesTh1(0, i) = linesOut1[i][0];
        linesTh1(1, i) = linesOut1[i][1];
        linesTh1(2, i) = linesOut1[i][2];
        linesTh1(3, i) = linesOut1[i][3];
    }

    linesTh2.resize(4, lineNum2);
    for (int i = 0; i < lineNum2; ++i) {
        linesTh2(0, i) = linesOut2[i][0];
        linesTh2(1, i) = linesOut2[i][1];
        linesTh2(2, i) = linesOut2[i][2];
        linesTh2(3, i) = linesOut2[i][3];
    }
}
/*
 * 获取离输入点最近打线段
 * x:输入点x坐标
 * y:输入点y坐标
 * lenth:输出点距离线段打距离
 * linefeatures:输入线段打特征信息
 * return:输出线段标签
 */
int LineDetection::pointInwhichLine(double x, double y, double& lenth,
                                    vector<lineSort_S>& linefeatures)
{
    int lnum = 0;
    double Lmin = 500;
    double L;
    int i = 0;
    // cout<<"x="<<x<<endl;
    for (vector<lineSort_S>::iterator keyline = linefeatures.begin(),
                                      keylineend = linefeatures.end();
         keyline != keylineend; ++keyline) {
        double dx = abs(keyline->star.x - keyline->end.x);
        double dy = abs(keyline->star.y - keyline->end.y);
        bool judge = dx > dy;
        if (judge) {
            if ((x > keyline->star.x - 5 && x < keyline->end.x + 5) ||
                (x < keyline->star.x + 5 && x > keyline->end.x - 5))
                goto part1; // FIXME 不要用goto
            else {
                ++i;
                continue;
            }
        } else {
            if ((y > keyline->star.y - 5 && y < keyline->end.y + 5) ||
                (y < keyline->star.y + 5 && y > keyline->end.y - 5))
                goto part1;
            else {
                ++i;
                continue;
            }
        }
    part1:
        double k = keyline->k;
        double b = keyline->b;
        if (isinf(k)) {
            L = abs(x - keyline->star.x);
        } else {
            L = abs(k * x - y + b) / sqrt(k * k + 1);
        }
        if (L < Lmin) {
            Lmin = L;
            lnum = i;
            // cout<<"n="<<i<<endl;
        }
        ++i;
    }
    lenth = Lmin;
    return lnum;
}

}  // namespace se2lam
