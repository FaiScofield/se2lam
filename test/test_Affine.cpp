#include "test_functions.hpp"

#define FIX_DELTA_FRAME 1
#if FIX_DELTA_FRAME
const int g_delta_frame = 15;
#endif

// string g_configPath = "/home/vance/dataset/se2/se2_config/";
string g_configPath = "/home/vance/dataset/rk/dibeaDataSet/se2_config/";
string g_orbVocFile = "/home/vance/slam_ws/ORB_SLAM2/Vocabulary/ORBvoc.bin";
string g_matchResult = "/home/vance/output/rk_se2lam/test_matchResult.txt";

Frame frameCur, frameRef;


//! FIXME 手写的 RANSAC 需要改进, 内点数不能太少, 应该要保持一个下限
//! 目前来看这种方法修正后得到的Affine可以达到最好的效果
Mat estimateAffineMatrix(const vector<KeyPoint>& kpRef, const vector<KeyPoint>& kpCur, vector<int>& matches12)
{
    vector<Point2f> ptRef, ptCur;
    vector<int> idxRef;
    idxRef.reserve(kpRef.size());
    ptRef.reserve(kpRef.size());
    ptCur.reserve(kpCur.size());
    for (int i = 0, iend = kpRef.size(); i < iend; ++i) {
        if (matches12[i] < 0)
            continue;
        idxRef.push_back(i);
        ptRef.push_back(kpRef[i].pt);
        ptCur.push_back(kpCur[matches12[i]].pt);
    }

    Se2 dOdom = frameCur.odom - frameRef.odom;
    Mat R = getRotationMatrix2D(Point2f(0, 0), dOdom.theta * 180.f / CV_PI, 1.).colRange(0, 2);

    const size_t N = ptCur.size();
    vector<uchar> inlineMask(N, 1);
    Mat t = Mat::zeros(2, 1, CV_64FC1);
    Mat J = -R;

    int inliers = N;
    double errorSumLast = 9999999;
    for (int it = 0; it < 10; ++it) {
        Mat H = Mat::zeros(2, 2, CV_64FC1);
        Mat b = Mat::zeros(2, 1, CV_64FC1);
        double errorSum = 0;
        for (size_t i = 0; i < N; ++i) {
            if (inlineMask[i] == 0)
                continue;

            Mat x1, x2;
            Mat(ptRef[i]).convertTo(x1, CV_64FC1);
            Mat(ptCur[i]).convertTo(x2, CV_64FC1);
            Mat e = x2 - R * x1 - t;
            H += J.t() * J;
            b += J.t() * e;
            errorSum += norm(e);

            if (it > 3 && norm(e) > 5.991) {
                inlineMask[i] = 0;
                inliers--;
            }
        }

        Mat dt = -H.inv() * b;
        t += dt;

        cout << "iter = " << it << ", inliers = " << inliers << ", ave chi = "
             << errorSum / inliers << ", t = " << t.t() << endl;
        if (errorSumLast < errorSum) {
            t -= dt;
            break;
        }
        if (errorSumLast - errorSum < 1e-6)
            break;
        errorSumLast = errorSum;
    }
    Mat A21(2, 3, CV_64FC1);
    R.copyTo(A21.colRange(0, 2));
    t.copyTo(A21.col(2));

    for (size_t i = 0; i < inlineMask.size(); ++i) {
        if (inlineMask[i] == 0)
            matches12[idxRef[i]] = -1;
    }


    return A21;
}

void calculateAffineMatrixSVD(const vector<Point2f>& ptRef, const vector<Point2f>& ptCur,
                              const vector<uchar>& inlineMask, Mat& Affine)
{
    assert(ptRef.size() == ptCur.size());
    assert(ptRef.size() == inlineMask.size());

    Affine = Mat::eye(2, 3, CV_64FC1);

    // 1.求质心
    Point2f p1_c(0.f, 0.f), p2_c(0.f, 0.f);
    size_t N = 0;
    for (size_t i = 0, iend = inlineMask.size(); i < iend; ++i) {
        if (inlineMask[i] == 0)
            continue;
        p1_c += ptRef[i];
        p2_c += ptCur[i];
        N++;
    }
    if (N < 5) {
        cerr << "内点数太少, 无法通过svd计算A" << endl;
        return;
    }

    p1_c.x /= N;
    p1_c.y /= N;
    p2_c.x /= N;
    p2_c.y /= N;

    // 2.构造超定矩阵A
    Mat A = Mat::zeros(2, 2, CV_32FC1);
    for (size_t i = 0; i < N; ++i) {
        if (inlineMask[i] == 0)
            continue;
        Mat p1_i = Mat(ptRef[i] - p1_c);
        Mat p2_i = Mat(ptCur[i] - p2_c);
        A += p1_i * p2_i.t();
    }

    // 3.用SVD分解求得R,t
    Mat U, W, Vt;
    SVD::compute(A, W, U, Vt);
    Mat R = U * Vt;
    Mat t = Mat(p1_c) - R * Mat(p2_c);

    // 求得的RT是A12, 即第2帧到第1帧的变换
    R.copyTo(Affine.colRange(0, 2));
    t.copyTo(Affine.col(2));

    //invertAffineTransform(Affine, Affine);
    //cerr << "根据" << N << "个内点计算出A12:" << endl << Affine << endl;
}

int removeOutliersWithRansac(const vector<KeyPoint>& kpRef, const vector<KeyPoint>& kpCur,
                             vector<int>& matches12, Mat& Asvd, int inlineTh = 3)
{
    assert(kpRef.size() == kpCur.size());
    assert(kpRef.size() >= 10);
    assert(!matches12.empty());

    vector<Point2f> ptRef, ptCur;
    vector<int> idxRef, matches12Good;
    idxRef.reserve(kpRef.size());
    ptRef.reserve(kpRef.size());
    ptCur.reserve(kpCur.size());
    for (int i = 0, iend = kpRef.size(); i < iend; ++i) {
        if (matches12[i] < 0)
            continue;
        idxRef.push_back(i);
        ptRef.push_back(kpRef[i].pt);
        ptCur.push_back(kpCur[matches12[i]].pt);
    }
    matches12Good = matches12;

    // Ransac
    int inliers = 0, lastInliers = 0;
    double error = 0, lastError = 99999;
    Mat Affine;
    Mat lastAffine = Mat::ones(2, 3, CV_64FC1);
    size_t N = ptRef.size(), i = 0;
    vector<uchar> inlineMask(ptRef.size(), 1);
    for (; i < 10; ++i) {
        inliers = 0;
        error = 0;

        calculateAffineMatrixSVD(ptRef, ptCur, inlineMask, Affine);  // Affine即A12

        for (size_t j = 0; j < N; ++j) {
            if (inlineMask[j] == 0)
                continue;

            const Mat pt2 = (Mat_<double>(3, 1) << ptCur[j].x, ptCur[j].y, 1);
            const Mat pt2W = Affine * pt2;
            const Point2f ptCurWarpj = Point2f(pt2W.at<double>(0), pt2W.at<double>(1));

            double ej = norm(ptRef[j] - ptCurWarpj);

            if (ej < sqrt(5.991)) {
                inlineMask[j] = 1;
                error += ej;
                inliers++;
            } else {
                inlineMask[j] = 0;
                matches12Good[idxRef[j]] = -1;
            }
        }

        cout << "#" << frameCur.id << " iter = " << i << ", 内外点数 = " << inliers << "/"
             << ptRef.size() - inliers << ", 当前平均误差为: " << error/inliers << endl;

        if (inliers == 0) {  // 如果没有内点
            cout << "#" << frameCur.id << " iter = " << i << ", 内点数为0, 即将退出循环." << endl;
            Affine = lastAffine;
            error = lastError;
            inliers = lastInliers;
            break;
        }

        error /= inliers;

        double e2 = 0;
        Mat A21, show;
        invertAffineTransform(Affine, A21);
        show = drawKPMatchesAGood(&frameRef, &frameCur, matches12, matches12Good, A21, e2);
        imshow("tmp svd", show);

        if (error > lastError) {  // 如果误差变大就用上一次的结果
            cout << "#" << frameCur.id << " iter = " << i << ", 误差变大结果变坏, 即将退出循环: last/curr error = "
                 << lastError << "/" << error << endl;
            Affine = lastAffine;
            error = lastError;
            inliers = lastInliers;
            break;
        }

        if (error < inlineTh) { // 如果误差足够小则退出
            cout << "#" << frameCur.id << " iter = " << i << ", 误差足够小, 即将退出循环: curr error = "
                 << error << endl;
            break;
        }

        if (lastError - error < 1e-4) {  // 如果误差下降不明显则退出
            cout << "#" << frameCur.id << " iter = " << i << ", 误差降低不明显, 即将退出循环: last/curr error = "
                 << lastError << "/" << error << endl;
            break;
        }

        lastError = error;
        lastAffine = Affine;
        lastInliers = inliers;
    }

    invertAffineTransform(Affine, Asvd);

    return inliers;
}


int main(int argc, char** argv)
{
    //! check input
    if (argc < 2) {
        fprintf(stderr, "Usage: test_pointDetection <dataPath> [number_frames_to_process]");
        exit(-1);
    }
    int num = INT_MAX;
    if (argc == 3) {
        num = atoi(argv[2]);
        cout << "set number_frames_to_process = " << num << endl << endl;
    }

    //! initialization
    Config::readConfig(g_configPath);

    string dataFolder = string(argv[1]) + "slamimg";
    vector<RK_IMAGE> imgFiles;
    readImagesRK(dataFolder, imgFiles);

    string odomRawFile = string(argv[1]) + "odo_raw.txt";  // [mm]
    ifstream rec(odomRawFile);
    if (!rec.is_open()) {
        cerr << "[Main ][Error] Please check if the odo_raw file exists!" << endl;
        rec.close();
        exit(-1);
    }

    ORBextractor* kpExtractor = new ORBextractor(500, 1, 1, 1, 20);
    ORBmatcher* kpMatcher = new ORBmatcher(0.9);  // 0.6
    ORBVocabulary* vocabulary = new ORBVocabulary();
    bool bVocLoad = vocabulary->loadFromBinaryFile(g_orbVocFile);
    if (!bVocLoad) {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Falied to open at: " << g_orbVocFile << endl;
        exit(-1);
    }
    cout << "Vocabulary loaded!" << endl << endl;


    //! main loop
    bool firstFrame = true;
    double projError1 = 0, projError2 = 0, projError3 = 0, projError4 = 0;
    //Frame frameCur, frameRef;
    PtrKeyFrame KFCur, KFRef;
    Mat imgUn, imgGray, imgCur, imgRef, imgWithFeatureCur, imgWithFeatureRef;
    Mat outImgWarpA, outImgWarpAP, outImgWarpRT, outImgWarpAsvd, outImgConcat;
    Mat A21 = Mat::eye(2, 3, CV_64FC1);
    Mat A21Partial = Mat::eye(2, 3, CV_64FC1);
    Mat RT = Mat::eye(2, 3, CV_64FC1);
    Mat Asvd = Mat::eye(2, 3, CV_64FC1);
    num = std::min(num, static_cast<int>(imgFiles.size()));
    int skipFrames = 50;
    for (int i = 0; i < num; ++i) {
        string line;
        Se2 odo;
        if (i < skipFrames) {
            std::getline(rec, line);
            continue;
        }
        std::getline(rec, line);
        istringstream iss(line);
        iss >> odo.x >> odo.y >> odo.theta;

        imgGray = imread(imgFiles[i].fileName, CV_LOAD_IMAGE_GRAYSCALE);
        if (imgGray.data == nullptr)
            continue;
        cv::undistort(imgGray, imgUn, Config::Kcam, Config::Dcam);
        Ptr<CLAHE> clahe = createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(imgUn, imgCur);

        //! ORB提取特征点
        frameCur = Frame(imgCur, odo, kpExtractor, imgFiles[i].timeStamp);
        KFCur = make_shared<KeyFrame>(frameCur);

        if (firstFrame) {
            imgCur.copyTo(imgRef);
            imgWithFeatureCur.copyTo(imgWithFeatureRef);
            frameRef = frameCur;
            KFRef = KFCur;
            firstFrame = false;
            continue;
        }


        //! 特征点匹配
        int nMatched1(0), nMatched2(0), nMatched3(0), nMatched4(0);
        int nInliers1(0), nInliers2(0), nInliers3(0), nInliers4(0);
        vector<int> matchIdxA, matchIdxAP, matchIdxRT, matchIdxAsvd;
        nMatched1 = kpMatcher->MatchByWindowWarp(frameRef, frameCur, A21, matchIdxA, 25);
        nMatched2 = kpMatcher->MatchByWindowWarp(frameRef, frameCur, A21Partial, matchIdxAP, 25);
        nMatched3 = kpMatcher->MatchByWindowWarp(frameRef, frameCur, RT, matchIdxRT, 25);
        nMatched4 = kpMatcher->MatchByWindowWarp(frameRef, frameCur, Asvd, matchIdxAsvd, 25);

        if (nMatched1 >= 10)
            nInliers1 = removeOutliersWithA(frameRef.mvKeyPoints, frameCur.mvKeyPoints, matchIdxA, A21, 0);
        if (nMatched2 >= 10)
            nInliers2 = removeOutliersWithA(frameRef.mvKeyPoints, frameCur.mvKeyPoints,  matchIdxAP, A21Partial, 1);
        if (nMatched3 >= 10) {
            //nInliers3 = removeOutliersWithA(frameRef.mvKeyPoints, frameCur.mvKeyPoints, matchIdxRT, RT, 2);
            RT = estimateAffineMatrix(frameRef.mvKeyPoints, frameCur.mvKeyPoints, matchIdxRT);
            nInliers3 = nMatched3;
        }
        if (nMatched4 >= 10)
            nInliers4 = removeOutliersWithRansac(frameRef.mvKeyPoints, frameCur.mvKeyPoints, matchIdxAsvd, Asvd);


//        cout << "A = " << endl << A21 << endl;
//        cout << "AP = " << endl << A21Partial << endl;
//        cout << "RT = " << endl << RT << endl;
//        cout << "Asvd = " << endl << Asvd << endl;

        // 对A SVD分解
//        Mat U, W, Vt;
//        SVD::compute(A12, W, U, Vt);
//        cout << " W of A (SVD) " << endl << W << endl;
//        cout << " U of A (SVD) " << endl << U << endl;
//        cout << " Vt of A (SVD) " << endl << Vt << endl;
//        cout << " U * Vt = " << endl << U * Vt << endl;

        //! 匹配情况可视化
        outImgWarpA = drawKPMatchesA(&frameRef, &frameCur, matchIdxA, A21, projError1);
        outImgWarpAP = drawKPMatchesA(&frameRef, &frameCur, matchIdxAP, A21Partial, projError2);
        outImgWarpRT = drawKPMatchesA(&frameRef, &frameCur, matchIdxRT, RT, projError3);
        outImgWarpAsvd = drawKPMatchesA(&frameRef, &frameCur, matchIdxAsvd, Asvd, projError4);

        char strMatches1[64], strMatches2[64], strMatches3[64], strMatches4[64];
        std::snprintf(strMatches1, 64, "A   F: %ld-%ld, M: %d/%d, E: %.2f", frameCur.id, frameRef.id, nInliers1, nMatched1, projError1);
        std::snprintf(strMatches2, 64, "AP  F: %ld-%ld, M: %d/%d, E: %.2f", frameCur.id, frameRef.id, nInliers2, nMatched2, projError2);
        std::snprintf(strMatches3, 64, "RT  F: %ld-%ld, M: %d/%d, E: %.2f", frameCur.id, frameRef.id, nInliers3, nMatched3, projError3);
        std::snprintf(strMatches4, 64, "Asvd  F: %ld-%ld, M: %d/%d, E: %.2f", frameCur.id, frameRef.id, nInliers4, nMatched4, projError4);
        putText(outImgWarpA, strMatches1, Point(15, 15), 1, 1, Scalar(0, 0, 255), 2);
        putText(outImgWarpAP, strMatches2, Point(15, 15), 1, 1, Scalar(0, 0, 255), 2);
        putText(outImgWarpRT, strMatches3, Point(15, 15), 1, 1, Scalar(0, 0, 255), 2);
        putText(outImgWarpAsvd, strMatches4, Point(15, 15), 1, 1, Scalar(0, 0, 255), 2);
        printf("#%ld 平均重投影误差A/AP/RT = %.2f/%.2f/%.2f/%.2f\n", frameCur.id, projError1, projError2, projError3, projError4);

        Mat imgTmp1, imgTmp2;
        vconcat(outImgWarpA, outImgWarpAP, imgTmp1);
        vconcat(outImgWarpRT, outImgWarpAsvd, imgTmp2);
        hconcat(imgTmp1, imgTmp2, outImgConcat);
        resize(outImgConcat, outImgConcat, Size2i(outImgConcat.cols * 1.5, outImgConcat.rows * 1.5));
        imshow("A & AP & RT Matches", outImgConcat);


//        // test A
//        Se2 dOdom = frameRef.odom - frameCur.odom;
//        Mat R0 = getRotationMatrix2D(Point2f(0, 0), dOdom.theta * 180.f / CV_PI, 1.);
//        cout << "delta angle = " << (dOdom.theta / CV_PI * 180) << endl;
//        cout << "R0 = " << endl << R0 << endl;

//        Mat Tc1c2 = Config::Tcb * dOdom.inv().toCvSE3() * Config::Tbc;  // 4x4
//        Mat Rc1c2 = Tc1c2.rowRange(0, 3).colRange(0, 3).clone();  // 3x3
//        Mat tc1c2 = Tc1c2.rowRange(0, 3).col(3).clone();  // 3x1
//        Mat R = Config::Kcam * Rc1c2 *
//                (Config::Kcam).inv();  // 3x3 这里相当于仿射变换里的A, 不过少了部分平移信息
//        Mat t = Config::Kcam * tc1c2 / 3000.f;  // 3x1 深度假设3m
//        cout << "R = " << endl << R.rowRange(0, 2) << endl;
//        cout << "R的缩放因子 = " << R.at<float>(0, 0) / cos(dOdom.theta) << endl;

//        // A1 - 通过相对变换关系计算得到
//        Mat A1;
//        R.rowRange(0, 2).convertTo(A1, CV_64FC1);
//        R0.colRange(0, 2).copyTo(A1.colRange(0, 2));  // 去掉尺度变换
//        A1.at<double>(0, 2) += t.at<float>(0, 0);  // 加上平移对图像造成的影响
//        A1.at<double>(1, 2) += t.at<float>(1, 0);
//        cout << "A1 = " << endl << A1 << endl;

//        // A2 - 旋转部分通过先验计算得到
//        Point2f rotationCenter;
//        rotationCenter.x = Config::cx - Config::Tbc.at<float>(1, 3) / 12;
//        rotationCenter.y = Config::cy - Config::Tbc.at<float>(0, 3) / 12;
//        Mat A2 = getRotationMatrix2D(rotationCenter, dOdom.theta * 180.f / CV_PI, 1.);
//        // A2.at<double>(0, 2) += t.at<float>(0, 0);
//        // A2.at<double>(1, 2) += t.at<float>(1, 0);
//        cout << "A2 = " << endl << A2 << endl;

//        // A21 - 通过特征点计算出来的仿射变换矩阵
//        Mat A21;
//        invertAffineTransform(A12, A21);
//        cout << "A21 = " << endl << A21 << endl;
//        cout << "A21的缩放因子 = " << A21.at<double>(0, 0) / cos(dOdom.theta) << endl;

//        Mat imgA1, imgA2, imgA3, imgC1, imgC2, imgC3;
//        warpAffine(imgCur, imgA1, A2, imgCur.size());
//        warpAffine(imgCur, imgA2, A1, imgCur.size());
//        warpAffine(imgCur, imgA3, A21, imgCur.size());
//        hconcat(imgA1, imgA2, imgC1);
//        hconcat(imgRef, imgA3, imgC2);
//        hconcat(imgC1, imgC2, imgC3);
//        imshow("TEST Affine", imgC3);

        string fileWarp = "/home/vance/output/rk_se2lam/warp/" + to_string(frameCur.id) + ".jpg";
        imwrite(fileWarp, outImgConcat);

        waitKey(30);


        //! 内点数太少要生成新的参考帧
#if FIX_DELTA_FRAME
        if (frameCur.id % g_delta_frame == 0) {
#else
        if (nInliers1 <= 12 || nInliers2 <= 12 || nInliers4 <= 10) {
#endif
            imgCur.copyTo(imgRef);
            imgWithFeatureCur.copyTo(imgWithFeatureRef);
            frameRef = frameCur;
            KFRef = KFCur;
            A21 = Mat::eye(2, 3, CV_64FC1);
            A21Partial = Mat::eye(2, 3, CV_64FC1);
            RT = Mat::eye(2, 3, CV_64FC1);
            Asvd = Mat::eye(2, 3, CV_64FC1);
        }
    }
    rec.close();

    delete kpExtractor;
    delete kpMatcher;
    delete vocabulary;
    return 0;
}


void writeMatchData(const string outFile, const vector<vector<int>>& vvMatches, const vector<vector<int>>& vvInliners)
{
    assert(vvMatches.size() == vvInliners.size());

    ofstream ofs(outFile);
    if (!ofs.is_open()) {
        cerr << "Open file error: " << outFile << endl;
        return;
    }

    int n = vvMatches.size();
    int sum[n] = {0}, sumIn[n] = {0};
    for (int i = 0; i < n; ++i) {
        ofs << i << " ";
        for (size_t j = 0; j < vvMatches[i].size(); ++j) {
            ofs << vvMatches[i][j] << " ";
            sum[i] += vvMatches[i][j];
        }
        ofs << endl;
    }
    ofs << endl;
    for (int i = 0; i < n; ++i)
        cout << "Tatal Match Average " << i << ": " << 1.0 * sum[i] / vvMatches[i].size() << endl;

    for (int i = 0; i < n; ++i) {
        ofs << i << " ";
        for (size_t j = 0; j < vvInliners[i].size(); ++j) {
            ofs << vvInliners[i][j] << " ";
            sumIn[i] += vvInliners[i][j];
        }
        ofs << endl;
    }
    ofs.close();
    for (int i = 0; i < n; ++i)
        cout << "Tatal Inliners Average " << i << ": " << 1.0 * sumIn[i] / vvInliners[i].size() << endl;

    cerr << "Write match data to file: " << outFile << endl;
}
