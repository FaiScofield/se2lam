#include "test_functions.hpp"

// string g_configPath = "/home/vance/dataset/se2/se2_config/";
string g_configPath = "/home/vance/dataset/rk/dibeaDataSet/se2_config/";
string g_orbVocFile = "/home/vance/slam_ws/ORB_SLAM2/Vocabulary/ORBvoc.bin";
string g_matchResult = "/home/vance/output/rk_se2lam/test_matchResult.txt";
Ptr<DescriptorMatcher> g_CVMatcher = DescriptorMatcher::create("BruteForce-Hamming");

int matchByCV(const Mat& desRef, const Mat& desCur, map<int, int>& matchCV)
{
    vector<DMatch> matches;
    g_CVMatcher->match(desRef, desCur, matches);

    // 过滤匹配点
    float max_dist = 0.f;
    float min_dist = 10000.f;

    for (int i = 0, iend = desRef.rows; i < iend; ++i) {
        float dist = matches[i].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }
    float delta_dist = std::max(std::min(2.f * min_dist, 0.5f * max_dist), 20.f);

    for (int i = 0, iend = desRef.rows; i < iend; ++i) {
        if (matches[i].distance < delta_dist)
            matchCV[matches[i].queryIdx] = matches[i].trainIdx;
    }

    return matchCV.size();
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
    Mat K = Config::Kcam, D = Config::Dcam;

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

    // string dataFolder = string(argv[1]) + "image/";
    // vector<string> imgFiles;
    // readImagesSe2(dataFolder, imgFiles);

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
    size_t nMatchesAll3(0), nMatchesAll4(0), nMatchesAll5(0);
    size_t nInliersAll3(0), nInliersAll4(0), nInliersAll5(0);
    float aveMatches3, aveMatches4, aveMatches5;
    float aveInliers3, aveInliers4, aveInliers5;

    Frame frameCur, frameRef;
    PtrKeyFrame KFCur, KFRef;
    Mat imgColor, imgGray, imgCur, imgRef, imgWithFeatureCur, imgWithFeatureRef;
    Mat outImgORBMatch, outImgWarpH, outImgWarpA, outImgBow, outImgCV, outImgConcat;
    Mat H12 = Mat::eye(3, 3, CV_64FC1);
    Mat A12 = Mat::eye(2, 3, CV_64FC1);
    Mat I3 = Mat::eye(3, 3, CV_64FC1);
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

        imgColor = imread(imgFiles[i].fileName, CV_LOAD_IMAGE_COLOR);
        // imgColor = imread(imgFiles[i], CV_LOAD_IMAGE_COLOR);
        if (imgColor.data == nullptr)
            continue;
        cv::undistort(imgColor, imgCur, Config::Kcam, Config::Dcam);
        cvtColor(imgCur, imgGray, CV_BGR2GRAY);

        //! 限制对比度自适应直方图均衡
        Mat imgClahe;
        Ptr<CLAHE> clahe = createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(imgGray, imgClahe);

        //! ORB提取特征点
        WorkTimer timer;
        frameCur = Frame(imgClahe, odo, kpExtractor, imgFiles[i].timeStamp);
        // frameCur = Frame(imgClahe, odo, kpExtractor, 0);
        KFCur = make_shared<KeyFrame>(frameCur);
        KFCur->computeBoW(vocabulary);
        imgCur.copyTo(imgWithFeatureCur);
        for (int i = 0, iend = frameCur.N; i < iend; ++i)
            circle(imgWithFeatureCur, frameCur.mvKeyPoints[i].pt, 3, Scalar(255, 0, 0));
        printf("#%ld 创建KF,提取特征点,计算词向量,标记KP共耗时%.2fms\n", frameCur.id, timer.count());

        if (firstFrame) {
            imgCur.copyTo(imgRef);
            imgWithFeatureCur.copyTo(imgWithFeatureRef);
            frameRef = frameCur;
            KFRef = KFCur;
            firstFrame = false;
            continue;
        }

        //! 特征点匹配
        int nMatched1(0), nMatched2(0), nMatched3(0), nMatched4(0), nMatched5(0);
        int nInliers1(0), nInliers2(0), nInliers3(0), nInliers4(0), nInliers5(0);
        vector<int> matchIdx, matchIdxH, matchIdxA;
        map<int, int> matchIdxBow, matchIdxCV;
        vector<Point2f> prevMatched;
        KeyPoint::convert(frameRef.mvKeyPoints, prevMatched);

        timer.start();
        nMatched1 = matchByCV(frameRef.mDescriptors, frameCur.mDescriptors, matchIdxCV);
        // printf("#%ld 匹配方式 matchByCV() 耗时%.2fms\n", frameCur.id, timer.count());

        timer.start();
        nMatched2 = kpMatcher->SearchByBoW(KFRef, KFCur, matchIdxBow, false);
        // printf("#%ld 匹配方式 SearchByBoW() 耗时%.2fms\n", frameCur.id, timer.count());

        timer.start();
        nMatched3 = kpMatcher->MatchByWindow(frameRef, frameCur, prevMatched, 25, matchIdx);
        // printf("#%ld 匹配方式 MatchByWindow() 耗时%.2fms\n", frameCur.id, timer.count());
        nMatchesAll3 += nMatched3;

        timer.start();
        nMatched4 = kpMatcher->MatchByWindowWarp(frameRef, frameCur, H12, matchIdxH, 25);
        // printf("#%ld 匹配方式 MatchByWindowWarpH() 耗时%.2fms\n", frameCur.id, timer.count());
        nMatchesAll4 += nMatched4;

        timer.start();
        nMatched5 = kpMatcher->MatchByWindowWarp(frameRef, frameCur, A12, matchIdxA, 25);
        // printf("#%ld 匹配方式 MatchByWindowWarpA() 耗时%.2fms\n", frameCur.id, timer.count());
        nMatchesAll5 += nMatched5;

        timer.start();
        if (nMatched1 >= 10)
            nInliers1 = removeOutliersWithHF(frameRef.mvKeyPoints, frameCur.mvKeyPoints, matchIdxCV, I3);
        if (nMatched2 >= 10)
            nInliers2 = removeOutliersWithHF(frameRef.mvKeyPoints, frameCur.mvKeyPoints, matchIdxBow, I3);
        if (nMatched3 >= 10) {
            nInliers3 = removeOutliersWithHF(frameRef.mvKeyPoints, frameCur.mvKeyPoints, matchIdx, I3);
            nInliersAll3 += nInliers3;
        }
        if (nMatched4 >= 10) {
            nInliers4 = removeOutliersWithHF(frameRef.mvKeyPoints, frameCur.mvKeyPoints, matchIdxH, H12);
            nInliersAll4 += nInliers4;
        } else
            H12 = Mat::eye(3, 3, CV_64FC1);
        if (nMatched5 >= 10) {
            nInliers5 = removeOutliersWithA(frameRef.mvKeyPoints, frameCur.mvKeyPoints, matchIdxA, A12);
            nInliersAll5 += nInliers5;
        }
        // printf("#%ld 剔除外点总耗时%.2fms\n", frameCur.id, timer.count());

        //! 存储匹配结果作分析H
        //        vvMatches[idx].push_back(nMatched2);
        //        vvInliners[idx].push_back(nInlines2);

        //! 匹配情况可视化
         outImgCV = drawKPMatches(KFRef, KFCur, imgWithFeatureRef, imgWithFeatureCur, matchIdxCV);
         outImgBow = drawKPMatches(KFRef, KFCur, imgWithFeatureRef, imgWithFeatureCur, matchIdxBow);
         outImgORBMatch = drawKPMatches(&frameRef, &frameCur, imgWithFeatureRef, imgWithFeatureCur, matchIdx);
         double projError1 = 0, projError2 = 0;
         outImgWarpH = drawKPMatchesH(&frameRef, &frameCur, imgWithFeatureRef, imgWithFeatureCur, matchIdxH, H12, projError1);
         outImgWarpA = drawKPMatchesA(&frameRef, &frameCur, imgWithFeatureRef, imgWithFeatureCur, matchIdxA, A12, projError2);

         char strMatches1[64], strMatches2[64], strMatches3[64], strMatches4[64], strMatches5[64];
         std::snprintf(strMatches1, 64, "CV    F: %ld-%ld, M: %d/%d", frameRef.id, frameCur.id,
                       nInliers1, nMatched1);
         std::snprintf(strMatches2, 64, "BOW   F: %ld-%ld, M: %d/%d", frameRef.id, frameCur.id,
                       nInliers2, nMatched2);
         std::snprintf(strMatches3, 64, "ORB   F: %ld-%ld, M: %d/%d", frameRef.id, frameCur.id,
                       nInliers3, nMatched3);
         std::snprintf(strMatches4, 64, "WarpH F: %ld-%ld, M: %d/%d", frameRef.id, frameCur.id,
                       nInliers4, nMatched4);
         std::snprintf(strMatches5, 64, "WarpA F: %ld-%ld, M: %d/%d", frameRef.id, frameCur.id,
                       nInliers5, nMatched5);
         putText(outImgCV, strMatches1, Point(15, 15), 1, 1, Scalar(0, 0, 255), 2);
         putText(outImgBow, strMatches2, Point(15, 15), 1, 1, Scalar(0, 0, 255), 2);
         putText(outImgORBMatch, strMatches3, Point(15, 15), 1, 1, Scalar(0, 0, 255), 2);
         putText(outImgWarpH, strMatches4, Point(15, 15), 1, 1, Scalar(0, 0, 255), 2);
         putText(outImgWarpA, strMatches5, Point(15, 15), 1, 1, Scalar(0, 0, 255), 2);

         Mat imgTmp1, imgTmp2, imgTmp3;
         hconcat(outImgCV, outImgBow, imgTmp1);
         hconcat(outImgORBMatch, outImgWarpH, imgTmp2);
         hconcat(imgTmp1, imgTmp2, imgTmp3);
         hconcat(imgTmp3, outImgWarpA, outImgConcat);
         imshow("CV & BOW & ORB & WarpH & WarpA Matches", outImgConcat);


        // test A
        Se2 dOdom = frameRef.odom - frameCur.odom;
        Mat R0 = getRotationMatrix2D(Point2f(0, 0), dOdom.theta * 180.f / CV_PI, 1.);
        cout << "delta angle = " << (dOdom.theta / CV_PI * 180) << endl;
        cout << "R0 = " << endl << R0 << endl;

        Mat Tc1c2 = Config::Tcb * dOdom.inv().toCvSE3() * Config::Tbc;  // 4x4
        Mat Rc1c2 = Tc1c2.rowRange(0, 3).colRange(0, 3).clone();  // 3x3
        Mat tc1c2 = Tc1c2.rowRange(0, 3).col(3).clone();    // 3x1
        Mat R = Config::Kcam * Rc1c2 * (Config::Kcam).inv();  // 3x3 这里相当于仿射变换里的A, 不过少了部分平移信息
        Mat t = Config::Kcam * tc1c2 / 3000.f;  // 3x1 深度假设3m
        cout << "R = " << endl << R.rowRange(0, 2) << endl;
        cout << "R的缩放因子 = " << R.at<float>(0, 0) / cos(dOdom.theta) << endl;

        // A1 - 通过相对变换关系计算得到
        Mat A1;
        R.rowRange(0, 2).convertTo(A1, CV_64FC1);
        R0.colRange(0, 2).copyTo(A1.colRange(0, 2));  // 去掉尺度变换
        A1.at<double>(0, 2) += t.at<float>(0, 0);  // 加上平移对图像造成的影响
        A1.at<double>(1, 2) += t.at<float>(1, 0);
        cout << "A1 = " << endl << A1 << endl;

        // A2 - 旋转部分通过先验计算得到
        Point2f rotationCenter;
        rotationCenter.x = Config::cx - Config::Tbc.at<float>(1, 3) / 12;
        rotationCenter.y = Config::cy - Config::Tbc.at<float>(0, 3) / 12;
        Mat A2 = getRotationMatrix2D(rotationCenter, dOdom.theta * 180.f / CV_PI, 1.);
        // A2.at<double>(0, 2) += t.at<float>(0, 0);
        // A2.at<double>(1, 2) += t.at<float>(1, 0);
        cout << "A2 = " << endl << A2 << endl;

        // A21 - 通过特征点计算出来的仿射变换矩阵
        Mat A21;
        invertAffineTransform(A12, A21);
        cout << "A21 = " << endl << A21 << endl;
        cout << "A21的缩放因子 = " << A21.at<double>(0, 0) / cos(dOdom.theta) << endl;

        Mat imgA1, imgA2, imgA3, imgC1, imgC2, imgC3;
        warpAffine(imgCur, imgA1, A2, imgCur.size());
        warpAffine(imgCur, imgA2, A1, imgCur.size());
        warpAffine(imgCur, imgA3, A21, imgCur.size());
        hconcat(imgA1, imgA2, imgC1);
        hconcat(imgRef, imgA3, imgC2);
        hconcat(imgC1, imgC2, imgC3);
        imshow("TEST Affine", imgC3);

        // Mat Trans = Config::Tcb * frameRef.odom.inv().toCvSE3() * frameCur.odom.toCvSE3() *
        // Config::Tbc; cout << "Trans 4*4: " << endl << Trans << endl;

        // Mat R, t;
        // R = Trans.rowRange(0, 3).colRange(0, 3);
        // t = Trans.rowRange(0, 3).col(3);
        // Mat AR = Config::Kcam * R * Config::Kcam.inv();
        // Mat At = Config::Kcam * t;

        // Mat TransA(2, 3, CV_64F);
        // TransA.colRange(0, 2) = AR.rowRange(0, 2).colRange(0, 2).clone();
        // TransA.col(2) = At.rowRange(0, 2).clone();
        // cout << "Trans 2*3: " << endl << TransA << endl;

        // Mat imgA11, imgA22, imgA33;
        // warpAffine(imgCur, imgA11, TransA, imgCur.size());
        // hconcat(imgCur, imgA11, imgA22);
        // hconcat(imgA22, imgRef, imgA33);
        // imshow("TEST TransA Affine", imgA33);

        waitKey(0);
        //        string fileWarp = "/home/vance/output/rk_se2lam/warp/warp-" + text + ".bmp";
        //        imwrite(fileWarp, outImgWarp);

        //! 内点数太少要生成新的参考帧
        if (nInliers4 <= 10 || nInliers5 <= 10) {
            imgCur.copyTo(imgRef);
            imgWithFeatureCur.copyTo(imgWithFeatureRef);
            frameRef = frameCur;
            KFRef = KFCur;
            H12 = Mat::eye(3, 3, CV_64FC1);
            A12 = Mat::eye(2, 3, CV_64FC1);
        }
    }
    rec.close();

    aveMatches3 = nMatchesAll3 / (num - skipFrames - 1.f);
    aveMatches4 = nMatchesAll4 / (num - skipFrames - 1.f);
    aveMatches5 = nMatchesAll5 / (num - skipFrames - 1.f);
    aveInliers3 = nInliersAll3 / (num - skipFrames - 1.f);
    aveInliers4 = nInliersAll4 / (num - skipFrames - 1.f);
    aveInliers5 = nInliersAll5 / (num - skipFrames - 1.f);
    printf("ORB匹配平均匹配点数: %.2f, 平均内点数: %.2f\n", aveMatches3, aveInliers3);
    printf("WarpH匹配平均匹配点数: %.2f, 平均内点数: %.2f\n", aveMatches4, aveInliers4);
    printf("WarpA匹配平均匹配点数: %.2f, 平均内点数: %.2f\n", aveMatches5, aveInliers5);


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
