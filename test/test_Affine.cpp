#include "test_functions.hpp"

// string g_configPath = "/home/vance/dataset/se2/se2_config/";
string g_configPath = "/home/vance/dataset/rk/dibeaDataSet/se2_config/";
string g_orbVocFile = "/home/vance/slam_ws/ORB_SLAM2/Vocabulary/ORBvoc.bin";
string g_matchResult = "/home/vance/output/rk_se2lam/test_matchResult.txt";


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
    double projError1 = 0, projError2 = 0, projError3 = 0;
    Frame frameCur, frameRef;
    PtrKeyFrame KFCur, KFRef;
    Mat imgColor, imgGray, imgCur, imgRef, imgWithFeatureCur, imgWithFeatureRef;
    Mat outImgWarpA, outImgWarpAP, outImgWarpRT, outImgConcat;
    Mat A12 = Mat::eye(2, 3, CV_64FC1);
    Mat A12Partial = Mat::eye(2, 3, CV_64FC1);
    Mat RT = Mat::eye(2, 3, CV_64FC1);
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
        if (imgColor.data == nullptr)
            continue;
        cv::undistort(imgColor, imgCur, Config::Kcam, Config::Dcam);
        cvtColor(imgCur, imgGray, CV_BGR2GRAY);

        //! 限制对比度自适应直方图均衡
        Mat imgClahe;
        Ptr<CLAHE> clahe = createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(imgGray, imgClahe);

        //! ORB提取特征点
        frameCur = Frame(imgClahe, odo, kpExtractor, imgFiles[i].timeStamp);
        KFCur = make_shared<KeyFrame>(frameCur);
        KFCur->computeBoW(vocabulary);
        imgCur.copyTo(imgWithFeatureCur);
        for (int i = 0, iend = frameCur.N; i < iend; ++i)
            circle(imgWithFeatureCur, frameCur.mvKeyPoints[i].pt, 2, Scalar(255, 0, 0));

        if (firstFrame) {
            imgCur.copyTo(imgRef);
            imgWithFeatureCur.copyTo(imgWithFeatureRef);
            frameRef = frameCur;
            KFRef = KFCur;
            firstFrame = false;
            continue;
        }

        //! 特征点匹配
        int nMatched1(0), nMatched2(0), nMatched3(0);
        int nInliers1(0), nInliers2(0), nInliers3(0);
        vector<int> matchIdxA, matchIdxAP, matchIdxRT;
        nMatched1 = kpMatcher->MatchByWindowWarp(frameRef, frameCur, A12, matchIdxA, 25);
        nMatched2 = kpMatcher->MatchByWindowWarp(frameRef, frameCur, A12Partial, matchIdxAP, 25);
        nMatched3 = kpMatcher->MatchByWindowWarp(frameRef, frameCur, RT, matchIdxRT, 25);

        if (nMatched1 >= 10)
            nInliers1 = removeOutliersWithA(frameRef.mvKeyPoints, frameCur.mvKeyPoints, matchIdxA, A12, 0);
        if (nMatched2 >= 10)
            nInliers2 = removeOutliersWithA(frameRef.mvKeyPoints, frameCur.mvKeyPoints,  matchIdxAP, A12Partial, 1);
        if (nMatched3>= 10)
            nInliers3 = removeOutliersWithA(frameRef.mvKeyPoints, frameCur.mvKeyPoints, matchIdxRT, RT, 2);

        cout << "A = " << endl << A12 << endl;
        cout << "AP = " << endl << A12Partial << endl;
        cout << "RT = " << endl << RT << endl;

        //! 匹配情况可视化
        outImgWarpA = drawKPMatchesA(&frameRef, &frameCur, imgWithFeatureRef, imgWithFeatureCur, matchIdxA, A12, projError1);
        outImgWarpAP = drawKPMatchesA(&frameRef, &frameCur, imgWithFeatureRef, imgWithFeatureCur, matchIdxAP, A12Partial, projError2);
        outImgWarpRT = drawKPMatchesA(&frameRef, &frameCur, imgWithFeatureRef, imgWithFeatureCur, matchIdxRT, RT, projError3);

        char strMatches1[64], strMatches2[64], strMatches3[64];
        std::snprintf(strMatches1, 64, "A   F: %ld-%ld, M: %d/%d, E: %.2f", frameRef.id, frameCur.id, nInliers1, nMatched1, projError1);
        std::snprintf(strMatches2, 64, "AP  F: %ld-%ld, M: %d/%d, E: %.2f", frameRef.id, frameCur.id, nInliers2, nMatched2, projError2);
        std::snprintf(strMatches3, 64, "RT  F: %ld-%ld, M: %d/%d, E: %.2f", frameRef.id, frameCur.id, nInliers3, nMatched3, projError3);
        putText(outImgWarpA, strMatches1, Point(15, 15), 1, 1, Scalar(0, 0, 255), 2);
        putText(outImgWarpAP, strMatches2, Point(15, 15), 1, 1, Scalar(0, 0, 255), 2);
        putText(outImgWarpRT, strMatches3, Point(15, 15), 1, 1, Scalar(0, 0, 255), 2);
        printf("平均重投影误差A/AP/RT = %.2f/%.2f/%.2f\n", projError1, projError2, projError3);

        Mat imgTmp1;
        vconcat(outImgWarpA, outImgWarpAP, imgTmp1);
        vconcat(imgTmp1, outImgWarpRT, outImgConcat);

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


        waitKey(0);
        //        string fileWarp = "/home/vance/output/rk_se2lam/warp/warp-" + text + ".bmp";
        //        imwrite(fileWarp, outImgWarp);

        //! 内点数太少要生成新的参考帧
        if (nInliers1 <= 12 || nInliers2 <= 12) {
            imgCur.copyTo(imgRef);
            imgWithFeatureCur.copyTo(imgWithFeatureRef);
            frameRef = frameCur;
            KFRef = KFCur;
            A12 = Mat::eye(2, 3, CV_64FC1);
            A12Partial = Mat::eye(2, 3, CV_64FC1);
            RT = Mat::eye(2, 3, CV_64FC1);
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
