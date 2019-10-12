#include "test_functions.hpp"

string g_configPath = "/home/vance/dataset/rk/dibeaDataSet/se2_config/";
string g_orbVocFile = "/home/vance/slam_ws/ORB_SLAM2/Vocabulary/ORBvoc.bin";
string g_matchResult = "/home/vance/output/rk_se2lam/test_matchResult.txt";
Ptr<DescriptorMatcher> g_CVMatcher = DescriptorMatcher::create("BruteForce-Hamming");

void writeMatchData(const string outFile,
                    const vector<vector<int>>& vvMatches,
                    const vector<vector<int>>& vvInliners)
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
        cout << "Tatal Match Average " << i << ": " << 1.0*sum[i]/vvMatches[i].size() << endl;

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
        cout << "Tatal Inliners Average " << i << ": " << 1.0*sumIn[i]/vvInliners[i].size() << endl;

    cerr << "Write match data to file: " << outFile << endl;
}

int matchByCV(const Mat& desRef, const Mat& desCur, map<int, int>& matchCV)
{
    vector<DMatch> matches;
    g_CVMatcher->match(desRef, desCur, matches);

    // 过滤匹配点
    float max_dist = 0.f;
    float min_dist = 10000.f;

    for (int i = 0, iend = desRef.rows; i < iend; ++i) {
        float dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }
    float delta_dist = std::max(std::min(2.f * min_dist, 0.5f * max_dist), 20.f);
//    printf("-- Max dist : %f \n", max_dist);
//    printf("-- Min dist : %f \n", min_dist);
//    printf("-- delta dist : %f \n", delta_dist);

    //-- Draw only "good" matches (i.e. whose distance is less than 0.6*max_dist )
    //-- PS.- radiusMatch can also be used here.

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
        fprintf(stderr, "Usage: test_pointDetection <rk_dataPath> [number_frames_to_process]");
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

    ORBextractor *kpExtractor = new ORBextractor(500, 1, 1, 1, 20);
    ORBmatcher *kpMatcher = new ORBmatcher(0.8); // 0.6
    ORBVocabulary *vocabulary = new ORBVocabulary();
    bool bVocLoad = vocabulary->loadFromBinaryFile(g_orbVocFile);
    if(!bVocLoad) {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Falied to open at: " << g_orbVocFile << endl;
        exit(-1);
    }
    cout << "Vocabulary loaded!" << endl << endl;


    //! main loop
    bool firstFrame = true;
//    vector<vector<int>> vvMatches(deltaKF);
//    vector<vector<int>> vvInliners(deltaKF);

    Frame frameCur, frameRef;
    PtrKeyFrame KFCur, KFRef;
    Mat imgColor, imgGray, imgCur, imgRef, imgWithFeatureCur, imgWithFeatureRef;
    Mat outImgORBMatch, outImgWarp, outImgBow, outImgCV, outImgConcat;
    Mat H12 = Mat::eye(3, 3, CV_64F);
    num = std::min(num, static_cast<int>(imgFiles.size()));
    int skipFrames = 50;
    for (int i = skipFrames; i < num; ++i) {
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
        WorkTimer timer;
        frameCur = Frame(imgClahe, imgFiles[i].timeStamp, Se2(), kpExtractor, K, D);
        KFCur = make_shared<KeyFrame>(frameCur);
        KFCur->ComputeBoW(vocabulary);
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
        int nMatched1(0), nMatched2(0), nMatched3(0), nMatched4(0);
        int nInlines1(0), nInlines2(0), nInlines3(0), nInlines4(0);
        vector<int> matchIdx, matchIdx12;
        map<int, int> matchBow, matchCV;
        vector<Point2f> prevMatched;
        KeyPoint::convert(frameRef.mvKeyPoints, prevMatched);

        timer.start();
        nMatched1 = kpMatcher->MatchByWindow(frameRef, frameCur, prevMatched, 25, matchIdx);
        printf("#%ld 匹配方式 MatchByWindow() 耗时%.2fms\n", frameCur.id, timer.count());

        timer.start();
        nMatched2 = kpMatcher->MatchByWindowWarp(frameRef, frameCur, H12, matchIdx12, 25);
        printf("#%ld 匹配方式 MatchByWindowWarp() 耗时%.2fms\n", frameCur.id, timer.count());

        timer.start();
        nMatched3 = kpMatcher->SearchByBoW(KFRef, KFCur, matchBow, false);
        printf("#%ld 匹配方式 SearchByBoW() 耗时%.2fms\n", frameCur.id, timer.count());

        timer.start();
        nMatched4 = matchByCV(frameRef.mDescriptors, frameCur.mDescriptors, matchCV);
        printf("#%ld 匹配方式 matchByCV() 耗时%.2fms\n", frameCur.id, timer.count());

        timer.start();
        if (nMatched1 >= 10) {
            Mat H = Mat::eye(3, 3, CV_64F);
            nInlines1 = removeOutliersWithHF(frameRef.mvKeyPoints, frameCur.mvKeyPoints, matchIdx, H);
        }
        if (nMatched2 >= 10) {
            nInlines2 = removeOutliersWithHF(frameRef.mvKeyPoints, frameCur.mvKeyPoints, matchIdx12, H12);
        } else {
            H12 = Mat::eye(3, 3, CV_64F);
        }
        if (nMatched3 >= 10) {
            Mat H = Mat::eye(3, 3, CV_64F);
            nInlines3 = removeOutliersWithHF(frameRef.mvKeyPoints, frameCur.mvKeyPoints, matchBow, H);
        }
        if (nMatched4 >= 10) {
            Mat H = Mat::eye(3, 3, CV_64F);
            nInlines4 = removeOutliersWithHF(frameRef.mvKeyPoints, frameCur.mvKeyPoints, matchCV, H);
        }
        printf("#%ld 剔除外点总耗时%.2fms\n", frameCur.id, timer.count());

        //! 存储匹配结果作分析H
//        vvMatches[idx].push_back(nMatched2);
//        vvInliners[idx].push_back(nInlines2);

        //! 匹配情况可视化
        outImgORBMatch = drawKPMatches(&frameRef, &frameCur, imgWithFeatureRef, imgWithFeatureCur, matchIdx);
        outImgWarp = drawKPMatches(&frameRef, &frameCur, imgWithFeatureRef, imgWithFeatureCur, matchIdx12, H12);
        outImgBow = drawKPMatches(KFRef, KFCur, imgWithFeatureRef, imgWithFeatureCur, matchBow);
        outImgCV = drawKPMatches(KFRef, KFCur, imgWithFeatureRef, imgWithFeatureCur, matchCV);

        char strMatches1[64], strMatches2[64], strMatches3[64], strMatches4[64];
        std::snprintf(strMatches1, 64, "ORB  F: %ld-%ld, M: %d/%d", frameRef.id, frameCur.id, nInlines1, nMatched1);
        std::snprintf(strMatches2, 64, "WARP F: %ld-%ld, M: %d/%d", frameRef.id, frameCur.id, nInlines2, nMatched2);
        std::snprintf(strMatches3, 64, "BOW  F: %ld-%ld, M: %d/%d", frameRef.id, frameCur.id, nInlines3, nMatched3);
        std::snprintf(strMatches4, 64, "CV   F: %ld-%ld, M: %d/%d", frameRef.id, frameCur.id, nInlines4, nMatched4);
        putText(outImgORBMatch, strMatches1, Point(15, 15), 1, 1, Scalar(0, 0, 255), 2);
        putText(outImgWarp, strMatches2, Point(15, 15), 1, 1, Scalar(0, 0, 255), 2);
        putText(outImgBow, strMatches3, Point(15, 15), 1, 1, Scalar(0, 0, 255), 2);
        putText(outImgCV, strMatches4, Point(15, 15), 1, 1, Scalar(0, 0, 255), 2);
        Mat imgTmp1, imgTmp2;
        hconcat(outImgORBMatch, outImgWarp, imgTmp1);
        hconcat(outImgCV, outImgBow, imgTmp2);
        hconcat(imgTmp2, imgTmp1, outImgConcat);
        imshow("BOW & ORB & Image Warp Match", outImgConcat);
        waitKey(100);
//        string fileWarp = "/home/vance/output/rk_se2lam/warp/warp-" + text + ".bmp";
//        string fileMatch = "/home/vance/output/rk_se2lam/warp/match-" + text + ".bmp";
//        imwrite(fileWarp, outImgWarp);
//        imwrite(fileMatch, outImgORBMatch);

        //! 内点数太少要生成新的参考帧
        if (nInlines2 <= 10) {
            imgCur.copyTo(imgRef);
            imgWithFeatureCur.copyTo(imgWithFeatureRef);
            frameRef = frameCur;
            KFRef = KFCur;
            H12 = Mat::eye(3, 3, CV_64F);
        }
    }

//    writeMatchData(g_matchResult, vvMatches, vvInliners);

    delete kpExtractor;
    delete kpMatcher;
    delete vocabulary;
    return 0;
}
