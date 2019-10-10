#include "test_functions.hpp"
#include "TestTrack.hpp"


string g_configPath = "/home/vance/dataset/rk/dibeaDataSet/se2_config/";
string g_orbVocFile = "/home/vance/slam_ws/ORB_SLAM2/Vocabulary/ORBvoc.bin";


int main(int argc, char** argv)
{
    ros::init(argc, argv, "test_MPManager");
    ros::start();

    //! check input
    if (argc < 2) {
        fprintf(stderr, "Usage: test_pointDetection <rk_dataPath> [number_frames_to_process]");
        ros::shutdown();
        exit(-1);
    }
    int num = INT_MAX;
    if (argc == 3) {
        num = atoi(argv[2]);
        cout << " - set number_frames_to_process = " << num << endl << endl;
    }


    //! initialization
    Config::readConfig(string(argv[1]));
    Mat K = Config::Kcam, D = Config::Dcam;

    string dataFolder = Config::DataPath + "/slamimg";
    vector<RK_IMAGE> allImages;
    readImagesRK(dataFolder, allImages);

    string odomRawFile = Config::DataPath + "/OdomRaw.txt";
    vector<Se2> allOdoms;
    readOdomsRK(odomRawFile, allOdoms);

    vector<Se2> alignedOdoms;
    dataAlignment(allImages, allOdoms, alignedOdoms);
    assert(allImages.size() == alignedOdoms.size());

    Map *pMap = new Map();
    ORBmatcher *kpMatcher = new ORBmatcher();
    ORBVocabulary *pVocabulary = new ORBVocabulary();
    bool bVocLoad = pVocabulary->loadFromBinaryFile(g_orbVocFile);
    if(!bVocLoad) {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Falied to open at: " << g_orbVocFile << endl;
        exit(-1);
    }
    cout << "Vocabulary loaded!" << endl << endl;


    //! main loop
    TestTrack tt;
    tt.setMap(pMap);

    int skipFrames = 30;
    num = std::min(num, static_cast<int>(allImages.size()));
    ros::Rate rate(Config::FPS);
    for (int i = skipFrames; i < num; ++i) {
        if (tt.mState == cvu::NO_READY_YET)
            tt.mState = cvu::FIRST_FRAME;
        tt.mLastState = tt.mState;

        Mat imgGray = imread(allImages[i].fileName, CV_LOAD_IMAGE_GRAYSCALE);
        if (imgGray.data == nullptr) {
            cerr << "Error in reading image file " << allImages[i].fileName << endl;
            continue;
        }

        WorkTimer timer;
        timer.start();

        if (tt.mState == cvu::FIRST_FRAME)
            tt.createFirstFrame(imgGray, allImages[i].timeStamp, alignedOdoms[i]);
        else
            tt.trackReferenceKF(imgGray, allImages[i].timeStamp, alignedOdoms[i]);

        tt.mpMap->setCurrentFramePose(tt.mCurrentFrame.getPose());

        timer.stop();
        fprintf(stderr, "[Track] #%ld Tracking consuming time: %fms\n", tt.mCurrentFrame.id, timer.time);

        //! 匹配情况可视化
        Mat outImgWarp;
        tt.drawMatchesForPub(outImgWarp);

        char strMatches[64];
        std::snprintf(strMatches, 64, "F: %ld-%ld, M: %d/%d", tt.mpReferenceKF->id,
                      tt.mCurrentFrame.id, tt.mnInliers, tt.mnMatchSum);
        putText(outImgWarp, strMatches, Point(15, 15), 1, 1, Scalar(0, 0, 255), 2);
        imshow("Image Warp Match", outImgWarp);
        waitKey(100);

        rate.sleep();
    }

    delete kpMatcher;
    delete pVocabulary;
    return 0;
}
