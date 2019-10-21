#include "test_functions.hpp"
#include "TestTrack.hpp"
#include "TestViewer.hpp"
#include "MapPublish.h"


string g_configPath = "/home/vance/dataset/rk/dibeaDataSet/se2_config/";
string g_orbVocFile = "/home/vance/slam_ws/ORB_SLAM2/Vocabulary/ORBvoc.bin";


int main(int argc, char** argv)
{
    ros::init(argc, argv, "test_MPManager");
    ros::start();

    //! check input
    if (argc < 2) {
        fprintf(stderr, "Usage: test_MapPointManager <rk_dataPath> [number_frames_to_process]");
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
    ORBmatcher *pMatcher = new ORBmatcher();
    ORBVocabulary *pVocabulary = new ORBVocabulary();
    bool bVocLoad = pVocabulary->loadFromBinaryFile(g_orbVocFile);
    if(!bVocLoad) {
        cerr << "[Track] Wrong path to vocabulary. " << endl;
        cerr << "[Track] Falied to open at: " << g_orbVocFile << endl;
        delete pMap;
        delete pMatcher;
        delete pVocabulary;
        exit(-1);
    }
    cout << "[Track] Vocabulary loaded!" << endl << endl;


    //! main loop
    TestTrack tt;
    tt.setMap(pMap);

    TestViewer tv(pMap);
    tv.setTracker(&tt);

    thread threadMapPub(&TestViewer::run, &tv);
    threadMapPub.detach();

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
        fprintf(stdout, "[Track] #%ld Tracking consuming time: %.2fms\n", tt.mCurrentFrame.id, timer.time);

        //! 匹配情况可视化
//        Mat outImgWarp = tt.drawMatchesForPub();
//        if (!outImgWarp.empty()) {
//            imshow("Image Warp Match", outImgWarp);
//            waitKey(0);
//        }

        rate.sleep();
    }
    tv.requestFinish();

    delete pMap;
    delete pMatcher;
    delete pVocabulary;
    return 0;
}
