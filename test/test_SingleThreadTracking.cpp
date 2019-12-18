#include "test_functions.hpp"
#include "TestTrack.h"
#include "MapPublish.h"
#include "MapStorage.h"

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
    int num = Config::ImgCount;
    if (argc == 3) {
        num = atoi(argv[2]);
        cout << " - set number_frames_to_process = " << num << endl << endl;
    }

    //! initialization
    Config::readConfig(string(argv[1]));

    string dataFolder = Config::DataPath + "slamimg";
    vector<RK_IMAGE> allImages;
    readImagesRK(dataFolder, allImages);

    string odomRawFile = Config::DataPath + "odo_raw.txt";
    vector<Se2> allOdoms;
    readOdomsRK(odomRawFile, allOdoms);
    if (allOdoms.empty())
        exit(-1);

    ORBmatcher *pMatcher = new ORBmatcher();
    ORBVocabulary *pVocabulary = new ORBVocabulary();
    bool bVocLoad = pVocabulary->loadFromBinaryFile(g_orbVocFile);
    if(!bVocLoad) {
        cerr << "[Track] Wrong path to vocabulary. " << endl;
        cerr << "[Track] Falied to open at: " << g_orbVocFile << endl;
        delete pMatcher;
        delete pVocabulary;
        exit(-1);
    }
    cout << "[Track] Vocabulary loaded!" << endl << endl;

    //! import class
    Map *pMap = new Map();
    MapPublish* pMapPub = new MapPublish(pMap);
    thread threadMapPub(&MapPublish::run, pMapPub);
    threadMapPub.detach();

    TestTrack* pTracker = new TestTrack();
    pTracker->setMap(pMap);
    pTracker->setMapPublisher(pMapPub);
    if (!pTracker->checkReady()) {
        cerr << "[Track] pTracker no read! " << endl;
        pMapPub->requestFinish();
        delete pMatcher;
        delete pVocabulary;
        delete pMap;
        delete pMapPub;
        delete pTracker;
        exit(-1);
    }


    //! main loop
    const int skipFrames = Config::ImgStartIndex;
    num = std::min(num, static_cast<int>(allImages.size()));
    ros::Rate rate(Config::FPS);
    for (int i = skipFrames; i < num; ++i) {
        Mat imgGray = imread(allImages[i].fileName, CV_LOAD_IMAGE_GRAYSCALE);
        if (imgGray.empty()) {
            cerr << "Error in reading image file " << allImages[i].fileName << endl;
            continue;
        }

        const double& timeStamp = allImages[i].timeStamp;
        const Se2& odo = allOdoms[i];

        pTracker->run(imgGray, odo, timeStamp);

        rate.sleep();
    }
    pMapPub->requestFinish();

    if (Config::SaveNewMap) {
        MapStorage* pMapStorage = new MapStorage();
        pMapStorage->setMap(pMap);
        pMapStorage->setFilePath(Config::MapFileStorePath, Config::WriteMapFileName);
        pMapStorage->saveMap();
    }

    delete pMatcher;
    delete pVocabulary;
    delete pMap;
    delete pMapPub;
    delete pTracker;
    return 0;
}
