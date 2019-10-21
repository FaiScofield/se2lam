/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#ifndef ODOSLAM_CPP
#define ODOSLAM_CPP
#include "OdoSLAM.h"
#include <opencv2/highgui/highgui.hpp>

#endif  // ODOSLAM_CPP

namespace se2lam
{

using namespace std;
using namespace cv;

OdoSLAM::OdoSLAM()
    : mpMap(nullptr), mpLocalMapper(nullptr), mpGlobalMapper(nullptr), mpFramePub(nullptr),
      mpMapPub(nullptr), mpTrack(nullptr), mpMapStorage(nullptr), mpLocalizer(nullptr),
      mpSensors(nullptr), mpVocabulary(nullptr), mbFinishRequested(false), mbFinished(false)
{}

OdoSLAM::~OdoSLAM()
{
    delete mpMapPub;
    delete mpLocalizer;
    delete mpTrack;
    delete mpLocalMapper;
    delete mpGlobalMapper;
    delete mpMap;
    delete mpMapStorage;
    delete mpFramePub;
    delete mpSensors;

    delete mpVocabulary;
}

void OdoSLAM::setVocFileBin(const char* strVoc)
{
    cout << "\n###\n"
         << "###  se2lam: On-SE(2) Localization and Mapping with SE(2)-XYZ Constraints.\n"
         << "###\n"
         << endl;

    cout << "[Syste][Info ] Set ORB Vocabulary to: " << strVoc << endl;
    cout << "[Syste][Info ] Loading ORB Vocabulary. This could take a while." << endl;

    // Init ORB BoW
    WorkTimer timer;

    mpVocabulary = new ORBVocabulary();

    string strVocFile(strVoc);
    bool bVocLoad = mpVocabulary->loadFromBinaryFile(strVocFile);
    if (!bVocLoad) {
        cerr << "[Syste][Error] Wrong path to vocabulary, Falied to open it." << endl;
        return;
    }

    printf("[Syste][Info ] Vocabulary loaded! Cost time: %.2fms\n", timer.count());
}

void OdoSLAM::setDataPath(const char* strDataPath)
{
    cout << "[Syste][Info ] Set Data Path to: " << strDataPath << endl;
    Config::readConfig(strDataPath);
}

cv::Mat OdoSLAM::getCurrentVehiclePose()
{
    return cvu::inv(mpMap->getCurrentFramePose()) * Config::Tcb;
}

cv::Mat OdoSLAM::getCurrentCameraPoseWC()
{
    return cvu::inv(mpMap->getCurrentFramePose());
}

cv::Mat OdoSLAM::getCurrentCameraPoseCW()
{
    return mpMap->getCurrentFramePose();
}

void OdoSLAM::start()
{
    if (mpVocabulary == nullptr) {
        std::cerr << "[Syste][Error] Please set vocabulary first!!" << std::endl;
        return;
    }

    // Construct the system
    mpMap = new Map;
    mpSensors = new Sensors;
    mpTrack = new Track;
    mpLocalMapper = new LocalMapper;
    mpGlobalMapper = new GlobalMapper;
    mpFramePub = new FramePublish(mpTrack, mpGlobalMapper);
    mpMapStorage = new MapStorage();
    mpMapPub = new MapPublish(mpMap);
    mpLocalizer = new Localizer();

    mpTrack->setLocalMapper(mpLocalMapper);
    mpTrack->setMap(mpMap);
    mpTrack->setSensors(mpSensors);

    mpLocalMapper->setMap(mpMap);
    mpLocalMapper->setGlobalMapper(mpGlobalMapper);

    mpGlobalMapper->setMap(mpMap);
    mpGlobalMapper->setLocalMapper(mpLocalMapper);
    mpGlobalMapper->setORBVoc(mpVocabulary);

    mpMapStorage->setMap(mpMap);

    mpLocalizer->setMap(mpMap);
    mpLocalizer->setORBVoc(mpVocabulary);
    mpLocalizer->setSensors(mpSensors);

    mpFramePub->setLocalizer(mpLocalizer);

    mpMapPub->setFramePub(mpFramePub);
    mpMapPub->setLocalizer(mpLocalizer);
    mpMapPub->setTracker(mpTrack);

    if (Config::UsePrevMap) {
        mpMapStorage->setFilePath(Config::MapFileStorePath, Config::ReadMapFileName);
        mpMapStorage->loadMap();
    }

    mbFinishRequested = false;
    mbFinished = false;

    if (Config::LocalizationOnly) {
        cerr << "[Syste][Info ] =====>> Localization-Only Mode <<=====" << endl;

        //! 注意标志位在这里设置的
        mpFramePub->mbIsLocalize = true;
        mpMapPub->mbIsLocalize = true;

        thread threadLocalizer(&Localizer::run, mpLocalizer);
        thread threadMapPub(&MapPublish::run, mpMapPub);

        threadLocalizer.detach();
        threadMapPub.detach();
    } else {  // SLAM case
        cout << "[Syste][Info ] =====>> Running SLAM <<=====" << endl;

        mpMapPub->mbIsLocalize = false;
        mpFramePub->mbIsLocalize = false;

        thread threadTracker(&Track::run, mpTrack);
        thread threadLocalMapper(&LocalMapper::run, mpLocalMapper);
        thread threadGlobalMapper(&GlobalMapper::run, mpGlobalMapper);
        if (Config::NeedVisualization) {
            thread threadMapPub(&MapPublish::run, mpMapPub);
            threadMapPub.detach();
        }

        threadTracker.detach();
        threadLocalMapper.detach();
        threadGlobalMapper.detach();
    }

    thread threadWait(&wait, this);
    threadWait.detach();
}

void OdoSLAM::wait(OdoSLAM* system)
{
    ros::Rate rate(Config::FPS * 2);
    while (1) {
        if (system->checkFinish()) {

            //! 系统其他模块的退出都是在这里发出信号的
            system->sendRequestFinish();

            break;
        }
        rate.sleep();
    }

    system->saveMap();

    system->checkAllExit();

    system->mbFinished = true;

    cerr << "[Syste][Info ] System is cleared .." << endl;
}

void OdoSLAM::saveMap()
{
    if (Config::LocalizationOnly)
        return;

    if (Config::SaveNewMap) {
        mpMapStorage->setFilePath(Config::MapFileStorePath, Config::WriteMapFileName);
        mpMapStorage->saveMap();
    }

    // Save keyframe trajectory
    string trajFile = Config::MapFileStorePath + Config::WriteTrajFileName;
    cerr << "\n[Syste][Info ] Saving keyframe trajectory to " << trajFile << endl;
    ofstream towrite(trajFile);
    if (!towrite.is_open())
        cerr << "[Syste][Error] Save trajectory error! Please check the trajectory file correct." << endl;
    towrite << "#format: id x y z theta" << endl;
    vector<PtrKeyFrame> vct = mpMap->getAllKF();
    for (size_t i = 0, iend = vct.size(); i != iend; ++i) {
        if (!vct[i]->isNull()) {
            Mat Twb = cvu::inv(Config::Tbc * vct[i]->getPose());
            Mat Rwb = Twb.rowRange(0, 3).colRange(0, 3);
            g2o::Vector3D euler = g2o::internal::toEuler(toMatrix3d(Rwb));
            towrite << vct[i]->id << " " << Twb.at<float>(0, 3) << " " << Twb.at<float>(1, 3) << " "
                    << Twb.at<float>(2, 3) << " " << euler(2) << endl;
        }
    }
    towrite.close();
}

void OdoSLAM::requestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool OdoSLAM::checkFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    if (Config::LocalizationOnly) {
        if (mpLocalizer->isFinished() || mpMapPub->isFinished()) {
            mbFinishRequested = true;
            return true;
        }
    } else {
        if (mpTrack->isFinished() || mpLocalMapper->isFinished() || mpGlobalMapper->isFinished() ||
            mpMapPub->isFinished()) {
            mbFinishRequested = true;
            return true;
        }
    }

    return mbFinishRequested;
}

void OdoSLAM::sendRequestFinish()
{
    if (Config::LocalizationOnly) {
        mpLocalizer->requestFinish();
        mpMapPub->requestFinish();
    } else {
        mpTrack->requestFinish();
        mpLocalMapper->requestFinish();
        mpGlobalMapper->requestFinish();
        mpMapPub->requestFinish();
    }
}

void OdoSLAM::checkAllExit()
{
    cout << "[Syste][Info ] Checking for all thread exited..." << endl;

    if (Config::LocalizationOnly) {
        while (1) {
            if (mpLocalizer->isFinished())
                break;
            else
                std::this_thread::sleep_for(std::chrono::microseconds(2));
        }
    } else {
        while (1) {
            if (mpTrack->isFinished() && mpLocalMapper->isFinished() && mpGlobalMapper->isFinished())
                break;
            else
                std::this_thread::sleep_for(std::chrono::microseconds(2));
        }
    }

    if (Config::NeedVisualization) {
        while (1) {
            if (mpMapPub->isFinished())
                break;
            else
                std::this_thread::sleep_for(std::chrono::microseconds(2));
        }
    }
}

void OdoSLAM::waitForFinish()
{
    while (1) {
        if (mbFinished) {
            break;
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(2));
        }
    }
    cerr << "[Syste][Info ] Wait for finish thread finished..." << endl;
    std::this_thread::sleep_for(std::chrono::microseconds(20));
}

bool OdoSLAM::ok()
{
    unique_lock<mutex> lock(mMutexFinish);
    return !mbFinishRequested;
}

}  // namespace se2lam
