/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#ifndef ODOSLAM_H
#define ODOSLAM_H

#include "cvutil.h"
#include "Track.h"
#include "LocalMapper.h"
#include "GlobalMapper.h"
#include "Map.h"
#include "Config.h"
#include "MapStorage.h"
#include "FramePublish.h"
#include "MapPublish.h"
#include "Localizer.h"
#include "Sensors.h"
#include "TrackKlt.h"

namespace  se2lam {

//#define UseKlt
class OdoSLAM {

public:
    OdoSLAM();

    ~OdoSLAM();

    void setDataPath(const char* strDataPath);

    void setVocFileBin(const char *strVoc);

    void start();

    inline void receiveOdoData(float x_, float y_, float z_, double time_ = 0.)
    {
        mpSensors->updateOdo(x_, y_, z_, time_);
    }

//    inline void receiveOdoDataSequence(std::vector<Se2> &odoDeque_)
//    {
//        mpSensors->updateOdoSequence(odoDeque_);
//    }

    inline void receiveImgData(const cv::Mat &img_, double time_ = 0.)
    {
        mpSensors->updateImg(img_, time_);
    }

    inline void receiveImuTheta(double theta_, double ctime_, bool useCeil_)
    {
        mpSensors->updateImu(theta_, ctime_, useCeil_);
    }

    void requestFinish();
    void waitForFinish();

    cv::Mat getCurrentVehiclePose();
    cv::Mat getCurrentCameraPoseWC();
    cv::Mat getCurrentCameraPoseCW();

    bool ok();

private:
    Map* mpMap;
    LocalMapper* mpLocalMapper;
    GlobalMapper* mpGlobalMapper;
    FramePublish* mpFramePub;
    MapPublish* mpMapPub;
//    Track* mpTrack;
#ifdef UseKlt
    TrackKlt* mpTrack;
#else
    Track* mpTrack;
#endif
    MapStorage* mpMapStorage;
    Localizer* mpLocalizer;
    Sensors* mpSensors;
    ORBVocabulary* mpVocabulary;

    bool mbFinishRequested;
    bool mbFinished;

    std::mutex mMutexFinish;

    bool checkFinish();
    void sendRequestFinish();
    void checkAllExit();

    void saveMap();

    static void wait(OdoSLAM* system);
};

}


#endif // ODOSLAM_H
