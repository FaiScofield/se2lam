/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#ifndef ODOSLAM_H
#define ODOSLAM_H

#include "Config.h"
#include "GlobalMapper.h"
#include "LocalMapper.h"
#include "Localizer.h"
#include "Map.h"
#include "MapPublish.h"
#include "MapStorage.h"
#include "Sensors.h"
#include "Track.h"
#include "TrackKlt.h"
#include "cvutil.h"

namespace se2lam
{

#define USEKLT

class OdoSLAM
{
public:
    OdoSLAM();

    ~OdoSLAM();

    void setDataPath(const char* strDataPath);

    void setVocFileBin(const char* strVoc);

    void start();

    inline void receiveOdoData(float x_, float y_, float z_, double time_ = 0.)
    {
        mpSensors->updateOdo(x_, y_, z_, time_);
    }

    inline void receiveImgData(const cv::Mat& img_, double time_ = 0.)
    {
        mpSensors->updateImg(img_, time_);
    }

    inline void receiveImuTheta(double theta_, double ctime_)
    {
        mpSensors->updateImu(theta_, ctime_);
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
    MapPublish* mpMapPub;
#ifdef USEKLT
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

}  // namespace se2lam

#endif  // ODOSLAM_H
