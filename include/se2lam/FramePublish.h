/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/


#ifndef FRAMEPUBLISH_H
#define FRAMEPUBLISH_H

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#define UseKlt
namespace se2lam {


#ifdef UseKlt
    class TrackKlt;
#else
    class Track;
#endif
class GlobalMapper;
class Localizer;

class FramePublish{
public:

    FramePublish();
#ifdef UseKlt
        FramePublish(TrackKlt* pTR, GlobalMapper* pGM);
#else
        FramePublish(Track* pTR, GlobalMapper* pGM);
#endif
    ~FramePublish();

    void run();

    cv::Mat drawMatchesInOneImg(const std::vector<cv::KeyPoint> queryKeys,
                                const cv::Mat &trainImg, const std::vector<cv::KeyPoint> trainKeys,
                                const std::vector<int> &matches);

    cv::Mat drawKeys(const std::vector<cv::KeyPoint> keys, const cv::Mat &mImg,
                     std::vector<int> matched);
    cv::Mat drawFrame();

    cv::Mat drawMatch();

    void setLocalizer(Localizer* localizer);

    bool mbIsLocalize;
    Localizer* mpLocalizer;

private:

#ifdef UseKlt
        TrackKlt* mpTrack;
#else
        Track* mpTrack;
#endif
    GlobalMapper* mpGM;

    std::vector<cv::KeyPoint> kp, kpRef;
    std::vector<int> matches;

    cv::Mat mImg, mImgRef;
    cv::Mat mImgOut;

    cv::Mat mImgMatch;

};


} // namespace se2lam

#endif // FRAMEPUBLISH_H
