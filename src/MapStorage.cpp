/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include "MapStorage.h"

#include "converter.h"
#include <iostream>
#include <mutex>
#include <opencv2/highgui/highgui.hpp>

namespace se2lam
{

using namespace std;
using namespace cv;

MapStorage::MapStorage()
{
    mvKFs.clear();
    mvMPs.clear();
}

void MapStorage::setFilePath(const string path, const string file)
{
    mMapPath = path;
    mMapFile = file;
}


void MapStorage::loadMap()
{
    loadMapPoints();

    loadKeyFrames();  // load MPs first

    //    loadObservations();

    //    loadCovisibilityGraph();

    loadOdoGraph();

    loadFtrGraph();

    loadToMap();

    printf("[MapStorage] Map loaded from: %s\n\n", (mMapPath + mMapFile).c_str());
}

void MapStorage::saveMap()
{

    mvKFs = mpMap->getAllKFs();
    mvMPs = mpMap->getAllMPs();

    sortKeyFrames();  // 将KF和MP从0开始重新编号.

    sortMapPoints();  // 去掉了视差不好的MP了

    saveKeyFrames();

    saveMapPoints();

    // saveObservations();

    // saveCovisibilityGraph();

    saveOdoGraph();

    saveFtrGraph();

    printf("\n[MapStorage] Map saved to '%s'\n", (mMapPath + mMapFile).c_str());
}

void MapStorage::sortKeyFrames()
{
    // Remove null KFs
    {
        vector<PtrKeyFrame> vKFs;
        vKFs.reserve(mvKFs.size());
        for (size_t i = 0, iend = mvKFs.size(); i != iend; ++i) {
            const PtrKeyFrame& pKF = mvKFs[i];
            if (pKF->isNull())
                continue;
            vKFs.push_back(pKF);
        }
        std::swap(vKFs, mvKFs);
    }

    // Change Id of KF to be vector index
    for (unsigned long i = 0, iend = mvKFs.size(); i != iend; ++i) {
        PtrKeyFrame& pKF = mvKFs[i];
        pKF->mIdKF = i;
    }
}

void MapStorage::sortMapPoints()
{
    // Remove null MPs
    {
        vector<PtrMapPoint> vMPs;
        vMPs.reserve(mvMPs.size());
        for (size_t i = 0, iend = mvMPs.size(); i != iend; ++i) {
            const PtrMapPoint& pMP = mvMPs[i];
            if (pMP->isNull() || !(pMP->isGoodPrl()))
                continue;
            vMPs.push_back(pMP);
        }
        std::swap(vMPs, mvMPs);
    }

    // Change Id of MP to be vector index
    for (unsigned long i = 0, iend = mvMPs.size(); i != iend; ++i) {
        PtrMapPoint& pMP = mvMPs[i];
        pMP->mId = i;
    }
}

// run after sortMapPoints()
void MapStorage::saveKeyFrames()
{
    // Save images to individual files
    if (Config::NeedVisualization) {
        for (size_t i = 0, iend = mvKFs.size(); i != iend; ++i) {
            const PtrKeyFrame& pKF = mvKFs[i];
            imwrite(mMapPath + to_string(i) + ".bmp", pKF->mImage);
        }
    }

    // Write data to file
    FileStorage file(mMapPath + mMapFile, FileStorage::WRITE);
    file << "KeyFrames"
         << "[";

    for (size_t i = 0, iend = mvKFs.size(); i != iend; ++i) {
        const PtrKeyFrame& pKF = mvKFs[i];

        file << "{";

        file << "Id" << i;

        file << "N" << pKF->N;

        file << "Pose" << pKF->getPose();

        file << "Odometry" << Point3f(pKF->odom.x, pKF->odom.y, pKF->odom.theta);

        file << "ScaleFactor" << pKF->mfScaleFactor;

        file << "KeyPoints"
             << "[";
        for (size_t j = 0, jend = pKF->mvKeyPoints.size(); j < jend; ++j) {
            const KeyPoint& kp = pKF->mvKeyPoints[j];
            file << "{";
            file << "pt" << kp.pt;
            file << "octave" << kp.octave;
            file << "angle" << kp.angle;
            file << "response" << kp.response;
            file << "}";
        }
        file << "]";

        file << "Descriptor" << pKF->mDescriptors;

        vector<PtrMapPoint> vViewMPs = pKF->getObservations();
        if (vViewMPs.size() != pKF->mvKeyPoints.size())
            cerr << "Wrong size of KP in saving" << endl;

        file << "ViewMPsId"
             << "[";
        for (size_t j = 0, jend = vViewMPs.size(); j < jend; ++j) {
            if (vViewMPs[j] == nullptr)
                file << -1;
            else
                file << vViewMPs[j]->mId;
        }
        file << "]";

        file << "ViewMPInfo"
             << "[";
        for (size_t j = 0, jend = pKF->mvViewMPsInfo.size(); j < jend; ++j) {
            file << toCvMat(pKF->mvViewMPsInfo[j]);
        }
        file << "]";

        file << "Covisibilities"
             << "[";
        map<PtrKeyFrame, int> vCosKFsWeight = pKF->getAllCovisibleKFsAndWeights();
        for (auto it = vCosKFsWeight.begin(), iend = vCosKFsWeight.end(); it != iend; ++it) {
            file << "{";

            file << "CovisibleKFID" << it->first->mIdKF;
            file << "CovisibleMPCount" << it->second;

            file << "}";
        }
        file << "]";


        file << "}";
    }
    file << "]";

    file.release();
}

void MapStorage::saveMapPoints()
{
    // Write data to file
    FileStorage file(mMapPath + mMapFile, FileStorage::APPEND);
    file << "MapPoints"
         << "[";

    for (size_t i = 0, iend = mvMPs.size(); i != iend; ++i) {
        const PtrMapPoint& pMP = mvMPs[i];
        file << "{";

        file << "Id" << i;
        file << "Pose" << pMP->getPos();

        file << "}";
    }
    file << "]";

    file.release();
}

void MapStorage::saveObservations()
{
    size_t sizeKF = mvKFs.size();
    size_t sizeMP = mvMPs.size();

    cv::Mat obs(sizeKF, sizeMP, CV_32SC1, Scalar(0));
    cv::Mat Index(sizeKF, sizeMP, CV_32SC1, Scalar(-1));

    for (size_t i = 0; i != sizeKF; ++i) {
        const PtrKeyFrame& pKF = mvKFs[i];
        for (size_t j = 0; j < sizeMP; ++j) {
            const PtrMapPoint& pMP = mvMPs[j];
            if (pKF->hasObservationByPointer(pMP)) {
                obs.at<int>(i, j) = 1;
                Index.at<int>(i, j) = pMP->getKPIndexInKF(pKF);
            }
        }
    }

    FileStorage file(mMapPath + mMapFile, FileStorage::APPEND);

    file << "Observations" << obs;

    file << "ObservationIndex" << Index;

    file.release();
}

void MapStorage::saveCovisibilityGraph()
{
    size_t sizeKF = mvKFs.size();

    cv::Mat CovisibilityGraph(sizeKF, sizeKF, CV_32SC1, Scalar(0));

    for (size_t i = 0; i != sizeKF; ++i) {
        const PtrKeyFrame& pKF = mvKFs[i];
        vector<PtrKeyFrame> sKFs = pKF->getAllCovisibleKFs();

        for (auto it = sKFs.begin(), itend = sKFs.end(); it != itend; ++it) {
            PtrKeyFrame covKF = *it;
            CovisibilityGraph.at<int>(i, covKF->mIdKF) = 1;
        }
    }

    FileStorage file(mMapPath + mMapFile, FileStorage::APPEND);
    file << "CovisibilityGraph" << CovisibilityGraph;
    file.release();
}

void MapStorage::saveOdoGraph()
{
    size_t sizeKF = mvKFs.size();

    mOdoNextId = vector<int>(sizeKF, -1);
    for (size_t i = 0; i != sizeKF; ++i) {
        const PtrKeyFrame& pKF = mvKFs[i];
        const PtrKeyFrame nextKF = pKF->mOdoMeasureFrom.first;

        if (nextKF != nullptr) {
            mOdoNextId[i] = nextKF->mIdKF;
        }
    }

    FileStorage file(mMapPath + mMapFile, FileStorage::APPEND);

    file << "OdoGraphNextKF"
         << "[";
    for (size_t i = 0; i != sizeKF; ++i) {
        const PtrKeyFrame& pKF = mvKFs[i];
        Mat measure = pKF->mOdoMeasureFrom.second.measure;
        Mat info = pKF->mOdoMeasureFrom.second.info;

        file << "{";

        file << "NextId" << mOdoNextId[i];
        file << "Measure" << measure;
        file << "Info" << info;

        file << "}";
    }
}

void MapStorage::saveFtrGraph()
{
    size_t sizeKF = mvKFs.size();

    FileStorage file(mMapPath + mMapFile, FileStorage::APPEND);

    file << "FtrGraphPairs"
         << "[";
    for (size_t i = 0; i != sizeKF; ++i) {
        const PtrKeyFrame& pKFi = mvKFs[i];
        unsigned long idi = pKFi->mIdKF;

        for (auto it = pKFi->mFtrMeasureFrom.begin(), itend = pKFi->mFtrMeasureFrom.end(); it != itend; ++it) {
            PtrKeyFrame pKFj = it->first;
            unsigned long idj = pKFj->mIdKF;
            Mat measure = it->second.measure;
            Mat info = it->second.info;

            file << "{";

            file << "PairId" << Point2i(idi, idj);
            file << "Measure" << measure;
            file << "Info" << info;

            file << "}";
        }
    }
    file << "]";

    file.release();
}

void MapStorage::loadKeyFrames()
{
    assert(!mvMPs.empty());

    mvKFs.clear();
    vector<CovisibleRelationship> vCovisRelation;

    FileStorage file(mMapPath + mMapFile, FileStorage::READ);
    FileNode nodeKFs = file["KeyFrames"];
    FileNodeIterator it = nodeKFs.begin(), itend = nodeKFs.end();

    for (; it != itend; ++it) {
        PtrKeyFrame pKF = make_shared<KeyFrame>();

        FileNode nodeKF = *it;

        nodeKF["Id"] >> pKF->mIdKF;
        nodeKF["N"] >> pKF->N;
        pKF->id = pKF->mIdKF;

        Mat pose;
        nodeKF["Pose"] >> pose;
        pKF->setPose(pose);

        Point3f odo;
        nodeKF["Odometry"] >> odo;
        pKF->odom = Se2(odo.x, odo.y, odo.z);

        nodeKF["ScaleFactor"] >> pKF->mfScaleFactor;

        pKF->mvKeyPoints.clear();
        FileNode nodeKP = nodeKF["KeyPoints"];
        {
            vector<KeyPoint> vKPs;
            vKPs.reserve(pKF->N);

            FileNodeIterator itKP = nodeKP.begin(), itKPend = nodeKP.end();
            for (; itKP != itKPend; itKP++) {
                KeyPoint kp;
                (*itKP)["pt"] >> kp.pt;
                kp.octave = (int)(*itKP)["octave"];
                kp.angle = (float)(*itKP)["angle"];
                kp.response = (float)(*itKP)["response"];
                vKPs.push_back(kp);
            }
            pKF->mvKeyPoints = vKPs;
        }

        nodeKF["Descriptor"] >> pKF->mDescriptors;


        FileNode nodeViewMPsId = nodeKF["ViewMPsId"];
        if (nodeKP.size() != nodeViewMPsId.size())
            cerr << "Wrong ViewMPs size in loading " << endl;

        vector<long> vViewMPsId;
        vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> vInfo;
        vViewMPsId.reserve(pKF->N);
        vInfo.reserve(pKF->N);

        {
            FileNodeIterator itMPId = nodeViewMPsId.begin(), itMPend = nodeViewMPsId.end();
            for (; itMPId != itMPend; itMPId++) {
                long id;
                (*itMPId) >> id;
                vViewMPsId.push_back(id);
            }
            assert(vViewMPsId.size() == pKF->N);
            for (size_t i = 0; i < pKF->N; ++i) {
                if (vViewMPsId[i] == -1)
                    pKF->setObservation(nullptr, i);
                else
                    pKF->setObservation(mvMPs[vViewMPsId[i]], i);
            }
        }

        pKF->mvViewMPsInfo.clear();
        FileNode nodeMPInfo = nodeKF["ViewMPInfo"];
        {
            FileNodeIterator itInfo = nodeMPInfo.begin(), itInfoend = nodeMPInfo.end();
            for (; itInfo != itInfoend; itInfo++) {
                Mat info;
                (*itInfo) >> info;
                vInfo.push_back(toMatrix3d(info));
            }
            pKF->mvViewMPsInfo = vInfo;
        }

        // 先记录共视关系, 等所有的KF都加载好并且排序后再添加
        FileNode nodeCovisKFs = nodeKF["Covisibilities"];
        {
            FileNodeIterator itCovisKF = nodeCovisKFs.begin(), itInfoend = nodeCovisKFs.end();
            for (; itCovisKF != itInfoend; itCovisKF++) {
                FileNode nodeCovis = *itCovisKF;
                CovisibleRelationship cr;
                nodeCovis["CovisibleKFID"] >> cr.thatKFid;
                nodeCovis["CovisibleMPCount"] >> cr.covisbleCount;
                cr.thisKFid = pKF->mIdKF;
                vCovisRelation.push_back(cr);
            }
        }

        mvKFs.push_back(pKF);
    }

    file.release();

    std::sort(mvKFs.begin(), mvKFs.end(),
              [](const PtrKeyFrame& lhs, const PtrKeyFrame& rhs) { return lhs->mIdKF < rhs->mIdKF; });
    for (const auto& cr : vCovisRelation) {
        PtrKeyFrame& pKF = mvKFs[cr.thisKFid];
        pKF->addCovisibleKF(mvKFs[cr.thatKFid], cr.covisbleCount);
    }

    if (Config::NeedVisualization) {
        for (size_t i = 0, iend = mvKFs.size(); i != iend; ++i) {
            const PtrKeyFrame& pKF = mvKFs[i];
            Mat img = imread(mMapPath + to_string(i) + ".bmp", CV_LOAD_IMAGE_GRAYSCALE);
            if (!img.data) {
                fprintf(stderr, "[MapStore] KeyFrame image '%ld.bmp' doesn't exist!!\n", i);
                continue;
            }
            img.copyTo(pKF->mImage);
        }
    }

    std::cout << "[MapStorage] Load " << mvKFs.size() << " KeyFrames." << std::endl;
}

void MapStorage::loadMapPoints()
{
    mvMPs.clear();

    FileStorage file(mMapPath + mMapFile, FileStorage::READ);
    FileNode nodeMPs = file["MapPoints"];
    FileNodeIterator it = nodeMPs.begin(), itend = nodeMPs.end();

    for (; it != itend; ++it) {
        PtrMapPoint pMP = make_shared<MapPoint>();

        FileNode nodeMP = *it;

        pMP->mId = (int)nodeMP["Id"];

        Point3f pos;
        nodeMP["Pose"] >> pos;
        pMP->setPos(pos);

        pMP->setGoodPrl(true);

        mvMPs.push_back(pMP);
    }

    file.release();

    std::sort(mvMPs.begin(), mvMPs.end(),
              [](const PtrMapPoint& lhs, const PtrMapPoint& rhs) { return lhs->mId < rhs->mId; });
    std::cout << "[MapStorage] Load " << mvMPs.size() << " MapPoints." << std::endl;
}

void MapStorage::loadObservations()
{
    FileStorage file(mMapPath + mMapFile, FileStorage::READ);
    Mat Index, Obs;

    file["Observations"] >> Obs;

    file["ObservationIndex"] >> Index;

    // mObservations = Obs;
    // Mat_<int> Index = Index_;

    int sizeKF = Obs.rows;
    int sizeMP = Obs.cols;

    for (int i = 0; i != sizeKF; ++i) {
        for (int j = 0; j < sizeMP; ++j) {
            if (Obs.at<int>(i, j)) {
                PtrKeyFrame& pKF = mvKFs[i];
                PtrMapPoint pMP = mvMPs[j];
                int idx = Index.at<int>(i, j);

                pKF->setObservation(pMP, idx);
                pMP->addObservation(pKF, idx);
            }
        }
    }

    file.release();
}

void MapStorage::loadCovisibilityGraph()
{
    FileStorage file(mMapPath + mMapFile, FileStorage::READ);

    Mat mCovisibilityGraph;

    file["CovisibilityGraph"] >> mCovisibilityGraph;

    int sizeKF = mCovisibilityGraph.rows;
    int sizeMP = mCovisibilityGraph.cols;

    for (int i = 0; i != sizeKF; ++i) {
        for (int j = 0; j < sizeMP; ++j) {
            if (mCovisibilityGraph.at<int>(i, j)) {
                PtrKeyFrame& pKFi = mvKFs[i];
                PtrKeyFrame& pKFj = mvKFs[j];
                pKFi->addCovisibleKF(pKFj);
                pKFj->addCovisibleKF(pKFi);
            }
        }
    }

    file.release();
}

void MapStorage::loadOdoGraph()
{
    FileStorage file(mMapPath + mMapFile, FileStorage::READ);

    FileNode nodeOdos = file["OdoGraphNextKF"];
    FileNodeIterator it = nodeOdos.begin(), itend = nodeOdos.end();

    for (size_t i = 0; it != itend; ++it, ++i) {
        FileNode nodeOdo = *it;

        int j = (int)nodeOdo["NextId"];
        if (j < 0)
            continue;

        Mat measure, info;
        nodeOdo["Measure"] >> measure;
        nodeOdo["Info"] >> info;

        cv::Mat Info = (info);

        PtrKeyFrame& pKFi = mvKFs[i];
        PtrKeyFrame& pKFj = mvKFs[j];
        pKFi->setOdoMeasureFrom(pKFj, measure, Info);
        pKFj->setOdoMeasureTo(pKFi, measure, Info);
    }
    file.release();
}

void MapStorage::loadFtrGraph()
{
    FileStorage file(mMapPath + mMapFile, FileStorage::READ);

    FileNode nodeFtrs = file["FtrGraphPairs"];
    FileNodeIterator it = nodeFtrs.begin(), itend = nodeFtrs.end();

    for (; it != itend; ++it) {
        FileNode nodeFtr = (*it);

        Point2i pairId;
        Mat measure, info;
        nodeFtr["PairId"] >> pairId;
        nodeFtr["Measure"] >> measure;
        nodeFtr["Info"] >> info;
        cv::Mat Info = (info);

        PtrKeyFrame& pKFi = mvKFs[pairId.x];
        PtrKeyFrame& pKFj = mvKFs[pairId.y];
        pKFi->addFtrMeasureFrom(pKFj, measure, Info);
        pKFj->addFtrMeasureTo(pKFi, measure, Info);
    }

    file.release();
}

void MapStorage::loadToMap()
{
    mpMap->clear();
    for (size_t i = 0, iend = mvKFs.size(); i != iend; ++i) {
        mpMap->insertKF(mvKFs[i]);
    }
    for (size_t i = 0, iend = mvMPs.size(); i != iend; ++i) {
        mpMap->insertMP(mvMPs[i]);
    }
}

void MapStorage::clearData()
{
    mvKFs.clear();
    mvMPs.clear();
    mCovisibilityGraph.release();
    mObservations.release();
    mOdoNextId.clear();
}

}  // namespace se2lam
