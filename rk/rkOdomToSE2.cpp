#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

using namespace std;
namespace bf = boost::filesystem;

struct OdomRaw {
    long long int timestamp;
    double x, y, theta;
    double linearVelX, AngularVelZ;
    double deltaDistance, deltaTheta;

    OdomRaw() {
        timestamp = 0;
        x = y = theta = 0.0;
        linearVelX = AngularVelZ = 0.0;
        deltaDistance = deltaTheta = 0.0;
    }
};

struct OdomFrame {
    long long int timestamp;
    double linearVelX, AngularVelZ;
    double deltaDistance, deltaTheta;

    OdomFrame() {
        timestamp = 0;
        linearVelX = AngularVelZ = 0.0;
        deltaDistance = deltaTheta = 0.0;
    }
};


vector<OdomFrame> readOdomeFrame(const string& file) {
    vector<OdomFrame> result;

    ifstream reader;
    reader.open(file.c_str());
    if (!reader) {
        fprintf(stderr, "%s file open error!\n", file.c_str());
        return result;
    }

    // get data
    while (reader.peek() != EOF) {
        OdomFrame ofr;
        reader >> ofr.timestamp >> ofr.deltaTheta >> ofr.AngularVelZ
               >> ofr.deltaDistance >> ofr.linearVelX;
        if (ofr.timestamp != 0)
            result.push_back(ofr);
    }

    reader.close();

    return result;
}

vector<OdomRaw> readOdomeRaw(const string& file) {
    vector<OdomRaw> result;

    ifstream reader;
    reader.open(file.c_str());
    if (!reader) {
        fprintf(stderr, "%s file open error!\n", file.c_str());
        return result;
    }

    // get data
    while (reader.peek() != EOF) {
        OdomRaw oraw;
        reader >> oraw.timestamp >> oraw.x >> oraw.y >> oraw.theta
               >> oraw.linearVelX >> oraw.AngularVelZ
               >> oraw.deltaDistance >> oraw.deltaTheta;
        if (oraw.timestamp != 0)
            result.push_back(oraw);
    }

    reader.close();

    return result;
}

vector<string> readFolderFiles(const string& folder) {
    vector<string> files;

    bf::path folderPath(folder);
    if (!bf::exists(folderPath))
        return files;

    bf::directory_iterator end_iter;
    for (bf::directory_iterator iter(folderPath); iter != end_iter; ++iter) {
        if (bf::is_directory(iter->status()))
            continue;

        if (bf::is_regular_file(iter->status()))
            files.push_back(iter->path().string());
    }

    return files;
}

vector<string> readImageTimestamp(const vector<string>& vTimes) {
    vector<string> times;
    for (const auto& ts : vTimes) {
        size_t s = ts.find_last_of('w');
        size_t e = ts.find_last_of('.');
        times.push_back(ts.substr(s+1, e-s-1));
    }
    return times;
}

double normalizeAngle(double angle) {
    return angle + 2*M_PI*floor((M_PI - angle)/(2*M_PI));
}

int main(int argc, char *argv[])
{
    if (argc < 1) {
        cerr << "[Error] Usage: " << argv[0] << " <rk_dataset_folder>" << endl;
        return -1;
    }

    string datasetFolder = string(argv[1]);
    if (!bf::exists(bf::path(datasetFolder))) {
        cerr << "[Error] " << datasetFolder << " folder dosn't exist!" << endl;
        return -1;
    }

    if (!boost::ends_with(datasetFolder, "/"))
        datasetFolder += "/";
    string odomRawFile = datasetFolder + "OdomRaw.txt";
    string imageFolder = datasetFolder + "slamimg/";
    if (!bf::exists(bf::path(imageFolder))) {
        cerr << "[Error] " << imageFolder << " folder dosn't exist!" << endl;
        return -1;
    }
    if (!bf::is_regular_file(bf::path(odomRawFile))) {
        cerr << "[Error] " << odomRawFile << " dosn't exist!" << endl;
        return -1;
    }

    vector<OdomRaw> vOdomRaws = readOdomeRaw(odomRawFile);
    vector<string> vImageNames = readFolderFiles(imageFolder);
    if (vImageNames.size() < 1) {
        cerr << "[Error] No images in this folder: " << datasetFolder << endl;
        return -1;
    }
    vector<string> vImageTimestamps = readImageTimestamp(vImageNames);
    if (vImageTimestamps.size() < 1) {
        cerr << "[Error] Read timestamps error!" << endl;
        return -1;
    }

    // upper_bound
    vector<long long int> timeOdomRaw, timeOdomFrame;
    for (auto & r : vOdomRaws)
        timeOdomRaw.push_back(r.timestamp);
    for (auto & f : vImageTimestamps)
        timeOdomFrame.push_back(stoll(f));
    sort(timeOdomFrame.begin(), timeOdomFrame.end());
    cout << "time stamp size: " << timeOdomFrame.size() << ", " << timeOdomRaw.size() << endl;

    string outOdomRaw = datasetFolder + "odo_raw.txt";
    string outOdomSyned = datasetFolder + "odomSyned.txt";
    ofstream ofs(outOdomRaw, ios_base::out);
    ofstream ofs2(outOdomSyned, ios_base::out);
    if (!ofs.is_open()) {
        cerr << "[Error] Wrong output file path." << endl;
        return -1;
    }
    for (auto & t : timeOdomFrame) {
        size_t r = static_cast<size_t>(
                    upper_bound(timeOdomRaw.begin(), timeOdomRaw.end(), t) - timeOdomRaw.begin());
        if (r > timeOdomRaw.size() - 1) {
            cout << "[Warning] 跳过此帧，因为找不到它的对应帧. " << t << endl;
            continue;
        }
        if (r == 0) {
            ofs << 1000*vOdomRaws[r].x << " " << 1000*vOdomRaws[r].y << " " << normalizeAngle(vOdomRaws[r].theta) << "\n";
            ofs2 << t << " " << vOdomRaws[r].timestamp << " " << vOdomRaws[r].timestamp << " "
                 << vOdomRaws[r].x << " " << vOdomRaws[r].y << " " << normalizeAngle(vOdomRaws[r].theta) << "\n";
            continue;
        }
        auto l = r - 1;
        double alpha = (t - vOdomRaws[l].timestamp)/(vOdomRaws[r].timestamp - vOdomRaws[l].timestamp);
        double x = vOdomRaws[l].x + alpha * (vOdomRaws[r].x - vOdomRaws[l].x);
        double y = vOdomRaws[l].y + alpha * (vOdomRaws[r].y - vOdomRaws[l].y);
        double theta = vOdomRaws[l].theta + alpha * (vOdomRaws[r].theta - vOdomRaws[l].theta);
        ofs << 1000*x << " " << 1000*y << " " << normalizeAngle(theta) << "\n";

        ofs2 << t << " " << vOdomRaws[r].timestamp << " " << vOdomRaws[l].timestamp
             << " " << x << " " << y << " " << normalizeAngle(theta) << "\n";
    }

    cout << "done." << endl;
    return 0;
}

