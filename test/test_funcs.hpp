#ifndef TEST_FUNCS_HPP
#define TEST_FUNCS_HPP

#include <boost/filesystem.hpp>

namespace bf = boost::filesystem;

struct IMAGE {
    IMAGE(const string& s, const double& t) : fileName(s), timeStamp(t) {}

    string fileName;
    double timeStamp;

    // for map container
    bool operator<(const IMAGE& that) const { return timeStamp < that.timeStamp; }
};

inline bool IMAGE_CMP_LT(const IMAGE& r1, const IMAGE& r2)
{
    return r1.timeStamp < r2.timeStamp;
}


void ReadImageNamesFromFolder(const std::string& folder, std::vector<std::string>& names)
{
    bf::path path(folder);
    if (!bf::exists(path)) {
        std::cerr << "[Error] Data folder doesn't exist! " << path << std::endl;
        return;
    }

    vector<string> vImageNames;
    vImageNames.reserve(100);
    bf::directory_iterator end_iter;
    for (bf::directory_iterator iter(path); iter != end_iter; ++iter) {
        if (bf::is_directory(iter->status()))
            continue;
        if (bf::is_regular_file(iter->status()))
            vImageNames.push_back(iter->path().string());
    }

    if (vImageNames.empty()) {
        std::cerr << "[Error] No image data in the folder!" << std::endl;
        return;
    } else {
        std::cout << "[INFO ] Read " << vImageNames.size() << " images in " << folder << std::endl;
    }

    vImageNames.shrink_to_fit();
    names.swap(vImageNames);
}

#endif // TEST_FUNCS_HPP
