#include <string>
#include <iostream>
#include <fstream>
#include <unistd.h>

// need pangolin for plotting trajectory
#include <pangolin/pangolin.h>
#include <Eigen/Core>

using namespace std;
using namespace Eigen;

//#include "matplotlibcpp.h"
//namespace plt = matplotlibcpp;

typedef vector<Vector2f, Eigen::aligned_allocator<Vector2f>> vector_poses;

// function for read trajectory file and get poses
void getPoses(const string& file1, const string& file2,
              vector_poses& p_e, vector_poses& p_g);

// fuction for ICP
void ICP_solve(const vector_poses& p_e, const vector_poses& p_g, Matrix2f& R, Vector2f& t);

// function for plotting trajectory, don't edit this code
// start point is red and end point is blue
void DrawTrajectory(const vector_poses& poses1, const vector_poses& poses2);

//void drawTrajectoryMatplot(const vector_poses& poses1, const vector_poses& poses2)
//{
//    vector<float> x, y, x1, y1;
//    for (size_t i = 0; i < poses1.size(); ++i) {
//        x.push_back(poses1[i][0]);
//        y.push_back(poses1[i][1]);
//        x1.push_back(poses2[i][0]);
//        y1.push_back(poses2[i][1]);
//    }

//    plt::plot(x, y, "r--");
//    plt::plot(x1, y1, "b-");
//    plt::legend();
//    plt::show();
//}

// main
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <odomVI> <odomRaw>\n", argv[0]);
        return -1;
    }

    vector_poses poses_e, poses_g;
    getPoses(argv[1], argv[2], poses_e, poses_g);
    // 重复一次可以删掉
    poses_e.pop_back();
    poses_g.pop_back();
    printf("get %ld omdoVI and %ld odomRaw poses.\n", poses_e.size(), poses_g.size());
//    DrawTrajectory(poses_e, poses_g);

    Eigen::Matrix2f R;
    Eigen::Vector2f t;
    ICP_solve(poses_e, poses_g, R, t);
    cout << "R: " << R << endl;
    cout << "t: " << t << endl;

//    for (auto& pe : poses_e) {
//        pe = R * pe + t;
//    }

//    DrawTrajectory(poses_e, poses_g);

    return 0;
}

/*******************************************************************************************/
void DrawTrajectory(const vector_poses& poses1, const vector_poses& poses2) {
    if (poses1.empty() || poses2.empty()) {
        cerr << "Trajectory is empty!" << endl;
        return;
    }

    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Data logger object
    pangolin::DataLog log;
    // Optionally add named labels
    std::vector<std::string> labels;
    labels.push_back(std::string("poses_estimate"));
    labels.push_back(std::string("poses_groundtruth"));
    log.SetLabels(labels);

    // Define Projection and initial ModelView matrix
    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));


    while ( !pangolin::ShouldQuit() ) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glLineWidth(2);
        // draw poses 1
        for (size_t i=0; i<poses1.size()-1; ++i) {
            glColor3f(1.0f, 0.0f, 0.0f);
            glBegin(GL_LINES);
            auto p1 = poses1[i], p2 = poses1[i + 1];
            glVertex2d(p1[0], p1[1]);
            glVertex2d(p2[0], p2[1]);
            glEnd();
        }

        // draw poses 2
        for (size_t j=0; j<poses2.size()-1; ++j) {
            glColor3f(0.0f, 1.0f, 0.0f);
            glBegin(GL_LINES);
            auto p1 = poses2[j], p2 = poses2[j + 1];
            glVertex2d(p1[0], p1[1]);
            glVertex2d(p2[0], p2[1]);
            glEnd();
        }

        pangolin::FinishFrame();
        usleep(5000000);
    }

    return;
}


void getPoses(const string& file1, const string& file2,
              vector_poses& p_e, vector_poses& p_g)
{
    // implement pose reading code
    Eigen::Vector2f t_g(0, 0), t_e(0, 0);
    float frameID, theta;
    string str;

    ifstream fin;
    fin.open(file1.c_str(), ios_base::in);
    if (!fin) {
        cerr << "odomVI file can not be found!" << endl;
        return;
    }
    while (!fin.eof()) {
        getline(fin, str);
        istringstream ss(str);
        ss >> frameID >> t_e[0] >> t_e[1] >> theta;
        t_e /= 1000.;
        p_e.push_back(t_e);
    }
    fin.close();

    fin.open(file2.c_str(), ios_base::in);
    if (!fin) {
        cerr << "odomRaw file can not be found!" << endl;
        return;
    }
    while (!fin.eof()) {
        getline(fin, str);
        istringstream ss(str);
        ss >> t_g[0] >> t_g[1] >> theta;
        t_g /= 1000.;
        p_g.push_back(t_g);
    }
    fin.close();
}


void ICP_solve(const vector_poses& p_e, const vector_poses& p_g,
               Eigen::Matrix2f& R, Eigen::Vector2f& t){
    // center mass
    Eigen::Vector2f center_e, center_g;
    int N = p_e.size();
    for (int i=0; i<N; ++i) {
        center_e += p_e[i];
        center_g += p_g[i];
    }
//    printf("taltal: (%f, %f) and (%f, %f)\n", center_e[0], center_e[1], center_g[0], center_g[1]);
    center_e /= N;
    center_g /= N;
//    printf("ave: (%f, %f) and (%f, %f)\n", center_e[0], center_e[1], center_g[0], center_g[1]);

    // remove the center
    vector<Eigen::Vector2f> t_e, t_g;
    for (int i=0; i<N; ++i) {
        t_e.push_back(p_e[i] - center_e);
        t_g.push_back(p_g[i] - center_g);
    }

    // compute t_e * t_g^T
    Eigen::Matrix2f W = Eigen::Matrix2f::Zero();
    for (int i=0; i<N; ++i) {
        W += t_g[i] * t_e[i].transpose();
    }
//    cout << "W = " << W << endl;

    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix2f> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix2f U = svd.matrixU();
    Eigen::Matrix2f V = svd.matrixV();
//    cout << "U = " << U << endl;
//    cout << "V = " << V << endl;

    R = U * V.transpose();
    t = center_g - R * center_e;
}



