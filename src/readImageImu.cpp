//
// Created by lmp on 19-10-23.
//

#include "readImageImu.h"

//根据输入路径和图像名字获取图像，获取时间戳
int readImageImu::playback_mydate(const string fname, int t_offset, cv::Mat &gimgf) {
    static bool b_open = false;
    static int count = 0;
    static string dir(fname);
    static ifstream fin(fname);

    double time_msec = 0;
    string str, name;
    size_t p, q;

    // Open file only once（获取图像所在文件夹路径）
    if (!b_open) {
        if (fin.fail()) {
            cout << "Error in opening a file: " << fname << endl;
            return false;
        }

        p = dir.rfind("/");
        if (p != string::npos) dir.erase(p + 1, dir.size() - p - 1);
        getline(fin, str);
        str = str.substr(0, str.length() - 1);
        dir += str + '/';

        b_open = true;
    }

    // Read image and timestamp
    if (getline(fin, str)) {
        //获取读图路径
        str = str.substr(0, str.length() - 1);
        name = dir + str;

        cv::Mat img = cv::imread(name, 1);
        if (img.empty()) {
            printf("Error in loading %s\n", name.c_str());
            return -1;
        }

        if (img.cols != gimgf.cols || img.rows != gimgf.rows) {
            printf("Error: Wrong image size, %d x %d\n", img.cols, img.rows);
            return -1;
        }

        //获取时间戳
        p = name.rfind("w");
        q = name.rfind(".");
        if (p == string::npos || q == string::npos) {
            cout << "Error in reading timestamp from image file name: " << name << endl;
            return -1;
        }
        string str = name.substr(p + 1, q - p - 1);
        time_msec = (atof(str.c_str()) + t_offset) / 1000;

        //printf("%s, %d\n", str.c_str(), time_msec);
        smooth_image_float(img, gimgf, 3);
    } else
        return -1;
    int time_msec_int = int(time_msec);
    return time_msec_int;
}

//图像平滑滤波
void readImageImu::smooth_image_float(cv::Mat img, cv::Mat &gimgf, int w) {
    cv::Mat gimg;
    if (img.channels() == 3) {
        cv::cvtColor(img, gimg, cv::COLOR_BGR2GRAY);
    } else if (img.channels() == 1) {
        img.copyTo(gimg);
    }
    cv::GaussianBlur(img, gimgf, cv::Size(3, 3), 1);
    return;
}


//根据IMU数据计算相邻两帧的夹角
bool readImageImu::compute_theta_IMU(int ptime, int ctime, const string fname, const string shape_fname,
                                     Out_IMU_Data &outImuData) {
    static bool b_open = false;
    static int index = 0;
    static vector<IMU_DATA> vdata;

    double wi, xi, yi;
    ifstream fin(fname);
    ifstream fin_sh(shape_fname);

    // Open file only once
    if (!b_open) {
        if (fin_sh.fail()) {
            cout << "Error in opening IMU shape file " << shape_fname << endl;
            return false;
        }

        double bias_acc[3], bias_gyro[3];
        double Ca[3][3], Cw[3][3];
        //未使用
        for (int i = 0; i < 3; i++) fin_sh >> bias_acc[i];
        for (int k = 0; k < 3; k++)
            for (int i = 0; i < 3; i++) fin_sh >> Ca[k][i];

        //只用了Cw,没有使用Ca;
        for (int i = 0; i < 3; i++) fin_sh >> bias_gyro[i];//陀螺仪偏差
        for (int k = 0; k < 3; k++)
            for (int i = 0; i < 3; i++) fin_sh >> Cw[k][i];//shape matrix

        if (fin.fail()) {
            cout << "Error in opening IMU file " << fname << endl;
            return false;
        }

        int t1;
        double odo[3], a[3], w[3], dtheta;
        IMU_DATA m;
        while (!fin.eof()) {
            w[1] = 0;
            w[0] = 0;
            fin >> t1 >> odo[0] >> odo[1] >> w[2] >> a[0] >> a[1] >> a[2] >> dtheta;
            for (int i = 0; i < 3; i++) a[i] = a[i] - bias_acc[i]; //未使用
            for (int i = 0; i < 3; i++) w[i] = w[i] - bias_gyro[i]; //使用z-v
            //w[2] = w[2] * 180 / 3.14;
            //AddGaussianDistribution(w);//添加高斯白噪声
            m.time_msec = t1 / 1000;
            m.acc[0] = Ca[0][0] * a[0] + Ca[0][1] * a[1] + Ca[0][2] * a[2];
            m.acc[1] = Ca[1][0] * a[0] + Ca[1][1] * a[1] + Ca[1][2] * a[2];
            m.acc[2] = Ca[2][0] * a[0] + Ca[2][1] * a[1] + Ca[2][2] * a[2];
            m.rate[0] = Cw[0][0] * w[0] + Cw[0][1] * w[1] + Cw[0][2] * w[2]; //S转置的逆*（z-v）=Rt_t+1
            m.rate[1] = Cw[1][0] * w[0] + Cw[1][1] * w[1] + Cw[1][2] * w[2];
            m.rate[2] = Cw[2][0] * w[0] + Cw[2][1] * w[1] + Cw[2][2] * w[2];
            m.x = odo[0];
            m.y = odo[1];
            vdata.push_back(m);

            //printf("%6d %6.1f %6.1f %6.1f\n", t1, m.rate[0], m.rate[1], m.rate[2]);
        }

        printf("IMU: %d samples are loaded\n", (int) vdata.size());

        b_open = true;
    }

    double q[4] = {1, 0, 0, 0};
    int deltaT = 2;
    int tt, n = (int) vdata.size();

    bool t = false;
    bool f = true;
    for (tt = ptime; tt < ctime;) {
        double a = 1.0;
        int ti = (int) vdata[index].time_msec;
        int tj = (int) vdata[index + 1].time_msec;

        if (tt < ti) {//图像时间戳在陀螺仪之前
            a = ((ti - tt) < 10) ? 1.0 : 0.0;//当时间差值小于30ms时a=1,否则为0
        } else if (tt >= ti && tt < tj) {//图像时间戳在当前陀螺仪与下一陀螺仪时间戳之间时
            a = (double) (tj - tt) / (double) (tj - ti);//根据距离哪个陀螺仪时间戳的远近确定权重
            t = true;
        } else if (tt >= tj) {
            if (index++ > n - 1) break;
            else continue;
        } else break;

        // first-order approximation
        double w[3], x, y;
        for (int i = 0; i < 3; i++) w[i] = a * vdata[index].rate[i] + (1 - a) * vdata[index + 1].rate[i];
        x = a * vdata[index].x + (1 - a) * vdata[index + 1].x;
        y = a * vdata[index].y + (1 - a) * vdata[index + 1].y;
        if (t) {
            if (f) {
                wi = w[2];
                f = false;
            }
        }

        //printf("INS: %d/%d %8d %6.2f %8.4f %8.4f %8.4f\n", index, vdata[index].time_msec, tt, 1-a, w[0], w[1], w[2]);
        outImuData.dtheta = w[2] - wi;
        outImuData.x = 1000 * x;
        outImuData.y = 1000 * y;
        outImuData.theta = normalizeAngle(w[2]);

        tt += deltaT;
    }

    return true;
}

double readImageImu::normalizeAngle(double angle) {
    return angle + 2 * M_PI * floor((M_PI - angle) / (2 * M_PI));
}
