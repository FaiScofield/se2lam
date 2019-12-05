/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include "EdgeSE2XYZ.h"
#include "Thirdparty/g2o/g2o/core/factory.h"
#include "Thirdparty/g2o/g2o/types/slam3d/isometry3d_mappings.h"
#include "converter.h"
#include <cmath>

namespace g2o
{

G2O_REGISTER_TYPE(EDGE_SE2:PIROR, PreEdgeSE2);
G2O_REGISTER_TYPE(EDGE_SE2:XYZ, EdgeSE2XYZ);

using namespace std;
using namespace Eigen;

Eigen::Matrix3d d_inv_d_se2(const SE2& _se2)
{
    double c = std::cos(_se2.rotation().angle());
    double s = std::sin(_se2.rotation().angle());
    double x = _se2.translation()(0);
    double y = _se2.translation()(1);
    Matrix3d ret;
    ret << -c, -s, s * x - c * y, s, -c, c * x + s * y, 0, 0, -1;
    return ret;
}

g2o::SE3Quat SE2ToSE3(const g2o::SE2& _se2)
{
    SE3Quat ret;
    ret.setTranslation(Eigen::Vector3d(_se2.translation()(0), _se2.translation()(1), 0));
    ret.setRotation(Eigen::Quaterniond(AngleAxisd(_se2.rotation().angle(), Vector3d::UnitZ())));
    return ret;
}

g2o::SE2 SE3ToSE2(const SE3Quat& _se3)
{
    Eigen::Vector3d eulers = g2o::internal::toEuler(_se3.rotation().matrix());
    return g2o::SE2(_se3.translation()(0), _se3.translation()(1), eulers(2));
}


EdgeSE2XYZ::EdgeSE2XYZ()
{}

EdgeSE2XYZ::~EdgeSE2XYZ()
{}

bool EdgeSE2XYZ::read(std::istream& is)
{
    Vector2d m;
    is >> m[0] >> m[1];
    setMeasurement(m);
    for (int i = 0; i < 2; i++) {
        for (int j = i; j < 2; j++) {
            is >> information()(i, j);
            if (i != j)
                information()(j, i) = information()(i, j);
        }
    }
    setExtParameter(se2lam::toSE3Quat(se2lam::Config::Tbc));
    return true;
}

bool EdgeSE2XYZ::write(std::ostream& os) const
{
    Vector2d m = measurement();
    os << m[0] << " " << m[1] << " ";
    for (int i = 0; i < 2; i++) {
        for (int j = i; j < 2; j++) {
            os << " " << information()(i, j);
        }
    }
    return os.good();
}

//! 计算重投影误差，2维
void EdgeSE2XYZ::computeError()
{
    VertexSE2* v1 = static_cast<VertexSE2*>(_vertices[0]);
    VertexSBAPointXYZ* v2 = static_cast<VertexSBAPointXYZ*>(_vertices[1]);

    //!@Vance: v1是Twb，所以这里要用逆
    SE3Quat Tbw = SE2ToSE3(v1->estimate().inverse());
    SE3Quat Tcw = Tcb * Tbw;

    // 地图点的观测在相机坐标系下的坐标
    Vector3d lc = Tcw.map(v2->estimate());

    // 误差=观测-投影, 这里是 投影-观测, 效果一样
    // cam_map()把相机坐标系下三维点用内参转换为图像坐标输出
    _error = cam->cam_map(lc) - Vector2d(_measurement);
}

//! 计算雅克比
void EdgeSE2XYZ::linearizeOplus()
{
    VertexSE2* v1 = static_cast<VertexSE2*>(_vertices[0]);
    VertexSBAPointXYZ* v2 = static_cast<VertexSBAPointXYZ*>(_vertices[1]);

    Vector3d vwb = v1->estimate().toVector();

    SE3Quat Tcw = Tcb * SE2ToSE3(v1->estimate().inverse());
    Matrix3d Rcw = Tcw.rotation().toRotationMatrix();

    Vector3d pi(vwb[0], vwb[1], 0);

    Vector3d lw = v2->estimate();
    Vector3d lc = Tcw.map(lw);
    double zc = lc(2);
    double zc_inv = 1. / zc;
    double zc_inv2 = zc_inv * zc_inv;

    const double& fx = cam->focal_length;

    // 误差对空间点在相机坐标系下的坐标的偏导
    Matrix23d J_pi;
    J_pi << fx * zc_inv, 0, -fx * lc(0) * zc_inv2, 0, fx * zc_inv, -fx * lc(1) * zc_inv2;

    // 误差对空间点的偏导2x3
    Matrix23d J_pi_Rcw = J_pi * Rcw;

    // 误差对相机位姿se2的偏导2x3
    _jacobianOplusXi.block<2, 2>(0, 0) = -J_pi_Rcw.block<2, 2>(0, 0);
    _jacobianOplusXi.block<2, 1>(0, 2) = (J_pi_Rcw * skew(lw - pi)).block<2, 1>(0, 2);

    _jacobianOplusXj = J_pi_Rcw;
}

bool PreEdgeSE2::read(std::istream& is)
{
    Vector3D m;
    is >> m[0] >> m[1] >> m[2];
    setMeasurement(m);
    for (int i = 0; i < 3; i++) {
        for (int j = i; j < 3; j++) {
            is >> information()(i, j);
            if (i != j)
                information()(j, i) = information()(i, j);
        }
    }
    return true;
}

bool PreEdgeSE2::write(std::ostream& os) const
{
    Vector3D m = measurement();
    os << m[0] << " " << m[1] << " " << m[2] << " ";
    for (int i = 0; i < 3; i++) {
        for (int j = i; j < 3; j++) {
            os << " " << information()(i, j);
        }
    }
    return os.good();
}
}
