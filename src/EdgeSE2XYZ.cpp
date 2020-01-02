/**
 * This file is part of se2lam
 *
 * Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
 */

#include "EdgeSE2XYZ.h"
#include <cmath>
#include <g2o/types/slam3d/isometry3d_mappings.h>

namespace g2o
{
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


EdgeSE2XYZ::EdgeSE2XYZ() {}

EdgeSE2XYZ::~EdgeSE2XYZ() {}


bool EdgeSE2XYZ::read(std::istream& is)
{
    return true;
}

bool EdgeSE2XYZ::write(std::ostream& os) const
{
    return true;
}

void EdgeSE2XYZ::computeError()
{
    const VertexSE2* v1 = dynamic_cast<VertexSE2*>(_vertices[0]);
    const VertexSBAPointXYZ* v2 = dynamic_cast<VertexSBAPointXYZ*>(_vertices[1]);

    const SE3Quat Tbw = SE2ToSE3(v1->estimate().inverse());
    const SE3Quat Tcw = Tcb * Tbw;

    const Vector3d lc = Tcw.map(v2->estimate());

    _error = cam->cam_map(lc) - Vector2d(_measurement);
}


void EdgeSE2XYZ::linearizeOplus()
{
    const VertexSE2* v1 = dynamic_cast<VertexSE2*>(_vertices[0]);
    const VertexSBAPointXYZ* v2 = dynamic_cast<VertexSBAPointXYZ*>(_vertices[1]);

    const Vector3d vwb = v1->estimate().toVector();
    const Vector3d pi(vwb[0], vwb[1], 0);

    const SE3Quat Tcw = Tcb * SE2ToSE3(v1->estimate().inverse());
    const Matrix3d Rcw = Tcw.rotation().toRotationMatrix();

    const Vector3d lw = v2->estimate();
    const Vector3d lc = Tcw.map(lw);
    const double zc = lc(2);
    const double zc_inv = 1. / zc;
    const double zc_inv2 = zc_inv * zc_inv;

    const double& fx = cam->focal_length;
    const double& fy = fx;

    Matrix23d de_dlc;
    de_dlc << fx * zc_inv, 0, -fx * lc(0) * zc_inv2, 0, fy * zc_inv, -fy * lc(1) * zc_inv2;

    _jacobianOplusXj = de_dlc * Rcw;

    _jacobianOplusXi.block<2, 2>(0, 0) = -_jacobianOplusXj.block<2, 2>(0, 0);
    _jacobianOplusXi.block<2, 1>(0, 2) = (_jacobianOplusXj * skew(lw - pi)).block<2, 1>(0, 2);
}


}  // namespace g2o
