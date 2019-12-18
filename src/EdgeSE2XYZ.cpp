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

G2O_REGISTER_TYPE(EDGE_SE2, PreEdgeSE2);
G2O_REGISTER_TYPE(EDGE_SE2:XYZ, EdgeSE2XYZ);

using namespace std;
using namespace Eigen;

Matrix3d d_inv_d_se2(const SE2& _se2)
{
    double c = cos(_se2.rotation().angle());
    double s = sin(_se2.rotation().angle());
    double x = _se2.translation()(0);
    double y = _se2.translation()(1);
    Matrix3d ret;
    ret << -c, -s, s * x - c * y, s, -c, c * x + s * y, 0, 0, -1;
    return ret;
}

SE3Quat SE2ToSE3(const SE2& _se2)
{
    SE3Quat ret;
    ret.setTranslation(Vector3d(_se2.translation()(0), _se2.translation()(1), 0));
    ret.setRotation(Quaterniond(AngleAxisd(_se2.rotation().angle(), Vector3d::UnitZ())));
    return ret;
}

SE2 SE3ToSE2(const SE3Quat& _se3)
{
    Vector3d eulers = internal::toEuler(_se3.rotation().matrix());
    return SE2(_se3.translation()(0), _se3.translation()(1), eulers(2));
}


bool EdgeSE2XYZ::read(istream& is)
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

bool EdgeSE2XYZ::write(ostream& os) const
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

    // v1是Twb，所以这里要用逆
    SE3Quat Tbw = SE2ToSE3(v1->estimate().inverse());
    SE3Quat Tcw = Tcb * Tbw;

    // 地图点的观测在相机坐标系下的坐标
    Vector3d lc = Tcw.map(v2->estimate());

    // 误差=观测-投影, 这里是 投影-观测, 效果一样
    // cam_map()把相机坐标系下三维点用内参转换为图像坐标输出
    _error = cam->cam_map(lc) - Vector2d(_measurement);
}
/*
#ifdef CUSTOMIZE_JACOBIAN_SE2XYZ
//! 计算雅克比
void EdgeSE2XYZ::linearizeOplus()
{
    VertexSE2* v1 = static_cast<VertexSE2*>(_vertices[0]);
    VertexSBAPointXYZ* v2 = static_cast<VertexSBAPointXYZ*>(_vertices[1]);

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

    // 误差对lc的偏导
    Matrix23d de_dlc;
    de_dlc << fx * zc_inv, 0, -fx * lc(0) * zc_inv2, 0, fy * zc_inv, -fy * lc(1) * zc_inv2;
    // 误差对空间点lw的偏导
    Matrix23d J_lw = de_dlc * Rcw;
    // lc对se2位姿的倒数
    Matrix32d dlc_dpi = -Tcb.rotation().toRotationMatrix().block<3, 2>(0, 0);
    Vector3d dlc_dtheta = Rcw * skew(lw - pi).block<3, 1>(0, 2);
    Matrix23d J_v1;
    J_v1.block<2, 2>(0, 0) = de_dlc * dlc_dpi;
    J_v1.block<2, 1>(0, 2) = de_dlc * dlc_dtheta;

    _jacobianOplusXi = J_v1;
    _jacobianOplusXj = J_lw;

    /*
    {  //! ------------- check jacobians -----------------
        // 数值法求解雅克比, 看结果是否跟自己解析写的一样
        const double eps = 1e-6;
        const double eps1[3] = {eps, 0, 0};
        const double eps2[3] = {0, eps, 0};
        const double eps3[3] = {0, 0, eps};
        cout << "EdgeSE2XYZ Jacobian 解析法 Xi = " << endl << _jacobianOplusXi << endl;
        Matrix23d J1, J2;
        for (int i = 0; i < 3; ++i) {
            const double epsi[3] = {eps1[i], eps2[i], eps3[i]};
            SE3Quat Tbw_eps = SE2ToSE3(v1->estimate() * SE2(epsi[0], epsi[1], epsi[2])).inverse();
            SE3Quat Tcw_eps = Tcb * Tbw_eps;
            Eigen::Vector3d lc_eps1 = Tcw_eps.map(v2->estimate());
            Eigen::Vector2d Ji = (cam->cam_map(lc_eps1) - cam->cam_map(lc)) / eps;
            J1.block<2, 1>(0, i) = Ji;
        }
        cout << "EdgeSE2XYZ Jacobian 数值法 Xi = " << endl << J1 << endl;

        cout << "EdgeSE2XYZ Jacobian 解析法 Xj = " << endl << _jacobianOplusXj << endl;
        for (int i = 0; i < 3; ++i) {
            const double epsi[3] = {eps1[i], eps2[i], eps3[i]};
            const Eigen::Vector3d lw_eps = lw + Eigen::Vector3d(epsi[0], epsi[1], epsi[2]);
            const Eigen::Vector3d lc_eps2 = Tcw.map(lw_eps);
            Eigen::Vector2d Ji = (cam->cam_map(lc_eps2) - cam->cam_map(lc)) / eps;
            J2.block<2, 1>(0, i) = Ji;
        }
        cout << "EdgeSE2XYZ Jacobian 数值法 Xj = " << endl << J2 << endl;
    }

}
#endif
*/
bool PreEdgeSE2::read(istream& is)
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

bool PreEdgeSE2::write(ostream& os) const
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

void PreEdgeSE2::computeError()
{
    const VertexSE2* v1 = static_cast<const VertexSE2*>(_vertices[0]);
    const VertexSE2* v2 = static_cast<const VertexSE2*>(_vertices[1]);
    Matrix2D Ri = v1->estimate().rotation().toRotationMatrix();
    Vector2D ri = v1->estimate().translation();
    double ai = v1->estimate().rotation().angle();
    double aj = v2->estimate().rotation().angle();
    Vector2D rj = v2->estimate().translation();

    Vector2D rm = _measurement.head<2>();
    Matrix2D Rm = Eigen::Rotation2Dd(_measurement[2]).toRotationMatrix();

    // FIXME 这里的残差应该是SE2减法!
    _error.head<2>() = Rm.transpose() * (Ri.transpose() * (rj - ri) - rm);
    _error[2] = normalize_theta(aj - ai - _measurement[2]);
}

//! FIXME 这里雅克比和g2o::EdgeSE2中一致了, 但与数值结果不一致. 可能是数值计算方式有误!
//! FIXME 这里的test_BA效果和g2o::EdgeSE2不一样.
//! @Vance 2019.12.18
/*
void PreEdgeSE2::linearizeOplus()
{
    const VertexSE2* v1 = static_cast<const VertexSE2*>(_vertices[0]);
    const VertexSE2* v2 = static_cast<const VertexSE2*>(_vertices[1]);
    Matrix2D Ri = v1->estimate().rotation().toRotationMatrix();
    Vector2D ri = v1->estimate().translation();
    Vector2D rj = v2->estimate().translation();
    Vector2D rij = rj - ri;
    Vector2D rij_x(-rij[1], rij[0]);

    _jacobianOplusXi.block<2, 2>(0, 0) = -Ri.transpose();
    _jacobianOplusXi.block<2, 1>(0, 2) = -Ri.transpose() * rij_x;
    _jacobianOplusXi.block<1, 2>(2, 0).setZero();
    _jacobianOplusXi(2, 2) = -1;  // 这里应该是-1

    _jacobianOplusXj.setIdentity();
    _jacobianOplusXj.block<2, 2>(0, 0) = Ri.transpose();

    Matrix3D Rm = Matrix3D::Identity();
    Rm.block<2, 2>(0, 0) = _inverseMeasurement.rotation().toRotationMatrix();
    _jacobianOplusXi = Rm.transpose() * _jacobianOplusXi;
    _jacobianOplusXj = Rm.transpose() * _jacobianOplusXj;

    /*
    {  //! ------------- check jacobians -----------------
        // 数值法求解雅克比, 看结果是否跟自己解析写的一样
        const double eps = 1e-6;
        const double eps1[3] = {eps, 0, 0};
        const double eps2[3] = {0, eps, 0};
        const double eps3[3] = {0, 0, eps};
        cout << "PreEdgeSE2 Jacobian 解析法 Xi = " << endl << _jacobianOplusXi << endl;
        Matrix3D J1, J2;
        for (int i = 0; i < 3; ++i) {
            const double epsi[3] = {eps1[i], eps2[i], eps3[i]};
            SE2 v1_eps = v1->estimate() * SE2(epsi[0], epsi[1], epsi[2]);
            SE2 delta_eps = _inverseMeasurement * (v1_eps.inverse() * v2->estimate());
            Vector3D Ji = (SE2(_error).inverse() * delta_eps).toVector() / eps;
            J1.block<3, 1>(0, i) = Ji;
        }
        J1 = Rm.transpose() * J1;
        cout << "PreEdgeSE2 Jacobian 数值法 Xi = " << endl << J1 << endl;

        cout << "PreEdgeSE2 Jacobian 解析法 Xj = " << endl << _jacobianOplusXj << endl;
        for (int j = 0; j < 3; ++j) {
            const double epsi[3] = {eps1[j], eps2[j], eps3[j]};
            SE2 v2_eps = v2->estimate() * SE2(epsi[0], epsi[1], epsi[2]);
            SE2 delta_eps = _inverseMeasurement * (v1->estimate().inverse() * v2_eps);
            Vector3D Jj = (SE2(_error).inverse() * delta_eps).toVector() / eps;
            J2.block<3, 1>(0, j) = Jj;
        }
        J2 *= Rm.transpose() * J2;
        cout << "PreEdgeSE2 Jacobian 数值法 Xj = " << endl << J2 << endl;
    }

}
*/
}  // namespace g2o
