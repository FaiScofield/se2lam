/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include "Config.h"
#include "EdgeSE2XYZ.h"
#include "Thirdparty/g2o/g2o/core/factory.h"
#include "Thirdparty/g2o/g2o/types/slam3d/isometry3d_mappings.h"
#include "converter.h"
#include <cmath>

namespace g2o
{

#define ORIGINAL_ERROR_PRESE2

G2O_REGISTER_TYPE(EDGE_SE2, PreEdgeSE2);  // for g2o_viewer
G2O_REGISTER_TYPE(EDGE_SE2:XYZ, EdgeSE2XYZ);

using namespace std;
using namespace Eigen;

Matrix3D d_inv_d_se2(const SE2& _se2)
{
    double c = cos(_se2.rotation().angle());
    double s = sin(_se2.rotation().angle());
    double x = _se2.translation()(0);
    double y = _se2.translation()(1);
    Matrix3D ret;
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
    Vector3D eulers = internal::toEuler(_se3.rotation().matrix());
    return SE2(_se3.translation()(0), _se3.translation()(1), eulers(2));
}


bool EdgeSE2XYZ::read(istream& is)
{
    Vector2D m;
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
    Vector2D m = measurement();
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
    const VertexSE2* v1 = dynamic_cast<VertexSE2*>(_vertices[0]);
    const VertexSBAPointXYZ* v2 = dynamic_cast<VertexSBAPointXYZ*>(_vertices[1]);

    // v1是Twb，所以这里要用逆
    const SE3Quat Tbw = SE2ToSE3(v1->estimate().inverse());
    const SE3Quat Tcw = Tcb * Tbw;

    // 地图点的观测在相机坐标系下的坐标
    const Vector3D lc = Tcw.map(v2->estimate());

    // 误差=观测-投影, 这里是 投影-观测, 效果一样
    // cam_map()把相机坐标系下三维点用内参转换为图像坐标输出
    _error = cameraProject(lc) - Vector2D(_measurement);
}

Vector2D EdgeSE2XYZ::cameraProject(const Vector3D& xyz) const
{
    Vector2D res;
    res[0] = xyz[0] / xyz[2] * fx + cx;
    res[1] = xyz[1] / xyz[2] * fy + cy;
    return res;
}

void EdgeSE2XYZ::setCameraParameter(const cv::Mat& K)
{
    fx = K.at<float>(0, 0);
    fy = K.at<float>(1, 1);
    cx = K.at<float>(0, 2);
    cy = K.at<float>(1, 2);
}

#ifdef CUSTOMIZE_JACOBIAN_SE2XYZ
void EdgeSE2XYZ::linearizeOplus()
{
    const VertexSE2* v1 = dynamic_cast<VertexSE2*>(_vertices[0]);
    const VertexSBAPointXYZ* v2 = dynamic_cast<VertexSBAPointXYZ*>(_vertices[1]);

    const Vector3D vwb = v1->estimate().toVector();
    const Vector3D pi(vwb[0], vwb[1], 0);

    const SE3Quat Tcw = Tcb * SE2ToSE3(v1->estimate().inverse());
    const Matrix3D Rcw = Tcw.rotation().toRotationMatrix();

    const Vector3D lw = v2->estimate();
    const Vector3D lc = Tcw.map(lw);
    const double zc = lc(2);
    const double zc_inv = 1. / zc;
    const double zc_inv2 = zc_inv * zc_inv;

    Matrix23D de_dlc;  // 误差对lc的偏导
    de_dlc << fx * zc_inv, 0, -fx * lc(0) * zc_inv2, 0, fy * zc_inv, -fy * lc(1) * zc_inv2;

    _jacobianOplusXj = de_dlc * Rcw;

    _jacobianOplusXi.block<2, 2>(0, 0) = -_jacobianOplusXj.block<2, 2>(0, 0);
    _jacobianOplusXi.block<2, 1>(0, 2) = (_jacobianOplusXj * skew(lw - pi)).block<2, 1>(0, 2);

    /*
    {  //! ------------- check jacobians -----------------
        //! 已验证, 结论正确. 201912
        // 数值法求解雅克比, 看结果是否跟自己解析写的一样
        const double eps = 1e-6;
        const double eps1[3] = {eps, 0, 0};
        const double eps2[3] = {0, eps, 0};
        const double eps3[3] = {0, 0, eps};
        cout << "EdgeSE2XYZ Jacobian 解析法 Xi = " << endl << _jacobianOplusXi << endl;
        Matrix23D J1, J2;
        for (int i = 0; i < 3; ++i) {
            const double epsi[3] = {eps1[i], eps2[i], eps3[i]};
            const Vector3D pose = v1->estimate().toVector();
            const SE2 pose_eps(pose + Vector3D(epsi));  //! 注意扰动要直接加在R3中的平移上
            const SE3Quat Tcw_eps = Tcb * SE2ToSE3(pose_eps.inverse());
            const Vector3D lc_eps1 = Tcw_eps.map(v2->estimate());
            const Vector2D Ji = (cam->cam_map(lc_eps1) - cam->cam_map(lc)) / eps;
            J1.block<2, 1>(0, i) = Ji;
        }
        cout << "EdgeSE2XYZ Jacobian 数值法 Xi = " << endl << J1 << endl;

        cout << "EdgeSE2XYZ Jacobian 解析法 Xj = " << endl << _jacobianOplusXj << endl;
        for (int i = 0; i < 3; ++i) {
            const double epsi[3] = {eps1[i], eps2[i], eps3[i]};
            const Vector3D lw_eps = lw + Vector3D(epsi[0], epsi[1], epsi[2]);
            const Vector3D lc_eps2 = Tcw.map(lw_eps);
            const Vector2D Ji = (cam->cam_map(lc_eps2) - cam->cam_map(lc)) / eps;
            J2.block<2, 1>(0, i) = Ji;
        }
        cout << "EdgeSE2XYZ Jacobian 数值法 Xj = " << endl << J2 << endl;
    }*/

}
#endif

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
    const Matrix2D Ri = v1->estimate().rotation().toRotationMatrix();
    const Vector2D ti = v1->estimate().translation();
    const Vector2D tj = v2->estimate().translation();
    double ai = v1->estimate().rotation().angle();
    double aj = v2->estimate().rotation().angle();

#ifdef ORIGINAL_ERROR_PRESE2
    _error.head<2>() = Ri.transpose() * (tj - ti) - _measurement.head<2>();
    _error[2] = aj - ai - _measurement[2];
#else
    const Vector2D tm = _measurement.head<2>();
    Matrix2D Rm = Eigen::Rotation2Dd(_measurement[2]).toRotationMatrix();

    //! NOTE 这里的残差应该是SE2的减法!
    _error.head<2>() = Rm.transpose() * (Ri.transpose() * (tj - ti) - tm);
    _error[2] = normalize_theta(aj - ai - _measurement[2]);
#endif
}

void PreEdgeSE2::linearizeOplus()
{
    const VertexSE2* vi = static_cast<const VertexSE2*>(_vertices[0]);
    const VertexSE2* vj = static_cast<const VertexSE2*>(_vertices[1]);
    const Matrix2D Ri = vi->estimate().rotation().toRotationMatrix();
    const Vector2D ti = vi->estimate().translation();
    const Vector2D tj = vj->estimate().translation();
    const Vector2D dt = tj - ti;
    const Vector2D dt_x(-dt[1], dt[0]);

    _jacobianOplusXi.setZero();
    _jacobianOplusXi.block<2, 2>(0, 0) = -Ri.transpose();
    _jacobianOplusXi.block<2, 1>(0, 2) = -Ri.transpose() * dt_x;
    _jacobianOplusXi(2, 2) = -1;  //! NOTE 这里应该是-1

    _jacobianOplusXj.setIdentity();
    _jacobianOplusXj.block<2, 2>(0, 0) = Ri.transpose();

#ifdef ORIGINAL_ERROR_PRESE2
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
            //! NOTE 状态变量的更新要和顶点中的一致!
            const SE2 vi_eps(vi->estimate().toVector() + Vector3D(epsi[0], epsi[1], epsi[2]));
            const Matrix2D Ri_eps = vi_eps.rotation().toRotationMatrix();
            const Vector2D ti_eps = vi_eps.translation();
            const double ai_eps = vi_eps.rotation().angle();
            const double aj = vj->estimate().rotation().angle();

            Vector3D delta_eps;
            delta_eps.head<2>() = Ri_eps.transpose() * (tj - ti_eps) - _measurement.head<2>();
            delta_eps[2] = aj - ai_eps - _measurement[2];

            //! NOTE 根据倒数定义, 这里就是普通减法, 不是广义减!
            Vector3D Ji = (delta_eps - _error) / eps;
            J1.block<3, 1>(0, i) = Ji;
        }
        cout << "PreEdgeSE2 Jacobian 数值法 Xi = " << endl << J1 << endl;

        cout << "PreEdgeSE2 Jacobian 解析法 Xj = " << endl << _jacobianOplusXj << endl;
        for (int j = 0; j < 3; ++j) {
            const double epsi[3] = {eps1[j], eps2[j], eps3[j]};
            const SE2 vj_eps(vj->estimate().toVector() + Vector3D(epsi[0], epsi[1], epsi[2]));
            const Vector2D tj_eps = vj_eps.translation();
            const double ai = vi->estimate().rotation().angle();
            const double aj_eps = vj_eps.rotation().angle();

            Vector3D delta_eps;
            delta_eps.head<2>() = Ri.transpose() * (tj_eps - ti) - _measurement.head<2>();
            delta_eps[2] = aj_eps - ai - _measurement[2];

            Vector3D Jj = (delta_eps - _error) / eps;
            J2.block<3, 1>(0, j) = Jj;
        }
        cout << "PreEdgeSE2 Jacobian 数值法 Xj = " << endl << J2 << endl;
    }
    */
#else
    Matrix3D Rm = Matrix3D::Identity();
    Rm.block<2, 2>(0, 0) = Eigen::Rotation2Dd(_measurement[2]).toRotationMatrix();
    _jacobianOplusXi = Rm.transpose() * _jacobianOplusXi;
    _jacobianOplusXj = Rm.transpose() * _jacobianOplusXj;

    /*  //! 已验证, 结论正确 20200106
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
            //! NOTE 状态变量的更新要和顶点中的一致!
            SE2 vi_eps(vi->estimate().toVector() + Vector3D(epsi[0], epsi[1], epsi[2]));
            SE2 delta_eps = SE2(_measurement).inverse() * (vi_eps.inverse() * vj->estimate());
            //! NOTE 根据倒数定义, 这里就是普通减法, 不是广义减!
            Vector3D Ji = (delta_eps.toVector() - _error) / eps;
            J1.block<3, 1>(0, i) = Ji;
        }
        cout << "PreEdgeSE2 Jacobian 数值法 Xi = " << endl << J1 << endl;

        cout << "PreEdgeSE2 Jacobian 解析法 Xj = " << endl << _jacobianOplusXj << endl;
        for (int j = 0; j < 3; ++j) {
            const double epsi[3] = {eps1[j], eps2[j], eps3[j]};
            SE2 vj_eps(vj->estimate().toVector() + Vector3D(epsi[0], epsi[1], epsi[2]));
            SE2 delta_eps = SE2(_measurement).inverse() * (vi->estimate().inverse() * vj_eps);
            Vector3D Jj = (delta_eps.toVector() - _error) / eps;
            J2.block<3, 1>(0, j) = Jj;
        }
        cout << "PreEdgeSE2 Jacobian 数值法 Xj = " << endl << J2 << endl;
    }
    */
#endif
}

}  // namespace g2o
