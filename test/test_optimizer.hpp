#ifndef TEST_OPTIMIZER_HPP
#define TEST_OPTIMIZER_HPP

#include "Config.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/eigen_types.h"
#include "Thirdparty/g2o/g2o/core/factory.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_dogleg.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_gauss_newton.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/core/sparse_optimizer.h"
#include "Thirdparty/g2o/g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "Thirdparty/g2o/g2o/solvers/csparse/linear_solver_csparse.h"
#include "Thirdparty/g2o/g2o/solvers/dense/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/solvers/eigen/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/solvers/pcg/linear_solver_pcg.h"
#include "Thirdparty/g2o/g2o/types/sba/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/types/slam2d/types_slam2d.h"
#include "Thirdparty/g2o/g2o/types/slam3d/types_slam3d.h"
//#include "Thirdparty/g2o/g2o/types/slam3d/dquat2mat.h"

// typedef g2o::BlockSolverX SlamBlockSolver;
// typedef g2o::LinearSolverCholmod<SlamBlockSolver::PoseMatrixType> SlamLinearSolverCholmod;
// typedef g2o::LinearSolverCSparse<SlamBlockSolver::PoseMatrixType> SlamLinearSolverCSparse;
// typedef g2o::LinearSolverPCG<SlamBlockSolver::PoseMatrixType> SlamLinearSolverPCG;
// typedef g2o::LinearSolverDense<SlamBlockSolver::PoseMatrixType> SlamLinearSolverDense;
// typedef g2o::LinearSolverEigen<SlamBlockSolver::PoseMatrixType> SlamLinearSolverEigen;
// typedef g2o::OptimizationAlgorithmLevenberg SlamAlgorithmLM;
// typedef g2o::OptimizationAlgorithmGaussNewton SlamAlgorithmGN;
// typedef g2o::OptimizationAlgorithmDogleg SlamAlgorithmDL;
// typedef g2o::SparseOptimizer SlamOptimizer;
// typedef g2o::CameraParameters CamPara;
// typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> SlamBlockSolver_3_1;

using std::cout;
using std::endl;

typedef Eigen::Matrix<double, 2, 3> Matrix23d;
typedef Eigen::Matrix<double, 3, 2> Matrix32d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

g2o::SE3Quat SE2ToSE3_(const g2o::SE2& _se2)
{
    g2o::SE3Quat ret;
    ret.setTranslation(Eigen::Vector3d(_se2.translation()(0), _se2.translation()(1), 0));
    ret.setRotation(
        Eigen::Quaterniond(Eigen::AngleAxisd(_se2.rotation().angle(), Eigen::Vector3d::UnitZ())));
    return ret;
}


class VertexSE2InSE3 : public g2o::VertexSE3Expmap
{
public:
    VertexSE2InSE3() {}

    virtual void setToOriginImpl() { _estimate = g2o::SE3Quat(); }

    virtual void setEstimate(const g2o::SE2& se2)
    {
        Vector6d v;
        v << se2[0], se2[1], 0., 0., 0., se2[2];
        _estimate = g2o::SE3Quat::exp(v);
    }

    virtual void setEstimate(const g2o::SE3Quat& se3) { _estimate = se3; }

    virtual void oplusImpl(const double* update_)
    {
        Eigen::Map<const Vector6d> update(update_);
        setEstimate(g2o::SE3Quat::exp(update) * estimate());
    }
};

//! 完成使命 201912
class EdgeSE2XYZCustom
    : public g2o::BaseBinaryEdge<2, g2o::Vector2D, g2o::VertexSE2, g2o::VertexSBAPointXYZ>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeSE2XYZCustom() {}
    ~EdgeSE2XYZCustom() {}

    bool read(std::istream& is)
    {
        Eigen::Vector2d m;
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
    bool write(std::ostream& os) const
    {
        Eigen::Vector2d m = measurement();
        os << m[0] << " " << m[1] << " ";
        for (int i = 0; i < 2; i++) {
            for (int j = i; j < 2; j++) {
                os << " " << information()(i, j);
            }
        }
        return os.good();
    }

    void computeError()
    {
        g2o::VertexSE2* v1 = dynamic_cast<g2o::VertexSE2*>(_vertices[0]);
        g2o::VertexSBAPointXYZ* v2 = dynamic_cast<g2o::VertexSBAPointXYZ*>(_vertices[1]);

        const g2o::SE3Quat Tcw = Tcb * SE2ToSE3_(v1->estimate().inverse());
        const Eigen::Vector3d lc = Tcw.map(v2->estimate());

        _error = cam->cam_map(lc) - g2o::Vector2D(_measurement);
    }

    virtual void linearizeOplus()
    {
        const g2o::VertexSE2* v1 = dynamic_cast<g2o::VertexSE2*>(_vertices[0]);
        const g2o::VertexSBAPointXYZ* v2 = dynamic_cast<g2o::VertexSBAPointXYZ*>(_vertices[1]);


        const Eigen::Vector3d vwb = v1->estimate().toVector();
        const Eigen::Vector3d pi(vwb[0], vwb[1], 0);

        const g2o::SE3Quat Tcw = Tcb * SE2ToSE3_(v1->estimate()).inverse();
        const Eigen::Matrix3d Rcw = Tcw.rotation().toRotationMatrix();

        const Eigen::Vector3d lw = v2->estimate();
        const Eigen::Vector3d lc = Tcw.map(lw);
        const double zc = lc(2);
        const double zc_inv = 1. / zc;
        const double zc_inv2 = zc_inv * zc_inv;

        const double& fx = cam->focal_length;
        const double& fy = fx;

        Matrix23d J_v1, J_v2;
        Matrix23d de_dlc;
        de_dlc << fx * zc_inv, 0, -fx * lc(0) * zc_inv2, 0, fy * zc_inv, -fy * lc(1) * zc_inv2;
        J_v2 = de_dlc * Rcw;

        //! 这里是把平移扰动加在SE2上的求导结果, 不适用.
        // J_v1.block<2, 2>(0, 0) = -(de_dlc * Tcb.rotation().toRotationMatrix()).block<2, 2>(0, 0);
        // J_v1.block<2, 1>(0, 2) = J_v2 * g2o::skew(lw - pi).block<3, 1>(0, 2);

        J_v1.block<2, 2>(0, 0) = -J_v2.block<2, 2>(0, 0);
        J_v1.block<2, 1>(0, 2) = (J_v2 * g2o::skew(lw - pi)).block<2, 1>(0, 2);
        _jacobianOplusXi = J_v1;
        _jacobianOplusXj = J_v2;

        /*
        {  //! ------------- check jacobians -----------------
            se2lam::WorkTimer timer2;
            // 数值法求解雅克比, 看结果是否跟自己解析写的一样
            const double eps = 1e-6;
            const double eps1[3] = {eps, 0, 0};
            const double eps2[3] = {0, eps, 0};
            const double eps3[3] = {0, 0, eps};
            cout << "EdgeSE2XYZCustom 解析雅克比 Jv1 = " << endl << _jacobianOplusXi << endl;
            Matrix23d J1, J2;
            for (int i = 0; i < 3; ++i) {
                const double epsi[3] = {eps1[i], eps2[i], eps3[i]};
                const g2o::SE2 pose = v1->estimate();  //! 注意扰动要直接加在SE3中的平移上,
        不能加在SE2上
                const g2o::SE2 pose_eps(pose[0] + epsi[0], pose[1] + epsi[1], pose[2] + epsi[2]);
                const g2o::SE3Quat Tcw_eps = Tcb * SE2ToSE3_(pose_eps.inverse());
                const Eigen::Vector3d lc_eps1 = Tcw_eps.map(v2->estimate());
                const Eigen::Vector2d Ji = (cam->cam_map(lc_eps1) - cam->cam_map(lc)) / eps;
                J1.block<2, 1>(0, i) = Ji;
            }
            cout << "EdgeSE2XYZCustom 数值雅克比 Jv1 = " << endl << J1 << endl;

            cout << "EdgeSE2XYZCustom 解析雅克比 Jv2 = " << endl << _jacobianOplusXj << endl;
            for (int i = 0; i < 3; ++i) {
                const double epsi[3] = {eps1[i], eps2[i], eps3[i]};
                const Eigen::Vector3d lw_eps = lw + Eigen::Vector3d(epsi[0], epsi[1], epsi[2]);
                const Eigen::Vector3d lc_eps2 = Tcw.map(lw_eps);
                Eigen::Vector2d Ji = (cam->cam_map(lc_eps2) - cam->cam_map(lc)) / eps;
                J2.block<2, 1>(0, i) = Ji;
            }
            cout << "EdgeSE2XYZCustom 数值雅克比 Jv2 = " << endl << J2 << endl;
            double t2 = timer2.count();
            cout << "EdgeSE2XYZCustom 雅克比求解时间: 解析/数值 = " << t1 << "/" << t2 << endl;
        }*/
    }

    inline void setCameraParameter(g2o::CameraParameters* _cam) { cam = _cam; }

    inline void setExtParameter(const g2o::SE3Quat& _Tbc)
    {
        Tbc = _Tbc;
        Tcb = Tbc.inverse();
    }

private:
    g2o::SE3Quat Tbc;
    g2o::SE3Quat Tcb;

    g2o::CameraParameters* cam;
};

//! 完成使命 20200106
class EdgeSE2Custom : public g2o::EdgeSE2
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeSE2Custom() {}

    bool read(std::istream& is)
    {
        g2o::Vector3D p;
        is >> p[0] >> p[1] >> p[2];
        setMeasurement(g2o::SE2(p));
        _inverseMeasurement = measurement().inverse();
        for (int i = 0; i < 3; ++i)
            for (int j = i; j < 3; ++j) {
                is >> information()(i, j);
                if (i != j)
                    information()(j, i) = information()(i, j);
            }
        return true;
    }

    bool write(std::ostream& os) const
    {
        g2o::Vector3D p = measurement().toVector();
        os << p.x() << " " << p.y() << " " << p.z();
        for (int i = 0; i < 3; ++i)
            for (int j = i; j < 3; ++j)
                os << " " << information()(i, j);
        return os.good();
    }

    void computeError()
    {
        const g2o::VertexSE2* vi = dynamic_cast<const g2o::VertexSE2*>(_vertices[0]);
        const g2o::VertexSE2* vj = dynamic_cast<const g2o::VertexSE2*>(_vertices[1]);
        const g2o::SE2 delta = _inverseMeasurement * (vi->estimate().inverse() * vj->estimate());
        _error = delta.toVector();
    }

    virtual void linearizeOplus()
    {
        const g2o::VertexSE2* vi = dynamic_cast<const g2o::VertexSE2*>(_vertices[0]);
        const g2o::VertexSE2* vj = dynamic_cast<const g2o::VertexSE2*>(_vertices[1]);
        const g2o::Matrix2D Ri = vi->estimate().rotation().toRotationMatrix();
        const g2o::Vector2D ti = vi->estimate().translation();
        const g2o::Vector2D tj = vj->estimate().translation();
        const g2o::Vector2D dt = tj - ti;
        const g2o::Vector2D dt_x(-dt[1], dt[0]);

        _jacobianOplusXi.setZero();
        _jacobianOplusXi.block<2, 2>(0, 0) = -Ri.transpose();
        _jacobianOplusXi.block<2, 1>(0, 2) = -Ri.transpose() * dt_x;
        _jacobianOplusXi(2, 2) = -1;

        _jacobianOplusXj.setIdentity();
        _jacobianOplusXj.block<2, 2>(0, 0) = Ri.transpose();

        g2o::Matrix3D Rm = g2o::Matrix3D::Identity();
        Rm.block<2, 2>(0, 0) = _measurement.rotation().toRotationMatrix();
        _jacobianOplusXi = Rm.transpose() * _jacobianOplusXi;
        _jacobianOplusXj = Rm.transpose() * _jacobianOplusXj;
        /*
        {  //! ------------- check jacobians -----------------
            // 数值法求解雅克比, 看结果是否跟自己解析写的一样
            const double eps = 1e-6;
            const double eps1[3] = {eps, 0, 0};
            const double eps2[3] = {0, eps, 0};
            const double eps3[3] = {0, 0, eps};
            cout << "EdgeSE2Custom Jacobian 解析法 Xi = " << endl << _jacobianOplusXi << endl;
            g2o::Matrix3D J1, J2;
            for (int i = 0; i < 3; ++i) {
                const double epsi[3] = {eps1[i], eps2[i], eps3[i]};
                //! NOTE 状态变量的更新要和顶点中的一致!
                g2o::SE2 vi_eps(vi->estimate().toVector() + g2o::Vector3D(epsi[0], epsi[1], epsi[2]));
                g2o::SE2 delta_eps = _measurement.inverse() * (vi_eps.inverse() * vj->estimate());
                //! NOTE 根据倒数定义, 这里就是普通减法, 不是广义减!
                g2o::Vector3D Ji = (delta_eps.toVector() - _error) / eps;
                J1.block<3, 1>(0, i) = Ji;
            }
            cout << "EdgeSE2Custom Jacobian 数值法 Xi = " << endl << J1 << endl;

            cout << "EdgeSE2Custom Jacobian 解析法 Xj = " << endl << _jacobianOplusXj << endl;
            for (int j = 0; j < 3; ++j) {
                const double epsi[3] = {eps1[j], eps2[j], eps3[j]};
                g2o::SE2 vj_eps(vj->estimate().toVector() + g2o::Vector3D(epsi[0], epsi[1], epsi[2]));
                g2o::SE2 delta_eps = _measurement.inverse() * (vi->estimate().inverse() * vj_eps);
                g2o::Vector3D Jj = (delta_eps.toVector() - _error) / eps;
                J2.block<3, 1>(0, j) = Jj;
            }
            cout << "EdgeSE2Custom Jacobian 数值法 Xj = " << endl << J2 << endl;
        }
        */
    }

};

G2O_REGISTER_TYPE(EDGE_SE2, EdgeSE2Custom);

#endif  // TEST_OPTIMIZER_HPP
