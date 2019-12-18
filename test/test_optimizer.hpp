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
};


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
        g2o::VertexSE2* v1 = static_cast<g2o::VertexSE2*>(_vertices[0]);
        g2o::VertexSBAPointXYZ* v2 = static_cast<g2o::VertexSBAPointXYZ*>(_vertices[1]);

        const g2o::SE3Quat Tcw = Tcb * SE2ToSE3_(v1->estimate().inverse());
        const Eigen::Vector3d lc = Tcw.map(v2->estimate());

        _error = cam->cam_map(lc) - g2o::Vector2D(_measurement);
    }

    virtual void linearizeOplus()
    {
        g2o::VertexSE2* v1 = static_cast<g2o::VertexSE2*>(_vertices[0]);
        g2o::VertexSBAPointXYZ* v2 = static_cast<g2o::VertexSBAPointXYZ*>(_vertices[1]);

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

        J_v1.block<2, 2>(0, 0) = -(de_dlc * Tcb.rotation().toRotationMatrix()).block<2, 2>(0, 0);
        J_v1.block<2, 1>(0, 2) = (J_v2 * g2o::skew(lw - pi)).block<2, 1>(0, 2);

        _jacobianOplusXi = J_v1;
        _jacobianOplusXj = J_v2;

        /*
        {  //! ------------- check jacobians -----------------
            // 数值法求解雅克比, 看结果是否跟自己解析写的一样
            const double eps = 1e-6;
            const double eps1[3] = {eps, 0, 0};
            const double eps2[3] = {0, eps, 0};
            const double eps3[3] = {0, 0, eps};
            cout << "解析 Jacobian Xi = " << endl << _jacobianOplusXi << endl;
            Matrix23d J1, J2;
            for (int i = 0; i < 3; ++i) {
                const double epsi[3] = {eps1[i], eps2[i], eps3[i]};
                g2o::SE3Quat Tbw_eps =
                    SE2ToSE3_(v1->estimate() * g2o::SE2(epsi[0], epsi[1], epsi[2])).inverse();
                g2o::SE3Quat Tcw_eps = Tcb * Tbw_eps;
                Eigen::Vector3d lc_eps1 = Tcw_eps.map(v2->estimate());
                Eigen::Vector2d Ji = (cam->cam_map(lc_eps1) - cam->cam_map(lc)) / eps;
                J1.block<2, 1>(0, i) = Ji;
            }
            cout << "数值 Jacobian Xi = " << endl << J1 << endl;

            cout << "解析 Jacobian Xj = " << endl << _jacobianOplusXj << endl;
            for (int i = 0; i < 3; ++i) {
                const double epsi[3] = {eps1[i], eps2[i], eps3[i]};
                const Eigen::Vector3d lw_eps = lw + Eigen::Vector3d(epsi[0], epsi[1], epsi[2]);
                const Eigen::Vector3d lc_eps2 = Tcw.map(lw_eps);
                Eigen::Vector2d Ji = (cam->cam_map(lc_eps2) - cam->cam_map(lc)) / eps;
                J2.block<2, 1>(0, i) = Ji;
            }
            cout << "数值 Jacobian Xj = " << endl << J2 << endl;
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

#endif  // TEST_OPTIMIZER_HPP
