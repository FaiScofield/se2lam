/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "Config.h"
#include "EdgeSE2XYZ.h"
#include "KeyFrame.h"
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
#include "Thirdparty/g2o/g2o/types/slam3d/dquat2mat.h"
#include "Thirdparty/g2o/g2o/types/slam3d/types_slam3d.h"

namespace se2lam
{

// class Frame;

typedef g2o::BlockSolverX SlamBlockSolver;
typedef g2o::LinearSolverCholmod<SlamBlockSolver::PoseMatrixType> SlamLinearSolverCholmod;
typedef g2o::LinearSolverCSparse<SlamBlockSolver::PoseMatrixType> SlamLinearSolverCSparse;
typedef g2o::LinearSolverPCG<SlamBlockSolver::PoseMatrixType> SlamLinearSolverPCG;
typedef g2o::LinearSolverDense<SlamBlockSolver::PoseMatrixType> SlamLinearSolverDense;
typedef g2o::LinearSolverEigen<SlamBlockSolver::PoseMatrixType> SlamLinearSolverEigen;
typedef g2o::OptimizationAlgorithmLevenberg SlamAlgorithmLM;
typedef g2o::OptimizationAlgorithmGaussNewton SlamAlgorithmGN;
typedef g2o::OptimizationAlgorithmDogleg SlamAlgorithmDL;
typedef g2o::SparseOptimizer SlamOptimizer;
typedef g2o::CameraParameters CamPara;
typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> SlamBlockSolver_3_1;

inline Eigen::Quaterniond toQuaterniond(const Eigen::Vector3d& rot_vector)
{
    double angle = rot_vector.norm();  // 返回squareNorm()的开方根,即向量的模
    if (angle <= 1e-14)
        return Eigen::Quaterniond(1, 0, 0, 0);
    else
        return Eigen::Quaterniond(Eigen::AngleAxisd(angle, rot_vector.normalized()));  // 归一化
}

inline Eigen::Vector3d toRotationVector(const Eigen::Quaterniond& q_)
{
    Eigen::AngleAxisd angle_axis(q_);
    return angle_axis.angle() * angle_axis.axis();
}

g2o::Matrix3D Jl(const g2o::Vector3D& v3d);
g2o::Matrix3D invJl(const g2o::Vector3D& v3d);
g2o::Matrix6d invJJl(const g2o::Vector6d& v6d);
g2o::Matrix6d AdjTR(const g2o::SE3Quat& pose);

/**
 * @brief  全局平面运动约束
 * 将当前KF的pose投射到平面运动空间作为measurement(z=0, alpha = beta = 0)
 * BaseUnaryEdge    一元边, KF之间的先验信息, KF的全局平面约束.
 * g2o::SE3Quat     误差变量, SE3李代数, 由r和t组成, r为四元素. 转向量Vector7d后平移在前,
 * 旋转在后(虚前实后)
 * VertexSE3Expmap  相机位姿节点，6个维度
 *
 * _error:       e = ln([Tm * T^(-1)])^v, 误差函数
 * _measurement: Tcw, 相机位姿测量值, 令 $z=0, \alpha=0, \beta=0$ 作为观测
 * _jacobian:    J = -Jl(-e)^(-1), 左扰动的结果, 如果是右扰动的话还需要乘T.adj()
 */
class G2O_TYPES_SBA_API EdgeSE3ExpmapPrior
    : public g2o::BaseUnaryEdge<6, g2o::SE3Quat, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3ExpmapPrior() : BaseUnaryEdge()
    {
        setMeasurement(g2o::SE3Quat());
        information().setIdentity();
    }

    bool read(istream& is) override
    {
        g2o::Vector7d meas;
        for (int i = 0; i < 7; i++)
            is >> meas[i];
        g2o::SE3Quat cam2world;
        cam2world.fromVector(meas);
        setMeasurement(cam2world.inverse());
        for (int i = 0; i < 6; i++) {
            for (int j = i; j < 6; j++) {
                is >> information()(i, j);
                if (i != j)
                    information()(j, i) = information()(i, j);
            }
        }
        return true;
    }

    bool write(ostream& os) const override
    {
        g2o::SE3Quat cam2world(measurement().inverse());
        for (int i = 0; i < 7; i++)
            os << cam2world[i] << " ";
        for (int i = 0; i < 6; i++) {
            for (int j = i; j < 6; j++) {
                os << " " << information()(i, j);
            }
        }
        return os.good();
    }

    void computeError() override
    {
        g2o::VertexSE3Expmap* v = static_cast<g2o::VertexSE3Expmap*>(_vertices[0]);
        g2o::SE3Quat err = _measurement * v->estimate().inverse();
        _error = err.log();
    }

    void setMeasurement(const g2o::SE3Quat& m) override
    {
        _measurement = m;  // 测量值Tcw
    }

    //! NOTE 0917改成和PPT上一致
    void linearizeOplus() override
    {
        _jacobianOplusXi = -invJJl(-_error);  // 右扰动的话还要再乘一个Adj(T)
    //    _jacobianOplusXi = -g2o::Matrix6d::Identity();  //?
    }
};

class EdgeSE3ProjectXYZOnlyPose
    : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3ProjectXYZOnlyPose() : fx(Config::fx), fy(Config::fy), cx(Config::cx), cy(Config::cy) {}

    bool read(istream& is) override
    {
        Eigen::Vector2d m;
        is >> m[0] >> m[1];
        setMeasurement(m);

        is >> Xw[0] >> Xw[1] >> Xw[2];
        for (int i = 0; i < 2; i++) {
            for (int j = i; j < 2; j++) {
                is >> information()(i, j);
                if (i != j)
                    information()(j, i) = information()(i, j);
            }
        }
        return true;
    }

    bool write(ostream& os) const override
    {
        Eigen::Vector2d m = measurement();
        os << m[0] << " " << m[1] << " ";
        os << Xw[0] << " " << Xw[1] << " " << Xw[2] << " ";
        for (int i = 0; i < 2; i++) {
            for (int j = i; j < 2; j++) {
                os << " " << information()(i, j);
            }
        }
        return os.good();
    }

    void computeError() override
    {
        const g2o::VertexSE3Expmap* v = dynamic_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        Eigen::Vector2d obs(_measurement);
        _error = obs - cam_project(v->estimate().map(Xw));
    }

    void linearizeOplus() override
    {
        const g2o::VertexSE3Expmap* v = dynamic_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        Eigen::Vector3d xyz_trans = v->estimate().map(Xw);

        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double invz = 1.0 / xyz_trans[2];
        double invz_2 = invz * invz;

        _jacobianOplusXi(0, 0) = x * y * invz_2 * fx;
        _jacobianOplusXi(0, 1) = -(1 + (x * x * invz_2)) * fx;
        _jacobianOplusXi(0, 2) = y * invz * fx;
        _jacobianOplusXi(0, 3) = -invz * fx;
        _jacobianOplusXi(0, 4) = 0;
        _jacobianOplusXi(0, 5) = x * invz_2 * fx;

        _jacobianOplusXi(1, 0) = (1 + y * y * invz_2) * fy;
        _jacobianOplusXi(1, 1) = -x * y * invz_2 * fy;
        _jacobianOplusXi(1, 2) = -x * invz * fy;
        _jacobianOplusXi(1, 3) = 0;
        _jacobianOplusXi(1, 4) = -invz * fy;
        _jacobianOplusXi(1, 5) = y * invz_2 * fy;
    }

    Eigen::Vector2d cam_project(const Eigen::Vector3d& trans_xyz) const
    {
        Eigen::Vector2d proj = trans_xyz.head<2>() / trans_xyz[2];
        Eigen::Vector2d res;
        res[0] = proj[0] * fx + cx;
        res[1] = proj[1] * fy + cy;
        return res;
    }

    bool isDepthPositive()
    {
        const g2o::VertexSE3Expmap* v1 = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        return (v1->estimate().map(Xw))(2) > 0.0;
    }

    Eigen::Vector3d Xw;
    double fx, fy, cx, cy;
};


void initOptimizer(SlamOptimizer& opt, bool verbose = false);

CamPara* addCamPara(SlamOptimizer& opt, const cv::Mat& K, int id);

// Camera Pose Vertex in SE3
g2o::VertexSE3Expmap* addVertexSE3Expmap(SlamOptimizer& opt, const g2o::SE3Quat& pose, int id, bool fixed = false);

// Landmark Pose Vertex in R^3
g2o::VertexSBAPointXYZ* addVertexSBAXYZ(SlamOptimizer& opt, const Eigen::Vector3d& xyz, int id, bool marginal = true,
                     bool fixed = false);

// Robot Pose Vertex in SE2
g2o::VertexSE2* addVertexSE2(SlamOptimizer& opt, const g2o::SE2& pose, int id, bool fixed = false);

g2o::VertexSE3* addVertexSE3(SlamOptimizer& opt, const g2o::Isometry3D& pose, int id, bool fixed = false);

g2o::VertexPointXYZ* addVertexXYZ(SlamOptimizer& opt, const g2o::Vector3D& xyz, int id, bool marginal = true);

g2o::EdgeSE3Prior* addVertexSE3AndEdgePlaneMotion(SlamOptimizer& opt, const g2o::Isometry3D& pose,
                                                  int id, const cv::Mat& extPara,
                                                  int paraSE3OffsetId, bool fixed = false);


EdgeSE3ExpmapPrior* addEdgeSE3ExpmapPlaneConstraint(SlamOptimizer& opt, const g2o::SE3Quat& pose,
                                                    int vId, const cv::Mat& extPara);

g2o::EdgeSE3Expmap* addEdgeSE3Expmap(SlamOptimizer& opt, const g2o::SE3Quat& measure, int id0, int id1,
                      const g2o::Matrix6d& info);

g2o::EdgeProjectXYZ2UV* addEdgeXYZ2UV(SlamOptimizer& opt, const Eigen::Vector2d& measure, int id0,
                                      int id1, int paraId, const Eigen::Matrix2d& info,
                                      double thHuber);

g2o::EdgeSE2XYZ* addEdgeSE2XYZ(SlamOptimizer& opt, const g2o::Vector2D& meas, int id0, int id1,
                               CamPara* campara, const g2o::SE3Quat& _Tbc,
                               const g2o::Matrix2D& info, double thHuber);

g2o::PreEdgeSE2* addPreEdgeSE2(SlamOptimizer& opt, const g2o::Vector3D& meas, int id0, int id1,
                               const g2o::Matrix3D& info);
g2o::EdgeSE2* addEdgeSE2(SlamOptimizer& opt, const g2o::Vector3D& meas, int id0, int id1,
                         const g2o::Matrix3D& info);

g2o::ParameterSE3Offset* addParaSE3Offset(SlamOptimizer& opt, const g2o::Isometry3D& se3offset,
                                          int id);


g2o::EdgeSE3* addEdgeSE3(SlamOptimizer& opt, const g2o::Isometry3D& measure, int id0, int id1,
                         const g2o::Matrix6d& info);

g2o::EdgeSE3PointXYZ* addEdgeSE3XYZ(SlamOptimizer& opt, const g2o::Vector3D& measure, int idse3,
                                    int idxyz, int paraSE3OffsetId, const g2o::Matrix3D& info,
                                    double thHuber);


g2o::SE2 estimateVertexSE2(SlamOptimizer& opt, int id);

g2o::Isometry3D estimateVertexSE3(SlamOptimizer& opt, int id);

Eigen::Vector3d estimateVertexXYZ(SlamOptimizer& opt, int id);

g2o::SE3Quat estimateVertexSE3Expmap(SlamOptimizer& opt, int id);

g2o::Vector3D estimateVertexSBAXYZ(SlamOptimizer& opt, int id);

void calcOdoConstraintCam(const Se2& dOdo, cv::Mat& Tc1c2, g2o::Matrix6d& Info_se3);
void calcSE3toXYZInfo(const cv::Point3f& Pc1, const cv::Mat& Tc1w, const cv::Mat& Tc2w,
                      Eigen::Matrix3d& info1, Eigen::Matrix3d& info2);
bool verifyInfo(const g2o::Matrix6d& info);
bool verifyInfo(const Eigen::Matrix3d& info);

void poseOptimization(Frame* pFrame, int& nCorrespondences, double& error);

}  // namespace se2lam

#endif  // OPTIMIZER_H
