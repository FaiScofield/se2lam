/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/

#include "optimizer.h"
#include "converter.h"
#include "cvutil.h"

namespace se2lam
{

using namespace g2o;
using namespace std;
using namespace Eigen;


EdgeSE2XYZ *addEdgeSE2XYZ(SlamOptimizer &opt, const Vector2D &meas, int id0, int id1,
                          CamPara *campara, const SE3Quat &_Tbc, const Matrix2D &info,
                          double thHuber)
{
    EdgeSE2XYZ *e = new EdgeSE2XYZ;
    e->vertices()[0] = opt.vertex(id0);
    e->vertices()[1] = opt.vertex(id1);
    e->setCameraParameter(campara);
    e->setExtParameter(_Tbc);
    e->setMeasurement(meas);
    e->setInformation(info);
    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
    rk->setDelta(thHuber);
    e->setRobustKernel(rk);
    opt.addEdge(e);
    return e;
}

VertexSE2 *addVertexSE2(SlamOptimizer &opt, const SE2 &pose, int id, bool fixed)
{
    VertexSE2 *v = new VertexSE2;
    v->setId(id);
    v->setEstimate(pose);
    v->setFixed(fixed);
    opt.addVertex(v);
    return v;
}

SE2 estimateVertexSE2(SlamOptimizer &opt, int id)
{
    VertexSE2 *v = static_cast<VertexSE2 *>(opt.vertex(id));
    return v->estimate();
}

PreEdgeSE2 *addEdgeSE2(SlamOptimizer &opt, const Vector3D &meas, int id0, int id1,
                       const Matrix3D &info)
{
    PreEdgeSE2 *e = new PreEdgeSE2;
    e->vertices()[0] = opt.vertex(id0);
    e->vertices()[1] = opt.vertex(id1);
    e->setMeasurement(meas);
    e->setInformation(info);
    opt.addEdge(e);
    return e;
}

Matrix3D Jl(const Vector3D &v3d)
{
    double theta = v3d.norm();
    double invtheta = 1. / theta;
    double sint = std::sin(theta);
    double cost = std::cos(theta);
    Vector3D a = v3d * invtheta;

    Matrix3D Jl = sint * invtheta * Matrix3D::Identity() +
                  (1 - sint * invtheta) * a * a.transpose() + (1 - cost) * invtheta * skew(a);
    return Jl;
}

Matrix3D invJl(const Vector3D &v3d)
{
    double theta = v3d.norm();
    double thetahalf = theta * 0.5;
    double invtheta = 1. / theta;
    double cothalf = std::tan(M_PI_2 - thetahalf);
    Vector3D a = v3d * invtheta;

    Matrix3D invJl = thetahalf * cothalf * Matrix3D::Identity() +
                     (1 - thetahalf * cothalf) * a * a.transpose() - thetahalf * skew(a);

    return invJl;
}

Matrix6d AdjTR(const g2o::SE3Quat &pose)
{
    Matrix3D R = pose.rotation().toRotationMatrix();
    Matrix6d res;
    res.block(0, 0, 3, 3) = R;
    res.block(3, 3, 3, 3) = R;
    res.block(0, 3, 3, 3) = skew(pose.translation()) * R;
    res.block(3, 0, 3, 3) = Matrix3D::Zero(3, 3);
    return res;
}

//! 计算BCH近似表达式中的inv(J_l),
//! @see: 视觉SLAM十四讲试(4.30)-(4.32)
Matrix6d invJJl(const Vector6d &v6d)
{

    //! rho: translation; phi: rotation
    //! vector order: [rot, trans]

    Vector3D rho, phi;
    for (int i = 0; i < 3; ++i) {
        phi[i] = v6d[i];
        rho[i] = v6d[i + 3];
    }
    double theta = phi.norm();
    Matrix3D Phi = skew(phi);
    Matrix3D Rho = skew(rho);
    double sint = sin(theta);
    double cost = cos(theta);
    double theta2 = theta * theta;
    double theta3 = theta * theta2;
    double theta4 = theta2 * theta2;
    double theta5 = theta4 * theta;
    double invtheta = 1. / theta;
    double invtheta3 = 1. / theta3;
    double invtheta4 = 1. / theta4;
    double invtheta5 = 1. / theta5;
    Matrix3D PhiRho = Phi * Rho;
    Matrix3D RhoPhi = Rho * Phi;
    Matrix3D PhiRhoPhi = PhiRho * Phi;
    Matrix3D PhiPhiRho = Phi * PhiRho;
    Matrix3D RhoPhiPhi = RhoPhi * Phi;
    Matrix3D PhiRhoPhiPhi = PhiRhoPhi * Phi;
    Matrix3D PhiPhiRhoPhi = Phi * PhiRhoPhi;

    double temp = (1. - 0.5 * theta2 - cost) * invtheta4;

    Matrix3D Ql = 0.5 * Rho + (theta - sint) * invtheta3 * (PhiRho + RhoPhi + PhiRhoPhi) -
                  temp * (PhiPhiRho + RhoPhiPhi - 3. * PhiRhoPhi) -
                  0.5 * (temp - (3. * (theta - sint) + theta3 * 0.5) * invtheta5) *
                      (PhiRhoPhiPhi + PhiPhiRhoPhi);


    double thetahalf = theta * 0.5;
    double cothalf = tan(M_PI_2 - thetahalf);
    Vector3D a = phi * invtheta;
    Matrix3D invJl = thetahalf * cothalf * Matrix3D::Identity() +
                     (1 - thetahalf * cothalf) * a * a.transpose() - thetahalf * skew(a);

    Matrix6d invJJl = Matrix6d::Zero();
    invJJl.block<3, 3>(0, 0) = invJl;
    invJJl.block<3, 3>(3, 0) = -invJl * Ql * invJl;
    invJJl.block<3, 3>(3, 3) = invJl;
    return invJJl;
}


EdgeSE3ExpmapPrior::EdgeSE3ExpmapPrior() : BaseUnaryEdge<6, SE3Quat, VertexSE3Expmap>()
{
    setMeasurement(SE3Quat());
    information().setIdentity();
}

/**
 * @brief EdgeSE3ExpmapPrior::computeError KF 全局平面运动约束
 * error: e = ln(Tm * T^(-1))^V
 * jacobian: J = -Jr(-e)^(-1)
 */
void EdgeSE3ExpmapPrior::computeError()
{
    VertexSE3Expmap *v = static_cast<VertexSE3Expmap *>(_vertices[0]);
    SE3Quat err = _measurement * v->estimate().inverse();
    _error = err.log();
}

void EdgeSE3ExpmapPrior::setMeasurement(const SE3Quat &m)
{
    _measurement = m;
}

void EdgeSE3ExpmapPrior::linearizeOplus()
{
    //! NOTE 0917改成和PPT上一致
    VertexSE3Expmap *v = static_cast<VertexSE3Expmap*>(_vertices[0]);
    Vector6d err = (_measurement * v->estimate().inverse()).log() ;
    _jacobianOplusXi = -invJJl(-err);
//    _jacobianOplusXi = -g2o::Matrix6d::Identity();  //! ?
}

bool EdgeSE3ExpmapPrior::read(istream &is)
{
    return true;
}

bool EdgeSE3ExpmapPrior::write(ostream &os) const
{
    return true;
}


void initOptimizer(SlamOptimizer &opt, bool verbose)
{
    SlamLinearSolver *linearSolver = new SlamLinearSolver();
    SlamBlockSolver *blockSolver = new SlamBlockSolver(linearSolver);
    SlamAlgorithm *solver = new SlamAlgorithm(blockSolver);
    opt.setAlgorithm(solver);
    opt.setVerbose(verbose);
}

CamPara *addCamPara(SlamOptimizer &opt, const cv::Mat &K, int id)
{
    Eigen::Vector2d principal_point(K.at<float>(0, 2), K.at<float>(1, 2));
    CamPara *campr = new CamPara(K.at<float>(0, 0), principal_point, 0.);
    campr->setId(id);
    opt.addParameter(campr);

    return campr;
}

g2o::ParameterSE3Offset *addParaSE3Offset(SlamOptimizer &opt, const g2o::Isometry3D &se3offset,
                                          int id)
{
    g2o::ParameterSE3Offset *para = new g2o::ParameterSE3Offset();
    para->setOffset(se3offset);
    para->setId(id);
    opt.addParameter(para);

    return para;
}

/**
 * @brief addVertexSE3Expmap 添加相机位姿节点, 6DOF, EdgeProjectXYZ2UV边连接的节点之一
 * @param opt   优化器
 * @param pose  相机位姿Tcw
 * @param id    KF id
 * @param fixed 要优化相机位姿, 这里是false
 */
void addVertexSE3Expmap(SlamOptimizer &opt, const g2o::SE3Quat &pose, int id, bool fixed)
{
    g2o::VertexSE3Expmap *v = new g2o::VertexSE3Expmap();
    v->setEstimate(pose);
    v->setId(id);
    v->setFixed(fixed);
    opt.addVertex(v);
}

/**
 * @brief addPlaneMotionSE3Expmap 给位姿节点添加平面运动约束
 * @param opt       optimizer
 * @param pose      KF's pose，Tcw
 * @param vId       vertex id
 * @param extPara   Config::bTc
 * @return
 */
EdgeSE3ExpmapPrior *addPlaneMotionSE3Expmap(SlamOptimizer &opt, const g2o::SE3Quat &pose, int vId,
                                            const cv::Mat &extPara)
{

//#define USE_EULER

#ifdef USE_EULER
    const cv::Mat bTc = extPara;
    const cv::Mat cTb = scv::inv(bTc);

    cv::Mat Tcw = toCvMat(pose);
    cv::Mat Tbw = bTc * Tcw;
    g2o::Vector3D euler = g2o::internal::toEuler(toMatrix3d(Tbw.rowRange(0, 3).colRange(0, 3)));
    float yaw = euler(2);

    // Fix pitch and raw to zero, only yaw remains
    cv::Mat Rbw = (cv::Mat_<float>(3, 3) << cos(yaw), -sin(yaw), 0, sin(yaw), cos(yaw), 0, 0, 0, 1);
    Rbw.copyTo(Tbw.rowRange(0, 3).colRange(0, 3));
    Tbw.at<float>(2, 3) = 0;  // Fix the height to zero

    Tcw = cTb * Tbw;
    //! Vector order: [rot, trans]
    g2o::Matrix6d Info_bw = g2o::Matrix6d::Zero();
    Info_bw(0, 0) = Config::PLANEMOTION_XROT_INFO;
    Info_bw(1, 1) = Config::PLANEMOTION_YROT_INFO;
    Info_bw(2, 2) = 1e-4;
    Info_bw(3, 3) = 1e-4;
    Info_bw(4, 4) = 1e-4;
    Info_bw(5, 5) = Config::PLANEMOTION_Z_INFO;
    g2o::Matrix6d J_bb_cc = toSE3Quat(bTc).adj();
    g2o::Matrix6d Info_cw = J_bb_cc.transpose() * Info_bw * J_bb_cc;
#else

    g2o::SE3Quat Tbc = toSE3Quat(extPara);
    g2o::SE3Quat Tbw = Tbc * pose;  // Tbc * Tcw

    // 令旋转只在Z轴上
    Eigen::AngleAxisd AngleAxis_bw(Tbw.rotation());
    Eigen::Vector3d Log_Rbw = AngleAxis_bw.angle() * AngleAxis_bw.axis();
    AngleAxis_bw = Eigen::AngleAxisd(Log_Rbw[2], Eigen::Vector3d::UnitZ());
    Tbw.setRotation(Eigen::Quaterniond(AngleAxis_bw));

    // 令Z方向平移为0
    Eigen::Vector3d xyz_bw = Tbw.translation();
    xyz_bw[2] = 0;
    Tbw.setTranslation(xyz_bw);

    g2o::SE3Quat Tcw = Tbc.inverse() * Tbw;

    //! Vector order: [rot, trans]
    // 确定信息矩阵，XY轴旋转为小量, Info_bw已经平方了
    g2o::Matrix6d Info_bw = g2o::Matrix6d::Zero();
    Info_bw(0, 0) = Config::PlaneMotionInfoXrot;  // 1e6
    Info_bw(1, 1) = Config::PlaneMotionInfoYrot;
    Info_bw(2, 2) = 1e-4;
    Info_bw(3, 3) = 1e-4;
    Info_bw(4, 4) = 1e-4;
    Info_bw(5, 5) = Config::PlaneMotionInfoZ;
    g2o::Matrix6d J_bb_cc = Tbc.adj();
    g2o::Matrix6d Info_cw = J_bb_cc.transpose() * Info_bw * J_bb_cc;
#endif

    // Make sure the infor matrix is symmetric. 确保信息矩阵正定
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < i; ++j)
            Info_cw(i, j) = Info_cw(j, i);

    EdgeSE3ExpmapPrior *planeConstraint = new EdgeSE3ExpmapPrior();
    planeConstraint->setInformation(Info_cw);
#ifdef USE_EULER
    planeConstraint->setMeasurement(toSE3Quat(Tcw));
#else
    planeConstraint->setMeasurement(Tcw);
#endif
    planeConstraint->vertices()[0] = opt.vertex(vId);
    opt.addEdge(planeConstraint);

    return planeConstraint;
}

/**
 * @brief addVertexSBAXYZ  添加MP观测节点, 3DOF, EdgeProjectXYZ2UV边连接的节点之一
 * @param opt       优化器
 * @param xyz       MP的三维坐标
 * @param id        节点对应的id
 * @param marginal  节点是否边缘化
 * @param fixed     节点是否固定, default = false
 */
void addVertexSBAXYZ(SlamOptimizer &opt, const Eigen::Vector3d &xyz, int id, bool marginal,
                     bool fixed)
{
    g2o::VertexSBAPointXYZ *v = new g2o::VertexSBAPointXYZ();
    v->setEstimate(xyz);
    v->setId(id);
    v->setMarginalized(marginal);
    v->setFixed(fixed);
    opt.addVertex(v);
}


// 6d vector (x,y,z,qx,qy,qz) (note that we leave out the w part of the quaternion)
void addVertexSE3(SlamOptimizer &opt, const g2o::Isometry3D &pose, int id, bool fixed)
{
    g2o::VertexSE3 *v = new g2o::VertexSE3();
    v->setEstimate(pose);
    v->setFixed(fixed);
    v->setId(id);
    opt.addVertex(v);
}

g2o::EdgeSE3Prior *addVertexSE3PlaneMotion(SlamOptimizer &opt, const g2o::Isometry3D &pose, int id,
                                           const cv::Mat &extPara, int paraSE3OffsetId, bool fixed)
{
    //#define USE_OLD_SE3_PRIOR_JACOB

    g2o::VertexSE3 *v = new g2o::VertexSE3();
    v->setEstimate(pose);
    v->setFixed(fixed);
    v->setId(id);

#ifdef USE_OLD_SE3_PRIOR_JACOB
    //! Notation: `w' for World, `b' for Robot, `c' for Camera

    const cv::Mat T_b_c = extPara;
    const cv::Mat T_c_b = scv::inv(T_b_c);
    cv::Mat T_w_c = toCvMat(pose);
    cv::Mat T_w_b = T_w_c * T_c_b;
    g2o::Vector3D euler = g2o::internal::toEuler(toMatrix3d(T_w_b.rowRange(0, 3).colRange(0, 3)));
    float yaw = euler(2);

    // fix pitch and raw to zero, only yaw remains
    cv::Mat R_w_b =
        (cv::Mat_<float>(3, 3) << cos(yaw), -sin(yaw), 0, sin(yaw), cos(yaw), 0, 0, 0, 1);
    R_w_b.copyTo(T_w_b.rowRange(0, 3).colRange(0, 3));
    T_w_b.at<float>(2, 3) = 0;  // fix height to zero

    // refine T_w_c
    T_w_c = T_w_b * T_b_c;

    /** Compute Information Matrix of Camera Error Compact Quaternion
     * Q_qbar_b_c: quaternion matrix of the inverse of q_b_c
     * Qbar_q_b_c: conjugate quaternion matrix of q_b_c
     */

    vector<float> vq_w_b = toQuaternion(T_w_b);
    cv::Mat q_w_b = (cv::Mat_<float>(4, 1) << vq_w_b[0], vq_w_b[1], vq_w_b[2], vq_w_b[3]);
    float norm_w_b = sqrt(vq_w_b[0] * vq_w_b[0] + vq_w_b[1] * vq_w_b[1] + vq_w_b[2] * vq_w_b[2] +
                          vq_w_b[3] * vq_w_b[3]);
    q_w_b = q_w_b / norm_w_b;

    cv::Mat Info_q_bm_b =
        (cv::Mat_<float>(4, 4) << 1e-6, 0, 0, 0, 0, 1e4, 0, 0, 0, 0, 1e4, 0, 0, 0, 0, 1e-6);


    //    Info_q_bm_b = Info_q_bm_b + 1e6 * q_w_b * q_w_b.t();

    cv::Mat R_b_c = T_b_c.rowRange(0, 3).colRange(0, 3);
    vector<float> q_b_c = toQuaternion(R_b_c);
    float norm_q_b_c =
        sqrt(q_b_c[0] * q_b_c[0] + q_b_c[1] * q_b_c[1] + q_b_c[2] * q_b_c[2] + q_b_c[3] * q_b_c[3]);
    for (int i = 0; i < 4; ++i)
        q_b_c[i] = q_b_c[i] / norm_q_b_c;

    cv::Mat Q_qbar_b_c = (cv::Mat_<float>(4, 4) << -q_b_c[0], -q_b_c[1], -q_b_c[2], -q_b_c[3],
                          q_b_c[1], -q_b_c[0], q_b_c[3], -q_b_c[2], q_b_c[2], -q_b_c[3], -q_b_c[0],
                          q_b_c[1], q_b_c[3], q_b_c[2], -q_b_c[1], -q_b_c[0]);
    cv::Mat Qbar_q_b_c = (cv::Mat_<float>(4, 4) << q_b_c[0], -q_b_c[1], -q_b_c[2], -q_b_c[3],
                          q_b_c[1], q_b_c[0], -q_b_c[3], q_b_c[2], q_b_c[2], q_b_c[3], q_b_c[0],
                          -q_b_c[1], q_b_c[3], -q_b_c[2], q_b_c[1], q_b_c[0]);
    cv::Mat J = Q_qbar_b_c * Qbar_q_b_c;
    cv::Mat Jinv = J.inv();
    cv::Mat Info_q_cm_c = Jinv.t() * Info_q_bm_b * Jinv;

    g2o::Matrix6d Info_pose = g2o::Matrix6d::Zero();

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            Info_pose(i + 3, j + 3) = Info_q_cm_c.at<float>(i + 1, j + 1);

    cv::Mat Info_t_bm_b = (cv::Mat_<float>(3, 3) << 1e-6, 0, 0, 0, 1e-6, 0, 0, 0, 1);

    cv::Mat R_w_c = T_w_c.rowRange(0, 3).colRange(0, 3);
    cv::Mat Info_t_cm_c = R_w_c.t() * Info_t_bm_b * R_w_c;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            Info_pose(i, j) = Info_t_cm_c.at<float>(i, j);
    g2o::Isometry3D Iso_w_c = toIsometry3D(T_w_c);


#else
    g2o::SE3Quat Tbc = toSE3Quat(extPara);
    g2o::SE3Quat Twc = toSE3Quat(pose);
    g2o::SE3Quat Twb = Twc * Tbc.inverse();

    Eigen::AngleAxisd AngleAxis_bw(Twb.rotation());
    Eigen::Vector3d Log_Rbw = AngleAxis_bw.angle() * AngleAxis_bw.axis();
    AngleAxis_bw = Eigen::AngleAxisd(Log_Rbw[2], Eigen::Vector3d::UnitZ());
    Twb.setRotation(Eigen::Quaterniond(AngleAxis_bw));

    Eigen::Vector3d xyz_wb = Twb.translation();
    xyz_wb[2] = 0;
    Twb.setTranslation(xyz_wb);

    Twc = Twb * Tbc;

    //! Vector order: [trans, rot]
    g2o::Matrix6d Info_wb = g2o::Matrix6d::Zero();
    Info_wb(3, 3) = Config::PlaneMotionInfoXrot;
    Info_wb(4, 4) = Config::PlaneMotionInfoYrot;
    Info_wb(5, 5) = 1e-4;
    Info_wb(0, 0) = 1e-4;
    Info_wb(1, 1) = 1e-4;
    Info_wb(2, 2) = Config::PlaneMotionInfoZ;
    g2o::Matrix6d J_bb_cc = AdjTR(Tbc);
    g2o::Matrix6d Info_pose = J_bb_cc.transpose() * Info_wb * J_bb_cc;

    Isometry3D Iso_w_c = Twc;

#endif
    g2o::EdgeSE3Prior *planeConstraint = new g2o::EdgeSE3Prior();
    planeConstraint->setInformation(Info_pose);
    planeConstraint->setMeasurement(Iso_w_c);
    planeConstraint->vertices()[0] = v;
    planeConstraint->setParameterId(0, paraSE3OffsetId);

    opt.addVertex(v);
    opt.addEdge(planeConstraint);

    return planeConstraint;
}

// 地图点观测节点
void addVertexXYZ(SlamOptimizer &opt, const g2o::Vector3D &xyz, int id, bool marginal)
{
    g2o::VertexPointXYZ *v = new g2o::VertexPointXYZ();
    v->setEstimate(xyz);
    v->setId(id);
    v->setMarginalized(marginal);
    opt.addVertex(v);
}

/**
 * @brief addEdgeSE3Expmap 添加相机位姿间的约束边, 连接两个相机的位姿节点VertexSE3Expmap
 * measure = Tc1c2 = Tcb * Tb1b2 * Tbc, C = SE3Quat(measure), _error = (T2.inv() * C * T1).log()
 *
 * @param opt       优化器
 * @param measure   两KF间相机的位姿变换Tc1c2
 * @param id0       T1的id
 * @param id1       T2的id
 * @param info      信息矩阵
 */
void addEdgeSE3Expmap(SlamOptimizer &opt, const g2o::SE3Quat &measure, int id0, int id1,
                      const g2o::Matrix6d &info)
{
    assert(verifyInfo(info));

    g2o::EdgeSE3Expmap *e = new g2o::EdgeSE3Expmap();
    e->setMeasurement(measure);
    e->vertices()[0] = opt.vertex(id0);  // from
    e->vertices()[1] = opt.vertex(id1);  // to

    // The input info is [trans rot] order, but EdgeSE3Expmap requires [rot trans]
    g2o::Matrix6d infoNew;
    infoNew.block(0, 0, 3, 3) = info.block(3, 3, 3, 3);
    infoNew.block(3, 0, 3, 3) = info.block(0, 3, 3, 3);
    infoNew.block(0, 3, 3, 3) = info.block(3, 0, 3, 3);
    infoNew.block(3, 3, 3, 3) = info.block(0, 0, 3, 3);

    e->setInformation(infoNew);
    opt.addEdge(e);
}

/**
 * @brief addEdgeXYZ2UV 添加重投影误差边, 连接路标节点VertexSBAPointXYZ和相机位姿节点VertexSE3Expmap
 * @param opt       优化器
 * @param measure   KP的像素坐标
 * @param id0       该MP的节点id
 * @param id1       和那个有观测关联的KF的节点id
 * @param paraId    相机id，这里为0
 * @param info      信息矩阵
 * @param thHuber   鲁棒核
 * @return
 */
g2o::EdgeProjectXYZ2UV *addEdgeXYZ2UV(SlamOptimizer &opt, const Eigen::Vector2d &measure, int id0,
                                      int id1, int paraId, const Eigen::Matrix2d &info,
                                      double thHuber)
{
    g2o::EdgeProjectXYZ2UV *e = new g2o::EdgeProjectXYZ2UV();
    e->vertices()[0] = opt.vertex(id0);
    e->vertices()[1] = opt.vertex(id1);
    e->setMeasurement(measure);
    e->setInformation(info);
    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
    rk->setDelta(thHuber);
    e->setRobustKernel(rk);
    e->setParameterId(0, paraId);
    opt.addEdge(e);

    return e;
}

g2o::EdgeSE3 *addEdgeSE3(SlamOptimizer &opt, const g2o::Isometry3D &measure, int id0, int id1,
                         const g2o::Matrix6d &info)
{
    g2o::EdgeSE3 *e = new g2o::EdgeSE3();
    e->setMeasurement(measure);
    e->vertices()[0] = opt.vertex(id0);
    e->vertices()[1] = opt.vertex(id1);
    e->setInformation(info);

    opt.addEdge(e);

    return e;
}

g2o::EdgeSE3PointXYZ *addEdgeSE3XYZ(SlamOptimizer &opt, const g2o::Vector3D &measure, int idse3,
                                    int idxyz, int paraSE3OffsetId, const g2o::Matrix3D &info,
                                    double thHuber)
{
    g2o::EdgeSE3PointXYZ *e = new g2o::EdgeSE3PointXYZ();
    e->vertices()[0] = opt.vertex(idse3);
    e->vertices()[1] = opt.vertex(idxyz);
    e->setMeasurement(measure);
    e->setParameterId(0, paraSE3OffsetId);
    e->setInformation(info);
    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
    rk->setDelta(thHuber);
    e->setRobustKernel(rk);
    opt.addEdge(e);

    return e;
}


g2o::Vector3D estimateVertexSBAXYZ(SlamOptimizer &opt, int id)
{
    g2o::VertexSBAPointXYZ *v = static_cast<g2o::VertexSBAPointXYZ *>(opt.vertex(id));
    return v->estimate();
}

g2o::SE3Quat estimateVertexSE3Expmap(SlamOptimizer &opt, int id)
{
    g2o::VertexSE3Expmap *v = static_cast<g2o::VertexSE3Expmap *>(opt.vertex(id));
    return v->estimate();
}

g2o::Isometry3D estimateVertexSE3(SlamOptimizer &opt, int id)
{
    g2o::VertexSE3 *v = static_cast<g2o::VertexSE3 *>(opt.vertex(id));
    return v->estimate();
}

g2o::Vector3D estimateVertexXYZ(SlamOptimizer &opt, int id)
{
    g2o::VertexPointXYZ *v = static_cast<g2o::VertexPointXYZ *>(opt.vertex(id));
    return v->estimate();
}

bool verifyInfo(const g2o::Matrix6d &info)
{
    bool symmetric = true;
    double th = 0.0001;
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < i; ++j)
            symmetric = (std::abs(info(i, j) - info(j, i)) < th) && symmetric;
    return symmetric;
}

bool verifyInfo(const Eigen::Matrix3d &info)
{
    double th = 0.0001;
    return (std::abs(info(0, 1) - info(1, 0)) < th && std::abs(info(0, 2) - info(2, 0)) < th &&
            std::abs(info(1, 2) - info(2, 1)) < th);
}


}  // namespace se2lam
