/**
* This file is part of se2lam
*
* Copyright (C) Fan ZHENG (github.com/izhengfan), Hengbo TANG (github.com/hbtang)
*/


#ifndef EDGE_SE2_XYZ_H
#define EDGE_SE2_XYZ_H

#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include "Thirdparty/g2o/g2o/core/base_binary_edge.h"
#include "Thirdparty/g2o/g2o/core/base_vertex.h"
#include "Thirdparty/g2o/g2o/types/sba/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/types/slam2d/vertex_se2.h"

#define CUSTOMIZE_JACOBIAN_SE2XYZ

namespace g2o
{

typedef Eigen::Matrix<double, 2, 3> Matrix23D;
typedef Eigen::Matrix<double, 3, 2> Matrix32D;

Matrix3D d_inv_d_se2(const SE2& _se2);

SE3Quat SE2ToSE3(const SE2& _se2);

SE2 SE3ToSE2(const SE3Quat& _se3);

//! 2元边
//! 2维测量，测量类型为Vector2d，代表像素的重投影误差
//! 两种顶点类型分别是VertexSE2、VertexSBAPointXYZ，即此2元边连接SE2李群位姿点和三维地图点
class EdgeSE2XYZ : public BaseBinaryEdge<2, Vector2D, VertexSE2, VertexSBAPointXYZ>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE2XYZ() {}

    // Useless functions we don't care
    bool read(std::istream& is);
    bool write(std::ostream& os) const;

    // Note: covariance are set here. Just set information to identity outside.
    void computeError();

#ifdef CUSTOMIZE_JACOBIAN_SE2XYZ
    //! 如果不提供解析雅克比, g2o会自动使用数值雅克比, 但会降低计算速度
    virtual void linearizeOplus();
#endif

    g2o::Vector2D cameraProject(const g2o::Vector3D& xyz) const;

    void setCameraParameter(const cv::Mat& K);

    void setExtParameter(const SE3Quat& _Tbc)
    {
        Tbc = _Tbc;
        Tcb = Tbc.inverse();
    }

private:
    SE3Quat Tbc;
    SE3Quat Tcb;

    double fx, fy, cx, cy;
};

//! 2元边
//! 误差为3维向量, 表示se2位姿的差异
//! 两种顶点的类型都是VertexSE2, 表示SE2位姿点
class PreEdgeSE2 : public BaseBinaryEdge<3, Vector3D, VertexSE2, VertexSE2>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PreEdgeSE2() {}

    virtual void setMeasurement(const Vector3D& m){
        _measurement = m;
        _inverseMeasurement = SE2(m).inverse();
    }

    void computeError();

    virtual void linearizeOplus();

    bool read(std::istream& is);
    bool write(std::ostream& os) const;

protected:
    SE2 _inverseMeasurement;
};

}  // namespace se2lam

#endif  // EDGE_SE2_XYZ_H
