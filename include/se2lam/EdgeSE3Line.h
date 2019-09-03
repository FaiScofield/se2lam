/////////////////////////////////////////////////////////////////////////////////
//
// LineSLAM, version 1.0
// Copyright (C) 2013-2015 Yan Lu, Dezhen Song
// Netbot Laboratory, Texas A&M University, USA
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.
//
//
/////////////////////////////////////////////////////////////////////////////////

#ifndef G2O_EDGE_SE3_LINE_ENDPTS_H_
#define G2O_EDGE_SE3_LINE_ENDPTS_H_

#include "g2o/core/base_binary_edge.h"

#include "g2o/types/slam3d/parameter_se3_offset.h"
#include "g2o/types/slam3d/vertex_se3.h"

#include "g2o/types/slam3d/g2o_types_slam3d_api.h"

namespace Eigen
{
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
}

namespace g2o
{


/**
 * \brief Vertex for a tracked point in space
 */
class VertexLine : public BaseVertex<6, Eigen::Vector6d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    VertexLine() {}
//    virtual bool read(std::istream &is);
//    virtual bool write(std::ostream &os) const;

    bool read(std::istream &is)
    {
        Vector6d lv;
        for (int i = 0; i < estimateDimension(); i++)
            is >> lv[i];
        setEstimate(lv);
        return true;
    }

    bool write(std::ostream &os) const
    {
        Vector6d lv = estimate();
        for (int i = 0; i < estimateDimension(); i++) {
            os << lv[i] << " ";
        }
        return os.good();
    }

    virtual void setToOriginImpl() { _estimate.fill(0.); }

    virtual void oplusImpl(const double *update_)
    {
        Map<const Vector6d> update(update_);
        _estimate += update;
    }

    virtual bool setEstimateDataImpl(const double *est)
    {
        Map<const Vector6d> _est(est);
        _estimate = _est;
        return true;
    }

    virtual bool getEstimateData(double *est) const
    {
        Map<Vector6d> _est(est);
        _est = _estimate;
        return true;
    }

    virtual int estimateDimension() const { return 6; }

    virtual bool setMinimalEstimateDataImpl(const double *est)
    {
        _estimate = Map<const Vector6d>(est);
        return true;
    }

    virtual bool getMinimalEstimateData(double *est) const
    {
        Map<Vector6d> v(est);
        v = _estimate;
        return true;
    }

    virtual int minimalEstimateDimension() const { return 6; }
};


// first two args are the measurement type, second two the connection classes
class EdgeSE3Line : public BaseBinaryEdge<6, Vector6d, VertexSE3, VertexLine>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3Line();
    virtual bool read(std::istream &is);
    virtual bool write(std::ostream &os) const;

    // return the error estimate as a 3-vector
    void computeError();

    // jacobian
    //   virtual void linearizeOplus();

    virtual void setMeasurement(const Vector6d &m) { _measurement = m; }

    virtual bool setMeasurementData(const double *d)
    {
        Map<const Vector6d> v(d);
        _measurement = v;
        return true;
    }

    virtual bool getMeasurementData(double *d) const
    {
        Map<Vector6d> v(d);
        v = _measurement;
        return true;
    }

    virtual int measurementDimension() const { return 6; }

    virtual bool setMeasurementFromState();

    virtual double initialEstimatePossible(const OptimizableGraph::VertexSet &from,
                                           OptimizableGraph::Vertex *to)
    {
        (void)to;
        return (from.count(_vertices[0]) == 1 ? 1.0 : -1.0);
    }

    virtual void initialEstimate(const OptimizableGraph::VertexSet &from,
                                 OptimizableGraph::Vertex *to);

    Eigen::Matrix<double, 6, 6> endptCov;
    Eigen::Matrix<double, 6, 6> endpt_AffnMat;  // to compute mahalanobis dist

private:
    Eigen::Matrix<double, 6, 6 + 6> J;  // jacobian before projection
    ParameterSE3Offset *offsetParam;
    CacheSE3Offset *cache;
    virtual bool resolveCaches();
};

}
#endif
