#ifndef ACT_TYPES_H
#define ACT_TYPES_H

#include "Eigen/Dense"

namespace act {

// Aliases for templated Eigen types used throughout the ACT backends.

template <typename Scalar>
using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

template <typename Scalar>
using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

// Parameter matrix: rows = K or dict_size, cols = 4 (tc, fc, logDt, c)
template <typename Scalar>
using ParamsMat = Eigen::Matrix<Scalar, Eigen::Dynamic, 4, Eigen::RowMajor>;

} // namespace act

#endif // ACT_TYPES_H
