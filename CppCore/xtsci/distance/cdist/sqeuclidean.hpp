#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include "xtensor/xexpression.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"

#include "xtensor-blas/xlinalg.hpp"

namespace xts {
namespace distance {
namespace cdist {

template <class E1, class E2>
auto sqeuclidean(const xt::xexpression<E1> &expr1,
                 const xt::xexpression<E2> &expr2) {
  const auto &mat1 = expr1.derived_cast();
  const auto &mat2 = expr2.derived_cast();

  if (mat1.dimension() != 2 || mat2.dimension() != 2) {
    throw std::runtime_error("Input tensors must be 2D");
  }

  if (mat1.shape()[1] != mat2.shape()[1]) {
    throw std::runtime_error("Incompatible shapes: the number of columns must "
                             "be the same for both matrices");
  }

  std::size_t m_rows = mat1.shape()[0];
  std::size_t p_rows = mat2.shape()[0];

  xt::xtensor<typename E1::value_type, 2> distances =
      xt::empty<typename E1::value_type>({m_rows, p_rows});

  for (std::size_t i = 0; i < m_rows; ++i) {
    for (std::size_t j = 0; j < p_rows; ++j) {
      auto diff = xt::row(mat1, i) - xt::row(mat2, j);
      auto dist = xt::linalg::norm(diff, 2);
      distances(i, j) = dist;
    }
  }
  distances = xt::pow(distances, 2);
  return distances;
}

} // namespace cdist
} // namespace distance
} // namespace xts
