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
namespace pdist {
template <class E> auto euclidean(const xt::xexpression<E> &expr) {
  const auto &mat = expr.derived_cast();
  if (mat.dimension() != 2) {
    throw std::runtime_error("Input tensor must be 2D");
  }
  std::size_t n_rows = mat.shape()[0];
  std::size_t m_entries = n_rows * (n_rows - 1) / 2;
  xt::xarray<typename E::value_type> distances =
      xt::empty<typename E::value_type>({m_entries});

  size_t k = 0;
  for (std::size_t i = 0; i < n_rows; ++i) {
    for (std::size_t j = i + 1; j < n_rows; ++j) {
      auto diff = xt::row(mat, i) - xt::row(mat, j);
      auto dist = xt::linalg::norm(diff, 2);
      distances(k) = dist;
      k++;
    }
  }
  return distances;
}
} // namespace pdist
} // namespace distance
} // namespace xts
