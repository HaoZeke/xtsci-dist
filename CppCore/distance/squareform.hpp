#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <variant>
// clang-format off
#include "xtensor/xfunctor_view.hpp"
#include "xtensor/xexpression.hpp"
#include "xtensor/xstrides.hpp"
// clang-format on
namespace xts {
namespace distance {

template <class E> auto squareform_1D(const xt::xexpression<E> &expr) {
  const auto &arr = expr.derived_cast();

  if (arr.dimension() != 1) {
    throw std::runtime_error("Input array must be 1D");
  }

  std::size_t m = arr.size();
  // Calculate n, where m = n * (n - 1) / 2
  std::size_t n =
      static_cast<std::size_t>((1.0 + std::sqrt(1 + 8.0 * m)) / 2.0);
  if (n * (n - 1) / 2 != m) {
    throw std::runtime_error("Invalid size of 1D array for squareform");
  }

  // TODO(rgoswami): We know the size, could return xtensor
  // but there are issues with the dispatching then
  xt::xarray<typename E::value_type> mat =
      xt::zeros<typename E::value_type>({n, n});

  std::size_t k = 0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      mat(i, j) = arr(k);
      mat(j, i) = arr(k);
      ++k;
    }
  }

  return mat;
}

template <class E> auto squareform_2D(const xt::xexpression<E> &expr) {
  const auto &arr = expr.derived_cast();
  if (arr.shape()[0] != arr.shape()[1]) {
    throw std::runtime_error("Input array must be square");
  }
  std::size_t n_rows = arr.shape()[0];
  std::size_t m_entries = n_rows * (n_rows - 1) / 2;
  // TODO(rgoswami): We know the size, could return xtensor
  // but there are issues with the dispatching then
  xt::xarray<typename E::value_type> distances =
      xt::empty<typename E::value_type>({m_entries});
  size_t k = 0;
  for (std::size_t i = 0; i < n_rows; ++i) {
    for (std::size_t j = i + 1; j < n_rows; ++j) {
      distances(k) = arr(i, j);
      k++;
    }
  }
  return distances;
}

template <class E> auto squareform(const xt::xexpression<E> &expr) {
  const auto &val = expr.derived_cast();
  if (val.dimension() == 1) {
    return squareform_1D(expr);
  } else if (val.dimension() == 2) {
    return squareform_2D(expr);
  } else {
    throw std::runtime_error("Input array must be 1D or 2D");
  }
}

} // namespace distance
} // namespace xts
