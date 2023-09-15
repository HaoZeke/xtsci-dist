#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
// clang-format off
#include <algorithm>
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"
// clang-format on

namespace xts {
namespace util {

template <class E, class T> void fill_diagonal(E &arr, T fill_value) {
  if (arr.dimension() < 2) {
    throw std::runtime_error("Array must have at least two dimensions");
  }

  auto shape = arr.shape();
  auto strides = arr.strides();

  std::size_t min_dim = std::min(shape[0], shape[1]);
  for (std::size_t i = 0; i < min_dim; ++i) {
    std::size_t idx = i * strides[0] + i * strides[1];
    arr.data()[idx] = fill_value;
  }
}
} // namespace util
} // namespace xts
