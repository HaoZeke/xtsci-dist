#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include "xtensor/xexpression.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"

#include "xtensor-blas/xlinalg.hpp"

#include "./euclidean.hpp"

namespace xts {
namespace distance {
namespace pdist {
template <class E> auto sqeuclidean(const xt::xexpression<E> &expr) {
  auto distances = xts::distance::pdist::euclidean(expr);
  distances = xt::pow(distances, 2);
  return distances;
}
} // namespace pdist
} // namespace distance
} // namespace xts
