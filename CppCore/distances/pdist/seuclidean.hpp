#include "xtensor/xexpression.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"

#include "xtensor-blas/xlinalg.hpp"

namespace xts {
namespace distances {
namespace pdist {

template <class E> auto seuclidean_1D(const xt::xexpression<E> &expr) {
  const auto &mat = expr.derived_cast();
  auto var = xt::variance(mat, /*axes=*/{0}, /*ddof=*/1);
  std::size_t n_rows = mat.shape()[0];
  std::size_t m_entries = n_rows * (n_rows - 1) / 2;
  xt::xarray<typename E::value_type> distances =
      xt::zeros<typename E::value_type>({m_entries});

  size_t k = 0;
  for (std::size_t i = 0; i < n_rows; ++i) {
    for (std::size_t j = i + 1; j < n_rows; ++j) {
      auto diff = xt::row(mat, i) - xt::row(mat, j);
      auto dist = xt::linalg::norm(diff, 2);
      distances(k) = dist;
      k++;
    }
  }
  distances /= sqrt(var);
  return distances;
}

template <class E> auto seuclidean_ND(const xt::xexpression<E> &expr) {
  const auto &mat = expr.derived_cast();
  auto var = xt::variance(mat, /*axes=*/{0}, /*ddof=*/1);
  auto sqrt_var_array = xt::eval(xt::sqrt(var));
  sqrt_var_array.reshape({1, sqrt_var_array.shape()[0]});
  std::size_t n_rows = mat.shape()[0];
  std::size_t m_entries = n_rows * (n_rows - 1) / 2;
  xt::xarray<typename E::value_type> distances =
      xt::zeros<typename E::value_type>({m_entries});

  size_t k = 0;
  for (std::size_t i = 0; i < n_rows; ++i) {
    for (std::size_t j = i + 1; j < n_rows; ++j) {
      auto diff = (xt::row(mat, i) - xt::row(mat, j)) / sqrt_var_array;
      auto dist = xt::linalg::norm(diff, 2);
      distances(k) = dist;
      k++;
    }
  }
  return distances;
}

template <class E> auto seuclidean(const xt::xexpression<E> &expr) {
  // TODO: Support passing in variance like scipy does
  const auto &mat = expr.derived_cast();
  if (mat.dimension() != 2) {
    throw std::runtime_error("Input tensor must be 2D");
  }
  auto var = xt::variance(mat, /*axes=*/{0}, /*ddof=*/1);
  std::size_t n_rows = mat.shape()[0];
  std::size_t m_entries = n_rows * (n_rows - 1) / 2;
  if (!var.dimension()) {
    return seuclidean_1D(expr);
  } else {
    return seuclidean_ND(expr);
  }
}

} // namespace pdist
} // namespace distances
} // namespace xts
