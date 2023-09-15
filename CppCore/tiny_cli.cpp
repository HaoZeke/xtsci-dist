#include <cmath>
#include <cstdlib>
#include <iostream>

#include <fmt/format.h>
#include <fmt/ostream.h>

#include "include/xtensor_fmt.hpp"
#include "xtensor/xarray.hpp"
#include "include/xtensor_pdist.hpp"

int main(int argc, char *argv[]) {
  // Eat warnings, also safer
  static_cast<void>(argc);
  static_cast<void>(argv);
  xt::xarray<double> feat = {0.57955012, 0.9640571, 0.96120518};
  feat.reshape({3, 1});
  fmt::print("feat: {}\n", feat);
  xt::xarray<double> xtpd_euc = xts::pdist_euclidean(feat);
  fmt::print("xtpd_euc: {}\n", xtpd_euc);
  xt::xarray<double> xtpd_seuc = xts::pdist_seuclidean(feat);
  fmt::print("xtpd_seuc: {}\n", xtpd_seuc);
  return EXIT_SUCCESS;
}
