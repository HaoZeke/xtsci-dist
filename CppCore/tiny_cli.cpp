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
  xt::xarray<double> xtpdist = xts::pdist_euclidean(feat);
  fmt::print("xtpdist: {}\n", xtpdist);
  return EXIT_SUCCESS;
}
