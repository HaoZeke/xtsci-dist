// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <fmt/format.h>
#include <fmt/ostream.h>

#include <cmath>
#include <cstdlib>

#include <iostream>

#include "distance/pdist/euclidean.hpp"
#include "distance/pdist/seuclidean.hpp"
#include "distance/squareform.hpp"
#include "include/xtensor_fmt.hpp"
#include "xtensor-io/xnpz.hpp"
#include "xtensor/xarray.hpp"

int main(int argc, char *argv[]) {
  // Eat warnings, also safer
  static_cast<void>(argc);
  static_cast<void>(argv);
  xt::xarray<double> feat = {0.57955012, 0.9640571, 0.96120518};
  feat.reshape({3, 1});
  // auto feat = xt::load_npz<double>("inp_3_4.npz", "A");
  fmt::print("feat: {}\n", feat);
  xt::xarray<double> xtpd_euc = xts::distance::pdist::euclidean(feat);
  fmt::print("xtpd_euc: {}\n", xtpd_euc);
  xt::xarray<double> xtpd_seuc = xts::distance::pdist::seuclidean(feat);
  fmt::print("xtpd_seuc: {}\n", xtpd_seuc);
  // Squareform
  auto sqform = xts::distance::squareform(xtpd_euc);
  fmt::print("sqform: {}\n", sqform);
  auto xtpd_seuc_re = xts::distance::squareform(sqform);
  fmt::print("xtpd_seuc_re: {}\n", xtpd_seuc_re);
  return EXIT_SUCCESS;
}
