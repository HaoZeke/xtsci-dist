// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <fmt/format.h>
#include <fmt/ostream.h>

#include <cmath>
#include <cstdlib>

#include <iostream>

#include "distances/pdist/euclidean.hpp"
#include "distances/pdist/seuclidean.hpp"
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
  xt::xarray<double> xtpd_euc = xts::distances::pdist::euclidean(feat);
  fmt::print("xtpd_euc: {}\n", xtpd_euc);
  xt::xarray<double> xtpd_seuc = xts::distances::pdist::seuclidean(feat);
  fmt::print("xtpd_seuc: {}\n", xtpd_seuc);
  return EXIT_SUCCESS;
}
