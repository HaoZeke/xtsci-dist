// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <fmt/format.h>
#include <fmt/ostream.h>

#include <cmath>
#include <cstdlib>

#include <iostream>

#include "distance/cdist/euclidean.hpp"
#include "distance/pdist/euclidean.hpp"
#include "distance/pdist/seuclidean.hpp"
#include "distance/pdist/sqeuclidean.hpp"
#include "distance/squareform.hpp"
#include "include/xtensor_fmt.hpp"
#include "xtensor-io/xnpz.hpp"
#include "xtensor/xarray.hpp"

int main(int argc, char *argv[]) {
  // Eat warnings, also safer
  static_cast<void>(argc);
  static_cast<void>(argv);
  // xt::xarray<double> feat = {0.57955012, 0.9640571, 0.96120518};
  // feat.reshape({3, 1});
  // auto feat = xt::load_npz<double>("inp_3_4.npz", "inp");
  // fmt::print("feat: {}\n", feat);
  // xt::xarray<double> xtpd_euc = xts::distance::pdist::euclidean(feat);
  // fmt::print("xtpd_euc: {}\n", xtpd_euc);
  // xt::xarray<double> xtpd_sqeuc = xts::distance::pdist::sqeuclidean(feat);
  // fmt::print("xtpd_sqeuc: {}\n", xtpd_sqeuc);
  // xt::xarray<double> xtpd_seuc = xts::distance::pdist::seuclidean(feat);
  // fmt::print("xtpd_seuc: {}\n", xtpd_seuc);
  // Squareform
  // auto sqform = xts::distance::squareform(xtpd_euc);
  // fmt::print("sqform: {}\n", sqform);
  // auto xtpd_seuc_re = xts::distance::squareform(sqform);
  // fmt::print("xtpd_seuc_re: {}\n", xtpd_seuc_re);

  // cdist
  auto inpa = xt::load_npz<double>("inpa_3_4.npz", "inpa");
  auto inpb = xt::load_npz<double>("inpb_2_4.npz", "inpb");
  auto eucCdist = xt::load_npz<double>("eucCdist_3_4_2_4.npz", "eucCdist");
  xt::xarray<double> eccdist = xts::distance::cdist::euclidean(inpa, inpb);
  fmt::print("eccdist: {}\n", eccdist);
  return EXIT_SUCCESS;
}
