// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include "xtensor/xarray.hpp"

#include "distance/pdist/euclidean.hpp"
#include "distance/squareform.hpp"

#include <catch2/catch_all.hpp>

TEST_CASE("Test xts::distance::squareform 1D to 2D",
          "[xts::distance::squareform]") {
  xt::xarray<double> input = {2.0, 3.0, 4.0};
  xt::xarray<double> expected = {
      {0.0, 2.0, 3.0}, {2.0, 0.0, 4.0}, {3.0, 4.0, 0.0}};

  auto output = xts::distance::squareform(input);
  REQUIRE(xt::all(xt::equal(output, expected)));
}

TEST_CASE("Test xts::distance::squareform 2D to 1D",
          "[xts::distance::squareform]") {
  xt::xarray<double> input = {
      {0.0, 2.0, 3.0}, {2.0, 0.0, 4.0}, {3.0, 4.0, 0.0}};
  xt::xarray<double> expected = {2.0, 3.0, 4.0};

  auto output = xts::distance::squareform(input);
  REQUIRE(xt::all(xt::equal(output, expected)));
}

TEST_CASE("Test xts::distance::squareform with non-square 2D array",
          "[xts::distance::squareform]") {
  xt::xarray<double> input = {{0.0, 2.0}, {2.0, 0.0}, {3.0, 4.0}};

  REQUIRE_THROWS(xts::distance::squareform(input));
}

TEST_CASE("Test xts::distance::squareform with 3D array",
          "[xts::distance::squareform]") {
  xt::xarray<double> input = xt::xarray<double>::from_shape({2, 2, 2});

  REQUIRE_THROWS(xts::distance::squareform(input));
}

TEST_CASE("Test xts::distance::squareform with incorrect 1D array",
          "[xts::distance::squareform]") {
  xt::xarray<double> input = {1.0, 2.0};

  REQUIRE_THROWS(xts::distance::squareform(input));
}

TEST_CASE("Test xts::distance::squareform round-tripping",
          "[xts::distance::squareform]") {
  xt::xarray<double> feat = {0.57955012, 0.9640571, 0.96120518};
  feat.reshape({3, 1});

  auto xtpd_euc = xts::distance::pdist::euclidean(feat);
  auto sqform = xts::distance::squareform(xtpd_euc);
  auto xtpd_euc_re = xts::distance::squareform(sqform);

  REQUIRE(xt::all(xt::equal(xtpd_euc, xtpd_euc_re)));
}
