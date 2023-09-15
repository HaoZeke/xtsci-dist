// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include "xtensor/xarray.hpp"

#include "xtsci/util.hpp"

#include <catch2/catch_all.hpp>

TEST_CASE("Test xts::util::fill_diagonal on 3x3 array",
          "[xts::util::fill_diagonal]") {
  xt::xtensor<double, 2> arr = {
      {0.0, 1.0, 2.0}, {3.0, 4.0, 5.0}, {6.0, 7.0, 8.0}};

  xts::util::fill_diagonal(arr, 9.0);

  REQUIRE(arr(0, 0) == Catch::Approx(9.0));
  REQUIRE(arr(1, 1) == Catch::Approx(9.0));
  REQUIRE(arr(2, 2) == Catch::Approx(9.0));
  REQUIRE(arr(0, 1) == Catch::Approx(1.0));
  REQUIRE(arr(0, 2) == Catch::Approx(2.0));
}

TEST_CASE("Test xts::util::fill_diagonal on 2x2 array",
          "[xts::util::fill_diagonal]") {
  xt::xtensor<double, 2> arr = {{1.0, 2.0}, {3.0, 4.0}};

  xts::util::fill_diagonal(arr, -1.0);

  REQUIRE(arr(0, 0) == Catch::Approx(-1.0));
  REQUIRE(arr(1, 1) == Catch::Approx(-1.0));
  REQUIRE(arr(0, 1) == Catch::Approx(2.0));
  REQUIRE(arr(1, 0) == Catch::Approx(3.0));
}
