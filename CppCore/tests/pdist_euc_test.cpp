// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include "xtensor-io/xnpz.hpp"
#include "xtensor/xarray.hpp"

#include "distances/pdist/euclidean.hpp"

#include <catch2/catch_all.hpp>

TEST_CASE("Testing distances::pdist::euclidean",
          "[distances::pdist::euclidean]") {
  SECTION("Test 1: Simple 3x1 matrix") {
    xt::xarray<double> mat = {0.57955012, 0.9640571, 0.96120518};
    mat.reshape({3, 1});

    auto result = xts::distances::pdist::euclidean(mat);

    REQUIRE(result.shape()[0] == 3);
    REQUIRE(result(0) == Catch::Approx(0.38450698));
    REQUIRE(result(1) == Catch::Approx(0.38165506));
    REQUIRE(result(2) == Catch::Approx(0.00285192));
  }

  SECTION("Test 2: Another 3x1 matrix") {
    xt::xarray<double> mat = {1.0, 2.0, 3.0};
    mat.reshape({3, 1});

    auto result = xts::distances::pdist::euclidean(mat);

    REQUIRE(result.shape()[0] == 3);
    REQUIRE(result(0) == Catch::Approx(1.0));
    REQUIRE(result(1) == Catch::Approx(2.0));
    REQUIRE(result(2) == Catch::Approx(1.0));
  }

  SECTION("Test 3: A 3x4 matrix") {
    xt::xarray<double> mat = xt::load_npz<double>("data/inp_3_4.npz", "inp");
    // TODO(rgoswami): Upstream bug, Layout not supported in immediate
    // reduction. auto tdat= xt::load_npz("data/inp_3_4.npz"); auto mat =
    // tdat["inp"].cast<double>();
    xt::xarray<double> expected =
        xt::load_npz<double>("data/eucDist_3_4.npz", "eucDist");
    auto result = xts::distances::pdist::euclidean(mat);
    REQUIRE(result.shape()[0] == 3);
    for (size_t idx{0}; idx < expected.shape()[0]; ++idx) {
      REQUIRE(result(idx) == Catch::Approx(expected(idx)));
    }
  }

  SECTION("Test 4: Input is not a 2D tensor") {
    xt::xarray<double> mat = {1.0, 2.0, 3.0};

    REQUIRE_THROWS_AS(xts::distances::pdist::euclidean(mat),
                      std::runtime_error);
  }
}
