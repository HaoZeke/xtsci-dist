// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include "xtensor-io/xnpz.hpp"
#include "xtensor/xarray.hpp"

#include "distance/cdist/sqeuclidean.hpp"

#include <catch2/catch_all.hpp>

TEST_CASE("Testing distance::cdist::sqeuclidean",
          "[distance::cdist::sqeuclidean]") {
  SECTION("Test xts::distance::cdist::sqeuclidean") {
    xt::xarray<double> A = {{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}};

    xt::xarray<double> B = {{0.5, 0.6}, {0.7, 0.8}, {0.9, 1.0}};

    xt::xarray<double> expected = {
        {0.32, 0.72, 1.28}, {0.08, 0.32, 0.72}, {0., 0.08, 0.32}};
    auto output = xts::distance::cdist::sqeuclidean(A, B);

    REQUIRE(output.shape()[0] == 3);
    REQUIRE(output.shape()[1] == 3);

    for (std::size_t i = 0; i < 3; ++i) {
      for (std::size_t j = 0; j < 3; ++j) {
        REQUIRE(output(i, j) == Catch::Approx(expected(i, j)).epsilon(0.0001));
      }
    }
  }
  SECTION("Test case: A 3x4 matrix") {
    xt::xarray<double> inpa = xt::load_npz<double>("inpa_3_4.npz", "inpa");
    xt::xarray<double> inpb = xt::load_npz<double>("inpb_2_4.npz", "inpb");
    // TODO(rgoswami): Upstream bug, Layout not supported in immediate
    // reduction. auto tdat= xt::load_npz("inp_3_4.npz"); auto mat =
    // tdat["inp"].cast<double>();
    xt::xarray<double> expected =
        xt::load_npz<double>("sqeucCdist_3_4_2_4.npz", "sqeucCdist");
    auto result = xts::distance::cdist::sqeuclidean(inpa, inpb);
    REQUIRE(result.shape()[0] == 3);
    for (size_t idx{0}; idx < expected.shape()[0]; ++idx) {
      REQUIRE(result(idx) == Catch::Approx(expected(idx)));
    }
  }
}
