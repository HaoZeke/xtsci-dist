#include "include/xtensor_pdist.hpp"
#include "xtensor/xarray.hpp"
#include <catch2/catch_all.hpp>

TEST_CASE("Testing pdist_seuclidean", "[pdist_seuclidean]") {

  SECTION("Test 1: Simple 3x1 matrix") {
    xt::xarray<double> mat = {0.57955012, 0.9640571, 0.96120518};
    mat.reshape({3, 1});

    auto result = xts::pdist_seuclidean(mat);

    REQUIRE(result.shape()[0] == 3);
    REQUIRE(result(0) == Catch::Approx(1.73846197));
    REQUIRE(result(1) == Catch::Approx(1.72556765));
    REQUIRE(result(2) == Catch::Approx(0.01289432));
  }

  SECTION("Test 2: Another 3x1 matrix") {
    xt::xarray<double> mat = {1.0, 2.0, 3.0};
    mat.reshape({3, 1});

    auto result = xts::pdist_seuclidean(mat);

    REQUIRE(result.shape()[0] == 3);
    REQUIRE(result(0) == Catch::Approx(1.0));
    REQUIRE(result(1) == Catch::Approx(2.0));
    REQUIRE(result(2) == Catch::Approx(1.0));
  }

  SECTION("Test 3: Input is not a 2D tensor") {
    xt::xarray<double> mat = {1.0, 2.0, 3.0};

    REQUIRE_THROWS_AS(xts::pdist_seuclidean(mat), std::runtime_error);
  }
}
