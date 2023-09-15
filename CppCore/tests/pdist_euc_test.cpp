#include "include/xtensor_pdist.hpp"
#include "xtensor/xarray.hpp"
#include <catch2/catch_all.hpp>

TEST_CASE("Testing pdist_euclidean", "[pdist_euclidean]") {

  SECTION("Test 1: Simple 3x1 matrix") {
    xt::xarray<double> mat = {0.57955012, 0.9640571, 0.96120518};
    mat.reshape({3, 1});

    auto result = xts::pdist_euclidean(mat);

    REQUIRE(result.shape()[0] == 3);
    REQUIRE(result(0) == Catch::Approx(0.38450698));
    REQUIRE(result(1) == Catch::Approx(0.38165506));
    REQUIRE(result(2) == Catch::Approx(0.00285192));
  }

  SECTION("Test 2: Another 3x1 matrix") {
    xt::xarray<double> mat = {1.0, 2.0, 3.0};
    mat.reshape({3, 1});

    auto result = xts::pdist_euclidean(mat);

    REQUIRE(result.shape()[0] == 3);
    REQUIRE(result(0) == Catch::Approx(1.0));
    REQUIRE(result(1) == Catch::Approx(2.0));
    REQUIRE(result(2) == Catch::Approx(1.0));
  }

  SECTION("Test 3: Input is not a 2D tensor") {
    xt::xarray<double> mat = {1.0, 2.0, 3.0};

    REQUIRE_THROWS_AS(xts::pdist_euclidean(mat), std::runtime_error);
  }
}
