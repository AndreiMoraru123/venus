
#include <catch2/catch_test_macros.hpp>

#include <core/tensor/shape.hpp>
#include <ranges>

using namespace venus;

TEST_CASE("Shape semantics", "[shape]") {

  SECTION("Shape construction") {
    constexpr auto shape = Shape<3>();
    STATIC_REQUIRE(shape[0] == 0);
    STATIC_REQUIRE(shape[1] == 0);
    STATIC_REQUIRE(shape[2] == 0);
  }

  SECTION("Shape argument deduction") {
    STATIC_REQUIRE(Shape<3>() == Shape(0, 0, 0));
    STATIC_REQUIRE(Shape<3>(1, 2, 3) == Shape(1, 2, 3));
  }

  SECTION("Shapes of different dimensions") {
    STATIC_REQUIRE(Shape<4>() != Shape<3>());
    STATIC_REQUIRE(Shape<4>() != Shape(0, 0, 0));
  }

  SECTION("Shape Count") {
    STATIC_REQUIRE(Shape(0, 0, 0).Count() == 0);
    STATIC_REQUIRE(Shape(1, 2, 3).Count() == 6);
    STATIC_REQUIRE(Shape().Count() == 1); // scalar
  }

  SECTION("Iterate Shape") {
    constexpr auto shape = Shape<4>();

    for (const auto dim : shape) {
      REQUIRE(dim == 0);
    }

    for (const auto dim : std::ranges::reverse_view(shape)) {
      REQUIRE(dim == 0);
    }
  }

  SECTION("Shape size") {
    constexpr auto shape = Shape<4>(1, 2, 3, 4);
    constexpr auto sz = std::ranges::size(shape);
    STATIC_REQUIRE(sz == 4);
  }
}