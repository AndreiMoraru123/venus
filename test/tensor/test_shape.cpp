
#include <catch2/catch_test_macros.hpp>

#include <core/tensor/shape.hpp>

using namespace venus;

TEST_CASE("Shape semantics", "[shape]") {

  SECTION("Shape construction") {
    const auto shape = Shape<3>();
    for (int i = 0; i <= 2; i++) {
      REQUIRE(shape[i] == 0);
    }
  }

  SECTION("Shape argument deduction") {
    REQUIRE(Shape<3>() == Shape(0, 0, 0));
    REQUIRE(Shape<3>(1, 2, 3) == Shape(1, 2, 3));
  }

  SECTION("Shapes of different dimensions") {
    REQUIRE(Shape<4>() != Shape<3>());
    REQUIRE(Shape<4>() != Shape(0, 0, 0));
  }
}