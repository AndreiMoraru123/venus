
#include <catch2/catch_test_macros.hpp>

#include <core/tensor/shape.hpp>

using namespace venus;

TEST_CASE("Shape semantics", "[shape]") {

  SECTION("Shape construction") {
    const auto shape = Shape<3>();
    for (int i = 0; i <= 2; i++) {
      REQUIRE(shape[i] == 0);
    }

    // const auto anotherShape = Shape(1, 2, 3);
  }
}