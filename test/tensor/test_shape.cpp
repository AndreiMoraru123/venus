
#include <algorithm>
#include <cassert>
#include <catch2/catch_test_macros.hpp>
#include <numeric>

#include <ranges>
#include <venus/tensor/shape.hpp>

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

  SECTION("Index to Offset") {
    constexpr auto shape = Shape<3>(3, 2, 2);
    STATIC_REQUIRE(shape.IndexToOffset(0, 0, 0) == 0);
    STATIC_REQUIRE(shape.IndexToOffset(0, 0, 1) == 1);
    STATIC_REQUIRE(shape.IndexToOffset(0, 1, 0) == 2);
    STATIC_REQUIRE(shape.IndexToOffset(1, 0, 0) == 4);
    STATIC_REQUIRE(shape.IndexToOffset(1, 0, 1) == 5);
    STATIC_REQUIRE(shape.IndexToOffset(1, 1, 0) == 6);
    STATIC_REQUIRE(shape.IndexToOffset(1, 1, 1) == 7);
    STATIC_REQUIRE(shape.IndexToOffset(2, 0, 0) == 8);
    STATIC_REQUIRE(shape.IndexToOffset(2, 0, 1) == 9);
    STATIC_REQUIRE(shape.IndexToOffset(2, 1, 0) == 10);
    STATIC_REQUIRE(shape.IndexToOffset(2, 1, 1) == 11);
  }

  SECTION("Offset to Index") {
    const auto shape = Shape<3>(3, 2, 2);
    size_t input = 0;
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 2; ++j) {
        for (size_t k = 0; k < 2; ++k) {
          auto res = shape.OffsetToIndex(input);
          assert(res[0] == i);
          assert(res[1] == j);
          assert(res[2] == k);
          ++input;
        }
      }
    }
  }

  SECTION("Shape as Range") {
    auto shape = Shape<3>();
    STATIC_REQUIRE(std::ranges::random_access_range<decltype(shape)>);

    std::ranges::fill(shape, 1);
    for (const auto dim : shape) {
      REQUIRE(dim == 1);
    }

    std::ranges::transform(shape, shape.begin(), [](auto &x) { return x * 2; });
    for (const auto dim : shape) {
      REQUIRE(dim == 2);
    }

#if _cpp_lib_ranges >= 202110L
    std::ranges::iota(shape, 1);
#else
    std::iota(shape.begin(), shape.end(), 1);
#endif
    REQUIRE(shape[0] == 1);
    REQUIRE(shape[1] == 2);
    REQUIRE(shape[2] == 3);

    bool all_positive =
        std::ranges::all_of(shape, [](const auto &x) { return x > 0; });
    REQUIRE(all_positive == true);
  }
}