#include "core/memory/device.hpp"
#include <cassert>
#include <catch2/catch_test_macros.hpp>
#include <print>

#include <cmath>
#include <core/tensor/tensor.hpp>
#include <numeric>

using namespace venus;

TEST_CASE("Tensor Ops", "[tensor][ops]") {
  SECTION("Scalar Addition") {
    auto x = Tensor<int, Device::CPU, 0>(5);
    auto y = Tensor<float, Device::CPU, 0>(2.5f);
    auto z = x + y;

    STATIC_REQUIRE(
        std::is_same_v<decltype(z)::ElementType, float>); // type promotion
    REQUIRE(z.Value() == 7.5f);
  }

  SECTION("Scalar Multiplication") {
    auto x = Tensor<int, Device::CPU, 0>(5);
    auto y = Tensor<float, Device::CPU, 0>(2.5f);
    auto z = x * y;

    STATIC_REQUIRE(
        std::is_same_v<decltype(z)::ElementType, float>); // type promotion
    REQUIRE(z.Value() == 12.5f);
  }

  SECTION("Tensor Addition") {
    auto x = Tensor<int, Device::CPU, 2>(3, 3);
    auto y = Tensor<float, Device::CPU, 2>(3, 3);
    auto expected = Tensor<int, Device::CPU, 2>(3, 3);

#if _cpp_lib_ranges >= 202110L
    std::ranges::iota(x, 1);
    std::ranges::iota(y, 1);
    std::ranges::iota(expected, 1);
#else
    std::iota(x.begin(), x.end(), 1);
    std::iota(y.begin(), y.end(), 1);
    std::iota(expected.begin(), expected.end(), 1);
#endif

    std::ranges::transform(expected, expected.begin(),
                           [](int val) { return val + val; });
    auto z = x + y;

    STATIC_REQUIRE(
        std::is_same_v<decltype(z)::ElementType, float>); // type promotion
    REQUIRE(z.Shape() == x.Shape());
    REQUIRE(std::ranges::equal(z, expected));
  }

  SECTION("Tensor Multiplication") {
    auto x = Tensor<int, Device::CPU, 2>(3, 3);
    auto y = Tensor<float, Device::CPU, 2>(3, 3);
    auto expected = Tensor<int, Device::CPU, 2>(3, 3);

#if _cpp_lib_ranges >= 202110L
    std::ranges::iota(x, 1);
    std::ranges::iota(y, 1);
    std::ranges::iota(expected, 1);
#else
    std::iota(x.begin(), x.end(), 1);
    std::iota(y.begin(), y.end(), 1);
    std::iota(expected.begin(), expected.end(), 1);
#endif

    std::ranges::transform(expected, expected.begin(),
                           [](int val) { return val * val; });
    auto z = x * y;

    STATIC_REQUIRE(
        std::is_same_v<decltype(z)::ElementType, float>); // type promotion
    REQUIRE(z.Shape() == x.Shape());
    REQUIRE(std::ranges::equal(z, expected));
  }

  SECTION("Tensor Addition (shape mismatch)") {
    auto x = Tensor<int, Device::CPU, 2>(3, 3);
    auto y = Tensor<int, Device::CPU, 2>(2, 2);

    REQUIRE_THROWS_AS(x + y, std::invalid_argument);
  }

  SECTION("Scalar Transform") {
    auto tensor = Tensor<float, Device::CPU, 0>(5.0);
    auto res = venus::ops::transform(tensor, [](auto &&t) { return t * 3; });

    STATIC_REQUIRE(std::is_same_v<decltype(res)::ElementType, float>);
    REQUIRE(res.Value() == 15.0f);
  }

  SECTION("Tensor Transform") {
    auto tensor = Tensor<float, Device::CPU, 2>(3, 3);

#if _cpp_lib_ranges >= 202110L
    std::ranges::iota(tensor, 1);
#else
    std::iota(tensor.begin(), tensor.end(), 1);
#endif

    auto res = venus::ops::transform(tensor, [](auto &&t) { return t * 3; });

    STATIC_REQUIRE(std::is_same_v<decltype(res)::ElementType, float>);
    REQUIRE(res[0, 0] == 3.0f);  // 1 * 3
    REQUIRE(res[1, 1] == 15.0f); // 5 * 3
    REQUIRE(res[2, 2] == 27.0f); // 9 * 3
  }
}