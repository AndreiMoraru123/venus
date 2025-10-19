#include "core/memory/device.hpp"
#include <cassert>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <functional>
#include <print>

#include <cmath>
#include <core/tensor/tensor.hpp>
#include <numeric>
#include <tuple>

using namespace venus;

TEST_CASE("Tensor Ops", "[tensor][ops]") {
  SECTION("Scalar Ops") {
    auto x = Tensor<int, Device::CPU, 0>(5);
    auto y = Tensor<float, Device::CPU, 0>(2.5f);

    using Op = std::function<decltype(x + y)(decltype(x), decltype(y))>;

    auto [name, op, expected] = GENERATE(
        std::make_tuple("Addition", Op([](auto a, auto b) { return a + b; }),
                        7.5f),
        std::make_tuple("Multiplication",
                        Op([](auto a, auto b) { return a * b; }), 12.5f),
        std::make_tuple("Subtraction", Op([](auto a, auto b) { return a - b; }),
                        2.5f),
        std::make_tuple("Division", Op([](auto a, auto b) { return a / b; }),
                        2.0f));

    DYNAMIC_SECTION("Scalar " << name) {
      auto result = op(x, y);

      STATIC_REQUIRE(std::is_same_v<decltype(result)::ElementType, float>);
      REQUIRE(result.Value() == expected);
    }
  }

  SECTION("Tensor Ops") {
    auto x = Tensor<int, Device::CPU, 2>(3, 3);
    auto y = Tensor<float, Device::CPU, 2>(3, 3);

    venus::ops::iota(x, 1);
    venus::ops::iota(y, 1);

    using TensorOp = std::function<decltype(x + y)(decltype(x), decltype(y))>;
    using ScalarOp = std::function<float(int, float)>;

    auto [name, tensor_op, scalar_op] = GENERATE(
        std::make_tuple("Addition",
                        TensorOp([](auto a, auto b) { return a + b; }),
                        ScalarOp([](int a, float b) { return a + b; })),
        std::make_tuple("Multiplication",
                        TensorOp([](auto a, auto b) { return a * b; }),
                        ScalarOp([](int a, float b) { return a * b; })),
        std::make_tuple("Subtraction",
                        TensorOp([](auto a, auto b) { return a - b; }),
                        ScalarOp([](int a, float b) { return a - b; })),
        std::make_tuple("Division",
                        TensorOp([](auto a, auto b) { return a / b; }),
                        ScalarOp([](int a, float b) { return a / b; })));

    DYNAMIC_SECTION("Tensor " << name) {
      auto result = tensor_op(x, y);
      auto expected = Tensor<int, Device::CPU, 2>(3, 3);
      std::ranges::transform(x, y, expected.begin(),
                             [&scalar_op](int x_val, float y_val) {
                               return scalar_op(x_val, y_val);
                             });

      REQUIRE(result.Shape() == x.Shape());
      REQUIRE(result.Shape() == y.Shape());
      REQUIRE(std::ranges::equal(result, expected));
    }
  }

  SECTION("Tensor Ops (Shape Mismatch)") {
    auto x = Tensor<int, Device::CPU, 2>(3, 3);
    auto y = Tensor<int, Device::CPU, 2>(2, 2);

    using Op = std::function<decltype(x + y)(decltype(x), decltype(y))>;

    auto [name, op] = GENERATE(
        std::make_tuple("Addition", Op([](auto a, auto b) { return a + b; })),
        std::make_tuple("Multiplication",
                        Op([](auto a, auto b) { return a * b; })),
        std::make_tuple("Subtraction",
                        Op([](auto a, auto b) { return a - b; })),
        std::make_tuple("Division", Op([](auto a, auto b) { return a / b; })));

    DYNAMIC_SECTION("Tensor " << name << " (Shape Mismatch)") {
      REQUIRE_THROWS_AS(op(x, y), std::invalid_argument);
    }
  }

  SECTION("Scalar Transform") {
    auto tensor = Tensor<float, Device::CPU, 0>(5.0);
    auto res = venus::ops::transform(tensor, [](auto &&t) { return t * 3; });

    STATIC_REQUIRE(std::is_same_v<decltype(res)::ElementType, float>);
    REQUIRE(res.Value() == 15.0f);
  }

  SECTION("Tensor Transform") {
    auto tensor = Tensor<float, Device::CPU, 2>(3, 3);

    venus::ops::iota(tensor, 1);

    auto res = venus::ops::transform(tensor, [](auto &&t) { return t * 3; });

    STATIC_REQUIRE(std::is_same_v<decltype(res)::ElementType, float>);
    REQUIRE(res[0, 0] == 3.0f);  // 1 * 3
    REQUIRE(res[1, 1] == 15.0f); // 5 * 3
    REQUIRE(res[2, 2] == 27.0f); // 9 * 3
  }

  SECTION("Dot product") {
    auto x = Tensor<int, Device::CPU, 1>(3);
    auto y = Tensor<float, Device::CPU, 1>(3);

    venus::ops::iota(x, 1);
    venus::ops::iota(y, 1);

    auto z = x.dot(y);

    STATIC_REQUIRE(
        std::is_same_v<decltype(z)::ElementType, float>); // type promotion
    REQUIRE(z.Value() == 14.0f); // 1 * 1 + 2 * 2 + 3 * 3
    REQUIRE(z == y.dot(x));      // commutative
  }
}