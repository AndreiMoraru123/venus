#include <cassert>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <functional>
#include <venus/memory/device.hpp>

#include <cmath>
#include <tuple>
#include <venus/tensor/tensor.hpp>

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
      REQUIRE(result.value() == expected);
    }
  }

  SECTION("Tensor Ops") {
    auto x = Tensor<int, Device::CPU, 2>(3, 3);
    auto y = Tensor<float, Device::CPU, 2>(3, 3);

    x.iota(1);
    y.iota(1);

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

      REQUIRE(result.shape() == x.shape());
      REQUIRE(result.shape() == y.shape());
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

  SECTION("Tensor Transform") {
    const auto tensor =
        Tensor<float, Device::CPU, 2>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto res = venus::ops::transform(tensor, [](auto &&t) { return t * 3; });

    STATIC_REQUIRE(std::is_same_v<decltype(res)::ElementType, float>);
    REQUIRE(res[0, 0] == 3.0f);  // 1 * 3
    REQUIRE(res[1, 1] == 15.0f); // 5 * 3
    REQUIRE(res[2, 2] == 27.0f); // 9 * 3
  }

  SECTION("Tensor Transform (in-place)") {
    auto tensor =
        Tensor<float, Device::CPU, 2>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    tensor.transform([](auto &&t) { return t * 3; });

    REQUIRE(tensor[0, 0] == 3.0f);  // 1 * 3
    REQUIRE(tensor[1, 1] == 15.0f); // 5 * 3
    REQUIRE(tensor[2, 2] == 27.0f); // 9 * 3
  }

  SECTION("Dot product") {
    auto x = Tensor<int, Device::CPU, 1>(3);
    auto y = Tensor<float, Device::CPU, 1>(3);

    x.iota(1);
    y.iota(1);

    auto z = x.dot(y);

    STATIC_REQUIRE(
        std::is_same_v<decltype(z)::ElementType, float>); // type promotion
    REQUIRE(z.value() == 14.0f); // 1 * 1 + 2 * 2 + 3 * 3
    REQUIRE(z == y.dot(x));      // commutative
  }

  SECTION("2D Matrix Multiplication") {
    auto A = Tensor<int, Device::CPU, 2>{{1, 2, 3}, {4, 5, 6}};
    auto B = Tensor<int, Device::CPU, 2>{{7, 8}, {9, 10}, {11, 12}};

    auto C = venus::ops::mm(A, B);

    auto [M, K] = A.shape();
    auto [K2, N] = B.shape();

    auto expected = Tensor<int, Device::CPU, 2>(M, N);

    for (std::size_t i = 0; i < M; ++i) {
      for (std::size_t j = 0; j < N; ++j) {
        for (std::size_t k = 0; k < K; ++k) {
          expected[i, j] += A[i, k] * B[k, j];
        }
      }
    }

    REQUIRE(venus::ops::equal(C, expected));
  }

  SECTION("Where - Condition Only") {
    auto x = Tensor<float, Device::CPU, 2>(3, 2);
    auto y = Tensor<float, Device::CPU, 2>(3, 2);

    x.iota(1);
    y.fill(1);

    auto z = venus::ops::where(x > 3);

    STATIC_REQUIRE(std::is_same_v<decltype(z)::ElementType, std::size_t>);
    for (std::size_t i = 0; i < z.shape().count(); ++i) {
      if (x.lowLevel().rawMemory()[i] > 3) {
        REQUIRE(z.lowLevel().rawMemory()[i] == i);
      } else {
        REQUIRE(z.lowLevel().rawMemory()[i] == 0);
      }
    }
  }

  SECTION("Where - Ternary") {
    auto x = Tensor<float, Device::CPU, 2>(3, 2);
    auto y = Tensor<float, Device::CPU, 2>(3, 2);

    x.iota(1);
    y.fill(1);

    auto z = venus::ops::where(x > 3, x, y);

    STATIC_REQUIRE(std::is_same_v<decltype(z)::ElementType, float>);
    for (std::size_t i = 0; i < z.shape().count(); ++i) {
      if (x.lowLevel().rawMemory()[i] > 3) {
        REQUIRE(z.lowLevel().rawMemory()[i] == x.lowLevel().rawMemory()[i]);
      } else {
        REQUIRE(z.lowLevel().rawMemory()[i] == y.lowLevel().rawMemory()[i]);
      }
    }
  }
}