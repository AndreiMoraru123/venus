#include "catch2/catch_template_test_macros.hpp"
#include <cassert>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <functional>
#include <venus/memory/device.hpp>

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
    auto res = venus::eager::transform(tensor, [](auto &&t) { return t * 3; });

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

    auto C = venus::eager::mm(A, B);

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

    REQUIRE(venus::eager::equal(C, expected));
  }

  SECTION("Where - Condition Only") {
    auto x = Tensor<float, Device::CPU, 2>(3, 2);
    auto y = Tensor<float, Device::CPU, 2>(3, 2);

    x.iota(1);
    y.fill(1);

    auto z = venus::eager::where(x > 3);

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

    auto z = venus::eager::where(x > 3, x, y);

    STATIC_REQUIRE(std::is_same_v<decltype(z)::ElementType, float>);
    for (std::size_t i = 0; i < z.shape().count(); ++i) {
      if (x.lowLevel().rawMemory()[i] > 3) {
        REQUIRE(z.lowLevel().rawMemory()[i] == x.lowLevel().rawMemory()[i]);
      } else {
        REQUIRE(z.lowLevel().rawMemory()[i] == y.lowLevel().rawMemory()[i]);
      }
    }
  }

  SECTION("Broadcasting") {
    // clang-format off
    auto a = Tensor<int, Device::CPU, 2>{{
        {0,  1,  2,  3},
        {4,  5,  6,  7},
        {8,  9,  10, 11},
        {12, 13, 14, 15}}};

    auto b_row = Tensor<int, Device::CPU, 2>{{4, 3, 2, 1}};
    auto b_col = Tensor<int, Device::CPU, 2>{
      {4},
      {3},
      {2},
      {1}};

    auto c = a + b_row;
    auto d = a + b_col;

    REQUIRE(venus::eager::equal(c, b_row + a));
    REQUIRE(venus::eager::equal(d, b_col + a));

    REQUIRE(c.shape() == a.shape());
    REQUIRE(d.shape() == a.shape());

    auto b_row_bc = Tensor<int, Device::CPU, 2>{{
      {4, 3, 2, 1},
      {4, 3, 2, 1},
      {4, 3, 2, 1},
      {4, 3, 2, 1}
    }};
    auto b_col_bc = Tensor<int, Device::CPU, 2>{{
      {4, 4, 4, 4},
      {3, 3, 3, 3},
      {2, 2, 2, 2},
      {1, 1, 1, 1}
    }};
    // clang-format on

    for (std::size_t i = 0; i < a.shape().count(); i++) {
      REQUIRE(c.lowLevel().rawMemory()[i] ==
              a.lowLevel().rawMemory()[i] + b_row_bc.lowLevel().rawMemory()[i]);
      REQUIRE(d.lowLevel().rawMemory()[i] ==
              a.lowLevel().rawMemory()[i] + b_col_bc.lowLevel().rawMemory()[i]);
    }
  }
}

TEMPLATE_TEST_CASE_SIG("Sum across a single dimension",
                       "[tensor][ops][sum_dim]",
                       ((std::size_t target_dim), target_dim), 0, 1, 2) {
  auto tensor = Tensor<int, Device::CPU, 3>(2, 3, 4);
  tensor.iota(1);

  auto result = venus::eager::sum_dim<target_dim>(tensor);
  STATIC_REQUIRE(tensor.rank == result.rank);

  auto [M, N, K] = tensor.shape();
  auto expected_shape = Shape(target_dim == 0 ? 1 : M, target_dim == 1 ? 1 : N,
                              target_dim == 2 ? 1 : K);
  REQUIRE(result.shape() == expected_shape);
  auto expected = Tensor<int, Device::CPU, 3>(expected_shape);

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      for (std::size_t k = 0; k < K; ++k) {
        std::size_t ei = (target_dim == 0) ? 0 : i;
        std::size_t ej = (target_dim == 1) ? 0 : j;
        std::size_t ek = (target_dim == 2) ? 0 : k;
        expected[ei, ej, ek] += tensor[i, j, k];
      }
    }
  }

  REQUIRE(venus::eager::equal(result, expected));
}

TEST_CASE("Sum across multiple dimensions", "[tensor][ops][sum_dims]") {
  auto tensor = Tensor<int, Device::CPU, 3>(2, 3, 4);
  tensor.iota(1);

  SECTION("Composition: sum_dims<0, 1> == sum_dim<0> + sum_dim<1>") {
    auto multi_sum = venus::eager::sum_dims<0, 1>(tensor);
    auto seq_sum = venus::eager::sum_dim<0>(venus::eager::sum_dim<1>(tensor));

    REQUIRE(multi_sum.shape() == Shape(1, 1, 4));
    REQUIRE(venus::eager::equal(multi_sum, seq_sum));
  }

  SECTION("Order invariance: sum_dims<0, 2> == sum_dims<2, 0>") {
    auto sum_02 = venus::eager::sum_dims<0, 2>(tensor);
    auto sum_20 = venus::eager::sum_dims<2, 0>(tensor);

    REQUIRE(sum_02.shape() == Shape(1, 3, 1));
    REQUIRE(venus::eager::equal(sum_02, sum_20));
  }

  SECTION("Total reduction") {
    auto total = venus::eager::sum_dims<0, 1, 2>(tensor);

    REQUIRE(total.shape() == Shape(1, 1, 1));
    REQUIRE(total.toScalar() == 300);
  }
}

TEST_CASE("Einsum", "[tensor][ops][einsum]") {

  SECTION("Vector Inner (Dot) Product (i,i->)") {
    auto a = Tensor<int, Device::CPU, 1>(3);
    a.iota(1);

    auto b = Tensor<int, Device::CPU, 1>(3);
    b.iota(1);

    auto result = venus::eager::einsum<"i,i->">(a, b);
    REQUIRE(result == 1 * 1 + 2 * 2 + 3 * 3);
  }

  SECTION("Vector Outer Product (i,j->)") {
    auto a = Tensor<int, Device::CPU, 1>(3);
    a.iota(1);

    auto b = Tensor<int, Device::CPU, 1>(3);
    b.iota(1);

    auto result = venus::eager::einsum<"i,j->">(a, b);
    REQUIRE(result == std::ranges::fold_left(a, 0, std::plus{}) *
                          std::ranges::fold_left(b, 0, std::plus{}));
  }

  SECTION("Vector Hadamard (Element-Wise) Product (i,j->)") {
    auto a = Tensor<int, Device::CPU, 1>(3);
    a.iota(1);

    auto b = Tensor<int, Device::CPU, 1>(3);
    b.iota(1);

    auto result = venus::eager::einsum<"i,i->i">(a, b);
    REQUIRE(result[0] == 1 * 1);
    REQUIRE(result[1] == 2 * 2);
    REQUIRE(result[2] == 3 * 3);
  }

  SECTION("Batched Matrix Multiplication (bij,bjk->bik)") {
    auto a = Tensor<int, Device::CPU, 3>(2, 2, 2);
    a.iota(1); // Batch 0: [[1, 2], [3, 4]]
               // Batch 1: [[5, 6], [7, 8]]

    auto b = venus::eager::eye_like(a);

    // Identity
    auto result = venus::eager::einsum<"bij,bjk->bik">(a, b);
    REQUIRE(result.shape() == Shape(2, 2, 2));
    REQUIRE(venus::eager::equal(result, a));
  }

  SECTION("Implicit Mode (ij,jk)") {
    auto a = Tensor<int, Device::CPU, 2>(2, 3);
    a.iota(1);
    auto b = Tensor<int, Device::CPU, 2>(3, 2);
    b.iota(1);

    // Because 'i' and 'k' appear once, implicit mode sorts them alphabetically
    // to "ik".
    auto result_implicit = venus::eager::einsum<"ij,jk">(a, b);
    auto result_explicit = venus::eager::einsum<"ij,jk->ik">(a, b);

    REQUIRE(result_implicit.shape() == Shape(2, 2));
    REQUIRE(venus::eager::equal(result_implicit, result_explicit));
  }

  // --- Diagonal not implemented for now ---
  // SECTION("Matrix Trace (ii->)") {
  //  auto matrix = Tensor<int, Device::CPU, 2>(3, 3);
  //  matrix.iota(1);
  //
  //  auto result = venus::eager::einsum<"ii->">(matrix);
  //  REQUIRE(result == 1 + 5 + 9);
  //}
  //
  // SECTION("Matrix Diagonal Extraction (ii->i)") {
  //  auto matrix = Tensor<int, Device::CPU, 2>(3, 3);
  //  matrix.iota(1);
  //
  //  auto result = venus::eager::einsum<"ii->i">(matrix);
  //  REQUIRE(result.shape() == Shape(3));
  //  REQUIRE(result[0] == 1);
  //  REQUIRE(result[1] == 5);
  //  REQUIRE(result[2] == 9);
  //}
}