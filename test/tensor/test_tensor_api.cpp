
#include <cassert>
#include <catch2/catch_test_macros.hpp>
#include <venus/memory/contiguous_memory.hpp>
#include <venus/memory/device.hpp>

#include <cmath>
#include <stdexcept>
#include <venus/tensor/tensor.hpp>

using namespace venus;

TEST_CASE("Tensor API", "[tensor][api]") {

  SECTION("Scalar Tensor") {
    auto scalar = Tensor<float, Device::CPU, 0>(10.0f);
    REQUIRE(scalar.value() == 10.0f);
    REQUIRE(bool(scalar) == true);
    REQUIRE(scalar.unique());

    scalar.setValue(100.0f);
    REQUIRE(scalar.value() == 100.0f);

    scalar.setValue(0.0f);
    REQUIRE(bool(scalar) == false);
  }

  SECTION("Scalar Boolean Tensor") {
    auto scalar = Tensor<bool, Device::CPU, 0>(true);
    REQUIRE(scalar.value() == true);
    REQUIRE(scalar == true);

    scalar.setValue(false);
    REQUIRE(scalar == false);
  }

  SECTION("Shared Memory") {
    auto shared_memo = ContiguousMemory<float, Device::CPU>(1);

    float *rawPtr = shared_memo.ptr();
    rawPtr[0] = 1.0f;

    const auto scalar1 = Tensor<float, Device::CPU, 0>(shared_memo);
    const auto scalar2 = Tensor<float, Device::CPU, 0>(shared_memo);

    REQUIRE(scalar1.value() == 1.0f);
    REQUIRE(scalar2.value() == 1.0f);

    REQUIRE_FALSE(scalar1.unique());
    REQUIRE_FALSE(scalar2.unique());
  }

  SECTION("Shared Shifted Memory") {
    auto shared_memo = ContiguousMemory<float, Device::CPU>(3);

    float *rawPtr = shared_memo.ptr();
    rawPtr[0] = 1.0f;
    rawPtr[2] = 3.0f;

    const auto scalar1 = Tensor<float, Device::CPU, 0>(shared_memo);
    const auto scalar2 = Tensor<float, Device::CPU, 0>(shared_memo.shift(2));

    REQUIRE(scalar1.value() == 1.0f);
    REQUIRE(scalar2.value() == 3.0f);

    REQUIRE_FALSE(scalar1.unique());
    REQUIRE_FALSE(scalar2.unique());
  }

  SECTION("Low Level Access") {
    auto tensor = Tensor<float, Device::CPU, 0>(10.0f);
    REQUIRE(tensor.unique());

    auto lowLevelTensor = tensor.lowLevel();
    REQUIRE(*lowLevelTensor.rawMemory() == 10.0f);
    REQUIRE(tensor.unique()); // non-owning access

    *lowLevelTensor.rawMemory() = 20.0f; // mutating is allowed
    REQUIRE(tensor.value() == 20.0f);
  }

  SECTION("Low Level Access Memory") {
    auto memo = ContiguousMemory<float, Device::CPU>(1);
    float *rawPtr = memo.ptr();
    rawPtr[0] = 10.0f;

    const auto tensor = Tensor<float, Device::CPU, 0>(memo);
    REQUIRE_FALSE(tensor.unique()); // internal copy of memo

    const auto lowLevelTensor = tensor.lowLevel();
    REQUIRE(lowLevelTensor.sharedMemory() == memo);
    REQUIRE(*lowLevelTensor.rawMemory() == 10.0f);
  }

  SECTION("Attempty to Build Tensor From Insufficient Memory") {
    constexpr auto rank = 3;
    constexpr auto shape = Shape<rank>(3, 2, 2);

    const auto memo = ContiguousMemory<float, Device::CPU>(10);
    REQUIRE_THROWS_AS((Tensor<float, Device::CPU, rank>(memo, shape)),
                      std::invalid_argument);
  }

  SECTION("Build Tensor From Memory and Shape") {
    constexpr auto rank = 3;
    constexpr auto shape = Shape<rank>(3, 2, 2);

    /** NOTE: shallow (bitwise) constness (can't point to different memory, but
     * has no control/restrictions over the thing it points to), so the tensor
     elements can be modified, so long as the tensor itself is not const
     */
    const auto memo = ContiguousMemory<float, Device::CPU>(12);

    const auto tensor = Tensor<float, Device::CPU, rank>(memo, shape);
    REQUIRE_FALSE(tensor.unique());
    REQUIRE(tensor.shape() == shape);
  }

  SECTION("Build Tensor From Shape") {
    constexpr auto rank = 3;
    constexpr auto shape = Shape<rank>(3, 2, 2);

    // memory layout is deduced automatically
    const auto tensor = Tensor<float, Device::CPU, rank>(shape);
    REQUIRE(tensor.unique());
    REQUIRE(tensor.shape() == shape);
  }

  SECTION("Tensor Indexing") {
    constexpr auto rank = 3;
    constexpr auto shape = Shape<rank>(3, 2, 2);

    auto memo = ContiguousMemory<float, Device::CPU>(12);
    auto tensor = Tensor<float, Device::CPU, rank>(std::move(memo), shape);
    REQUIRE(tensor.unique());

    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 2; ++j) {
        for (int k = 0; k < 2; ++k) {
          tensor[i, j, k] = i * 100 + j * 10 + k;
        }
      }
    }

    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 2; ++j) {
        for (int k = 0; k < 2; ++k) {
          tensor[i, j, k]++;
          tensor[i, j, k]--;
          ++tensor[i, j, k];
          --tensor[i, j, k];
        }
      }
    }

    REQUIRE(tensor[0, 0, 0] == 0);
    REQUIRE(tensor[0, 0, 1] == 1);
    REQUIRE(tensor[2, 0, 0] == 200);
    REQUIRE(tensor[2, 1, 1] == 211);

    REQUIRE_THROWS_AS((tensor[3, 0, 0]), std::out_of_range);
    REQUIRE_THROWS_AS((tensor[0, 2, 0]), std::out_of_range);
    REQUIRE_THROWS_AS((tensor[0, 0, 2]), std::out_of_range);
  }

  SECTION("Attempting To Write To Shared Tensor") {
    constexpr auto rank = 3;
    constexpr auto shape = Shape<rank>(3, 2, 2);

    auto memo = ContiguousMemory<float, Device::CPU>(12);
    std::fill_n(memo.ptr(), 12, 0.0f);

    auto tensor = Tensor<float, Device::CPU, rank>(memo, shape);
    REQUIRE_FALSE(tensor.unique());

    // writing is forbidden
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 2; ++j) {
        for (int k = 0; k < 2; ++k) {
          REQUIRE_THROWS_AS((tensor[i, j, k] += 1.0f), std::runtime_error);
        }
      }
    }

    // reading is allowed
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 2; ++j) {
        for (int k = 0; k < 2; ++k) {
          REQUIRE(tensor[i, j, k] == 0.0f);
        }
      }
    }
  }

  SECTION("Shared Memory Indexing") {
    constexpr auto rank = 3;
    constexpr auto shape = Shape<rank>(3, 2, 2);

    auto memo = ContiguousMemory<float, Device::CPU>(12);
    std::fill_n(memo.ptr(), 12, 0.0f);

    auto tensor1 = Tensor<float, Device::CPU, rank>(memo, shape);
    auto tensor2 = Tensor<float, Device::CPU, rank>(memo, shape);

    REQUIRE_FALSE(tensor1.unique());
    REQUIRE_FALSE(tensor2.unique());

    REQUIRE(tensor1[0, 0, 0] == 0.0f);
    REQUIRE(tensor2[0, 0, 0] == 0.0f);
  }

  SECTION("Tensor To Scalar") {
    auto x = Tensor<float, Device::CPU, 1>(1);
    auto y = Tensor<float, Device::CPU, 1>(2);

    x.iota(1);
    y.iota(1);

    REQUIRE_THROWS_AS(y.toScalar(), std::runtime_error);

    auto scalar = x.toScalar();

    REQUIRE(x.unique());
    REQUIRE(scalar.unique());
    REQUIRE(scalar.value() == x[0]);
  }

  SECTION("Tensor From Initializer List") {
    auto tensor = Tensor<int, Device::CPU, 1>({1, 2, 3, 4, 5});

    REQUIRE(tensor.shape() == Shape<1>(5));
    REQUIRE(tensor[0] == 1);
    REQUIRE(tensor[1] == 2);
    REQUIRE(tensor[2] == 3);
    REQUIRE(tensor[3] == 4);
    REQUIRE(tensor[4] == 5);
  }

  SECTION("2D Tensor From Nested Initializer List") {
    auto tensor = Tensor<int, Device::CPU, 2>{{1, 2, 3}, {4, 5, 6}};

    REQUIRE(tensor.shape() == Shape<2>(2, 3));
    REQUIRE(tensor[0, 0] == 1);
    REQUIRE(tensor[0, 1] == 2);
    REQUIRE(tensor[0, 2] == 3);
    REQUIRE(tensor[1, 0] == 4);
    REQUIRE(tensor[1, 1] == 5);
    REQUIRE(tensor[1, 2] == 6);
  }

  SECTION("2D Tensor From Uneven Nested Initializer List") {
    REQUIRE_THROWS_AS((Tensor<int, Device::CPU, 2>{{1, 2}, {4, 5, 6}}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS((Tensor<int, Device::CPU, 2>{{1, 2, 3}, {4, 5}}),
                      std::invalid_argument);
  }

  SECTION("3D Tensor From Nested Initializer List") {
    auto tensor = Tensor<int, Device::CPU, 3>{{{1, 2, 3}, {4, 5, 6}},
                                              {{7, 8, 9}, {10, 11, 12}}};

    REQUIRE(tensor.shape() == Shape<3>(2, 2, 3));
    REQUIRE(tensor[0, 0, 0] == 1);
    REQUIRE(tensor[0, 0, 1] == 2);
    REQUIRE(tensor[0, 0, 2] == 3);
    REQUIRE(tensor[0, 1, 0] == 4);
    REQUIRE(tensor[0, 1, 1] == 5);
    REQUIRE(tensor[0, 1, 2] == 6);
    REQUIRE(tensor[1, 0, 0] == 7);
    REQUIRE(tensor[1, 0, 1] == 8);
    REQUIRE(tensor[1, 0, 2] == 9);
    REQUIRE(tensor[1, 1, 0] == 10);
    REQUIRE(tensor[1, 1, 1] == 11);
    REQUIRE(tensor[1, 1, 2] == 12);
  }

  SECTION("3D Tensor From Uneven Nested Initializer List") {
    REQUIRE_THROWS_AS((Tensor<int, Device::CPU, 3>{{{1, 2}, {4, 5, 6}},
                                                   {{7, 8, 9}, {10, 11, 12}}}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS((Tensor<int, Device::CPU, 3>{{{1, 2, 3}, {4, 5}},
                                                   {{7, 8, 9}, {10, 11, 12}}}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS((Tensor<int, Device::CPU, 3>{{{1, 2, 3}, {4, 5, 6}},
                                                   {{7, 8}, {10, 11, 12}}}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS((Tensor<int, Device::CPU, 3>{{{1, 2, 3}, {4, 5, 6}},
                                                   {{7, 8, 9}, {10, 11}}}),
                      std::invalid_argument);
  }

  SECTION("Tensor Reshape") {
    auto tensor_2d = Tensor<float, Device::CPU, 2>(2, 6);
    tensor_2d.iota(1);

    auto tensor_1d = tensor_2d.reshape<1>(Shape<1>{12});
    REQUIRE(tensor_1d.shape() == Shape<1>{12});
    for (std::size_t i{}; i < 12; i++) {
      REQUIRE(tensor_1d[i] == tensor_2d.lowLevel().rawMemory()[i]);
    }

    auto tensor_3d = tensor_2d.reshape<3>(Shape<3>{2, 2, 3});
    REQUIRE(tensor_3d.shape() == Shape<3>{2, 2, 3});
    for (std::size_t i{}; i < 2; i++) {
      for (std::size_t j{}; j < 2; j++) {
        for (std::size_t k{}; k < 3; k++) {
          REQUIRE(tensor_3d[i, j, k] == tensor_2d[i, j * 3 + k]);
        }
      }
    }
  }
}

TEST_CASE("Tensor Creation", "[tensor][ctor]") {

  SECTION("Copy Ctor") {
    auto x = Tensor<float, Device::CPU, 2>(2, 3);
    x.iota(1);
    void *x_ptr = static_cast<void *>(x.data());

    auto y(x);
    REQUIRE(y.unique());
    REQUIRE(static_cast<void *>(y.data()) != x_ptr); // new memory is allocated
    REQUIRE(y.shape() == x.shape());
  }

  SECTION("Move Ctor") {
    auto x = Tensor<float, Device::CPU, 2>(2, 3);
    x.iota(1);
    void *x_ptr = static_cast<void *>(x.data());

    auto y(std::move(x));
    REQUIRE(y.unique());
    REQUIRE(static_cast<void *>(y.data()) == x_ptr); // no new memory allocation
    REQUIRE(y.shape() == x.shape());
  }

  SECTION("Copy Assign (Same Size, Reuses Memory)") {
    auto x = Tensor<float, Device::CPU, 2>(2, 3);
    x.iota(1);

    auto y = Tensor<float, Device::CPU, 2>(2, 3);
    void *y_ptr_before = static_cast<void *>(y.data());

    y = x;
    REQUIRE(y.unique());
    // sizes match exactly, y should reuse its existing memory
    REQUIRE(static_cast<void *>(y.data()) == y_ptr_before);
    REQUIRE(static_cast<void *>(y.data()) != static_cast<void *>(x.data()));
  }

  SECTION("Copy Assign (Different Size, Reallocates Memory)") {
    auto x = Tensor<float, Device::CPU, 2>(10, 10);
    x.iota(1);

    auto y = Tensor<float, Device::CPU, 2>(2, 3);
    void *y_ptr_before = static_cast<void *>(y.data());

    y = x;
    REQUIRE(y.unique());
    // sizes do not match, y should reallocate
    REQUIRE(static_cast<void *>(y.data()) != y_ptr_before);
    REQUIRE(static_cast<void *>(y.data()) != static_cast<void *>(x.data()));
    REQUIRE(y.shape() == Shape(10, 10));
  }

  SECTION("Copy Assign (Shared Memory, Detaches and Reallocates)") {
    auto x = Tensor<float, Device::CPU, 2>(10, 10);
    x.iota(1);

    auto y = Tensor<float, Device::CPU, 2>(2, 3);
    void *y_ptr_before = static_cast<void *>(y.data());

    auto y_view = y.view();

    REQUIRE_FALSE(y.unique());
    REQUIRE_FALSE(y_view.unique());

    // y is not unique, so it must allocate memory, even though sizes match
    y = x;

    // y should have detached and allocated new memory
    REQUIRE(y.unique());
    REQUIRE(static_cast<void *>(y.data()) != y_ptr_before);
    REQUIRE(static_cast<void *>(y.data()) != static_cast<void *>(x.data()));
    REQUIRE(y.shape() == Shape(10, 10));

    // y_view should now be the sole owner of the original memory
    REQUIRE(y_view.unique());
    REQUIRE(static_cast<void *>(y_view.data()) == y_ptr_before);
    REQUIRE(y_view.shape() == Shape(2, 3));
  }

  SECTION("Move Assign") {
    auto x = Tensor<float, Device::CPU, 2>(2, 3);
    void *x_ptr = static_cast<void *>(x.data());
    x.iota(1);

    auto y = Tensor<float, Device::CPU, 2>(5, 5);

    y = std::move(x);
    REQUIRE(y.unique());
    // z takes x's memory diectly
    REQUIRE(static_cast<void *>(y.data()) == x_ptr);
    REQUIRE(y.shape() == Shape(2, 3));
  }
}