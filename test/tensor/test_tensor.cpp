
#include "core/memory/contiguous_memory.hpp"
#include "core/memory/device.hpp"
#include <cassert>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <core/tensor/tensor.hpp>
#include <stdexcept>

using namespace venus;

TEST_CASE("Tensor Ops", "[tensor]") {

  SECTION("Scalar Tensor") {
    auto scalar = Tensor<float, Device::CPU, 0>(10.0f);
    REQUIRE(scalar.Value() == 10.0f);
    REQUIRE(scalar.HasUniqueMemory());

    scalar.SetValue(100.0f);
    REQUIRE(scalar.Value() == 100.0f);
  }

  SECTION("Shared Memory") {
    const auto shared_memo = ContiguousMemory<float, Device::CPU>(1);

    float *rawPtr = shared_memo.RawMemory();
    rawPtr[0] = 1.0f;

    const auto scalar1 = Tensor<float, Device::CPU, 0>(shared_memo);
    const auto scalar2 = Tensor<float, Device::CPU, 0>(shared_memo);

    REQUIRE(scalar1.Value() == 1.0f);
    REQUIRE(scalar2.Value() == 1.0f);

    REQUIRE_FALSE(scalar1.HasUniqueMemory());
    REQUIRE_FALSE(scalar2.HasUniqueMemory());
  }

  SECTION("Shared Shifted Memory") {
    const auto shared_memo = ContiguousMemory<float, Device::CPU>(3);

    float *rawPtr = shared_memo.RawMemory();
    rawPtr[0] = 1.0f;
    rawPtr[2] = 3.0f;

    const auto scalar1 = Tensor<float, Device::CPU, 0>(shared_memo);
    const auto scalar2 = Tensor<float, Device::CPU, 0>(shared_memo.Shift(2));

    REQUIRE(scalar1.Value() == 1.0f);
    REQUIRE(scalar2.Value() == 3.0f);

    REQUIRE_FALSE(scalar1.HasUniqueMemory());
    REQUIRE_FALSE(scalar2.HasUniqueMemory());
  }

  SECTION("Low Level Access") {
    const auto tensor = Tensor<float, Device::CPU, 0>(10.0f);
    REQUIRE(tensor.HasUniqueMemory());

    const auto lowLevelTensor = tensor.LowLevel();
    REQUIRE(*lowLevelTensor.RawMemory() == 10.0f);
    REQUIRE_FALSE(tensor.HasUniqueMemory());
  }

  SECTION("Low Level Access Discard") {
    const auto tensor = Tensor<float, Device::CPU, 0>(10.0f);
    REQUIRE(tensor.HasUniqueMemory());

    (void)tensor.LowLevel(); // discard
    REQUIRE(tensor.HasUniqueMemory());
  }

  SECTION("Low Level Access Memory") {
    const auto memo = ContiguousMemory<float, Device::CPU>(1);
    float *rawPtr = memo.RawMemory();
    rawPtr[0] = 10.0f;

    const auto tensor = Tensor<float, Device::CPU, 0>(memo);
    REQUIRE_FALSE(tensor.HasUniqueMemory()); // internal copy of memo

    const auto lowLevelTensor = tensor.LowLevel();
    REQUIRE(lowLevelTensor.SharedMemory() == memo);
    REQUIRE(*lowLevelTensor.RawMemory() == 10.0f);
  }

  SECTION("Attempty to Build Tensor From Insufficient Memory") {
    constexpr auto NUM_DIMS = 3;
    constexpr auto shape = Shape<NUM_DIMS>(3, 2, 2);

    const auto memo = ContiguousMemory<float, Device::CPU>(10);
    REQUIRE_THROWS_AS((Tensor<float, Device::CPU, NUM_DIMS>(memo, shape)),
                      std::invalid_argument);
  }

  SECTION("Build Tensor From Memory and Shape") {
    constexpr auto NUM_DIMS = 3;
    constexpr auto shape = Shape<NUM_DIMS>(3, 2, 2);

    /** NOTE: shallow (bitwise) constness (can't point to different memory, but
     * has no control/restrictions over the thing it points to), so the tensor
     elements can be modified, so long as the tensor itself is not const
     */
    const auto memo = ContiguousMemory<float, Device::CPU>(12);

    const auto tensor = Tensor<float, Device::CPU, NUM_DIMS>(memo, shape);
    REQUIRE_FALSE(tensor.HasUniqueMemory());
    REQUIRE(tensor.Shape() == shape);
  }

  SECTION("Build Tensor From Shape") {
    constexpr auto NUM_DIMS = 3;
    constexpr auto shape = Shape<NUM_DIMS>(3, 2, 2);

    // memory layout is deduced automatically
    const auto tensor = Tensor<float, Device::CPU, NUM_DIMS>(shape);
    REQUIRE(tensor.HasUniqueMemory());
    REQUIRE(tensor.Shape() == shape);
  }

  SECTION("Tensor Indexing") {
    constexpr auto NUM_DIMS = 3;
    constexpr auto shape = Shape<NUM_DIMS>(3, 2, 2);

    auto memo = ContiguousMemory<float, Device::CPU>(12);
    auto tensor = Tensor<float, Device::CPU, NUM_DIMS>(std::move(memo), shape);
    REQUIRE(tensor.HasUniqueMemory());

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
    constexpr auto NUM_DIMS = 3;
    constexpr auto shape = Shape<NUM_DIMS>(3, 2, 2);

    auto memo = ContiguousMemory<float, Device::CPU>(12);
    std::fill_n(memo.RawMemory(), 12, 0.0f);

    auto tensor = Tensor<float, Device::CPU, NUM_DIMS>(memo, shape);
    REQUIRE_FALSE(tensor.HasUniqueMemory());

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
    constexpr auto NUM_DIMS = 3;
    constexpr auto shape = Shape<NUM_DIMS>(3, 2, 2);

    auto memo = ContiguousMemory<float, Device::CPU>(12);
    std::fill_n(memo.RawMemory(), 12, 0.0f);

    auto tensor1 = Tensor<float, Device::CPU, NUM_DIMS>(memo, shape);
    auto tensor2 = Tensor<float, Device::CPU, NUM_DIMS>(memo, shape);

    REQUIRE_FALSE(tensor1.HasUniqueMemory());
    REQUIRE_FALSE(tensor2.HasUniqueMemory());

    REQUIRE(tensor1[0, 0, 0] == 0.0f);
    REQUIRE(tensor2[0, 0, 0] == 0.0f);
  }
}