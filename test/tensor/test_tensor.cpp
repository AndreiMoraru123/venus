
#include "core/memory/contiguous_memory.hpp"
#include "core/memory/device.hpp"
#include "core/memory/lower_access.hpp"
#include <catch2/catch_test_macros.hpp>

#include <core/tensor/tensor.hpp>

using namespace venus;

TEST_CASE("Tensor Ops", "[tensor]") {

  SECTION("Scalar Tensor") {
    auto scalar = Tensor<float, Device::CPU, 0>();
    REQUIRE(scalar.Value() == 0.0f);
    REQUIRE(scalar.AvailableForWrite());

    scalar.SetValue(10.0f);
    REQUIRE(scalar.Value() == 10.0f);
  }

  SECTION("Shared Memory") {
    const auto shared_memo = ContiguousMemory<float, Device::CPU>(1);

    float *rawPtr = shared_memo.RawMemory();
    rawPtr[0] = 1.0f;

    const auto scalar1 = Tensor<float, Device::CPU, 0>(shared_memo);
    const auto scalar2 = Tensor<float, Device::CPU, 0>(shared_memo);

    REQUIRE(scalar1.Value() == 1.0f);
    REQUIRE(scalar2.Value() == 1.0f);

    REQUIRE_FALSE(scalar1.AvailableForWrite());
    REQUIRE_FALSE(scalar2.AvailableForWrite());
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

    REQUIRE_FALSE(scalar1.AvailableForWrite());
    REQUIRE_FALSE(scalar2.AvailableForWrite());
  }

  SECTION("Low Level Access") {
    const auto memo = ContiguousMemory<float, Device::CPU>(1);
    float *rawPtr = memo.RawMemory();
    rawPtr[0] = 10.0f;

    const auto tensor = Tensor<float, Device::CPU, 0>(memo);
    REQUIRE_FALSE(tensor.AvailableForWrite()); // internal copy of memo

    const auto lowLevelTensor = LowLevel(std::move(tensor));
    REQUIRE(lowLevelTensor.SharedMemory() == memo);
    REQUIRE(*lowLevelTensor.RawMemory() == 10.0f);
  }
}