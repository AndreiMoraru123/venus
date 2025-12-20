
#include <algorithm>
#include <cassert>
#include <print>
#include <venus/memory/device.hpp>
#include <venus/tensor/tensor.hpp>

using namespace venus;

// clang-format off
auto main() -> int {
  {
    auto tensor = Tensor<float, Device::CPU, 2>(2, 3);
    assert(tensor.HasUniqueMemory());

    std::ranges::fill(tensor, 10.0f);
    std::println("{}", tensor); // venus::Tensor([10.00, 10.00, 10.00, 10.00, 10.00, 10.00], shape=(2, 3))

    auto tensor2 = Tensor<float, Device::CPU, 2>(2, 3);
    assert(tensor2.HasUniqueMemory());

    std::println("{}", tensor); // venus::Tensor([0.00, 0.00, 0.00, 0.00, 0.00, 0.00], shape=(2, 3))
  }
  {
    const auto memo = ContiguousMemory<float, Device::CPU>(6);

    auto tensor = Tensor<float, Device::CPU, 2>(memo, Shape(2, 3));
    assert(not tensor.HasUniqueMemory());

    std::ranges::fill(tensor, 5.0f);
    std::println("{}", tensor); // venus::Tensor([5.00, 5.00, 5.00, 5.00, 5.00, 5.00], shape=(2, 3))

    auto tensor2 = Tensor<float, Device::CPU, 2>(memo, Shape(2, 3));
    assert(not tensor2.HasUniqueMemory());

    std::println("{}", tensor2); // venus::Tensor([5.00, 5.00, 5.00, 5.00, 5.00, 5.00], shape=(2, 3))
  }
}