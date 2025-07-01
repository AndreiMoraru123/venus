#include "core/tensor/tensor.hpp"
#include <print>

using namespace venus;

auto main() -> int {
  {
    auto tensor = Tensor<int, Device::CPU, 2>(3, 3);
    std::ranges::iota(tensor, 1); // 1, 2, 3, 4, 5, 6, 7, 8, 9
    for (int el : tensor) {
      std::print("{}, ", el);
    }
  }

  std::print("\n");

  {
    const auto tensor2 = Tensor<int, Device::CPU, 2>(3, 3);
    for (auto el : tensor2) {
      std::print("{}, ", el); // 0, 0, 0, 0, 0, 0, 0, 0, 0
    }
  }
}