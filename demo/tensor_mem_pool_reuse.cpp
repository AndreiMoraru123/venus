#include "core/tensor/tensor.hpp"
#include <print>

using namespace venus;

auto main() -> int {
  {
    auto tensor = Tensor<int, Device::CPU, 2>(3, 3);
    std::ranges::iota(tensor, 1); // 1, 2, 3, 4, 5, 6, 7, 8, 9
    for (auto el : tensor) {
      std::print("{} ", static_cast<int>(el));
    }
  }

  std::print("\n");

  {
    auto tensor2 = Tensor<int, Device::CPU, 2>(3, 3);
    for (auto el : tensor2) {
      std::print("{} ", static_cast<int>(el)); // these should all be 0s
    }
  }
}