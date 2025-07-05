
#include "core/tensor/tensor.hpp"
#include <cassert>
#include <print>

using namespace venus;

auto main() -> int {
  auto scalar = Tensor<int, Device::CPU, 0>(10.0f);
  auto tensor = Tensor<int, Device::CPU, 3>(1, 2, 3);

  std::println("{}", scalar); // venus::Tensor(10)
  std::println("{}",
               tensor); // venus::Tensor([0, 0, 0, 0, 0, 0], shape=(1, 2, 3))
}