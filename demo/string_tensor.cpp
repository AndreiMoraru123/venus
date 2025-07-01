#include <print>
#include <string>

#include "core/tensor/tensor.hpp"
#include <cassert>

using namespace venus;

auto main() -> int {
  auto tensor = Tensor<std::string, Device::CPU, 1>(3);
  std::ranges::fill(tensor, "hello");

  for (auto el : tensor) {
    std::print("{} ", static_cast<std::string>(el)); // hello hello hello
  }
}