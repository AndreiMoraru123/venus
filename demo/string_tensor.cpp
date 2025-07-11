#include <print>
#include <string>

#include "core/tensor/tensor.hpp"

using namespace venus;

auto main() -> int {
  auto tensor = Tensor<std::string, Device::CPU, 1>(3);
  std::ranges::fill(tensor, "hello");

  // clang-format off
  std::println("{}", tensor); // venus::Tensor(["hello", "hello", "hello"], shape=(3))

  for (auto el : tensor) {
    std::print("{} ", el); // hello hello hello
  }
}