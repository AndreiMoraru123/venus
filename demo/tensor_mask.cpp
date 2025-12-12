#include "core/tensor/tensor.hpp"
#include <print>

using namespace venus;

auto main() -> int {
  auto x = Tensor<float, Device::CPU, 2>(3, 2);
  auto y = Tensor<float, Device::CPU, 2>(3, 2);

  venus::ops::iota(x, 1);
  venus::ops::fill(y, 1);
  std::println("{}", x);
  std::println("{}", y);

  auto condition = x > 3;
  std::println("{}", condition);

  auto z = venus::ops::where(x > 3);
  std::println("{}", z);
}