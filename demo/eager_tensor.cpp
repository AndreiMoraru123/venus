
#include "core/tensor/tensor.hpp"
#include <cassert>
#include <print>

using namespace venus;

// clang-format off
auto main() -> int {
  auto x = Tensor<float, Device::CPU, 3>(1, 2, 3); // heap alloc 1x2x3 Tensor
  auto y = Tensor<float, Device::CPU, 3>(1, 2, 3); // heap alloc 1x2x3 Tensor

  std::ranges::iota(x, 1); // 1, 2, 3, 4, 5, 6
  std::ranges::iota(y, 2); // 2, 3, 4, 5, 6, 7

  std::println("{}", x + y); // venus::Tensor([3.00, 5.00, 7.00, 9.00, 11.00, 13.00], shape=(1, 2, 3))
  std::println("{}", x - y); // venus::Tensor([-1.00, -1.00, -1.00, -1.00, -1.00, -1.00], shape=(1, 2, 3))
  std::println("{}", x * y); // venus::Tensor([2.00, 6.00, 12.00, 20.00, 30.00, 42.00], shape=(1, 2, 3))
  std::println("{}", x / y); // venus::Tensor([0.50, 0.67, 0.75, 0.80, 0.83, 0.86], shape=(1, 2, 3))

  assert(x + y == venus::ops::add(x, y));
  assert(x - y == venus::ops::sub(x, y));
  assert(x * y == venus::ops::mul(x, y));
  assert(x / y == venus::ops::div(x, y));
}