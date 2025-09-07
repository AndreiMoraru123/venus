
#include "core/tensor/tensor.hpp"
#include <cassert>
#include <print>

using namespace venus;

// clang-format off
auto main() -> int {
  auto x = Tensor<float, Device::CPU, 3>(1, 2, 3); // heap alloc 1x2x3 Tensor
  auto y = Tensor<float, Device::CPU, 3>(1, 2, 3); // heap alloc 1x2x3 Tensor
  auto s = Tensor<float, Device::CPU, 0>(12.5f);

  std::ranges::iota(x, 1); // 1, 2, 3, 4, 5, 6
  std::ranges::iota(y, 2); // 2, 3, 4, 5, 6, 7

  std::println("{}", x + y); // venus::Tensor([3.00, 5.00, 7.00, 9.00, 11.00, 13.00], shape=(1, 2, 3))
  std::println("{}", x - y); // venus::Tensor([-1.00, -1.00, -1.00, -1.00, -1.00, -1.00], shape=(1, 2, 3))
  std::println("{}", x * y); // venus::Tensor([2.00, 6.00, 12.00, 20.00, 30.00, 42.00], shape=(1, 2, 3))
  std::println("{}", x / y); // venus::Tensor([0.50, 0.67, 0.75, 0.80, 0.83, 0.86], shape=(1, 2, 3))
  std::println("{}", "");

  std::println("{}", x + s); // venus::Tensor([13.50, 14.50, 15.50, 16.50, 17.50, 18.50], shape=(1, 2, 3))
  std::println("{}", x - s); // venus::Tensor([-11.50, -10.50, -9.50, -8.50, -7.50, -6.50], shape=(1, 2, 3))
  std::println("{}", x * s); // venus::Tensor([12.50, 25.00, 37.50, 50.00, 62.50, 75.00], shape=(1, 2, 3))
  std::println("{}", x / s); // venus::Tensor([0.08, 0.16, 0.24, 0.32, 0.40, 0.48], shape=(1, 2, 3))
  std::println("{}", "");

  std::println("{}", s + x); // venus::Tensor([13.50, 14.50, 15.50, 16.50, 17.50, 18.50], shape=(1, 2, 3))
  std::println("{}", s - x); // venus::Tensor([11.50, 10.50, 9.50, 8.50, 7.50, 6.50], shape=(1, 2, 3))
  std::println("{}", s * x); // venus::Tensor([12.50, 25.00, 37.50, 50.00, 62.50, 75.00], shape=(1, 2, 3))
  std::println("{}", s / x); // venus::Tensor([12.50, 6.25, 4.17, 3.12, 2.50, 2.08], shape=(1, 2, 3))
  std::println("{}", "");

  std::println("{}", x + 2); // venus::Tensor([3.00, 4.00, 5.00, 6.00, 7.00, 8.00], shape=(1, 2, 3))
  std::println("{}", x - 2); // venus::Tensor([-1.00, 0.00, 1.00, 2.00, 3.00, 4.00], shape=(1, 2, 3))
  std::println("{}", x * 2); // venus::Tensor([2.00, 4.00, 6.00, 8.00, 10.00, 12.00], shape=(1, 2, 3))
  std::println("{}", x / 2); // venus::Tensor([0.50, 1.00, 1.50, 2.00, 2.50, 3.00], shape=(1, 2, 3))
  std::println("{}", "");

  std::println("{}", 2 + x); // venus::Tensor([3.00, 4.00, 5.00, 6.00, 7.00, 8.00], shape=(1, 2, 3))
  std::println("{}", 2 - x); // venus::Tensor([1.00, 0.00, -1.00, -2.00, -3.00, -4.00], shape=(1, 2, 3))
  std::println("{}", 2 * x); // venus::Tensor([2.00, 4.00, 6.00, 8.00, 10.00, 12.00], shape=(1, 2, 3))
  std::println("{}", 2 / x); // venus::Tensor([2.00, 1.00, 0.67, 0.50, 0.40, 0.33], shape=(1, 2, 3))

  assert(x + y == venus::ops::add(x, y));
  assert(x - y == venus::ops::sub(x, y));
  assert(x * y == venus::ops::mul(x, y));
  assert(x / y == venus::ops::div(x, y));
}