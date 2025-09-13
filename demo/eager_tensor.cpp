
#include "core/tensor/tensor.hpp"
#include <cassert>
#include <print>

using namespace venus;

auto main() -> int {
  auto x = Tensor<float, Device::CPU, 1>(7);
  auto y = Tensor<float, Device::CPU, 1>(7);

  std::ranges::iota(x, 1); // 1, 2, 3, 4, 5, 6, 7
  std::ranges::iota(y, 1); // 1, 2, 3, 4, 5, 6, 7

  auto z = x.dot(y);
  std::println("{}", z); // venus::Tensor(140.00)

  assert(z == y.dot(x));
  assert(z == venus::ops::dot(x, y));

  assert(x + y == venus::ops::add(x, y));
  assert(x - y == venus::ops::sub(x, y));
  assert(x * y == venus::ops::mul(x, y));
  assert(x / y == venus::ops::div(x, y));
}