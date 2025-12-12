
#include "core/tensor/eager_ops.hpp"
#include "core/tensor/tensor.hpp"
#include <cassert>
#include <print>

using namespace venus;

auto main() -> int {
  auto x = Tensor<float, Device::CPU, 1>(7);
  auto y = Tensor<float, Device::CPU, 1>(7);

  venus::ops::iota(x, 1);
  venus::ops::iota(y, 1);

  auto z = x.dot(y);
  std::println("{}", z); // venus::Tensor(140.00)

  assert(z == y.dot(x));
  assert(z == venus::ops::dot(x, y));

  assert(venus::ops::all_equal(x + y, venus::ops::add(x, y)));
  assert(venus::ops::all_equal(x - y, venus::ops::sub(x, y)));
  assert(venus::ops::all_equal(x * y, venus::ops::mul(x, y)));
  assert(venus::ops::all_equal(x / y, venus::ops::div(x, y)));
}