#include "core/tensor/tensor.hpp"
#include <print>

using namespace venus;

// clang-format off
auto main() -> int {
  auto x = Tensor<float, Device::CPU, 2>(3, 2);
  auto y = Tensor<float, Device::CPU, 2>(3, 2);

  venus::ops::iota(x, 1); // venus::Tensor([1.00, 2.00, 3.00, 4.00, 5.00, 6.00], shape=(3, 2))
  venus::ops::fill(y, 1); // venus::Tensor([1.00, 1.00, 1.00, 1.00, 1.00, 1.00], shape=(3, 2))

  auto condition = x > 3;
  std::println("{}", condition); // venus::Tensor([0, 0, 0, 1, 1, 1], shape=(3, 2))

  auto z = venus::ops::where(x > 3);
  std::println("{}", z); // venus::Tensor([0, 0, 0, 3, 4, 5], shape=(3, 2))

  auto w = venus::ops::where(x > 3, x, y); 
  std::println("{}", w); // venus::Tensor([1.00, 1.00, 1.00, 4.00, 5.00, 6.00], shape=(3, 2))

  auto k = venus::ops::where(x > 3, 2.0f, -1.0f); 
  std::println("{}", k); // venus::Tensor([-1.00, -1.00, -1.00, 2.00, 2.00, 2.00], shape=(3, 2))

  auto l = venus::ops::where(x > 3, y, -1.0f);
  std::println("{}", l);  // venus::Tensor([-1.00, -1.00, -1.00, 1.00, 1.00, 1.00], shape=(3, 2))
}