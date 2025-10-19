#include "core/tensor/tensor.hpp"
#include <print>

using namespace venus;

// clang-format off
auto main() -> int {
  auto x = Tensor<int, Device::CPU, 2>(3, 3); // heap alloc 3x3 Tensor
  auto y = Tensor<int, Device::CPU, 2>(3, 3);

  venus::ops::iota(x, 1); // venus::Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], shape=(3, 3))
  venus::ops::iota(y, 1); // venus::Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], shape=(3, 3))

  auto z = x * y;
  std::println("{}", z); // venus::Tensor([1, 4, 9, 16, 25, 36, 49, 64, 81], shape=(3, 3))

  auto w = Tensor<int, Device::CPU, 2>(3, 3);

#pragma omp parallel
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      w[i, j] = x[i, j] * y[i, j];
    }

  std::println("{}", w); // venus::Tensor([1, 4, 9, 16, 25, 36, 49, 64, 81], shape=(3, 3))
}
