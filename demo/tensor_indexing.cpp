#include <print>
#include <venus/tensor/tensor.hpp>

using namespace venus;

// clang-format off
auto main() -> int {
  auto tensor = Tensor<int, Device::CPU, 3>(3, 2, 2);
  tensor.iota(1); // venus::Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], shape=(3, 2, 2))

  const int x = tensor[0, 1, 1];
  const int y = tensor[0, 0, 1];
  const int z = tensor[1, 0, 1];

  std::println("{}", x); // 4
  std::println("{}", y); // 2
  std::println("{}", z); // 6
}
