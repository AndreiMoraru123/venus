
#include <cassert>
#include <print>
#include <venus/tensor/tensor.hpp>

using namespace venus;

auto main() -> int {
  auto a = Tensor<int, Device::CPU, 2>{
      {0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}};
  auto b = Tensor<int, Device::CPU, 1>{4, 3, 2, 1};

  auto c = a + b;
  std::println("{}", a);
  std::println("{}", b);
  std::println("{}", c);
}
