
#include <cassert>
#include <print>
#include <venus/tensor/tensor.hpp>

using namespace venus;

auto main() -> int {
  auto a = Tensor<int, Device::CPU, 2>{
      {0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}};

  auto b_row = Tensor<int, Device::CPU, 2>{{4, 3, 2, 1}};
  auto b_col = Tensor<int, Device::CPU, 2>{{4}, {3}, {2}, {1}};

  std::println("{}", a);
  std::println("{}", b_row);
  std::println("{}", b_col);

  auto c = a + b_row;
  std::println("{}", c);

  auto d = a + b_col;
  std::println("{}", d);
}
