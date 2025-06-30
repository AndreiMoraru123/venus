#include "core/tensor/tensor.hpp"
#include <cassert>
#include <ranges>

using namespace venus;

auto main() -> int {
  auto tensor = Tensor<int, Device::CPU, 2>(3, 3); // heap alloc

  std::ranges::iota(tensor, 1); // 1, 2, 3, 4, 5, 6, 7, 8, 9

  auto pipeline =
      tensor |
      std::views::filter([](int x) { return x % 2 == 0; }) | // 2, 4, 6, 8
      std::views::transform([](int x) { return x * x; }) |   // 4, 16, 36, 64
      std::views::take(2);                                   // 4, 16

  auto result = std::ranges::fold_left(pipeline, 0, std::plus{}); // 20
  assert(result == 20);
}