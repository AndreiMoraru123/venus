
#include <cassert>
#include <ranges>
#include <venus/tensor/tensor.hpp>

using namespace venus;

// clang-format off
auto main() -> int {
  auto tensor = Tensor<int, Device::CPU, 2>(3, 3); // heap alloc 3x3 Tensor

  tensor.iota(1); // venus::Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], shape=(3, 3))

  auto pipeline =
      tensor |
      std::views::filter([](int x) { return x % 2 == 0; }) | // 2, 4, 6, 8
      std::views::transform([](int x) { return x * x; }) |   // 4, 16, 36, 64
      std::views::take(2);                                   // 4, 16

  auto result = std::ranges::fold_left(pipeline, 0, std::plus{}); // 20
  assert(result == 20); // lazy eval
}