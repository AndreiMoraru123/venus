#include <print>
#include <venus/tensor/tensor.hpp>

using namespace venus;

auto check_order(auto ints) {
  return venus::ops::where(venus::ops::sort(ints) != ints);
}

auto main() -> int {
  auto ints = Tensor<int, Device::CPU, 1>{5, 2, 4, 3, 1};
  std::println("Difference at indices: {}",
               check_order(ints)); // venus::Tensor([0, 0, 2, 3, 4], shape=(5))
}