
#include "core/memory/device.hpp"
#include "core/tensor/tensor.hpp"
#include <cassert>
#include <print>

using namespace venus;

auto main() -> int {
  auto scalar = Tensor<int, Device::CPU, 0>(10.0f);
  auto tensor = Tensor<int, Device::CPU, 3>(1, 2, 3);
  auto ch_tensor = Tensor<char, Device::CPU, 3>(1, 2, 3);
  auto str_tensor = Tensor<std::string, Device::CPU, 3>(1, 2, 3);

  std::println("{}", scalar); // venus::Tensor(10)
  std::println("{}",
               tensor); // venus::Tensor([0, 0, 0, 0, 0, 0], shape=(1, 2, 3))
  std::println(
      "{}",
      ch_tensor); // venus::Tensor(['', '', '', '', '', ''], shape=(1, 2, 3))
  std::println(
      "{}",
      str_tensor); // venus::Tensor(["", "", "", "", "", ""], shape=(1, 2, 3))
}