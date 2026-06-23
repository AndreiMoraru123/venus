#include <cassert>
#include <print>
#include <venus/tensor/eager.hpp>
#include <venus/tensor/tensor.hpp>

using namespace venus;

auto main() -> int {
  auto x = Tensor<float, Device::CPU, 2>(2, 3);
  auto y = Tensor<float, Device::CPU, 2>(3, 4);

  x.iota(1);
  y.iota(1);

  auto z = eager::einsum<"ij,jk->ik">(x, y);
  std::println("{}", z);
}
