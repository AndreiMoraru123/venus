#include <cassert>
#include <print>
#include <venus/tensor/eager.hpp>
#include <venus/tensor/tensor.hpp>

using namespace venus;

auto main() -> int {
  auto x = Tensor<float, Device::CPU, 1>(7);
  auto y = Tensor<float, Device::CPU, 1>(7);

  x.iota(1);
  y.iota(1);

  auto z = eager::sum_dim<0>(x); // corrent, 28
  std::println("{}", z);

  eager::einsum<"i,i->i">(x, x);
}
