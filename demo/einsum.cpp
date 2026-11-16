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

  std::println("{}", x);
  auto x_vec = x.reshape(6);

  auto x_sum = eager::einsum<"i->">(x_vec);
  std::println("{}", x_sum);

  auto x_dot = eager::einsum<"i,i->">(x_vec, x_vec);
  std::println("{}", x_dot);

  auto z = eager::einsum<"ij,jk->ik">(x, y);
  std::println("{}", z);
}
