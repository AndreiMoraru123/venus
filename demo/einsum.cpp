#include <cassert>
#include <print>
#include <venus/tensor/eager.hpp>
#include <venus/tensor/tensor.hpp>

using namespace venus;

auto main() -> int {
  auto x = Tensor<float, Device::CPU, 1>(7);
  auto m = Tensor<float, Device::CPU, 2>(2, 3);

  x.iota(1);
  m.iota(1);

  auto z = eager::sum_dim<0>(x); // correct, 28
  std::println("{}", z);

  eager::einsum<"i,i->i">(x, x);

  auto r = eager::sum_dim<0>(m);
  auto c = eager::sum_dim<1>(m);

  auto rc = eager::sum_dims<0, 1>(m);
  auto cr = eager::sum_dims<1, 0>(m);

  std::println("{}", r);
  std::println("{}", c);

  std::println("{}", rc);
  std::println("{}", cr);
}
