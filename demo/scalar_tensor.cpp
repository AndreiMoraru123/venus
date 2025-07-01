
#include "core/tensor/tensor.hpp"
#include <cassert>

using namespace venus;

auto main() -> int {
  auto tensor = Tensor<float, Device::CPU, 0>(10.0f);
  assert(tensor.Value() == 10.0f);

  auto res = (tensor == 10.0f);
  assert(res.Value() == true);
}