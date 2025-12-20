
#include <cassert>
#include <venus/tensor/tensor.hpp>

using namespace venus;

auto main() -> int {
  auto tensor = Tensor<float, Device::CPU, 0>(10.0f);
  assert(tensor.Value() == 10.0f);

  tensor.SetValue(12.0f);

  auto good = (tensor == 12.0f);
  assert(good == true);

  auto bad = (tensor != 12.0f);
  assert(bad == false);
}