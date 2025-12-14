# Venus: Zero-Cost Tensors for Modern Deep Learning

### _A deep learning library, written in C++, for the love of C++_

Venus leverages C++20 ranges, providing both powerful optimizations and beautiful mathematical expressiveness:

```cpp
// demo/tensor_range.cpp
```

#### Solving the problem from https://www.youtube.com/watch?v=EEwREnUdbFs

in PyTorch:

```py
import torch


def check_order(ints):
    return torch.where(torch.sort(ints)[0] != ints)[0]

ints = torch.tensor([5, 2, 4, 3, 1])
print(check_order(ints) # tensor([0, 2, 3, 4])
```

in Venus:

```cpp
#include "core/tensor/tensor.hpp"
#include <print>

using namespace venus;

auto check_order(auto ints) {
  return venus::ops::where(venus::ops::sort(ints) != ints);
}

auto main() -> int {
  auto ints = Tensor<int, Device::CPU, 1>{5, 2, 4, 3, 1};
  std::println("{}", check_order(ints)); // venus::Tensor([0, 0, 2, 3, 4], shape=(5))
}
```
