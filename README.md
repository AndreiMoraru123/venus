# Venus: Zero-Cost Tensors for Modern Deep Learning

### _A deep learning library, written in C++, for C++, out the love for C++_

Venus leverages C++20 ranges, providing both powerful optimizations and beautiful mathematical expressiveness:

```cpp
// demo/tensor_range.cpp


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
```

#### [1 Problem, 7 Libraries](https://www.youtube.com/watch?v=EEwREnUdbFs)

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
#include <print>
#include <venus/tensor/tensor.hpp>

using namespace venus;

auto check_order(const auto &ints) {
  return venus::ops::where(venus::ops::sort(ints) != ints);
}

auto main() -> int {
  const auto ints = Tensor<int, Device::CPU, 1>{5, 2, 4, 3, 1};
  std::println("{}", check_order(ints)); // venus::Tensor([0, 0, 2, 3, 4], shape=(5))
}
```

#### Interactive Venus

Venus offers a lightweight implementation that works via [Cling](https://github.com/root-project/cling), allowing it to run interactively in shell mode:

```sh
cmake --build build --target venus-interactive
```

```cpp
[0/2] Re-checking globbed directories...
[0/2] Starting Cling-based interactive Venus interpreter. Include <single_include/venus.hpp>, have fun!

****************** CLING ******************
* Type C++ code and press enter to run it *
*             Type .q to exit             *
*******************************************
[cling]$ #include <single_include/venus.hpp>
[cling]$ using namespace venus;
[cling]$ using namespace venus::ops;
[cling]$ auto check_order(const auto& ints) { return where(sort(ints) != ints); }
[cling]$ auto ints = Tensor<int, Device::CPU, 1>{5, 2, 4, 3, 1};
[cling]$ check_order(ints)
(venus::Tensor<std::size_t, venus::Device::CPU, 1UL>) { 0, 0, 2, 3, 4 }
[cling]$
```
