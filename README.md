# Venus: Zero-Cost Tensors for Modern Deep Learning

| Linux (GCC 16) | Linux (Clang 22) | macOS (Apple Clang) | Windows (Clang-CL 22) | Windows (Native MSVC) |
| :---: | :---: | :---: | :---: | :---: |
| [![GCC](https://img.shields.io/github/actions/workflow/status/AndreiMoraru123/venus/gcc-linux.yml?logo=gnu&logoColor=white&label=GCC)](https://github.com/AndreiMoraru123/venus/actions/workflows/gcc-linux.yml) | [![Clang](https://img.shields.io/github/actions/workflow/status/AndreiMoraru123/venus/clang-linux.yml?logo=linux&logoColor=white&label=Clang)](https://github.com/AndreiMoraru123/venus/actions/workflows/clang-linux.yml) | [![macOS](https://img.shields.io/github/actions/workflow/status/AndreiMoraru123/venus/clang-apple.yml?logo=apple&logoColor=white&label=Apple%20Clang)](https://github.com/AndreiMoraru123/venus/actions/workflows/clang-apple.yml) | [![Clang-CL](https://img.shields.io/github/actions/workflow/status/AndreiMoraru123/venus/clang-cl-win.yml?logo=llvm&logoColor=white&label=Clang-CL)](https://github.com/AndreiMoraru123/venus/actions/workflows/clang-cl-win.yml) | [![MSVC](https://img.shields.io/github/actions/workflow/status/AndreiMoraru123/venus/msvc-win.yml?logo=cplusplus&logoColor=white&label=MSVC)](https://github.com/AndreiMoraru123/venus/actions/workflows/msvc-win.yml) |

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
print(check_order(ints)  # tensor([0, 2, 3, 4])
```

in Venus:

```cpp
#include <print>
#include <venus/tensor/tensor.hpp>

using namespace venus;

auto check_order(const auto &ints) {
  return venus::eager::where(venus::eager::sort(ints) != ints);
}

auto main() -> int {
  const auto ints = Tensor<int, Device::CPU, 1>{5, 2, 4, 3, 1};
  std::println("{}", check_order(ints)); // venus::Tensor([0, 0, 2, 3, 4], shape=(5))
}
```

#### Interactive Venus

Venus offers a lightweight implementation that works via [Clang-Repl](https://clang.llvm.org/docs/ClangRepl.html?utm_source=chatgpt.com), allowing it to
run interactively in shell mode:

```sh
cmake --build build --target venus-interactive
```

```cpp
clang-repl> using namespace venus;
clang-repl> using namespace venus::eager;
clang-repl> auto check_order(const auto& ints) { return where(sort(ints) != ints); }
clang-repl> auto ints = Tensor<int, Device::CPU, 1>{5, 2, 4, 3, 1};
clang-repl> auto res = check_order(ints);
clang-repl> print(res) // built-in macro
venus::Tensor([0, 0, 2, 3, 4], shape=(5))
(int) 0
clang-repl>
```

### Special Thanks and Author's Note

Venus draws heavy inspiration from [MetaNN](https://github.com/liwei-cpp/MetaNN), both the repository and the accompanying book _C++ Template Metaprogramming in Practice_ by Li Wei.

This is primarily a hobby project that I work on in my spare time, but I genuinely aim to build meaningfully on the original ideas rather than simply replicate them.

MetaNN was developed in a different era of the ecosystem, when static graph compilation and C++17 were the norm. The book references Theano multiple times, TensorFlow was still new, and PyTorch wasn't even a thing.

In many ways, this historical context still reflects how modern deep learning systems are structured today - the core execution models have remained relatively consistent, while most evolution has happened at the Python interface level.

One of the core design goals of Venus is to revisit this balance: to bring more of that evolution back into the native C++ layer, rather than relying on a higher-level bindings to provide usability and expressiveness.

While the original design is an impressive take on template metaprogramming, Venus aims to extend and modernize these ideas with improvements such as:

- Eager execution
- Use of modern C++ features (concepts, ranges)
- Heterogeneous compute, with a stronger focus on inference rather than training
- Beautiful, expressive APIs (as far as C++ allows)

Although TMP is still used where it fits naturally, it is no longer the central design goal.

On that note, Venus is intentionally designed as a C++-first library and does not aim to provide Python bindings.
There are already excellent projects in the Python ecosystem, such as PyTorch, Triton, and TVM, to name a few.

The goal of Venus is to explore what a modern, native C++ approach to tensor computation can offer.
