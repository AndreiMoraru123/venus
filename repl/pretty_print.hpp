#pragma once
#ifdef VENUS_INTERPRETER
#include <iostream>

template <typename T, typename TDevice, std::size_t Rank>
void __clang_repl__Display__(const venus::Tensor<T, TDevice, Rank> &t) {
  std::cout << t << '\n';
}

template <std::size_t Rank>
void __clang_repl__Display__(const venus::Shape<Rank> &s) {
  std::cout << s << '\n';
}

// I have to do this until clang-repl fixes the interception of the two above...
#define print(x) (std::cout << (x) << std::endl, 0)

#endif