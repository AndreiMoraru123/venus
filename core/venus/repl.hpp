#ifdef VENUS_INTERPRETER
#include <iostream>
namespace venus {
template <typename T, Device D, std::size_t Rank>
void __clang_repl__Display__(const Tensor<T, D, Rank> &t) {
  std::cout << t << "\n";
}
template <std::size_t Rank> void __clang_repl__Display__(const Shape<Rank> &s) {
  std::cout << s << "\n";
}
} // namespace venus
#endif