
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <print>
#include <string_view>

template <std::size_t N> struct ConstexprString {
  char data[N]{};

  constexpr ConstexprString(const char (&str)[N]) { std::copy_n(str, N, data); }

  constexpr auto operator<=>(const ConstexprString &) const = default;

  [[nodiscard]] constexpr auto view() const -> std::string_view {
    return {data, N - 1};
  }
};

template <std::size_t N> ConstexprString(const char (&)[N]) -> ConstexprString<N>;

template <ConstexprString Str> void print_str() {
  std::println("{}", Str.view());
  std::cout << Str.data << "\n";
}