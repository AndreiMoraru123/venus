
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <print>
#include <string_view>

template <std::size_t N> struct VenusStr {
  char data[N]{};

  constexpr VenusStr(const char (&str)[N]) { std::copy_n(str, N, data); }

  constexpr auto operator<=>(const VenusStr &) const = default;

  [[nodiscard]] constexpr auto view() const -> std::string_view {
    return {data, N - 1};
  }
};

template <std::size_t N> VenusStr(const char (&)[N]) -> VenusStr<N>;

template <VenusStr Str> void print_str() {
  std::println("{}", Str.view());
  std::cout << Str.data << "\n";
}