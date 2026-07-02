#pragma once

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <string_view>

namespace venus {
template <std::size_t N> struct ConstexprString {
  std::array<char, N> data{};

  constexpr ConstexprString(const char (&str)[N]) : data(std::to_array(str)) {}

  constexpr auto operator<=>(const ConstexprString &) const = default;

  [[nodiscard]] constexpr auto view() const -> std::string_view {
    return {data.data(), N - 1};
  }
};

template <std::size_t N>
ConstexprString(const char (&)[N]) -> ConstexprString<N>;
} // namespace venus
