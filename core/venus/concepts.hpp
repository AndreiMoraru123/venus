#pragma once

#ifndef VENUS_MODULE
#include <concepts>
#include <cstddef>
#include <type_traits>
#endif

namespace venus {
template <typename T>
concept VenusTensor = requires {
  typename std::remove_cvref_t<T>::ElementType;
  typename std::remove_cvref_t<T>::DeviceType;
  { std::remove_cvref_t<T>::rank } -> std::convertible_to<std::size_t>;
};

template <typename T>
concept Scalar = std::is_arithmetic_v<std::remove_cvref_t<T>>;

template <typename T>
concept ScalarTensor =
    VenusTensor<std::remove_cvref_t<T>> && (std::remove_cvref_t<T>::rank == 0);

template <typename T>
concept MDTensor =
    VenusTensor<std::remove_cvref_t<T>> && (std::remove_cvref_t<T>::rank > 0);

template <typename T>
concept BoolTensor =
    VenusTensor<std::remove_cvref_t<T>> &&
    std::is_convertible_v<typename std::remove_cvref_t<T>::ElementType, bool>;

template <typename T>
concept StringLike = requires(T str) {
  { str.c_str() } -> std::convertible_to<const char *>;
} or std::convertible_to<T, const char *>;

template <typename T>
concept CharLike = std::same_as<T, char> || std::same_as<T, signed char> ||
                   std::same_as<T, unsigned char> || std::same_as<T, wchar_t> ||
                   std::same_as<T, char8_t> || std::same_as<T, char16_t> ||
                   std::same_as<T, char32_t>;
} // namespace venus
