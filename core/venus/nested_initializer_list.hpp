#pragma once

#include <cstddef>
#include <initializer_list>

template <typename T, std::size_t Depth> struct nested_initializer_list {
  using type = std::initializer_list<
      typename nested_initializer_list<T, Depth - 1>::type>;
};

template <typename T> struct nested_initializer_list<T, 1> {
  using type = std::initializer_list<T>;
};

template <typename T, std::size_t Depth>
using nested_initializer_list_t = nested_initializer_list<T, Depth>::type;
