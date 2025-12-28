#pragma once

#include <cstddef>
#include <initializer_list>

template <class T, std::size_t Depth> struct nested_initializer_list {
  using type = std::initializer_list<
      typename nested_initializer_list<T, Depth - 1>::type>;
};

template <class T> struct nested_initializer_list<T, 1> {
  using type = std::initializer_list<T>;
};

template <class T, std::size_t Depth>
using nested_initializer_list_t =
    typename nested_initializer_list<T, Depth>::type;
