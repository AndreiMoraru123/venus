#pragma once

#include <array>
#include <bits/ranges_algo.h>
#include <cassert>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <type_traits>
namespace venus {

namespace detail {
template <std::size_t idx, typename TShape> constexpr void Fill(TShape &shape) {
  return;
}

template <std::size_t idx, typename TShape, typename TCurrParam,
          typename... TShapeParameter>
constexpr void Fill(TShape &shape, TCurrParam currParam,
                    TShapeParameter... shapes) {
  shape[idx] = static_cast<std::size_t>(currParam);
  Fill<idx + 1>(shape, shapes...);
}
} // namespace detail

template <typename... TIntTypes>
concept SizeTLike = (std::is_convertible_v<TIntTypes, std::size_t> and ...);

template <std::size_t Dim> class Shape {
  static_assert(Dim > 0);

public:
  static constexpr std::size_t dimNum = Dim;

  constexpr explicit Shape() = default;

  template <SizeTLike... TIntTypes>
  constexpr explicit Shape(TIntTypes... shapes) {
    static_assert(sizeof...(TIntTypes) == Dim);
    detail::Fill<0>(m_dims, shapes...);
  }

  constexpr auto operator==(const Shape &val) const -> bool {
    return m_dims == val.m_dims;
  }

  template <size_t otherDim>
  auto constexpr operator==(const Shape<otherDim> &) const -> bool {
    return false;
  }

  constexpr std::size_t Count() const {
    return std::ranges::fold_left(m_dims, static_cast<std::size_t>(1),
                                  std::multiplies<>());
  }

  constexpr auto operator[](size_t idx) const -> std::size_t {
    if constexpr (std::is_constant_evaluated()) {
      if (idx >= dimNum) {
        throw std::out_of_range("Index out of bounds for Shape");
      }
    } else {
      assert(idx < dimNum);
    }
    return m_dims[idx];
  }

private:
  std::array<std::size_t, Dim> m_dims{};
};

template <> class Shape<0> {
public:
  static constexpr std::size_t dimNum = 0;

  explicit Shape() = default;

  constexpr std::size_t Count() const { return 1; }
};

template <SizeTLike... TShapeParameter>
explicit Shape(TShapeParameter...) -> Shape<sizeof...(TShapeParameter)>;

} // namespace venus