#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <type_traits>
namespace venus {

namespace detail {
template <std::size_t idx, typename TShape> void Fill(TShape &shape) { return; }

template <std::size_t idx, typename TShape, typename TCurrParam,
          typename... TShapeParameter>
void Fill(TShape &shape, TCurrParam currParam, TShapeParameter... shapes) {
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

  explicit Shape() = default;

  template <SizeTLike... TIntTypes> explicit Shape(TIntTypes... shapes) {
    static_assert(sizeof...(TIntTypes) == Dim);
    detail::Fill<0>(m_dims, shapes...);
  }

  bool operator==(const Shape &val) const { return m_dims == val.m_dims; }

  template <size_t otherDim> bool operator==(const Shape<otherDim> &) const {
    return false;
  }

  auto operator[](size_t idx) const -> std::size_t {
    assert(idx < dimNum);
    return m_dims[idx];
  }

private:
  std::array<std::size_t, Dim> m_dims{};
};

template <> class Shape<0> {
  static constexpr std::size_t dimNum = 0;
  explicit Shape() = default;
};

template <SizeTLike... TShapeParameter>
explicit Shape(TShapeParameter...) -> Shape<sizeof...(TShapeParameter)>;

} // namespace venus