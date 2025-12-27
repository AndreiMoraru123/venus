#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <experimental/mdspan>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace venus {

template <typename... TIntTypes>
concept SizeTLike = (std::is_convertible_v<TIntTypes, std::size_t> and ...);

template <std::size_t Dim> class Shape {
  static_assert(Dim > 0);

public:
  static constexpr std::size_t dimNum = Dim;

  constexpr explicit Shape() = default;

  template <SizeTLike... TIntTypes>
    requires(sizeof...(TIntTypes) == Dim)
  constexpr explicit Shape(TIntTypes... shapes)
      : m_dims({static_cast<std::size_t>(shapes)...}) {}

  template <SizeTLike... TIntTypes>
    requires(sizeof...(TIntTypes) != Dim)
  constexpr explicit Shape(TIntTypes...) = delete;

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
    if (std::is_constant_evaluated()) {
      if (idx >= dimNum) {
        // TODO: This won't actually throw, do I really need comptime? (shape)
        throw std::out_of_range("Index out of bounds for Shape");
      }
    } else {
      assert(idx < dimNum);
    }
    return m_dims[idx];
  }

  constexpr auto OffsetToIndex(std::size_t offset) const
      -> std::array<std::size_t, dimNum> {
    std::array<std::size_t, dimNum> result{};
    for (int i = (int)dimNum - 1; i >= 0 && offset > 0; --i) {
      result[i] = offset % m_dims[i];
      offset /= m_dims[i];
    }
    if (offset != 0) {
      throw std::runtime_error("Offset out of bounds!");
    }
    return result;
  }

  template <SizeTLike... TIntTypes>
  constexpr auto IndexToOffset(TIntTypes... indices) const -> std::size_t {
    static_assert(sizeof...(TIntTypes) == dimNum, "Wrong number of indices");

    // TODO: The accessor policy in mdspan should be able to perform this (???)
    // bounds checking
    const std::size_t idx_array[] = {static_cast<std::size_t>(indices)...};
    for (std::size_t i = 0; i < dimNum; ++i) {
      if (idx_array[i] >= m_dims[i]) {
        throw std::out_of_range("Index out of bounds in Shape::IndexToOffset");
      }
    }

    auto span = CreateMdSpan(std::make_index_sequence<dimNum>{});
    return span.mapping()(indices...);
  }

  // Range Ops
  constexpr auto begin() { return m_dims.begin(); }
  constexpr auto end() { return m_dims.end(); }

  constexpr auto begin() const { return m_dims.begin(); }
  constexpr auto end() const { return m_dims.end(); }

  constexpr auto cbegin() const { return m_dims.begin(); }
  constexpr auto cend() const { return m_dims.end(); }

  constexpr auto size() const { return m_dims.size(); }

private:
  std::array<std::size_t, Dim> m_dims{};

  template <std::size_t... Is>
  constexpr auto CreateMdSpan(std::index_sequence<Is...>) const {
    return std::mdspan<int, std::dextents<std::size_t, dimNum>>(0,
                                                                m_dims[Is]...);
  }
};

template <> class Shape<0> {
public:
  static constexpr std::size_t dimNum = 0;

  explicit Shape() = default;

  constexpr std::size_t Count() const { return 1; }

  constexpr auto operator==(const Shape &val) const -> bool { return true; }

  template <size_t otherDim>
  auto constexpr operator==(const Shape<otherDim> &) const -> bool {
    return false;
  }
};

template <std::size_t Dim>
std::ostream &operator<<(std::ostream &os, const venus::Shape<Dim> &shape) {
  os << "(";
  std::size_t count = 0;
  for (auto dim : shape) {
    if (count > 0)
      os << ", ";
    count++;
    os << dim;
  }
  return os << ")";
}

template <std::size_t Dim>
std::ostream &operator<<(std::ostream &os, const venus::Shape<0> &shape) {
  return os << "()";
}

template <SizeTLike... TShapeParameter>
explicit Shape(TShapeParameter...) -> Shape<sizeof...(TShapeParameter)>;

} // namespace venus