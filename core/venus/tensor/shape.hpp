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

template <typename... Dimensions>
concept SizeTLike = (std::is_convertible_v<Dimensions, std::size_t> and ...);

template <std::size_t Rank> class Shape {
  static_assert(Rank > 0);

public:
  static constexpr std::size_t rank = Rank;

  constexpr explicit Shape() = default;

  template <SizeTLike... Dimensions>
    requires(sizeof...(Dimensions) == Rank)
  constexpr explicit Shape(Dimensions... shapes)
      : m_dims({static_cast<std::size_t>(shapes)...}) {}

  template <SizeTLike... Dimensions>
    requires(sizeof...(Dimensions) != Rank)
  constexpr explicit Shape(Dimensions...) = delete;

  constexpr auto operator==(const Shape &val) const -> bool {
    return m_dims == val.m_dims;
  }

  template <size_t otherRank>
  auto constexpr operator==(const Shape<otherRank> & /*unused*/) const -> bool {
    return false;
  }

  [[nodiscard]] constexpr auto count() const -> std::size_t {
    return std::ranges::fold_left(m_dims, static_cast<std::size_t>(1),
                                  std::multiplies<>());
  }

  constexpr auto operator[](size_t idx) const -> std::size_t {
    if (std::is_constant_evaluated()) {
      if (idx >= rank) {
        // TODO: This won't actually throw, do I really need comptime? (shape)
        throw std::out_of_range("Index out of bounds for Shape");
      }
    } else {
      assert(idx < rank);
    }
    return m_dims[idx];
  }

  constexpr auto offsetToIdx(std::size_t offset) const
      -> std::array<std::size_t, rank> {
    std::array<std::size_t, rank> result{};
    for (int i = (int)rank - 1; i >= 0 && offset > 0; --i) {
      result[i] = offset % m_dims[i];
      offset /= m_dims[i];
    }
    if (offset != 0) {
      throw std::runtime_error("Offset out of bounds!");
    }
    return result;
  }

  template <SizeTLike... Dimensions>
  constexpr auto idxToOffset(Dimensions... indices) const -> std::size_t {
    static_assert(sizeof...(Dimensions) == rank, "Wrong number of indices");

    // TODO: The accessor policy in mdspan should be able to perform this (???)
    // bounds checking
    const std::array<std::size_t, rank> idx_array = {
        static_cast<std::size_t>(indices)...};
    for (std::size_t i = 0; i < rank; ++i) {
      if (idx_array[i] >= m_dims[i]) {
        throw std::out_of_range("Index out of bounds in Shape::IndexToOffset");
      }
    }

    auto span = createSpan(std::make_index_sequence<rank>{});
    return span.mapping()(indices...);
  }

  constexpr static auto fromNestedInitializerList(auto nested_init_list)
      -> Shape<rank> {
    Shape<rank> shape;

    auto extract = [](const auto &list, std::size_t level,
                      std::array<std::size_t, rank> &dims,
                      const auto &self_ref) -> void {
      if constexpr (requires {
                      list.size();
                      list.size() > 0;
                    }) {
        dims[level] = list.size();

        if (level + 1 < rank) {
          if constexpr (requires { (*list.begin()).size(); }) {
            const auto expected_size = (*list.begin()).size();

            // Horizontal: Check all sibling lists at this level
            for (const auto &sublist : list) {
              if (sublist.size() != expected_size) {
                throw std::invalid_argument(
                    std::format("Inconsistent dimensions at dimension {}: "
                                "expected size {}, got {}",
                                level + 2, expected_size, sublist.size()));
              }
            }
            // Vertical: Go deeper into each sublist
            for (const auto &sublist : list) {
              self_ref(sublist, level + 1, dims, self_ref);
            }
          }
        }
      }
    };

    extract(nested_init_list, 0, shape.m_dims, extract);
    return shape;
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
  std::array<std::size_t, Rank> m_dims{};

  template <std::size_t... Is>
  constexpr auto createSpan(std::index_sequence<Is...> /*unused*/) const {
    return std::mdspan<int, std::dextents<std::size_t, rank>>(0, m_dims[Is]...);
  }
};

template <> class Shape<0> {
public:
  static constexpr std::size_t rank = 0;

  explicit Shape() = default;

  static constexpr auto count() -> std::size_t { return 1; }

  constexpr auto operator==(const Shape &val) const -> bool { return true; }

  template <size_t otherRank>
  auto constexpr operator==(const Shape<otherRank> & /*unused*/) const -> bool {
    return false;
  }
};

template <std::size_t Rank>
auto operator<<(std::ostream &os, const Shape<Rank> &shape) -> std::ostream & {
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

template <std::size_t Rank>
auto operator<<(std::ostream &os, const Shape<0> &shape) -> std::ostream & {
  return os << "()";
}

template <SizeTLike... TShapeParameter>
explicit Shape(TShapeParameter...) -> Shape<sizeof...(TShapeParameter)>;

} // namespace venus

template <std::size_t Rank> struct std::formatter<venus::Shape<Rank>> {
  constexpr auto parse(std::format_parse_context &ctx) { return ctx.begin(); }

  auto format(const venus::Shape<Rank> &shape, std::format_context &ctx) const {
    ostringstream oss;
    oss << shape;
    return std::format_to(ctx.out(), "{}", oss.str());
  }
};