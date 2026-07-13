#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <format>
#include <functional>
#include <mdspan>
#include <sstream>
#include <stdexcept>
#include <string_view>
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
  constexpr explicit Shape(Dimensions... dims)
      : m_dims({static_cast<std::size_t>(dims)...}) {}

  template <SizeTLike... Dimensions>
    requires(sizeof...(Dimensions) != Rank)
  constexpr explicit Shape(Dimensions...) = delete;

  constexpr explicit Shape(std::array<std::size_t, Rank> dims) noexcept
      : m_dims(std::move(dims)) {}

  constexpr auto operator==(const Shape &val) const -> bool {
    return m_dims == val.m_dims;
  }

  template <std::size_t otherRank>
  auto constexpr operator==(const Shape<otherRank> & /*unused*/) const -> bool {
    return false;
  }

  [[nodiscard]] constexpr auto count() const -> std::size_t {
    return std::ranges::fold_left(m_dims, static_cast<std::size_t>(1),
                                  std::multiplies<>());
  }

  constexpr auto operator[](std::size_t idx) const -> std::size_t {
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

  constexpr auto
  idxToOffset(const std::array<std::size_t, rank> &idx_array) const
      -> std::size_t {
    for (std::size_t i = 0; i < rank; ++i) {
      if (idx_array[i] >= m_dims[i]) {
        throw std::out_of_range("Index out of bounds in Shape::idxToOffset");
      }
    }
    std::size_t offset = 0;
    std::size_t stride = 1;
    for (int i = (int)rank - 1; i >= 0; --i) {
      offset += idx_array[i] * stride;
      stride *= m_dims[i];
    }
    return offset;
  }

  template <SizeTLike... Dimensions>
  constexpr auto idxToOffset(Dimensions... indices) const -> std::size_t {
    static_assert(sizeof...(Dimensions) == rank, "Wrong number of indices");

    // ? The accessor policy in mdspan should be able to perform this (???)
    // TODO: Move bounds checking up to tensor logic when mdspan::at lands
    // https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p3383r0.html
    const std::array<std::size_t, rank> idx_array = {
        static_cast<std::size_t>(indices)...};
    for (std::size_t i = 0; i < rank; ++i) {
      if (idx_array[i] >= m_dims[i]) {
        throw std::out_of_range("Index out of bounds in Shape::idxToOffset");
      }
    }

    auto mapping = createMapping(std::make_index_sequence<rank>{});
    return mapping(indices...);
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
  constexpr auto begin(this auto &&self) {
    return std::forward<decltype(self)>(self).m_dims.begin();
  }
  constexpr auto end(this auto &&self) {
    return std::forward<decltype(self)>(self).m_dims.end();
  }

  constexpr auto cbegin(this auto &&self) {
    return std::as_const(self).m_dims.begin();
  }
  constexpr auto cend(this auto &&self) {
    return std::as_const(self).m_dims.end();
  }

  constexpr auto size() const { return m_dims.size(); }

  template <std::size_t SubRank, std::size_t StartOffset = 0>
    requires(StartOffset + SubRank <= rank)
  constexpr auto slice() const -> Shape<SubRank> {
    std::array<std::size_t, SubRank> sub_dims{};
    for (std::size_t i = 0; i < SubRank; i++) {
      sub_dims[i] = m_dims[StartOffset + i];
    }
    return Shape<SubRank>(sub_dims);
  }

private:
  std::array<std::size_t, Rank> m_dims{};

  template <std::size_t... Is>
  constexpr auto createMapping(std::index_sequence<Is...> /*unused*/) const {
    using Extents = std::dextents<std::size_t, rank>;
    using Mapping = std::layout_right::mapping<Extents>;
    return Mapping{Extents{m_dims[Is]...}};
  }
};

template <> class Shape<0> {
public:
  static constexpr std::size_t rank = 0;

  explicit Shape() = default;

  static constexpr auto count() -> std::size_t { return 1; }

  constexpr auto operator==(const Shape &val) const -> bool { return true; }

  template <std::size_t otherRank>
  auto constexpr operator==(const Shape<otherRank> & /*unused*/) const -> bool {
    return false;
  }
};

template <std::size_t Rank>
auto operator<<(std::ostream &os, const Shape<Rank> &shape) -> std::ostream & {
  os << std::string_view{"("};
  std::size_t count = 0;
  for (auto dim : shape) {
    if (count > 0)
      os << std::string_view{", "};
    count++;
    os << dim;
  }
  return os << std::string_view{")"};
}

template <std::size_t Rank = 0>
auto operator<<(std::ostream &os, const Shape<0> &shape) -> std::ostream & {
  return os << std::string_view{"()"};
}

template <SizeTLike... TShapeParameter>
explicit Shape(TShapeParameter...) -> Shape<sizeof...(TShapeParameter)>;

template <std::size_t N, std::size_t Rank>
constexpr auto get(const Shape<Rank> &shape) noexcept -> std::size_t {
  static_assert(N < Rank, "Index out of bounds in Shape::get");
  return shape[N];
}

// -------------------------- BROADCASTING --------------------------
template <std::size_t RankOut, std::size_t RankIn>
constexpr auto
project_broadcast_idx(const std::array<std::size_t, RankOut> &in_idx,
                      const Shape<RankIn> &in_shape) {
  std::array<std::size_t, RankIn> out_idx{};
  constexpr auto offset = RankOut - RankIn;

  for (std::size_t i = 0; i < RankIn; i++) {
    out_idx[i] = in_shape[i] == 1 ? 0 : in_idx[i + offset];
  }

  return out_idx;
}

template <std::size_t RankOut, std::size_t Rank1, std::size_t Rank2>
constexpr auto broadcast(const Shape<Rank1> &s1, const Shape<Rank2> &s2)
    -> Shape<RankOut> {
  std::array<std::size_t, RankOut> out{};

  for (std::size_t i = 0; i < RankOut; i++) {
    auto d1 = std::size_t{1};
    auto d2 = std::size_t{1};

    if (i >= RankOut - Rank1)
      d1 = s1[i - (RankOut - Rank1)];
    if (i >= RankOut - Rank2)
      d2 = s2[i - (RankOut - Rank2)];

    if (d1 != d2 && d1 != 1 && d2 != 1) {
      throw std::invalid_argument(
          std::format("Tensor shapes are not broadcastable, t1 has shape {}, "
                      "whereas t2 has shape {}.",
                      s1, s2));
    }

    out[i] = std::max(d1, d2);
  }

  return Shape<RankOut>(out);
}

template <std::size_t RankOut, std::size_t Rank1, std::size_t Rank2,
          std::size_t Rank3>
constexpr auto broadcast(const Shape<Rank1> &s1, const Shape<Rank2> &s2,
                         const Shape<Rank3> &s3) {
  constexpr auto Rank12 = std::max(Rank1, Rank2);
  auto s12 = broadcast<Rank12>(s1, s2);
  return broadcast<RankOut>(s12, s3);
}

// ------------------------------------------------------------------

} // namespace venus

template <std::size_t Rank> struct std::formatter<venus::Shape<Rank>> {
  constexpr auto parse(std::format_parse_context &ctx) { return ctx.begin(); }

  auto format(const venus::Shape<Rank> &shape, std::format_context &ctx) const {
    std::ostringstream oss;
    oss << shape;
    return std::format_to(ctx.out(), "{}", oss.str());
  }
};

namespace std {
template <std::size_t Rank>
struct tuple_size<venus::Shape<Rank>>
    : std::integral_constant<std::size_t, Rank> {};

template <std::size_t N, std::size_t Rank>
struct tuple_element<N, venus::Shape<Rank>> {
  static_assert(N < Rank, "Index out of bounds in tuple_elements for Shape");
  using type = std::size_t;
};

} // namespace std