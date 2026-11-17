#pragma once

#ifndef VENUS_MODULE
#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <format>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>
#endif
#include <venus/concepts.hpp>
#include <venus/memory/contiguous_memory.hpp>
#include <venus/memory/device.hpp>
#include <venus/tensor/eager.hpp>
#include <venus/tensor/shape.hpp>
#include <venus/tensor/tensor_iterator.hpp>

namespace venus {

template <typename TElem, typename TDevice, std::size_t Rank> class Tensor;
template <typename TElem, typename TDevice, std::size_t Rank> class TensorView;
template <typename TElem, typename TDevice, std::size_t Rank> class Selection;

// Selectors for t[...]. An integer drops its dimension, range(a, b) keeps
// positions a to b - 1 and all keeps every position
struct all_t {
  explicit constexpr all_t() = default;
};
inline constexpr all_t all{};

struct Range {
  std::size_t start;
  std::size_t stop;
};

constexpr auto range(std::size_t start, std::size_t stop) -> Range {
  return Range{start, stop};
}

template <typename T>
concept Selector =
    std::same_as<T, all_t> || std::same_as<T, Range> || SizeTLike<T>;

// At least one selector keeps a dimension, so t[...] is a view
template <typename... Selectors>
concept Slicing = (Selector<Selectors> && ...) && !SizeTLike<Selectors...>;

template <typename... Selectors>
inline constexpr std::size_t kept_rank =
    (std::size_t{0} + ... + static_cast<std::size_t>(!SizeTLike<Selectors>));

template <typename T> struct IsTensorView : std::false_type {};
template <typename TElem, typename TDevice, std::size_t Rank>
struct IsTensorView<TensorView<TElem, TDevice, Rank>> : std::true_type {};
template <typename TElem, typename TDevice, std::size_t Rank>
struct IsTensorView<Selection<TElem, TDevice, Rank>> : std::true_type {};

template <typename T>
concept AnyTensorView = IsTensorView<std::remove_cvref_t<T>>::value;

namespace detail {
template <std::size_t Rank>
constexpr auto row_major_strides(const Shape<Rank> &shape)
    -> std::array<std::size_t, Rank> {
  std::array<std::size_t, Rank> strides{};
  std::size_t stride = 1;
  for (std::size_t i = Rank; i-- > 0;) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

template <std::size_t Rank>
constexpr auto extents(const Shape<Rank> &shape)
    -> std::array<std::size_t, Rank> {
  std::array<std::size_t, Rank> out{};
  std::ranges::copy(shape, out.begin());
  return out;
}
} // namespace detail

// Reads another tensor's memory through a shape and non-negative strides.
// Copying a view shares the memory; clone() copies the elements to a Tensor
template <typename TElem, typename TDevice, std::size_t Rank>
class TensorView : public std::ranges::view_base {
  static_assert(std::is_same_v<std::remove_cvref_t<TElem>, TElem>);
  static_assert(std::is_same_v<TDevice, Device::CPU>,
                "Tensor views exist only on CPU");

public:
  using ElementType = TElem;
  using DeviceType = TDevice;
  using Strides = std::array<std::size_t, Rank>;
  static constexpr std::size_t rank = Rank;

  // mem starts at the view's first element
  explicit TensorView(ContiguousMemory<TElem, TDevice> mem, Shape<Rank> shape,
                      Strides strides)
      : m_mem(std::move(mem)), m_shape(shape), m_strides(strides) {
    std::size_t last = 0;
    for (std::size_t i = 0; i < Rank; ++i) {
      if (m_shape[i] == 0) {
        throw std::invalid_argument(std::format(
            "A tensor view needs every extent above 0, got {}", m_shape));
      }
      last += (m_shape[i] - 1) * m_strides[i];
    }
    if (last >= m_mem.size()) {
      throw std::invalid_argument(std::format(
          "A view of shape {} reaches element {}, but the memory holds {}",
          m_shape, last, m_mem.size()));
    }
  }

  [[nodiscard]] auto shape() const noexcept -> Shape<Rank> { return m_shape; }
  [[nodiscard]] auto strides() const noexcept -> Strides { return m_strides; }
  [[nodiscard]] auto size() const -> std::size_t { return m_shape.count(); }
  [[nodiscard]] auto unique() const -> bool { return not m_mem.isShared(); }

  // The first element. It is the start of a row-major array only when
  // isContiguous() holds
  [[nodiscard]] auto data() const -> const TElem * { return m_mem.ptr(); }

  // The elements sit in row-major order with no gaps, so a kernel can read
  // data() as a flat array. Extents of 1 take any stride
  [[nodiscard]] auto isContiguous() const -> bool {
    std::size_t expected = 1;
    for (std::size_t i = Rank; i-- > 0;) {
      if (m_shape[i] == 1) {
        continue;
      }
      if (m_strides[i] != expected) {
        return false;
      }
      expected *= m_shape[i];
    }
    return true;
  }

  auto begin() const -> TensorViewIterator<TElem, Rank> {
    return {data(), detail::extents(m_shape), m_strides, 0};
  }

  auto end() const -> TensorViewIterator<TElem, Rank> {
    return {data(), detail::extents(m_shape), m_strides, size()};
  }

  template <typename... Indices>
    requires(sizeof...(Indices) == Rank) && SizeTLike<Indices...>
  auto operator[](Indices... indices) const -> const TElem & {
    return (*this)[Strides{static_cast<std::size_t>(indices)...}];
  }

  auto operator[](const std::array<std::size_t, Rank> &indices) const
      -> const TElem & {
    std::size_t offset = 0;
    for (std::size_t i = 0; i < Rank; ++i) {
      if (indices[i] >= m_shape[i]) {
        throw std::out_of_range("Index out of bounds in TensorView");
      }
      offset += indices[i] * m_strides[i];
    }
    return data()[offset];
  }

  template <Selector... Selectors>
    requires(sizeof...(Selectors) == Rank) && Slicing<Selectors...>
  auto operator[](Selectors... selectors) const
      -> TensorView<TElem, TDevice, kept_rank<Selectors...>> {
    constexpr std::size_t NewRank = kept_rank<Selectors...>;
    std::array<std::size_t, NewRank> shape{};
    std::array<std::size_t, NewRank> strides{};
    std::size_t offset = 0;
    std::size_t dim = 0;
    std::size_t kept = 0;
    auto place = [&]<typename S>(S selector) {
      const auto extent = m_shape[dim];
      const auto stride = m_strides[dim];
      if constexpr (std::same_as<S, all_t>) {
        shape[kept] = extent;
        strides[kept++] = stride;
      } else if constexpr (std::same_as<S, Range>) {
        if (selector.stop > extent) {
          throw std::out_of_range(std::format(
              "range({}, {}) is out of bounds for dimension {} of extent {}",
              selector.start, selector.stop, dim, extent));
        }
        if (selector.start >= selector.stop) {
          throw std::invalid_argument(std::format(
              "range({}, {}) selects nothing", selector.start, selector.stop));
        }
        offset += selector.start * stride;
        shape[kept] = selector.stop - selector.start;
        strides[kept++] = stride;
      } else {
        const auto index = static_cast<std::size_t>(selector);
        if (index >= extent) {
          throw std::out_of_range(
              std::format("Index {} is out of bounds for dimension {} of "
                          "extent {}",
                          index, dim, extent));
        }
        offset += index * stride;
      }
      ++dim;
    };
    (place(selectors), ...);
    return TensorView<TElem, TDevice, NewRank>(m_mem.shift(offset),
                                               Shape<NewRank>(shape), strides);
  }

  auto view() const -> TensorView { return *this; }

  auto clone() const -> Tensor<TElem, TDevice, Rank> {
    auto out = Tensor<TElem, TDevice, Rank>(m_shape);
    std::ranges::copy(*this, out.data());
    return out;
  }

  template <typename... Dimensions>
    requires SizeTLike<Dimensions...>
  auto reshape(Dimensions... dims) const {
    return reshape(Shape(dims...));
  }

  // Shares the memory when the view is contiguous and copies it otherwise
  template <std::size_t NewRank>
  auto reshape(Shape<NewRank> new_shape) const
      -> TensorView<TElem, TDevice, NewRank> {
    if (new_shape.count() != size()) {
      throw std::invalid_argument(std::format(
          "Cannot reshape tensor of size {} to new shape of size {}", size(),
          new_shape.count()));
    }
    if (not isContiguous()) {
      return clone().view().reshape(new_shape);
    }
    return TensorView<TElem, TDevice, NewRank>(
        m_mem, new_shape, detail::row_major_strides(new_shape));
  }

  // Dimension i of the result is dimension dims[i] of this view
  template <typename... Dims>
    requires(sizeof...(Dims) == Rank) && SizeTLike<Dims...>
  auto permute(Dims... dims) const -> TensorView {
    const std::array<std::size_t, Rank> order{
        static_cast<std::size_t>(dims)...};
    std::array<bool, Rank> seen{};
    std::array<std::size_t, Rank> shape{};
    Strides strides{};
    for (std::size_t i = 0; i < Rank; ++i) {
      const auto dim = order[i];
      if (dim >= Rank || seen[dim]) {
        throw std::invalid_argument(
            std::format("permute takes each dimension below {} once", Rank));
      }
      seen[dim] = true;
      shape[i] = m_shape[dim];
      strides[i] = m_strides[dim];
    }
    return TensorView(m_mem, Shape<Rank>(shape), strides);
  }

  template <typename... Dimensions>
    requires(sizeof...(Dimensions) >= Rank) && SizeTLike<Dimensions...>
  auto expand(Dimensions... dims) const {
    return expand(Shape(dims...));
  }

  // Repeats extents of 1 and adds leading dimensions, all with stride 0
  template <std::size_t NewRank>
    requires(NewRank >= Rank)
  auto expand(Shape<NewRank> new_shape) const
      -> TensorView<TElem, TDevice, NewRank> {
    constexpr auto lead = NewRank - Rank;
    std::array<std::size_t, NewRank> strides{};
    for (std::size_t i = 0; i < Rank; ++i) {
      if (m_shape[i] == new_shape[lead + i]) {
        strides[lead + i] = m_strides[i];
      } else if (m_shape[i] != 1) {
        throw std::invalid_argument(
            std::format("Cannot expand shape {} to {}", m_shape, new_shape));
      }
    }
    return TensorView<TElem, TDevice, NewRank>(m_mem, new_shape, strides);
  }

  template <std::size_t Dim>
    requires(Dim < Rank)
  auto squeeze() const -> TensorView<TElem, TDevice, Rank - 1> {
    if (m_shape[Dim] != 1) {
      throw std::invalid_argument(
          std::format("Cannot squeeze dimension {} of shape {}, its extent "
                      "is not 1",
                      Dim, m_shape));
    }
    if constexpr (Rank == 1) {
      return TensorView<TElem, TDevice, 0>(m_mem);
    } else {
      std::array<std::size_t, Rank - 1> shape{};
      std::array<std::size_t, Rank - 1> strides{};
      for (std::size_t i = 0, j = 0; i < Rank; ++i) {
        if (i != Dim) {
          shape[j] = m_shape[i];
          strides[j++] = m_strides[i];
        }
      }
      return TensorView<TElem, TDevice, Rank - 1>(m_mem, Shape<Rank - 1>(shape),
                                                  strides);
    }
  }

  template <std::size_t Dim>
    requires(Dim <= Rank)
  auto unsqueeze() const -> TensorView<TElem, TDevice, Rank + 1> {
    std::array<std::size_t, Rank + 1> shape{};
    std::array<std::size_t, Rank + 1> strides{};
    for (std::size_t i = 0, j = 0; i <= Rank; ++i) {
      if (i == Dim) {
        shape[i] = 1;
        strides[i] = Dim < Rank ? m_strides[Dim] * m_shape[Dim] : 1;
      } else {
        shape[i] = m_shape[j];
        strides[i] = m_strides[j++];
      }
    }
    return TensorView<TElem, TDevice, Rank + 1>(m_mem, Shape<Rank + 1>(shape),
                                                strides);
  }

protected:
  ContiguousMemory<TElem, TDevice> m_mem;
  Shape<Rank> m_shape;
  Strides m_strides;
};

template <typename TElem, typename TDevice>
class TensorView<TElem, TDevice, 0> {
  static_assert(std::is_same_v<std::remove_cvref_t<TElem>, TElem>);
  static_assert(std::is_same_v<TDevice, Device::CPU>,
                "Tensor views exist only on CPU");

public:
  using ElementType = TElem;
  using DeviceType = TDevice;
  using Strides = std::array<std::size_t, 0>;
  static constexpr std::size_t rank = 0;

  explicit TensorView(ContiguousMemory<TElem, TDevice> mem)
      : m_mem(std::move(mem)) {}

  [[nodiscard]] auto shape() const noexcept -> Shape<0> { return Shape<0>(); }
  [[nodiscard]] auto strides() const noexcept -> Strides { return {}; }
  [[nodiscard]] constexpr auto size() const -> std::size_t { return 1; }
  [[nodiscard]] auto unique() const -> bool { return not m_mem.isShared(); }
  [[nodiscard]] auto data() const -> const TElem * { return m_mem.ptr(); }
  [[nodiscard]] constexpr auto isContiguous() const -> bool { return true; }

  auto value() const -> const TElem & { return *data(); }

  auto view() const -> TensorView { return *this; }

  auto clone() const -> Tensor<TElem, TDevice, 0> {
    return Tensor<TElem, TDevice, 0>(value());
  }

private:
  ContiguousMemory<TElem, TDevice> m_mem;
};

// The view that t[...] returns on a tensor that may be written. Assigning to
// it writes the tensor's elements, and compiles only on the temporary.
// It holds a share of the memory like any view, so a held selection blocks
// writes to the tensor and outlives it
template <typename TElem, typename TDevice, std::size_t Rank>
class Selection : public TensorView<TElem, TDevice, Rank> {
  using Base = TensorView<TElem, TDevice, Rank>;
  template <typename, typename, std::size_t> friend class Tensor;

  explicit Selection(Base view) : Base(std::move(view)) {}

public:
  Selection(const Selection &) = default;
  Selection(Selection &&) noexcept = default;

  template <typename TValue>
    requires(!VenusTensor<TValue>) && std::convertible_to<TValue, TElem>
  void operator=(const TValue &value) && {
    requireSoleOwner();
    write([&value](std::size_t) -> const TValue & { return value; });
  }

  // other broadcasts to the selection's shape
  template <VenusTensor TOther>
    requires(TOther::rank <= Rank)
  void operator=(const TOther &other) && {
    requireSoleOwner();
    if constexpr (TOther::rank == 0) {
      write([value = other.value()](std::size_t) { return value; });
    } else {
      const auto source = other.view().expand(this->m_shape);
      write([it = source.begin()](std::size_t i) { return it[i]; });
    }
  }

  void operator=(const Selection &other) && {
    std::move(*this) = static_cast<const Base &>(other);
  }

private:
  // The tensor and this selection own the memory. Any other owner is a view
  // that would see the write
  void requireSoleOwner() const {
    if (this->m_mem.useCount() > 2) {
      throw std::runtime_error("Cannot write to shared tensor");
    }
  }

  template <typename Fn> void write(Fn element) {
    auto *base = this->m_mem.ptr();
    const auto ext = detail::extents(this->m_shape);
    for (std::size_t i = 0; i < this->size(); ++i) {
      base[strided_offset(i, ext, this->m_strides)] = element(i);
    }
  }
};

#define REGISTER_VIEW_OPERATOR(op, fn)                                         \
  template <AnyTensorView TView, typename TOther>                              \
    requires(Scalar<TOther> || VenusTensor<TOther>)                            \
  auto operator op(const TView &view, const TOther &other) {                   \
    return eager::fn(view, other);                                             \
  }                                                                            \
  template <Scalar TScalar, AnyTensorView TView>                               \
  auto operator op(const TScalar &scalar, const TView &view) {                 \
    return eager::fn(scalar, view);                                            \
  }

REGISTER_VIEW_OPERATOR(+, add)
REGISTER_VIEW_OPERATOR(-, sub)
REGISTER_VIEW_OPERATOR(*, mul)
REGISTER_VIEW_OPERATOR(/, div)
REGISTER_VIEW_OPERATOR(>, gt)
REGISTER_VIEW_OPERATOR(>=, gte)
REGISTER_VIEW_OPERATOR(<, lt)
REGISTER_VIEW_OPERATOR(<=, lte)
REGISTER_VIEW_OPERATOR(==, eq)
REGISTER_VIEW_OPERATOR(!=, neq)

#undef REGISTER_VIEW_OPERATOR

} // namespace venus
