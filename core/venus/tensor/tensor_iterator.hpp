#pragma once

#ifndef VENUS_MODULE
#include <array>
#include <compare>
#include <cstddef>
#include <iterator>
#endif

namespace venus {

template <typename T> class TensorIterator {
public:
  using iterator_category = std::contiguous_iterator_tag;
  using value_type = T::ElementType;
  using difference_type = std::ptrdiff_t;
  using pointer =
      std::conditional_t<std::is_const_v<T>, const value_type *, value_type *>;
  using reference =
      std::conditional_t<std::is_const_v<T>, const value_type &, value_type &>;

private:
  T *m_tensor;
  std::size_t m_offset;

public:
  constexpr TensorIterator() : m_tensor(nullptr), m_offset(0) {}
  constexpr TensorIterator(T *tensor, std::size_t offset)
      : m_tensor(tensor), m_offset(offset) {}

  constexpr auto operator*() const -> reference {
    return m_tensor->data()[m_offset];
  };

  constexpr auto operator->() const -> pointer {
    return &(m_tensor->data()[m_offset]);
  }

  constexpr auto operator++() -> TensorIterator & {
    ++m_offset;
    return *this;
  }

  constexpr auto operator++(int) -> TensorIterator {
    auto temp = *this;
    ++m_offset;
    return temp;
  }

  constexpr auto operator--() -> TensorIterator & {
    --m_offset;
    return *this;
  }

  constexpr auto operator--(int) -> TensorIterator {
    auto temp = *this;
    --m_offset;
    return temp;
  }

  constexpr auto operator+=(difference_type n) -> TensorIterator & {
    m_offset += n;
    return *this;
  }

  constexpr auto operator-=(difference_type n) -> TensorIterator & {
    m_offset -= n;
    return *this;
  }

  constexpr auto operator+(difference_type n) -> TensorIterator {
    return TensorIterator(m_tensor, m_offset + n);
  }

  constexpr auto operator+(difference_type n) const -> TensorIterator {
    return TensorIterator(m_tensor, m_offset + n);
  }

  constexpr auto operator-(difference_type n) -> TensorIterator {
    return TensorIterator(m_tensor, m_offset - n);
  }

  constexpr auto operator-(difference_type n) const -> TensorIterator {
    return TensorIterator(m_tensor, m_offset - n);
  }

  constexpr auto operator-(const TensorIterator &other) const
      -> difference_type {
    return static_cast<difference_type>(m_offset) -
           static_cast<difference_type>(other.m_offset);
  }

  constexpr auto operator==(const TensorIterator &other) const -> bool {
    return m_tensor == other.m_tensor && m_offset == other.m_offset;
  }

  constexpr auto operator<=>(const TensorIterator &other) const {
    if (m_tensor != other.m_tensor) {
      // comparing different tensors alltogether
      return std::compare_three_way{}(m_tensor, other.m_tensor);
    }
    // comparing offsets on the same tensor
    return m_offset <=> other.m_offset;
  }

  constexpr auto operator[](difference_type n) const -> reference {
    return *(*this + n);
  }
};

namespace detail {
// Offset of the element at position flat in row-major order over extents
template <std::size_t Rank>
constexpr auto strided_offset(std::size_t flat,
                              const std::array<std::size_t, Rank> &extents,
                              const std::array<std::size_t, Rank> &strides)
    -> std::size_t {
  std::size_t offset = 0;
  for (std::size_t i = Rank; i-- > 0;) {
    offset += (flat % extents[i]) * strides[i];
    flat /= extents[i];
  }
  return offset;
}
} // namespace detail

// Walks a tensor view in logical order. It copies the view's layout, so it
// stays valid while any owner keeps the memory alive
template <typename TElem, std::size_t Rank> class TensorViewIterator {
public:
  using iterator_concept = std::random_access_iterator_tag;
  using iterator_category = std::random_access_iterator_tag;
  using value_type = TElem;
  using difference_type = std::ptrdiff_t;
  using pointer = const TElem *;
  using reference = const TElem &;

private:
  using Layout = std::array<std::size_t, Rank>;
  const TElem *m_base = nullptr;
  Layout m_extents{};
  Layout m_strides{};
  std::size_t m_index = 0;

public:
  constexpr TensorViewIterator() = default;
  constexpr TensorViewIterator(const TElem *base, const Layout &extents,
                               const Layout &strides, std::size_t index)
      : m_base(base), m_extents(extents), m_strides(strides), m_index(index) {}

  constexpr auto operator*() const -> reference {
    return m_base[detail::strided_offset(m_index, m_extents, m_strides)];
  }

  constexpr auto operator->() const -> pointer { return &**this; }

  constexpr auto operator[](difference_type n) const -> reference {
    return *(*this + n);
  }

  constexpr auto operator++() -> TensorViewIterator & {
    ++m_index;
    return *this;
  }

  constexpr auto operator++(int) -> TensorViewIterator {
    auto temp = *this;
    ++m_index;
    return temp;
  }

  constexpr auto operator--() -> TensorViewIterator & {
    --m_index;
    return *this;
  }

  constexpr auto operator--(int) -> TensorViewIterator {
    auto temp = *this;
    --m_index;
    return temp;
  }

  constexpr auto operator+=(difference_type n) -> TensorViewIterator & {
    m_index += static_cast<std::size_t>(n);
    return *this;
  }

  constexpr auto operator-=(difference_type n) -> TensorViewIterator & {
    m_index -= static_cast<std::size_t>(n);
    return *this;
  }

  friend constexpr auto operator+(TensorViewIterator it, difference_type n)
      -> TensorViewIterator {
    return it += n;
  }

  friend constexpr auto operator+(difference_type n, TensorViewIterator it)
      -> TensorViewIterator {
    return it += n;
  }

  friend constexpr auto operator-(TensorViewIterator it, difference_type n)
      -> TensorViewIterator {
    return it -= n;
  }

  friend constexpr auto operator-(const TensorViewIterator &lhs,
                                  const TensorViewIterator &rhs)
      -> difference_type {
    return static_cast<difference_type>(lhs.m_index) -
           static_cast<difference_type>(rhs.m_index);
  }

  friend constexpr auto operator==(const TensorViewIterator &lhs,
                                   const TensorViewIterator &rhs) -> bool {
    return lhs.m_base == rhs.m_base && lhs.m_index == rhs.m_index;
  }

  friend constexpr auto operator<=>(const TensorViewIterator &lhs,
                                    const TensorViewIterator &rhs) {
    return lhs.m_index <=> rhs.m_index;
  }
};

} // namespace venus
