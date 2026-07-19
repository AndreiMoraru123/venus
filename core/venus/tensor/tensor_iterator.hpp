#pragma once

#include <compare>
#include <iterator>

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
} // namespace venus