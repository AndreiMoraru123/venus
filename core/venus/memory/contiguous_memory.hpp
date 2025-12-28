#pragma once

#include <cassert>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <venus/memory/allocators.hpp>

namespace venus {

template <typename TElem, typename TDevice> class ContiguousMemory {
  static_assert(std::is_same_v<std::remove_cvref_t<TElem>, TElem>);
  using ElementType = TElem;

#ifdef VENUS_INTERPRETER
  template <typename, typename, std::size_t> friend class Tensor;

private:
#else
public:
#endif
  explicit ContiguousMemory(std::size_t p_size)
      : m_mem(Allocator<TDevice>::template Allocate<ElementType>(p_size)),
        m_size(p_size) {
    if (p_size == 0) {
      throw std::invalid_argument("Cannot allocate zero-sized memory.");
    }
  }

  auto Shift(size_t pos) const {
    assert(pos < m_size);
    return ContiguousMemory(
        std::shared_ptr<ElementType>(m_mem, m_mem.get() + pos), m_size - pos);
  }

public:
  auto RawMemory() -> ElementType * { return m_mem.get(); }
  auto RawMemory() const -> const ElementType * { return m_mem.get(); }
  [[nodiscard]] auto IsShared() const -> bool { return m_mem.use_count() > 1; }
  [[nodiscard]] auto Size() const -> std::size_t { return m_size; }

  auto operator==(const ContiguousMemory &val) const -> bool {
    return (m_mem == val.m_mem) and (m_size == val.m_size);
  }

private:
  ContiguousMemory(std::shared_ptr<ElementType> ptr, std::size_t size)
      : m_mem(std::move(ptr)), m_size(size) {}
  std::shared_ptr<ElementType> m_mem;
  std::size_t m_size;
};

} // namespace venus