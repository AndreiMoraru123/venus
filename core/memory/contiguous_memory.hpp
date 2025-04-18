#pragma once

#include "../traits.hpp"
#include "allocators.hpp"
#include <cassert>
#include <cstddef>
#include <memory>
#include <type_traits>

namespace venus {

template <typename TElem, typename TDevice> class ContiguousMemory {
  static_assert(std::is_same_v<RemoveConstRef<TElem>, TElem>);
  using ElementType = TElem;

public:
  explicit ContiguousMemory(std::size_t p_size)
      : m_mem(Allocator<TDevice>::template Allocate<ElementType>(p_size)),
        m_size(p_size) {}

  auto Shift(size_t pos) const {
    assert(pos < m_size);
    return ContiguousMemory(
        std::shared_ptr<ElementType>(m_mem, m_mem.get() + pos), m_size - pos);
  }

  auto RawMemory() const { return m_mem.get(); }
  bool IsShared() const { return m_mem.use_count() > 1; }
  std::size_t Size() const { return m_size; }

  bool operator==(const ContiguousMemory &val) const {
    return (m_mem == val.m_mem) and (m_size == val.m_size);
  }

  ContiguousMemory(ContiguousMemory &&other) noexcept
      : m_mem(std::move(other.m_mem)), m_size(other.m_size) {
    other.m_size = 0;
    other.m_mem.reset();
  }

private:
  ContiguousMemory(std::shared_ptr<ElementType> ptr, std::size_t sz)
      : m_mem(std::move(ptr)), m_size(sz) {}
  std::shared_ptr<ElementType> m_mem;
  std::size_t m_size;
};

} // namespace venus