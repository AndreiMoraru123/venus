#pragma once
#include "../memory/contiguous_memory.hpp"
#include "../memory/lower_access.hpp"
#include "../traits.hpp"
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace venus {

template <typename TElem, typename TDevice, std::size_t Dim> class Tensor {
  static_assert(std::is_same_v<RemoveConstRef<TElem>, TElem>);
  static_assert(Dim > 0);

public:
  using ElementType = TElem;
  using DeviceType = TDevice;

  friend struct LowLevelAccess<Tensor>;

  auto AvailableForWrite() const -> bool { return not m_mem.IsShared(); }

  void SetValue(ElementType val) {
    assert(AvailableForWrite());
    m_mem.RawMemory()[0] = val;
  }

  auto Value() const noexcept { return m_mem.RawMemory()[0]; }

  auto operator==(const Tensor &tensor) const noexcept -> bool {
    return Value() == tensor.Value();
  }

  auto EvalRegister() const;

  auto LowLevel() const {
    using ThisType = RemoveConstRef<decltype(*this)>;
    return LowLevelAccess<ThisType>(*this);
  }

private:
  ContiguousMemory<ElementType, DeviceType> m_mem;
};

template <typename TElem, typename TDevice> class Tensor<TElem, TDevice, 0> {
  static_assert(std::is_same_v<RemoveConstRef<TElem>, TElem>);

public:
  using ElementType = TElem;
  using DeviceType = TDevice;

  friend struct LowLevelAccess<Tensor>;

  explicit Tensor(ElementType elem = ElementType()) : m_mem(1) {
    SetValue(elem);
  }

  explicit Tensor(ContiguousMemory<ElementType, DeviceType> p_mem)
      : m_mem(std::move(p_mem)) {
    assert(m_mem.Size() >= 1);
  }

  auto AvailableForWrite() const -> bool { return not m_mem.IsShared(); }

  void SetValue(ElementType val) {
    assert(AvailableForWrite());
    m_mem.RawMemory()[0] = val;
  }

  auto Value() const noexcept { return m_mem.RawMemory()[0]; }

  auto operator==(const Tensor &tensor) const noexcept -> bool {
    return Value() == tensor.Value();
  }

  auto EvalRegister() const;

  auto LowLevel() const {
    using ThisType = RemoveConstRef<decltype(*this)>;
    return LowLevelAccess<ThisType>(*this);
  }

private:
  ContiguousMemory<ElementType, DeviceType> m_mem;
};

template <typename TElem, typename TDevice, std::size_t Dim>
struct LowLevelAccess<Tensor<TElem, TDevice, Dim>> {
  LowLevelAccess(Tensor<TElem, TDevice, Dim> p) : m_tensor(std::move(p)) {}

  auto RawMemory() const -> const TElem * { return m_tensor.m_mem.RawMemory(); }
  auto SharedMemory() const { return m_tensor.m_mem; }

private:
  Tensor<TElem, TDevice, Dim> m_tensor;
};

} // namespace venus
