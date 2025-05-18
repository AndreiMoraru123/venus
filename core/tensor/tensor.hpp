#pragma once
#include "../memory/contiguous_memory.hpp"
#include "../memory/lower_access.hpp"
#include "../traits.hpp"
#include "core/memory/device.hpp"
#include "core/tensor/shape.hpp"
#include <cassert>
#include <cstddef>
#include <format>
#include <stdexcept>
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

  explicit Tensor(const venus::Shape<Dim> &shape)
      : m_shape(shape), m_mem(shape.Count()) {}

  explicit Tensor(ContiguousMemory<ElementType, DeviceType> p_mem,
                  venus::Shape<Dim> p_shape)
      : m_shape(std::move(p_shape)), m_mem(std::move(p_mem)) {
    if (m_mem.Size() < m_shape.Count()) {
      throw std::invalid_argument(
          std::format("Insufficient memory for tensor shape: need {} elements, "
                      "but only {} provided",
                      m_shape.Count(), m_mem.Size()));
    }
  }

  const auto &Shape() const noexcept { return m_shape; }

  auto AvailableForWrite() const -> bool { return not m_mem.IsShared(); }

  void SetValue(ElementType val) {
    assert(AvailableForWrite());
    m_mem.RawMemory()[0] = val;
  }

  auto operator==(const Tensor &tensor) const -> bool {
    return (m_shape == tensor.m_shape) && (m_mem == tensor.m_mem);
  }

  // Tensor indexing for reading
  template <typename... Indices>
    requires(sizeof...(Indices) == Dim)
  constexpr auto operator[](Indices... indices) const -> const ElementType & {
    static_assert(std::is_same_v<DeviceType, Device::CPU>,
                  "Indexing is currently only supported on CPU");
    const auto offset =
        m_shape.IndexToOffset(static_cast<std::size_t>(indices)...);
    return (m_mem.RawMemory())[offset];
  }

  // Tensor indexing for assignment
  template <typename... Indices>
    requires(sizeof...(Indices) == Dim)
  constexpr auto operator[](Indices... indices) -> ElementType & {
    static_assert(std::is_same_v<DeviceType, Device::CPU>,
                  "Indexing is currently only supported on CPU");
    const auto offset =
        m_shape.IndexToOffset(static_cast<std::size_t>(indices)...);
    return (m_mem.RawMemory())[offset];
  }

  auto EvalRegister() const;

  auto LowLevel() const {
    using ThisType = RemoveConstRef<decltype(*this)>;
    return LowLevelAccess<ThisType>(*this);
  }

private:
  ContiguousMemory<ElementType, DeviceType> m_mem;
  venus::Shape<Dim> m_shape;
};

// Scalar Tensor ===============================================
template <typename TElem, typename TDevice> class Tensor<TElem, TDevice, 0> {
  static_assert(std::is_same_v<RemoveConstRef<TElem>, TElem>);

public:
  using ElementType = TElem;
  using DeviceType = TDevice;

  friend struct LowLevelAccess<Tensor>;

  explicit Tensor(ElementType elem = ElementType()) : m_mem(1) {
    SetValue(elem);
  }

  explicit Tensor(venus::Shape<0>) : Tensor() {};

  explicit Tensor(ContiguousMemory<ElementType, DeviceType> p_mem)
      : m_mem(std::move(p_mem)) {
    assert(m_mem.Size() >= 1);
  }

  const auto &Shape() const noexcept {
    static const venus::Shape<0> shape;
    return shape;
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
