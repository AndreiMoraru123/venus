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

#define DEFINE_COMPOUND_OPERATOR(op)                                           \
  ElementProxy &operator op##=(const ElementType & value) {                    \
    if (!m_tensor.HasUniqueMemory()) {                                         \
      throw std::runtime_error("Cannot write to shared tensor");               \
    }                                                                          \
    m_element op## = value;                                                    \
    return *this;                                                              \
  }

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

  auto HasUniqueMemory() const -> bool { return not m_mem.IsShared(); }

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

  //* Proxy pattern for indexing elements (know when I'm reading vs writing)
  //? Price to pay: have to specify all possible operator overloads that I want
  class ElementProxy {
  private:
    Tensor &m_tensor;
    ElementType &m_element;

  public:
    ElementProxy(Tensor &tensor, ElementType &element)
        : m_tensor(tensor), m_element(element) {}

    // reading
    operator ElementType() const { return m_element; }

    // writing
    ElementProxy &operator=(const ElementType &value) {
      if (not m_tensor.HasUniqueMemory()) {
        //? Do I want to throw here or do I want copy on write (cow)
        throw std::runtime_error("Cannot write to shared tensor");
        //     const std::size_t offset = &m_element -
        //     m_tensor.m_mem.RawMemory();
        // auto new_mem =
        //     ContiguousMemory<ElementType, DeviceType>(m_tensor.m_mem.Size());
        // std::copy_n(m_tensor.m_mem.RawMemory(), m_tensor.m_mem.Size(),
        //             new_mem.RawMemory());
        // m_tensor.m_mem = std::move(new_mem);
        // m_tensor.m_mem.RawMemory()[offset] = value;
        // return *this;
      }
      m_element = value;
      return *this;
    }

    DEFINE_COMPOUND_OPERATOR(+)
    DEFINE_COMPOUND_OPERATOR(-)
    DEFINE_COMPOUND_OPERATOR(*)
    DEFINE_COMPOUND_OPERATOR(/)
    DEFINE_COMPOUND_OPERATOR(%)
  };

  // Tensor indexing for assignment
  template <typename... Indices>
    requires(sizeof...(Indices) == Dim)
  constexpr auto operator[](Indices... indices) -> ElementProxy {
    static_assert(std::is_same_v<DeviceType, Device::CPU>,
                  "Indexing is currently only supported on CPU");
    const auto offset =
        m_shape.IndexToOffset(static_cast<std::size_t>(indices)...);
    return ElementProxy(*this, (m_mem.RawMemory())[offset]);
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

  auto HasUniqueMemory() const -> bool { return not m_mem.IsShared(); }

  void SetValue(ElementType val) {
    assert(HasUniqueMemory());
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
