#pragma once

#ifndef VENUS_MODULE
#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <format>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <utility>
#endif
#include <venus/concepts.hpp>
#include <venus/memory/contiguous_memory.hpp>
#include <venus/memory/lower_access.hpp>
#include <venus/nested_initializer_list.hpp>
#include <venus/runtime.hpp>
#include <venus/tensor/cpu.hpp>
#include <venus/tensor/eager.hpp>
#include <venus/tensor/ops.hpp>
#include <venus/tensor/shape.hpp>
#include <venus/tensor/storage.hpp>
#include <venus/tensor/tensor_iterator.hpp>

#define REGISTER_PRE_OPERATOR(op)                                              \
  auto operator op()->ElementProxy & {                                         \
    if (not m_tensor.unique()) {                                               \
      throw std::runtime_error("Cannot write to shared tensor");               \
    }                                                                          \
    op m_element;                                                              \
    return *this;                                                              \
  }

#define REGISTER_POST_OPERATOR(op)                                             \
  auto operator op(int)->ElementType {                                         \
    if (not m_tensor.unique()) {                                               \
      throw std::runtime_error("Cannot write to shared tensor");               \
    }                                                                          \
    ElementType old_value = m_element;                                         \
    m_element op;                                                              \
    return old_value;                                                          \
  }

#define REGISTER_OPERATOR_EQUAL(op)                                            \
  auto operator op## = (const ElementType &value)->ElementProxy & {            \
    if (not m_tensor.unique()) {                                               \
      throw std::runtime_error("Cannot write to shared tensor");               \
    }                                                                          \
    m_element op## = value;                                                    \
    return *this;                                                              \
  }

namespace venus {

template <typename TElem, typename TRuntime, std::size_t Rank> class Tensor {
  static_assert(std::is_same_v<std::remove_cvref_t<TElem>, TElem>);
  static_assert(Rank > 0);

public:
  using ElementType = TElem;
  using RuntimeType = TRuntime;
  using StorageType = storage_t<TElem, TRuntime, Rank>;
  static constexpr std::size_t rank = Rank;

  friend struct LowLevelAccess<Tensor>;
  friend struct LowLevelAccess<const Tensor>;

  explicit Tensor(Shape<Rank> shape)
    requires std::constructible_from<StorageType, std::size_t>
      : m_shape(std::move(shape)), m_mem(shape.count()) {}

  explicit Tensor(StorageType p_mem, Shape<Rank> p_shape)
      : m_shape(std::move(p_shape)), m_mem(std::move(p_mem)) {
    if constexpr (requires { m_mem.size(); }) {
      if (m_mem.size() < m_shape.count()) {
        throw std::invalid_argument(std::format(
            "Insufficient memory for tensor shape: need {} elements, "
            "but only {} provided",
            m_shape.count(), m_mem.size()));
      }
    }
  }

  explicit Tensor(nested_initializer_list_t<ElementType, Rank> init_list)
    requires Writable<TRuntime>
      : m_shape(Shape<Rank>::fromNestedInitializerList(init_list)),
        m_mem(m_shape.count()) {

    auto flatten = [](const auto &list, ElementType *output_ptr,
                      const auto &self_ref) -> ElementType * {
      if constexpr (std::same_as<std::decay_t<decltype(*list.begin())>,
                                 ElementType>) {
        return std::ranges::copy(list, output_ptr).out;
      } else {
        for (const auto &inner_list : list) {
          output_ptr = self_ref(inner_list, output_ptr, self_ref);
        }
        return output_ptr;
      }
    };

    flatten(init_list, data(), flatten);
  }

  template <std::size_t D = Rank>
    requires(D == 1) && Writable<TRuntime>
  explicit Tensor(std::initializer_list<ElementType> init_list)
      : m_shape(init_list.size()), m_mem(init_list.size()) {
    std::ranges::copy(init_list, data());
  }

  template <std::size_t D = Rank>
    requires(D != 1)
  explicit Tensor(std::initializer_list<ElementType>) = delete;

  template <typename... Dims>
    requires(sizeof...(Dims) == Rank) &&
            (std::is_convertible_v<Dims, std::size_t> && ...) &&
            std::constructible_from<StorageType, std::size_t>
  explicit Tensor(Dims &&...dimensions)
      : Tensor(Shape<Rank>(std::forward<Dims>(dimensions)...)) {}

  template <typename... Dims>
    requires(sizeof...(Dims) != Rank) &&
                (std::is_convertible_v<Dims, std::size_t> && ...)
  explicit Tensor(Dims &&...) = delete;

  // A copy on a writable runtime copies the elements. Elsewhere it copies
  // the storage, which shares what it refers to
  auto operator=(const Tensor &other) -> Tensor &
    requires Writable<TRuntime>
  {
    if (this != &other) {
      if (not unique() || m_shape.count() != other.m_shape.count()) {
        m_mem = StorageType(other.m_shape.count());
      }
      m_shape = other.m_shape;
      std::ranges::copy(other, this->begin());
    }
    return *this;
  }

  auto operator=(Tensor &&other) noexcept -> Tensor & {
    if (this != &other) {
      m_shape = other.m_shape;
      m_mem = std::move(other.m_mem);
    }
    return *this;
  }

  auto operator=(const Tensor &other) -> Tensor & = default;

  Tensor(const Tensor &other)
    requires Writable<TRuntime>
      : m_shape(other.m_shape), m_mem(m_shape.count()) {
    std::ranges::copy(other, this->begin());
  }

  Tensor(const Tensor &other) = default;

  Tensor(Tensor &&other) noexcept
      : m_shape(other.m_shape), m_mem(std::move(other.m_mem)) {}

  ~Tensor() = default; // give back to mem-pool

  [[nodiscard]] auto shape() const noexcept -> Shape<rank> { return m_shape; }

  // What a runtime's actions read and build tensors from
  [[nodiscard]] auto storage() const noexcept -> const StorageType & {
    return m_mem;
  }

  [[nodiscard]] auto unique() const -> bool
    requires Writable<TRuntime>
  {
    return not m_mem.isShared();
  }

  auto clone() const -> Tensor { return Tensor(*this); }

  auto asScalar() const -> Tensor<TElem, TRuntime, 0>
    requires Writable<TRuntime>
  {
    if (size() != 1) {
      throw std::runtime_error(std::format(
          "asScalar can only be called on tensors with exactly 1 element."
          "Tensor size is {}, while the size of a scalar is 1.",
          size()));
    }
    return Tensor<TElem, TRuntime, 0>(*(*this).data());
  }

  // Inner product
  template <typename Other> auto inner(const Other &other) const {
    return venus::inner(*this, other);
  }

  // Dot product
  template <typename Other>
    requires(Rank == 1)
  auto dot(const Other &other) const {
    return venus::dot(*this, other);
  }

  // In-Place Transform
  template <typename Fn>
  void transform(this auto &&self, Fn &&fn)
    requires(!std::is_const_v<std::remove_reference_t<decltype(self)>>) &&
            Writable<TRuntime>
  {
    std::ranges::transform(self, self.begin(), std::forward<Fn>(fn));
  }

  // In-Place Sort
  void sort(this auto &&self)
    requires(!std::is_const_v<std::remove_reference_t<decltype(self)>>) &&
            Writable<TRuntime>
  {
    std::ranges::sort(self);
  }

  // In-Place Fill
  template <venus::Scalar Idx>
  void fill(this auto &&self, Idx i)
    requires(!std::is_const_v<std::remove_reference_t<decltype(self)>>) &&
            Writable<TRuntime>
  {
#if _cpp_lib_ranges >= 202110L
    std::ranges::fill(self, i);
#else
    std::fill(self.begin(), self.end(), i);
#endif
  }

  // In-Place Arange
  template <venus::Scalar Idx>
  void iota(this auto &&self, Idx i)
    requires(!std::is_const_v<std::remove_reference_t<decltype(self)>>) &&
            Writable<TRuntime>
  {
#if _cpp_lib_ranges >= 202110L
    std::ranges::iota(self, i);
#else
    std::iota(self.begin(), self.end(), i);
#endif
  }

  // In-Place Identity
  void eye(this auto &&self)
    requires(!std::is_const_v<std::remove_reference_t<decltype(self)>>) &&
            Writable<TRuntime>
  {
    for (auto [flat, val] :
         std::views::zip(std::views::iota(std::size_t{0}, self.size()), self)) {
      auto midx = self.m_shape.offsetToIdx(flat);
      if (midx[Rank - 2] == midx[Rank - 1]) {
        val = 1;
      } else {
        val = 0;
      }
    }
  }

  static auto eye(const Shape<rank> &shape) {
    auto tensor = Tensor(shape);
    tensor.eye();
    return tensor;
  }

  // * Proxy pattern for indexing elements (know when I'm reading vs writing)
  // ? Price to pay: have to specify all possible operator overloads that I want
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
    auto operator=(const ElementType &value) -> ElementProxy & {
      if (not m_tensor.unique()) {
        // TODO: Do I want to throw here or do I want copy on write (cow)
        throw std::runtime_error("Cannot write to shared tensor");
        //     const std::size_t offset = &m_element -
        //     m_tensor.m_mem.ptr();
        // auto new_mem =
        //     ContiguousMemory<ElementType,
        //     RuntimeType>(m_tensor.m_mem.Size());
        // std::copy_n(m_tensor.m_mem.ptr(), m_tensor.m_mem.Size(),
        //             new_mem.ptr());
        // m_tensor.m_mem = std::move(new_mem);
        // m_tensor.m_mem.ptr()[offset] = value;
        // return *this;
      }
      m_element = value;
      return *this;
    }

    auto operator=(ElementType &&value) -> ElementProxy & {
      if (not m_tensor.unique()) {
        throw std::runtime_error("Cannot write to shared tensor");
      }
      m_element = std::move(value);
      return *this;
    }

    // Required for modifying the tensor through range algos
    template <typename U>
      requires std::convertible_to<U, ElementType>
    auto operator=(U &&value) -> ElementProxy & {
      if (not m_tensor.unique()) {
        throw std::runtime_error("Cannot write to shared tensor");
      }
      m_element = std::forward<U>(value);
      return *this;
    }

    // explicit conversion (to extract the element by type)
    template <typename U> explicit operator U() const {
      return static_cast<U>(m_element);
    }

    REGISTER_OPERATOR_EQUAL(+)
    REGISTER_OPERATOR_EQUAL(-)
    REGISTER_OPERATOR_EQUAL(*)
    REGISTER_OPERATOR_EQUAL(/)
    REGISTER_OPERATOR_EQUAL(%)

    REGISTER_PRE_OPERATOR(++)
    REGISTER_PRE_OPERATOR(--)
    REGISTER_POST_OPERATOR(++)
    REGISTER_POST_OPERATOR(--)
  };

#ifdef VENUS_INTERPRETER
  // Simple direct indexing for interpreter mode (no proxy)
  template <typename... Indices>
    requires(sizeof...(Indices) == Rank) && Readable<TRuntime>
  constexpr auto operator[](this auto &&self, Indices... indices)
      -> ElementType {
    const auto offset =
        self.m_shape.idxToOffset(static_cast<std::size_t>(indices)...);
    return self.data()[offset];
  }

  auto operator[](this auto &&self,
                  const std::array<std::size_t, Rank> &indices)
    requires Readable<TRuntime>
  {
    const auto offset = self.m_shape.idxToOffset(indices);
    return self.data()[offset];
  }

#else
  // Tensor indexing
  template <typename... Indices>
    requires(sizeof...(Indices) == Rank) && Readable<TRuntime>
  auto operator[](this auto &&self, Indices... indices) {
    const auto offset =
        self.m_shape.idxToOffset(static_cast<std::size_t>(indices)...);
    if constexpr (std::is_const_v<std::remove_reference_t<decltype(self)>> ||
                  !writable<TRuntime>) {
      return self.data()[offset];
    } else {
      return ElementProxy(self, self.data()[offset]);
    }
  }

  auto operator[](this auto &&self,
                  const std::array<std::size_t, Rank> &indices)
    requires Readable<TRuntime>
  {
    const auto offset = self.m_shape.idxToOffset(indices);
    if constexpr (std::is_const_v<std::remove_reference_t<decltype(self)>> ||
                  !writable<TRuntime>) {
      return self.data()[offset];
    } else {
      return ElementProxy(self, self.data()[offset]);
    }
  }
#endif

  auto evalRegister() const;

  auto lowLevel(this auto &&self) {
    using Self = std::remove_reference_t<decltype(self)>;
    return LowLevelAccess<Self>(self);
  }

private:
  Shape<Rank> m_shape;
  StorageType m_mem;

public:
  constexpr auto begin(this auto &&self)
    requires Readable<TRuntime>
  {
    using Self = std::remove_reference_t<decltype(self)>;
    return TensorIterator<Self>(&self, 0);
  }

  constexpr auto end(this auto &&self)
    requires Readable<TRuntime>
  {
    using Self = std::remove_reference_t<decltype(self)>;
    return TensorIterator<Self>(&self, self.m_shape.count());
  }

  constexpr auto cbegin(this auto &&self) {
    return std::as_const(self).m_dims.begin();
  }
  constexpr auto cend(this auto &&self) {
    return std::as_const(self).m_dims.end();
  }

  [[nodiscard]] constexpr auto size() const -> std::size_t {
    return m_shape.count();
  }

  auto data(this auto &&self)
    requires Readable<TRuntime>
  {
    return std::forward<decltype(self)>(self).m_mem.ptr();
  }

  template <SizeTLike... Dimensions>
  auto reshape(this auto &&self, Dimensions... dims)
    requires Readable<TRuntime>
  {
    const auto new_shape = Shape(dims...);
    if (new_shape.count() != self.size()) {
      throw std::invalid_argument(std::format(
          "Cannot reshape tensor of size {} to new shape of size {}",
          self.size(), new_shape.count()));
    }

    return Tensor<ElementType, RuntimeType, new_shape.rank>(self.m_mem,
                                                            new_shape);
  }

  template <std::size_t NewRank>
  auto reshape(this auto &&self, Shape<NewRank> new_shape)
    requires Readable<TRuntime>
  {
    if (new_shape.count() != self.size()) {
      throw std::invalid_argument(std::format(
          "Cannot reshape tensor of size {} to new shape of size {}",
          self.size(), new_shape.count()));
    }

    return Tensor<ElementType, RuntimeType, NewRank>(self.m_mem, new_shape);
  }

  auto view(this auto &&self)
    requires Readable<TRuntime>
  {
    return Tensor<ElementType, RuntimeType, Rank>(self.m_mem, self.m_shape);
  }
};

// Scalar Tensor ===============================================
template <typename TElem, typename TRuntime> class Tensor<TElem, TRuntime, 0> {
  static_assert(std::is_same_v<std::remove_cvref_t<TElem>, TElem>);

public:
  using ElementType = TElem;
  using RuntimeType = TRuntime;
  using StorageType = storage_t<TElem, TRuntime, 0>;
  static constexpr std::size_t rank = 0;

  friend struct LowLevelAccess<Tensor>;
  friend struct LowLevelAccess<const Tensor>;

  explicit Tensor(ElementType value = ElementType())
    requires Writable<TRuntime>
      : m_mem(1) {
    assign(value);
  }

  explicit Tensor(Shape<0> /*unused*/)
    requires Writable<TRuntime>
      : Tensor() {};

  explicit Tensor(StorageType p_mem) : m_mem(std::move(p_mem)) {}

  // The same form as the higher ranks, so an action builds any rank alike
  explicit Tensor(StorageType p_mem, Shape<0> /*unused*/)
      : m_mem(std::move(p_mem)) {}

  auto operator=(const Tensor &other) -> Tensor &
    requires Writable<TRuntime>
  {
    if (this != &other) {
      if (not unique()) {
        m_mem = StorageType(1);
      }
      assign(other.value());
    }
    return *this;
  }

  auto operator=(Tensor &&other) noexcept -> Tensor & {
    if (this != &other) {
      m_mem = std::move(other.m_mem);
    }
    return *this;
  }

  auto operator=(const Tensor &other) -> Tensor & = default;

  Tensor(const Tensor &other)
    requires Writable<TRuntime>
      : m_mem(1) {
    assign(other.value());
  }

  Tensor(const Tensor &other) = default;

  Tensor(Tensor &&other) noexcept : m_mem(std::move(other.m_mem)) {}

  ~Tensor() = default; // give back to mem-pool

  [[nodiscard]] constexpr auto shape() const noexcept -> Shape<rank> {
    return Shape<rank>{};
  }

  [[nodiscard]] auto storage() const noexcept -> const StorageType & {
    return m_mem;
  }

  [[nodiscard]] auto unique() const -> bool
    requires Writable<TRuntime>
  {
    return not m_mem.isShared();
  }

  void assign(ElementType value) const = delete;

  void assign(ElementType value)
    requires Writable<TRuntime>
  {
    if (not unique()) {
      throw std::runtime_error("Cannot write to shared scalar tensor.");
    }
    data()[0] = value;
  }

  auto value() const noexcept
    requires Readable<TRuntime>
  {
    return data()[0];
  }

  // Same-type comparison reads both values. Other comparisons dispatch
  auto operator==(const Tensor &tensor) const noexcept -> bool
    requires Readable<TRuntime>
  {
    return value() == tensor.value();
  }

  // Inner product
  template <typename Other> auto inner(const Other &other) const {
    return venus::inner(*this, other);
  }

  operator bool() const noexcept
    requires Readable<TRuntime>
  {
    if constexpr (std::is_same_v<ElementType, bool>) {
      return value();
    } else {
      return value() != ElementType{};
    }
  }

  auto evalRegister() const;

  auto lowLevel(this auto &&self) {
    using Self = std::remove_reference_t<decltype(self)>;
    return LowLevelAccess<Self>(self);
  }

  [[nodiscard]] constexpr auto size() const { return 1; }

  auto data(this auto &&self)
    requires Readable<TRuntime>
  {
    return std::forward<decltype(self)>(self).m_mem.ptr();
  }

  auto view(this auto &&self)
    requires Readable<TRuntime>
  {
    return Tensor<ElementType, RuntimeType, 0>(self.m_mem);
  }

private:
  StorageType m_mem;
};

template <typename TElem, typename TRuntime, std::size_t Rank>
struct LowLevelAccess<Tensor<TElem, TRuntime, Rank>> {
  LowLevelAccess(Tensor<TElem, TRuntime, Rank> &tensor) : m_tensor(tensor) {}
  auto rawMemory() -> TElem * { return m_tensor.m_mem.ptr(); }
  auto sharedMemory() const { return m_tensor.m_mem; }

private:
  Tensor<TElem, TRuntime, Rank> &m_tensor;
};

template <typename TElem, typename TRuntime, std::size_t Rank>
struct LowLevelAccess<const Tensor<TElem, TRuntime, Rank>> {
  LowLevelAccess(const Tensor<TElem, TRuntime, Rank> &tensor)
      : m_tensor(tensor) {}
  auto rawMemory() const -> const TElem * { return m_tensor.m_mem.ptr(); }
  auto sharedMemory() const { return m_tensor.m_mem; }

private:
  const Tensor<TElem, TRuntime, Rank> &m_tensor;
};

// difference_type + iterator (for random_access_range | addition commutative)
template <typename T>
constexpr auto operator+(typename TensorIterator<T>::difference_type n,
                         const TensorIterator<T> &it) -> TensorIterator<T> {
  return it + n;
}

} // namespace venus

#undef REGISTER_OPERATOR_EQUAL
#undef REGISTER_POST_OPERATOR
#undef REGISTER_PRE_OPERATOR
