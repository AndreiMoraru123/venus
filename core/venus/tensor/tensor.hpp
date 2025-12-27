#pragma once
#include <algorithm>
#include <cassert>
#include <compare>
#include <concepts>
#include <cstddef>
#include <format>
#include <iomanip>
#include <ios>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <venus/memory/contiguous_memory.hpp>
#include <venus/memory/device.hpp>
#include <venus/memory/lower_access.hpp>
#include <venus/tensor/eager_ops.hpp>
#include <venus/tensor/shape.hpp>

#define REGISTER_SCALAR_OP(op)                                                 \
  auto operator op(const ElementType &element) const noexcept                  \
      -> Tensor<bool, DeviceType, 0> {                                         \
    return Tensor<bool, DeviceType, 0>(Value() op element);                    \
  }

#define REGISTER_PRE_OPERATOR(op)                                              \
  ElementProxy &operator op() {                                                \
    if (not m_tensor.Unique()) {                                               \
      throw std::runtime_error("Cannot write to shared tensor");               \
    }                                                                          \
    op m_element;                                                              \
    return *this;                                                              \
  }

#define REGISTER_POST_OPERATOR(op)                                             \
  ElementType operator op(int) {                                               \
    if (not m_tensor.Unique()) {                                               \
      throw std::runtime_error("Cannot write to shared tensor");               \
    }                                                                          \
    ElementType old_value = m_element;                                         \
    m_element op;                                                              \
    return old_value;                                                          \
  }

#define REGISTER_OPERATOR_EQUAL(op)                                            \
  ElementProxy &operator op## = (const ElementType &value) {                   \
    if (not m_tensor.Unique()) {                                               \
      throw std::runtime_error("Cannot write to shared tensor");               \
    }                                                                          \
    m_element op## = value;                                                    \
    return *this;                                                              \
  }

namespace venus {

template <typename T> class tensor_iterator {
public:
  using iterator_category = std::contiguous_iterator_tag; // do I need this?
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
  constexpr tensor_iterator() : m_tensor(nullptr), m_offset(0) {}
  constexpr tensor_iterator(T *tensor, std::size_t offset)
      : m_tensor(tensor), m_offset(offset) {}

  constexpr reference operator*() const {
    return m_tensor->LowLevel().RawMemory()[m_offset];
  };

  constexpr pointer operator->() const {
    return &(m_tensor->LowLevel().RawMemory()[m_offset]);
  }

  constexpr tensor_iterator &operator++() {
    ++m_offset;
    return *this;
  }

  constexpr tensor_iterator operator++(int) {
    auto temp = *this;
    ++m_offset;
    return temp;
  }

  constexpr tensor_iterator &operator--() {
    --m_offset;
    return *this;
  }

  constexpr tensor_iterator operator--(int) {
    auto temp = *this;
    --m_offset;
    return temp;
  }

  constexpr tensor_iterator &operator+=(difference_type n) {
    m_offset += n;
    return *this;
  }

  constexpr tensor_iterator &operator-=(difference_type n) {
    m_offset -= n;
    return *this;
  }

  constexpr tensor_iterator operator+(difference_type n) {
    return tensor_iterator(m_tensor, m_offset + n);
  }

  constexpr tensor_iterator operator+(difference_type n) const {
    return tensor_iterator(m_tensor, m_offset + n);
  }

  constexpr tensor_iterator operator-(difference_type n) {
    return tensor_iterator(m_tensor, m_offset - n);
  }

  constexpr tensor_iterator operator-(difference_type n) const {
    return tensor_iterator(m_tensor, m_offset - n);
  }

  constexpr difference_type operator-(const tensor_iterator &other) const {
    return static_cast<difference_type>(m_offset) -
           static_cast<difference_type>(other.m_offset);
  }

  constexpr bool operator==(const tensor_iterator &other) const {
    return m_tensor == other.m_tensor && m_offset == other.m_offset;
  }

  constexpr auto operator<=>(const tensor_iterator &other) const {
    if (m_tensor != other.m_tensor) {
      // comparing different tensors alltogether
      return std::compare_three_way{}(m_tensor, other.m_tensor);
    }
    // comparing offsets on the same tensor
    return m_offset <=> other.m_offset;
  }

  constexpr reference operator[](difference_type n) const {
    return *(*this + n);
  }
};

template <typename TElem, typename TDevice, std::size_t Dim> class Tensor {
  static_assert(std::is_same_v<std::remove_cvref_t<TElem>, TElem>);
  static_assert(Dim > 0);

public:
  using ElementType = TElem;
  using DeviceType = TDevice;
  static constexpr std::size_t Dimension = Dim;

  friend struct LowLevelAccess<Tensor>;
  friend struct LowLevelAccess<const Tensor>;

  explicit Tensor(venus::Shape<Dim> shape)
      : m_shape(std::move(shape)), m_mem(shape.Count()) {}

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

  template <std::size_t D = Dim>
    requires(D == 1)
  explicit Tensor(std::initializer_list<ElementType> init_list)
      : m_shape(init_list.size()), m_mem(init_list.size()) {
    std::copy(init_list.begin(), init_list.end(), m_mem.RawMemory());
  }

  template <std::size_t D = Dim>
    requires(D != 1)
  explicit Tensor(std::initializer_list<ElementType>) = delete;

  template <typename... Dims>
    requires(sizeof...(Dims) == Dim) &&
            (std::is_convertible_v<Dims, std::size_t> && ...)
  explicit Tensor(Dims &&...dimensions)
      : Tensor(venus::Shape<Dim>(std::forward<Dims>(dimensions)...)) {}

  template <typename... Dims>
    requires(sizeof...(Dims) != Dim) &&
                (std::is_convertible_v<Dims, std::size_t> && ...)
  explicit Tensor(Dims &&...) = delete;

  const auto &Shape() const noexcept { return m_shape; }

  auto Unique() const -> bool { return not m_mem.IsShared(); }

  auto Clone() const -> Tensor {
    Tensor copy_tensor(m_shape);
    std::ranges::copy(*this, copy_tensor.begin());
    return copy_tensor;
  }

  // Addition
  template <typename OtherType> auto operator+(OtherType &&other) const {
    return venus::ops::add(*this, std::forward<OtherType>(other));
  }

  // Subtraction
  template <typename OtherType> auto operator-(OtherType &&other) const {
    return venus::ops::sub(*this, std::forward<OtherType>(other));
  }

  // Multiplication
  template <typename OtherType> auto operator*(OtherType &&other) const {
    return venus::ops::mul(*this, std::forward<OtherType>(other));
  }

  // Division
  template <typename OtherType> auto operator/(OtherType &&other) const {
    return venus::ops::div(*this, std::forward<OtherType>(other));
  }

  // Greater than
  template <typename OtherType> auto operator>(OtherType &&other) const {
    return venus::ops::gt(*this, other);
  }

  // Greater or equal
  template <typename OtherType> auto operator>=(OtherType &&other) const {
    return venus::ops::gte(*this, other);
  }

  // Less than
  template <typename OtherType> auto operator<(OtherType &&other) const {
    return venus::ops::lt(*this, other);
  }

  // Less or equal
  template <typename OtherType> auto operator<=(OtherType &&other) const {
    return venus::ops::lte(*this, other);
  }

  // Equal
  template <typename OtherType> auto operator==(OtherType &&other) const {
    return venus::ops::eq(*this, other);
  }

  // Not equal
  template <typename OtherType> auto operator!=(OtherType &&other) const {
    return venus::ops::neq(*this, other);
  }

  // Dot product
  template <typename OtherElementType>
  auto dot(const Tensor<OtherElementType, DeviceType, Dim> &other) const {
    return venus::ops::dot(*this, other);
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
    const ElementProxy &operator=(const ElementType &value) const {
      if (not m_tensor.Unique()) {
        // TODO: Do I want to throw here or do I want copy on write (cow)
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

    const ElementProxy &operator=(ElementType &&value) const {
      if (not m_tensor.Unique()) {
        throw std::runtime_error("Cannot write to shared tensor");
      }
      m_element = std::move(value);
      return *this;
    }

    // Required for modifying the tensor through range algos
    template <typename U>
      requires std::convertible_to<U, ElementType>
    const ElementProxy &operator=(U &&value) const {
      if (not m_tensor.Unique()) {
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
    requires(sizeof...(Indices) == Dim)
  constexpr auto operator[](Indices... indices) -> ElementType {
    static_assert(std::is_same_v<DeviceType, Device::CPU>,
                  "Indexing is currently only supported on CPU");
    const auto offset =
        m_shape.IndexToOffset(static_cast<std::size_t>(indices)...);
    return m_mem.RawMemory()[offset];
  }
#else
  // Tensor indexing
  template <typename... Indices>
    requires(sizeof...(Indices) == Dim)
  constexpr auto operator[](Indices... indices) -> ElementProxy {
    static_assert(std::is_same_v<DeviceType, Device::CPU>,
                  "Indexing is currently only supported on CPU");
    const auto offset =
        m_shape.IndexToOffset(static_cast<std::size_t>(indices)...);
    return ElementProxy(*this, (m_mem.RawMemory())[offset]);
  }
#endif

  auto EvalRegister() const;

  auto LowLevel() { return LowLevelAccess<Tensor>(*this); }
  auto LowLevel() const { return LowLevelAccess<const Tensor>(*this); }

private:
  ContiguousMemory<ElementType, DeviceType> m_mem;
  venus::Shape<Dim> m_shape;

public:
  constexpr auto begin() {
    static_assert(std::is_same_v<DeviceType, Device::CPU>,
                  "Range iteration is currently only supported on CPU");
    return tensor_iterator<Tensor>(this, 0);
  }

  constexpr auto end() {
    static_assert(std::is_same_v<DeviceType, Device::CPU>,
                  "Range iteration is currently only supported on CPU");
    return tensor_iterator<Tensor>(this, m_shape.Count());
  }

  constexpr auto begin() const {
    static_assert(std::is_same_v<DeviceType, Device::CPU>,
                  "Range iteration is currently only supported on CPU");
    return tensor_iterator<const Tensor>(this, 0);
  }

  constexpr auto end() const {
    static_assert(std::is_same_v<DeviceType, Device::CPU>,
                  "Range iteration is currently only supported on CPU");
    return tensor_iterator<const Tensor>(this, m_shape.Count());
  }

  constexpr auto cbegin() const { return begin(); }
  constexpr auto cend() const { return end(); }

  constexpr std::size_t size() const { return m_shape.Count(); }
};

// Scalar Tensor ===============================================
template <typename TElem, typename TDevice> class Tensor<TElem, TDevice, 0> {
  static_assert(std::is_same_v<std::remove_cvref_t<TElem>, TElem>);

public:
  using ElementType = TElem;
  using DeviceType = TDevice;
  static constexpr std::size_t Dimension = 0;

  friend struct LowLevelAccess<Tensor>;
  friend struct LowLevelAccess<const Tensor>;

  explicit Tensor(ElementType value = ElementType()) : m_mem(1) {
    SetValue(value);
  }

  explicit Tensor(venus::Shape<0>) : Tensor() {};

  explicit Tensor(ContiguousMemory<ElementType, DeviceType> p_mem)
      : m_mem(std::move(p_mem)) {}

  const auto &Shape() const noexcept {
    static const venus::Shape<Dimension> shape;
    return shape;
  }

  auto Unique() const -> bool { return not m_mem.IsShared(); }

  void SetValue(ElementType value) const = delete;

  void SetValue(ElementType value) {
    if (not Unique()) {
      throw std::runtime_error("Cannot write to shared scalar tensor.");
    }
    m_mem.RawMemory()[0] = value;
  }

  auto Value() const noexcept { return m_mem.RawMemory()[0]; }

  auto operator==(const Tensor &tensor) const noexcept -> bool {
    return Value() == tensor.Value();
  }

  // Addition
  template <typename OtherType> auto operator+(OtherType &&other) const {
    return venus::ops::add(*this, std::forward<OtherType>(other));
  }

  // Subtraction
  template <typename OtherType> auto operator-(OtherType &&other) const {
    return venus::ops::sub(*this, std::forward<OtherType>(other));
  }

  // Multiplication
  template <typename OtherType> auto operator*(OtherType &&other) const {
    return venus::ops::mul(*this, std::forward<OtherType>(other));
  }

  // Division
  template <typename OtherType> auto operator/(OtherType &&other) const {
    return venus::ops::div(*this, std::forward<OtherType>(other));
  }

  // Dot product
  template <typename OtherElementType>
  auto dot(const Tensor<OtherElementType, DeviceType, Dimension> &other) const {
    return venus::ops::dot(*this, other);
  }

  operator bool() const noexcept {
    if constexpr (std::is_same_v<ElementType, bool>) {
      return Value();
    } else {
      return Value() != ElementType{};
    }
  }

  REGISTER_SCALAR_OP(==)
  REGISTER_SCALAR_OP(!=)
  REGISTER_SCALAR_OP(<)
  REGISTER_SCALAR_OP(<=)
  REGISTER_SCALAR_OP(>)
  REGISTER_SCALAR_OP(>=)

  auto EvalRegister() const;

  auto LowLevel() { return LowLevelAccess<Tensor>(*this); }
  auto LowLevel() const { return LowLevelAccess<const Tensor>(*this); }

private:
  ContiguousMemory<ElementType, DeviceType> m_mem;
};

template <typename TElem, typename TDevice, std::size_t Dim>
struct LowLevelAccess<Tensor<TElem, TDevice, Dim>> {
  LowLevelAccess(Tensor<TElem, TDevice, Dim> &p) : m_tensor(p) {}
  auto RawMemory() -> TElem * { return m_tensor.m_mem.RawMemory(); }
  auto SharedMemory() const { return m_tensor.m_mem; }

private:
  Tensor<TElem, TDevice, Dim> &m_tensor;
};

template <typename TElem, typename TDevice, std::size_t Dim>
struct LowLevelAccess<const Tensor<TElem, TDevice, Dim>> {
  LowLevelAccess(const Tensor<TElem, TDevice, Dim> &p) : m_tensor(p) {}
  auto RawMemory() const -> const TElem * { return m_tensor.m_mem.RawMemory(); }
  auto SharedMemory() const { return m_tensor.m_mem; }

private:
  const Tensor<TElem, TDevice, Dim> &m_tensor;
};

// difference_type + iterator (for random_access_range | addition commutative)
template <typename T>
constexpr tensor_iterator<T>
operator+(typename tensor_iterator<T>::difference_type n,
          const tensor_iterator<T> &it) {
  return it + n;
}

// Print Tensor ================================================
template <typename T>
concept StringLike = requires(T t) {
  { t.c_str() } -> std::convertible_to<const char *>;
} or std::convertible_to<T, const char *>;

template <typename T>
concept CharLike = std::same_as<T, char> || std::same_as<T, signed char> ||
                   std::same_as<T, unsigned char> || std::same_as<T, wchar_t> ||
                   std::same_as<T, char8_t> || std::same_as<T, char16_t> ||
                   std::same_as<T, char32_t>;

template <typename TElem, typename TDevice, std::size_t Dim>
std::ostream &operator<<(std::ostream &os,
                         const Tensor<TElem, TDevice, Dim> &tensor) {
  os << "venus::Tensor([";

  std::size_t count = 0;
  for (auto elem : tensor) {
    if (count > 0)
      os << ", ";
    count++;
    if constexpr (StringLike<TElem>) {
      os << "\"" << static_cast<TElem>(elem) << "\"";
    } else if constexpr (CharLike<TElem>) {
      os << "'" << static_cast<TElem>(elem) << "'";
    } else if constexpr (std::floating_point<TElem>) {
      os << std::fixed << std::setprecision(2) << static_cast<TElem>(elem);
    } else {
      os << static_cast<TElem>(elem);
    }
  }

  return os << "], " << "shape=" << tensor.Shape() << ")";
}

template <typename TElem, typename TDevice>
std::ostream &operator<<(std::ostream &os,
                         const Tensor<TElem, TDevice, 0> &tensor) {
  if constexpr (StringLike<TElem>) {
    return os << "venus::Tensor(" << "\"" << tensor.Value() << "\"" << ")";
  } else if constexpr (CharLike<TElem>) {
    return os << "venus::Tensor(" << "'" << tensor.Value() << "'" << ")";
  } else if constexpr (std::floating_point<TElem>) {
    return os << std::fixed << std::setprecision(2) << "venus::Tensor("
              << tensor.Value() << ")";
  } else {
    return os << "venus::Tensor(" << tensor.Value() << ")";
  }
}

} // namespace venus

template <typename TElem, typename TDevice, std::size_t Dim>
struct std::formatter<venus::Tensor<TElem, TDevice, Dim>> {
  constexpr auto parse(std::format_parse_context &ctx) { return ctx.begin(); }

  auto format(const venus::Tensor<TElem, TDevice, Dim> &tensor,
              std::format_context &ctx) const {
    ostringstream oss;
    oss << tensor;
    return std::format_to(ctx.out(), "{}", oss.str());
  }
};

// Scalar-first Addition
template <venus::Scalar Scalar, venus::Scalar TElem, typename TDevice,
          std::size_t Dim>
auto operator+(const Scalar &scalar,
               const venus::Tensor<TElem, TDevice, Dim> &tensor) {
  return venus::ops::add(scalar, tensor);
}

// Scalar-first Substraction
template <venus::Scalar Scalar, venus::Scalar TElem, typename TDevice,
          std::size_t Dim>
auto operator-(const Scalar &scalar,
               const venus::Tensor<TElem, TDevice, Dim> &tensor) {
  return venus::ops::sub(scalar, tensor);
}

// Scalar-first Multiplication
template <venus::Scalar Scalar, venus::Scalar TElem, typename TDevice,
          std::size_t Dim>
auto operator*(const Scalar &scalar,
               const venus::Tensor<TElem, TDevice, Dim> &tensor) {
  return venus::ops::mul(scalar, tensor);
}

// Scalar-fist Division
template <venus::Scalar Scalar, venus::Scalar TElem, typename TDevice,
          std::size_t Dim>
auto operator/(const Scalar &scalar,
               const venus::Tensor<TElem, TDevice, Dim> &tensor) {
  return venus::ops::div(scalar, tensor);
}

#undef REGISTER_OPERATOR_EQUAL
#undef REGISTER_POST_OPERATOR
#undef REGISTER_PRE_OPERATOR
#undef REGISTER_SCALAR_OP