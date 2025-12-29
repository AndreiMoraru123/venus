#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <numeric>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>
#include <venus/memory/device.hpp>

namespace venus {
template <typename T>
concept VenusTensor = requires {
  typename std::remove_cvref_t<T>::ElementType;
  typename std::remove_cvref_t<T>::DeviceType;
  { std::remove_cvref_t<T>::Dimension } -> std::convertible_to<std::size_t>;
};

template <typename T>
concept Scalar = std::is_arithmetic_v<std::remove_cvref_t<T>>;

template <typename T>
concept ScalarTensor = VenusTensor<std::remove_cvref_t<T>> &&
                       (std::remove_cvref_t<T>::Dimension == 0);

template <typename T>
concept MDTensor = VenusTensor<std::remove_cvref_t<T>> &&
                   (std::remove_cvref_t<T>::Dimension > 0);

template <typename T>
concept BoolTensor =
    VenusTensor<std::remove_cvref_t<T>> &&
    std::is_convertible_v<typename std::remove_cvref_t<T>::ElementType, bool>;
} // namespace venus

#define REGISTER_BINARY_OP(op_name, std_op, op_symbol)                         \
  template <typename T1, typename T2>                                          \
    requires(Scalar<T1> || VenusTensor<T1>) && (Scalar<T2> || VenusTensor<T2>) \
  auto op_name(T1 &&t1, T2 &&t2) {                                             \
    /* Tensor op Tensor */                                                     \
    if constexpr (MDTensor<T1> && MDTensor<T2>) {                              \
      return detail::binary_elementwise_op(std::std_op{}, t1, t2);             \
    } /* Tensor op Scalar */                                                   \
    else if constexpr (MDTensor<T1> && Scalar<T2>) {                           \
      return transform(t1, [s = t2](auto &&t) { return t op_symbol s; });      \
    } /* Scalar op Tensor */                                                   \
    else if constexpr (Scalar<T1> && MDTensor<T2>) {                           \
      return transform(t2, [s = t1](auto &&t) { return s op_symbol t; });      \
    } /* Tensor op ScalarTensor */                                             \
    else if constexpr (MDTensor<T1> && ScalarTensor<T2>) {                     \
      return op_name(t1, t2.value());                                          \
    } /* ScalarTensor op Tensor */                                             \
    else if constexpr (ScalarTensor<T1> && MDTensor<T2>) {                     \
      return op_name(t1.value(), t2);                                          \
    } /* ScalarTensor op ScalarTensor */                                       \
    else if constexpr (ScalarTensor<T1> && ScalarTensor<T2>) {                 \
      return detail::binary_elementwise_op(std::std_op{}, t1, t2);             \
    }                                                                          \
  }

namespace venus::ops {

// Details =====================================================
namespace detail {

template <template <typename, typename, std::size_t> class Tensor,
          typename Elem1, typename Dev1, std::size_t Dim1, typename Elem2,
          typename Dev2, std::size_t Dim2>
void validate_binary_op(const Tensor<Elem1, Dev1, Dim1> &t1,
                        const Tensor<Elem2, Dev2, Dim2> &t2) {
  static_assert(Dim1 == Dim2, "Tensor dimensions must match");
  static_assert(std::is_same_v<Dev1, Dev2>,
                "Tensors must be on the same device");
  static_assert(std::is_same_v<Dev1, Device::CPU>,
                "Operation is currently only supported on CPU");

  if constexpr (Dim1 > 0) {
    if (t1.shape() != t2.shape()) {
      throw std::invalid_argument("Tensor shapes must match");
    }
  }
}

template <typename Op, template <typename, typename, std::size_t> class Tensor,
          typename Elem1, typename Dev1, std::size_t Dim1, typename Elem2,
          typename Dev2, std::size_t Dim2>
auto binary_elementwise_op(Op op, const Tensor<Elem1, Dev1, Dim1> &t1,
                           const Tensor<Elem2, Dev2, Dim2> &t2) {

  validate_binary_op(t1, t2);
  using ResultElementType = std::common_type_t<Elem1, Elem2>;

  if constexpr (Dim1 == 0 && Dim2 == 0) {
    return Tensor<ResultElementType, Dev1, 0>(op(t1.value(), t2.value()));
  } else {
    if (t1.shape() != t2.shape()) {
      throw std::invalid_argument("Tensor shapes must match");
    }

    using ResultTensor = Tensor<ResultElementType, Dev1, Dim1>;
    ResultTensor result(t1.shape());
    auto computation =
        std::views::zip(t1, t2) | std::views::transform([op](auto &&tuple) {
          return std::apply(op, tuple);
        });
    std::ranges::copy(computation, result.begin());
    return result;
  }
}

template <typename Op, template <typename, typename, std::size_t> class Tensor,
          typename Elem1, typename Dev1, std::size_t Dim1, typename Elem2,
          typename Dev2, std::size_t Dim2, typename Elem3, typename Dev3,
          std::size_t Dim3>
auto ternary_elementwise_op(Op op, const Tensor<Elem1, Dev1, Dim1> &t1,
                            const Tensor<Elem2, Dev2, Dim2> &t2,
                            const Tensor<Elem3, Dev3, Dim3> &t3) {

  validate_binary_op(t1, t2);
  validate_binary_op(t2, t3);
  using ResultElementType = std::common_type_t<Elem1, Elem2, Elem3>;

  if constexpr (Dim1 == 0 && Dim2 == 0 && Dim3 == 0) {
    return Tensor<ResultElementType, Dev1, 0>(
        op(t1.value(), t2.value(), t3.value()));
  } else {
    if (t1.shape() != t2.shape() || t2.shape() != t3.shape()) {
      throw std::invalid_argument("Tensor shapes must match");
    }

    using ResultTensor = Tensor<ResultElementType, Dev1, Dim1>;
    ResultTensor result(t1.shape());
    auto computation =
        std::views::zip(t1, t2, t3) | std::views::transform([op](auto &&tuple) {
          return std::apply(op, tuple);
        });
    std::ranges::copy(computation, result.begin());
    return result;
  }
}

} // namespace detail

// Transform
template <template <typename, typename, std::size_t> class Tensor, Scalar Elem,
          typename Dev, std::size_t Dim, typename Fn>
  requires VenusTensor<Tensor<Elem, Dev, Dim>>
auto transform(const Tensor<Elem, Dev, Dim> &tensor, Fn &&fn) {
  static_assert(std::is_same_v<Dev, Device::CPU>,
                "Transform is currently only supported on CPU");

  using ResultElementType = std::invoke_result_t<Fn, Elem>;

  if constexpr (Dim == 0) {
    return Tensor<ResultElementType, Dev, 0>(fn(tensor.value()));
  } else {
    using ResultTensor = Tensor<ResultElementType, Dev, Dim>;
    ResultTensor result(tensor.shape());
    auto computation =
        tensor | std::views::transform(
                     [f = std::forward<Fn>(fn)](auto &&t) { return f(t); });
    std::ranges::copy(computation, result.begin());
    return result;
  }
}

REGISTER_BINARY_OP(add, plus, +)
REGISTER_BINARY_OP(sub, minus, -)
REGISTER_BINARY_OP(mul, multiplies, *)
REGISTER_BINARY_OP(div, divides, /)
REGISTER_BINARY_OP(gt, greater, >)
REGISTER_BINARY_OP(gte, greater_equal, >=)
REGISTER_BINARY_OP(lt, less, <)
REGISTER_BINARY_OP(lte, less_equal, <=)
REGISTER_BINARY_OP(eq, equal_to, ==)
REGISTER_BINARY_OP(neq, not_equal_to, !=)

// Sort
template <template <typename, typename, std::size_t> class Tensor, Scalar Elem,
          typename Dev, std::size_t Dim>
  requires VenusTensor<Tensor<Elem, Dev, Dim>>
auto sort(const Tensor<Elem, Dev, Dim> &tensor) {
  static_assert(std::is_same_v<Dev, Device::CPU>,
                "Sort is currently only supported on CPU");
  auto copy = tensor.clone();
  std::ranges::sort(copy);
  return copy;
}

// All equal
template <template <typename, typename, std::size_t> class Tensor, Scalar Elem1,
          typename Dev1, std::size_t Dim1, Scalar Elem2, typename Dev2,
          std::size_t Dim2>
  requires VenusTensor<Tensor<Elem1, Dev1, Dim1>> &&
           VenusTensor<Tensor<Elem2, Dev2, Dim2>>
auto all_equal(const Tensor<Elem1, Dev1, Dim1> &t1,
               const Tensor<Elem2, Dev2, Dim2> &t2) -> bool {
  detail::validate_binary_op(t1, t2);
  if (t1.shape() != t2.shape()) {
    return false;
  }
  return std::ranges::equal(t1, t2);
}

// Dot product
template <template <typename, typename, std::size_t> class Tensor, Scalar Elem1,
          typename Dev1, Scalar Elem2, typename Dev2, std::size_t Dim1,
          std::size_t Dim2>
  requires VenusTensor<Tensor<Elem1, Dev1, Dim1>> &&
           VenusTensor<Tensor<Elem2, Dev2, Dim2>>
auto dot(const Tensor<Elem1, Dev1, Dim1> &t1,
         const Tensor<Elem2, Dev2, Dim2> &t2) {
  detail::validate_binary_op(t1, t2);
  using ResultElementType = std::common_type_t<Elem1, Elem2>;
  auto product =
      std::inner_product(t1.begin(), t1.end(), t2.begin(), ResultElementType{});
  return Tensor<ResultElementType, Dev1, 0>(product);
}

template <template <typename, typename, std::size_t> class Tensor, Scalar Elem,
          Scalar Idx, typename Dev, std::size_t Dim>
  requires VenusTensor<Tensor<Elem, Dev, Dim>>
auto iota(Tensor<Elem, Dev, Dim> tensor, Idx i) {
#if _cpp_lib_ranges >= 202110L
  std::ranges::iota(tensor, i);
#else
  std::iota(tensor.begin(), tensor.end(), i);
#endif
}

template <template <typename, typename, std::size_t> class Tensor, Scalar Elem,
          Scalar Idx, typename Dev, std::size_t Dim>
  requires VenusTensor<Tensor<Elem, Dev, Dim>>
auto fill(Tensor<Elem, Dev, Dim> tensor, Idx i) {
#if _cpp_lib_ranges >= 202110L
  std::ranges::fill(tensor, i);
#else
  std::fill(tensor.begin(), tensor.end(), i);
#endif
}

template <template <typename, typename, std::size_t> class Tensor, Scalar Elem,
          typename Dev, std::size_t Dim>
  requires BoolTensor<Tensor<Elem, Dev, Dim>>
auto where(const Tensor<Elem, Dev, Dim> &condition) {
  using ResultTensor = Tensor<std::size_t, Dev, Dim>;
  ResultTensor result(condition.shape());

  auto result_ptr = std::ranges::data(result);
  auto indices = std::views::iota(std::size_t{0}, condition.size());
  std::ranges::for_each(std::views::zip(condition, indices),
                        [result_ptr](auto &&pair) {
                          const auto &[cond_val, idx] = pair;
                          if (static_cast<bool>(cond_val)) {
                            result_ptr[idx] = idx;
                          }
                        });

  return result;
}

template <typename T1, typename T2, typename T3>
  requires VenusTensor<T1> && (VenusTensor<T2> || Scalar<T2>) &&
           (VenusTensor<T3> || Scalar<T3>)
auto where(T1 &&t1, T2 &&t2, T3 &&t3) {
  auto v1 = [&] {
    if constexpr (ScalarTensor<T1>) {
      return t1.value();
    } else {
      return std::forward<T1>(t1);
    }
  }();

  auto v2 = [&] {
    if constexpr (ScalarTensor<T2>) {
      return t2.value();
    } else {
      return std::forward<T2>(t2);
    }
  }();

  auto v3 = [&] {
    if constexpr (ScalarTensor<T3>) {
      return t3.value();
    } else {
      return std::forward<T3>(t3);
    }
  }();

  // Tensor, Tensor, Tensor
  if constexpr (MDTensor<T1> && MDTensor<T2> && MDTensor<T3>) {
    return detail::ternary_elementwise_op(
        [](auto &&a, auto &&b, auto &&c) { return a ? b : c; }, v1, v2, v3);
  }

  // Tensor, Scalar, Scalar
  else if constexpr (MDTensor<T1> && Scalar<T2> && Scalar<T3>) {
    return transform(v1, [s2 = v2, s3 = v3](auto &&a) { return a ? s2 : s3; });
  }

  // Tensor, Tensor, Scalar
  else if constexpr (MDTensor<T1> && MDTensor<T2> && Scalar<T3>) {
    return detail::binary_elementwise_op(
        [s3 = v3](auto &&a, auto &&b) { return a ? b : s3; }, v1, v2);
  }

  // Tensor, Scalar, Tensor
  else if constexpr (MDTensor<T1> && Scalar<T2> && MDTensor<T3>) {
    return detail::binary_elementwise_op(
        [s2 = v2](auto &&a, auto &&c) { return a ? s2 : c; }, v1, v3);
  }
}

} // namespace venus::ops

#undef REGISTER_BINARY_OP