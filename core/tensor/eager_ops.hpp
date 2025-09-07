#pragma once

#include "core/memory/device.hpp"
#include <algorithm>
#include <concepts>
#include <functional>
#include <ranges>
#include <tuple>
#include <type_traits>

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
} // namespace venus

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
    if (t1.Shape() != t2.Shape()) {
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

  if constexpr (Dim1 == 0 && Dim2 == 0) {
    using ResultElementType = std::common_type_t<Elem1, Elem2>;
    return Tensor<ResultElementType, Dev1, 0>(op(t1.Value(), t2.Value()));
  } else {
    if (t1.Shape() != t2.Shape()) {
      throw std::invalid_argument("Tensor shapes must match");
    }

    using ResultTensor = Tensor<std::common_type_t<Elem1, Elem2>, Dev1, Dim1>;
    ResultTensor result(t1.Shape());
    auto computation =
        std::views::zip(t1, t2) | std::views::transform([op](auto &&tuple) {
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
    return Tensor<ResultElementType, Dev, 0>(fn(tensor.Value()));
  } else {
    using ResultTensor = Tensor<ResultElementType, Dev, Dim>;
    ResultTensor result(tensor.Shape());
    auto computation =
        tensor | std::views::transform(
                     [f = std::forward<Fn>(fn)](auto &&t) { return f(t); });
    std::ranges::copy(computation, result.begin());
    return result;
  }
}

// Addition
template <typename T1, typename T2>
  requires(Scalar<T1> || VenusTensor<T1>) && (Scalar<T2> || VenusTensor<T2>)
auto add(T1 &&t1, T2 &&t2) {
  // Tensor + Tensor
  if constexpr (MDTensor<T1> && MDTensor<T2>) {
    return detail::binary_elementwise_op(std::plus{}, t1, t2);
  }
  // Tensor + Scalar
  else if constexpr (MDTensor<T1> && Scalar<T2>) {
    return transform(t1, [s = t2](auto &&t) { return t + s; });
  }
  // Scalar + Tensor
  else if constexpr (Scalar<T1> && MDTensor<T2>) {
    return transform(t2, [s = t1](auto &&t) { return s + t; });
  }
  // Tensor + ScalarTensor
  else if constexpr (MDTensor<T1> && ScalarTensor<T2>) {
    return add(t1, t2.Value());
  }
  // ScalarTensor + Tensor
  else if constexpr (ScalarTensor<T1> && MDTensor<T2>) {
    return add(t1.Value(), t2);
  }
  // ScalarTensor + ScalarTensor
  else if constexpr (ScalarTensor<T1> && ScalarTensor<T2>) {
    return detail::binary_elementwise_op(std::plus{}, t1, t2);
  }
}

// Subtraction
template <typename T1, typename T2>
  requires(Scalar<T1> || VenusTensor<T1>) && (Scalar<T2> || VenusTensor<T2>)
auto sub(T1 &&t1, T2 &&t2) {
  // Tensor - Tensor
  if constexpr (MDTensor<T1> && MDTensor<T2>) {
    return detail::binary_elementwise_op(std::minus{}, t1, t2);
  }
  // Tensor - Scalar
  else if constexpr (MDTensor<T1> && Scalar<T2>) {
    return transform(t1, [s = t2](auto &&t) { return t - s; });
  }
  // Scalar - Tensor
  else if constexpr (Scalar<T1> && MDTensor<T2>) {
    return transform(t2, [s = t1](auto &&t) { return s - t; });
  }
  // Tensor - ScalarTensor
  else if constexpr (MDTensor<T1> && ScalarTensor<T2>) {
    return sub(t1, t2.Value());
  }
  // ScalarTensor - Tensor
  else if constexpr (ScalarTensor<T1> && MDTensor<T2>) {
    return sub(t1.Value(), t2);
  }
  // ScalarTensor - ScalarTensor
  else if constexpr (ScalarTensor<T1> && ScalarTensor<T2>) {
    return detail::binary_elementwise_op(std::minus{}, t1, t2);
  }
}

// Multiplication
template <typename T1, typename T2>
  requires(Scalar<T1> || VenusTensor<T1>) && (Scalar<T2> || VenusTensor<T2>)
auto mul(T1 &&t1, T2 &&t2) {
  // Tensor * Tensor
  if constexpr (MDTensor<T1> && MDTensor<T2>) {
    return detail::binary_elementwise_op(std::multiplies{}, t1, t2);
  }
  // Tensor * Scalar
  else if constexpr (MDTensor<T1> && Scalar<T2>) {
    return transform(t1, [s = t2](auto &&t) { return t * s; });
  }
  // Scalar * Tensor
  else if constexpr (Scalar<T1> && MDTensor<T2>) {
    return transform(t2, [s = t1](auto &&t) { return s * t; });
  }
  // Tensor * ScalarTensor
  else if constexpr (MDTensor<T1> && ScalarTensor<T2>) {
    return mul(t1, t2.Value());
  }
  // ScalarTensor * Tensor
  else if constexpr (ScalarTensor<T1> && MDTensor<T2>) {
    return mul(t1.Value(), t2);
  }
  // ScalarTensor * ScalarTensor
  else if constexpr (ScalarTensor<T1> && ScalarTensor<T2>) {
    return detail::binary_elementwise_op(std::multiplies{}, t1, t2);
  }
}

// Division
template <typename T1, typename T2>
  requires(Scalar<T1> || VenusTensor<T1>) && (Scalar<T2> || VenusTensor<T2>)
auto div(T1 &&t1, T2 &&t2) {
  // Tensor / Tensor
  if constexpr (MDTensor<T1> && MDTensor<T2>) {
    return detail::binary_elementwise_op(std::divides{}, t1, t2);
  }
  // Tensor / Scalar
  else if constexpr (MDTensor<T1> && Scalar<T2>) {
    return transform(t1, [s = t2](auto &&t) { return t / s; });
  }
  // Scalar / Tensor
  else if constexpr (Scalar<T1> && MDTensor<T2>) {
    return transform(t2, [s = t1](auto &&t) { return s / t; });
  }
  // Tensor / ScalarTensor
  else if constexpr (MDTensor<T1> && ScalarTensor<T2>) {
    return div(t1, t2.Value());
  }
  // ScalarTensor / Tensor
  else if constexpr (ScalarTensor<T1> && MDTensor<T2>) {
    return div(t1.Value(), t2);
  }
  // ScalarTensor / ScalarTensor
  else if constexpr (ScalarTensor<T1> && ScalarTensor<T2>) {
    return detail::binary_elementwise_op(std::divides{}, t1, t2);
  }
}
} // namespace venus::ops