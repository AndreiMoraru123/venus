#pragma once

#include "core/memory/device.hpp"
#include <algorithm>
#include <functional>
#include <ranges>
#include <tuple>
#include <type_traits>

namespace venus {
template <typename T>
concept VenusTensor = requires {
  typename T::ElementType;
  typename T::DeviceType;
};
} // namespace venus

namespace venus::ops {

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
                "Operation currently only supported on CPU");

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

template <template <typename, typename, std::size_t> class Tensor,
          typename Elem1, typename Dev1, std::size_t Dim1, typename Elem2,
          typename Dev2, std::size_t Dim2>
  requires VenusTensor<Tensor<Elem1, Dev1, Dim1>> &&
           VenusTensor<Tensor<Elem2, Dev2, Dim2>> &&
           std::is_arithmetic_v<Elem1> && std::is_arithmetic_v<Elem2>
auto add(const Tensor<Elem1, Dev1, Dim1> &t1,
         const Tensor<Elem2, Dev2, Dim2> &t2) {
  return detail::binary_elementwise_op(std::plus{}, t1, t2);
}

template <template <typename, typename, std::size_t> class Tensor,
          typename Elem1, typename Dev1, std::size_t Dim1, typename Elem2,
          typename Dev2, std::size_t Dim2>
  requires VenusTensor<Tensor<Elem1, Dev1, Dim1>> &&
           VenusTensor<Tensor<Elem2, Dev2, Dim2>> &&
           std::is_arithmetic_v<Elem1> && std::is_arithmetic_v<Elem2>
auto multiply(const Tensor<Elem1, Dev1, Dim1> &t1,
              const Tensor<Elem2, Dev2, Dim2> &t2) {
  return detail::binary_elementwise_op(std::multiplies{}, t1, t2);
}
} // namespace venus::ops