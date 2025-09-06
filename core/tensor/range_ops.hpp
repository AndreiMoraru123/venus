#pragma once

#include "core/memory/device.hpp"
#include <algorithm>
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

struct ElementWiseFn {
  template <typename F, typename... Tensors>
  constexpr auto operator()(F &&f, Tensors &&...tensors) const {
    return std::views::zip(tensors...) |
           std::views::transform([f = std::forward<F>(f)](auto &&tuple) {
             return std::apply(f, tuple);
           });
  }
};

inline constexpr ElementWiseFn elementwise{};
} // namespace detail

template <template <typename, typename, std::size_t> class Tensor,
          typename Elem1, typename Dev1, std::size_t Dim1, typename Elem2,
          typename Dev2, std::size_t Dim2>
  requires VenusTensor<Tensor<Elem1, Dev1, Dim1>> &&
           VenusTensor<Tensor<Elem2, Dev2, Dim2>> &&
           std::is_arithmetic_v<Elem1> && std::is_arithmetic_v<Elem2>
auto add(const Tensor<Elem1, Dev1, Dim1> &t1,
         const Tensor<Elem2, Dev2, Dim2> &t2) {
  static_assert(Dim1 == Dim2, "Tensor dimensions must match");
  static_assert(std::is_same_v<Dev1, Dev2>,
                "Tensors must be on the same device");
  static_assert(std::is_same_v<Dev1, Device::CPU>,
                "Addition is currently only supported on CPU");

  if constexpr (Dim1 == 0 && Dim2 == 0) {
    using ResultElementType = std::common_type_t<Elem1, Elem2>;
    return Tensor<ResultElementType, Dev1, 0>(t1.Value() + t2.Value());
  } else {
    if (t1.Shape() != t2.Shape()) {
      throw std::invalid_argument("Tensor shapes must match");
    }

    using ResultTensor = Tensor<std::common_type_t<Elem1, Elem2>, Dev1, Dim1>;
    ResultTensor result(t1.Shape());
    auto computation = detail::elementwise(std::plus{}, t1, t2);
    std::ranges::copy(computation, result.begin());
    return result;
  }
}
} // namespace venus::ops