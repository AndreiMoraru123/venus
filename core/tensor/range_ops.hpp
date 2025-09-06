#include <ranges>
#include <tuple>

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

template <typename... Tensors> auto add(Tensors &&...tensors) {
  return detail::elementwise(std::plus{}, tensors...);
}
} // namespace venus::ops
