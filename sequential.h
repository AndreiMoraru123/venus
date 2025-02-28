#pragma once

#include <cstddef>

namespace venus::Sequential {

// Details =====================================================
namespace detail {

// At details ==================================================
template <std::size_t N, typename T, typename... Rest> struct TypeAt {
  using type = typename TypeAt<N - 1, Rest...>::type;
};
template <typename T, typename... Rest> struct TypeAt<0, T, Rest...> {
  using type = T;
};
// =============================================================

// Order details ===============================================
template <typename T, typename... Types> struct FindTypeIndex;

template <typename T, typename U, typename... Rest>
struct FindTypeIndex<T, U, Rest...> {
  constexpr static int value = 1 + FindTypeIndex<T, Rest...>::value;
};

template <typename T, typename... Rest> struct FindTypeIndex<T, T, Rest...> {
  constexpr static int value = 0;
};
// =============================================================

} // namespace detail

// =============================================================

// At ==========================================================
template <typename TCon, int N> struct At_;

template <template <typename...> typename TCon, typename... TParams, int N>
struct At_<TCon<TParams...>, N> {
  static_assert(N < sizeof...(TParams), "index out of bounds");
  using type = typename detail::TypeAt<N, TParams...>::type;
};

template <typename TCon, int N> using At = typename At_<TCon, N>::type;

template <typename TCon, typename TReq> struct Order_ {};
// =============================================================

// Order =======================================================
template <template <typename...> typename TCon, typename... TParams,
          typename TReq>
struct Order_<TCon<TParams...>, TReq> {
  constexpr static int value = detail::FindTypeIndex<TReq, TParams...>::value;
};

template <typename TCon, typename TReq>
constexpr static int Order = Order_<TCon, TReq>::value;
// =============================================================

} // namespace venus::Sequential