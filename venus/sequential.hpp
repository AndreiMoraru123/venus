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

// Set details =================================================
template <typename TCon, std::size_t N, typename TValue, typename Processed,
          typename Remain>
struct SetImpl;

// Base case: Processed N elements and now I want to insert TValue
template <template <typename...> typename TCon, typename TValue,
          typename... Processed, typename Current, typename... Remaining>
struct SetImpl<TCon<>, 0, TValue, TCon<Processed...>,
               TCon<Current, Remaining...>> {
  using type = TCon<Processed..., TValue, Remaining...>;
};

// Recursive case: Move elements from Remaining to Processed until I reach Nth
template <template <typename...> typename TCon, std::size_t N, typename TValue,
          typename... Processed, typename Current, typename... Remaining>
  requires(N > 0)
struct SetImpl<TCon<>, N, TValue, TCon<Processed...>,
               TCon<Current, Remaining...>> {
  using type =
      typename SetImpl<TCon<>, N - 1, TValue, TCon<Processed..., Current>,
                       TCon<Remaining...>>::type;
};
// =============================================================

} // namespace detail

// At ==========================================================
template <typename TCon, std::size_t N> struct At_;

template <template <typename...> typename TCon, typename... TParams,
          std::size_t N>
struct At_<TCon<TParams...>, N> {
  static_assert(N < sizeof...(TParams), "index out of bounds");
  using type = typename detail::TypeAt<N, TParams...>::type;
};

template <typename TCon, std::size_t N> using At = typename At_<TCon, N>::type;

template <typename TCon, typename TReq> struct Order_ {};
// =============================================================

// Order =======================================================
template <template <typename...> typename TCon, typename... TParams,
          typename TReq>
struct Order_<TCon<TParams...>, TReq> {
  static constexpr int value = detail::FindTypeIndex<TReq, TParams...>::value;
};

template <typename TCon, typename TReq>
static constexpr int Order = Order_<TCon, TReq>::value;
// =============================================================

// Set =========================================================
template <typename TCon, std::size_t N, typename TValue> struct Set_;

template <template <typename...> typename TCont, std::size_t N, typename TValue,
          typename... TParams>
struct Set_<TCont<TParams...>, N, TValue> {
  static_assert(N < sizeof...(TParams), "index out of bounds");
  using type = typename detail::SetImpl<TCont<>, N, TValue, TCont<>,
                                        TCont<TParams...>>::type;
};

template <typename TCon, std::size_t N, typename TValue>
using Set = typename Set_<TCon, N, TValue>::type;
// =============================================================

} // namespace venus::Sequential