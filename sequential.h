#pragma once

#include "helpers.h"
#include <cstddef>

namespace venus::Sequential {

// At ==========================================================
// namespace NSAt {
// template <typename ignore> struct impl;

// template <int... ignore> struct impl<Helper::IndexSequence<ignore...>> {
//   template <typename nth>
//   static nth apply(decltype(ignore, (void *)nullptr)..., nth *, ...);
// };
// } // namespace NSAt

// template <typename TCon, int N> struct At_;

// template <template <typename...> typename TCon, typename... TParams, int N>
// struct At_<TCon<TParams...>, N> {
//   static_assert(N < sizeof...(TParams), "index out of bounds");
//   using type = decltype(NSAt::impl<Helper::MakeIndexSequence<N>>::apply(
//       (TParams *)nullptr...));
// };

// template <typename TCon, int N> using At = typename At_<TCon, N>::type;

// Details =====================================================
namespace detail {
template <std::size_t N, typename T, typename... Rest> struct TypeAt {
  using type = typename TypeAt<N - 1, Rest...>::type;
};
template <typename T, typename... Rest> struct TypeAt<0, T, Rest...> {
  using type = T;
};

template <typename T, typename... Types> struct FindTypeIndex;

template <typename T, typename U, typename... Rest>
struct FindTypeIndex<T, U, Rest...> {
  constexpr static int value = 1 + FindTypeIndex<T, Rest...>::value;
};

template <typename T, typename... Rest> struct FindTypeIndex<T, T, Rest...> {
  constexpr static int value = 0;
};

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

// namespace NSOrder {
// template <typename TIndexCont, typename TTypeCont> struct impl;

// template <template <typename...> typename TTypeCont, typename... TTypes,
//           int... index>
// struct impl<Helper::IndexSequence<index...>, TTypeCont<TTypes...>>
//     : Helper::KVBinder<TTypes, Helper::Int_<index>>... {
//   using Helper::KVBinder<TTypes, Helper::Int_<index>>::apply...;
// };
// } // namespace NSOrder
// template <typename TCon, typename TReq> struct Order_;

template <typename TCon, typename TReq> struct Order_ {};

// Order =======================================================
template <template <typename...> typename TCon, typename... TParams,
          typename TReq>
struct Order_<TCon<TParams...>, TReq> {
  // using IndexSeq = Helper::MakeIndexSequence<sizeof...(TParams)>;
  // using LookUpTable = NSOrder::impl<IndexSeq, TCon<TParams...>>;
  // using AimType = decltype(LookUpTable::apply((TReq *)nullptr));
  // constexpr static int value = AimType::value;
  constexpr static int value = detail::FindTypeIndex<TReq, TParams...>::value;
};

template <typename TCon, typename TReq>
constexpr static int Order = Order_<TCon, TReq>::value;
// =============================================================

} // namespace venus::Sequential