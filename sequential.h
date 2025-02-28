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

namespace detail {
template <std::size_t N, typename T, typename... Rest> struct TypeAt {
  using type = typename TypeAt<N - 1, Rest...>::type;
};
template <typename T, typename... Rest> struct TypeAt<0, T, Rest...> {
  using type = T;
};
} // namespace detail

template <typename TCon, int N> struct At_;

template <template <typename...> typename TCon, typename... TParams, int N>
struct At_<TCon<TParams...>, N> {
  static_assert(N < sizeof...(TParams), "index out of bounds");
  using type = typename detail::TypeAt<N, TParams...>::type;
};

template <typename TCon, int N> using At = typename At_<TCon, N>::type;

// =============================================================

// Order =======================================================
namespace NSOrder {
template <typename TIndexCont, typename TTypeCont> struct impl;

template <template <typename...> typename TTypeCont, typename... TTypes,
          int... index>
struct impl<Helper::IndexSequence<index...>, TTypeCont<TTypes...>>
    : Helper::KVBinder<TTypes, Helper::Int_<index>>::apply... {
  using Helper::KVBinder<TTypes, Helper::Int_<index>>::apply...;
};
} // namespace NSOrder
// =============================================================

} // namespace venus::Sequential