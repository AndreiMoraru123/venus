#pragma once

#include <cstddef>
#include <type_traits>
#include <venus/null_param.hpp>

namespace venus::Sequential {

// Details =====================================================
namespace detail {

// At details ==================================================
template <std::size_t N, typename T, typename... Rest> struct TypeAt {
  using type = TypeAt<N - 1, Rest...>::type;
};
template <typename T, typename... Rest> struct TypeAt<0, T, Rest...> {
  using type = T;
};
// =============================================================

// Order details ===============================================
template <typename T, typename... Types> struct FindTypeIndex;

template <typename T, typename U, typename... Rest>
struct FindTypeIndex<T, U, Rest...> {
  constexpr static std::size_t value = 1 + FindTypeIndex<T, Rest...>::value;
};

template <typename T, typename... Rest> struct FindTypeIndex<T, T, Rest...> {
  constexpr static std::size_t value = 0;
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
struct SetImpl<TCon<>, N, TValue, TCon<Processed...>,
               TCon<Current, Remaining...>> {
  using type = SetImpl<TCon<>, N - 1, TValue, TCon<Processed..., Current>,
                       TCon<Remaining...>>::type;
};
// =============================================================

// Fold details ================================================
template <typename TState, template <typename, typename> typename Fn,
          typename... TRemain>
struct FoldImpl {
  using type = TState;
};

template <typename TState, template <typename, typename> typename Fn,
          typename T0, typename... TRemain>
struct FoldImpl<TState, Fn, T0, TRemain...> {
  using type = FoldImpl<Fn<TState, T0>, Fn, TRemain...>::type;
};
// =============================================================

} // namespace detail

// Create ======================================================
// TODO: Might have to make this binary if recursion gets too deep
template <std::size_t N, template <typename...> typename TCont, typename... T>
struct Create_ {
  using type = Create_<N - 1, TCont, NullParameter, T...>::type;
};

template <template <typename...> class TCont, typename... T>
struct Create_<0, TCont, T...> {
  using type = TCont<T...>;
};

template <std::size_t N, template <typename...> typename TCon, typename... T>
using Create = Create_<N, TCon, T...>::type;
// =============================================================

// At ==========================================================
template <typename TCon, std::size_t N> struct At_;

template <template <typename...> typename TCon, typename... TParams,
          std::size_t N>
struct At_<TCon<TParams...>, N> {
  static_assert(N < sizeof...(TParams), "index out of bounds");
  using type = detail::TypeAt<N, TParams...>::type;
};

template <typename TCon, std::size_t N> using At = At_<TCon, N>::type;
// =============================================================

// Order =======================================================
template <typename TCon, typename TReq> struct Order_ {};

template <template <typename...> typename TCon, typename... TParams,
          typename TReq>
struct Order_<TCon<TParams...>, TReq> {
  static constexpr std::size_t value =
      detail::FindTypeIndex<TReq, TParams...>::value;
};

template <typename TCon, typename TReq>
static constexpr std::size_t Order = Order_<TCon, TReq>::value;
// =============================================================

// Set =========================================================
template <typename TCon, std::size_t N, typename TValue> struct Set_;

template <template <typename...> typename TCont, std::size_t N, typename TValue,
          typename... TParams>
struct Set_<TCont<TParams...>, N, TValue> {
  static_assert(N < sizeof...(TParams), "index out of bounds");
  using type =
      detail::SetImpl<TCont<>, N, TValue, TCont<>, TCont<TParams...>>::type;
};

template <typename TCon, std::size_t N, typename TValue>
using Set = Set_<TCon, N, TValue>::type;
// =============================================================

// PushBack ====================================================
template <typename TCon, typename... TValue> struct PushBack_;

template <template <typename...> typename TCon, typename... TParams,
          typename... TValue>
struct PushBack_<TCon<TParams...>, TValue...> {
  using type = TCon<TParams..., TValue...>;
};

template <typename TCon, typename... TValue>
using PushBack = PushBack_<TCon, TValue...>::type;
// =============================================================

// Fold ========================================================
template <typename TInitState, typename TInputCont,
          template <typename, typename> typename Fn>
struct Fold_;

template <typename TInitState, template <typename...> typename TCont,
          typename... TParams, template <typename, typename> typename Fn>
struct Fold_<TInitState, TCont<TParams...>, Fn> {
  template <typename S, typename I> using Fun = Fn<S, I>::type;
  using type = detail::FoldImpl<TInitState, Fun, TParams...>::type;
};

template <typename TInitState, typename TInputCont,
          template <typename, typename> typename Fn>
using Fold = Fold_<TInitState, TInputCont, Fn>::type;
// =============================================================

// Size ========================================================
template <typename T> struct Size_;

template <template <typename...> typename TCont, typename... T>
struct Size_<TCont<T...>> {
  static constexpr size_t value = sizeof...(T);
};

template <typename T>
static constexpr size_t Size = Size_<std::remove_cvref_t<T>>::value;
// =============================================================

// Head ========================================================
template <typename TCont> using Head = At<TCont, 0>;
// =============================================================

// Tail ========================================================
template <typename TCont> struct Tail_;

template <template <typename...> typename TCont, typename H,
          typename... TRemain>
struct Tail_<TCont<H, TRemain...>> {
  using type = TCont<TRemain...>;
};

template <typename TCont> using Tail = Tail_<TCont>::type;
// =============================================================

// Last ========================================================
template <typename TCont> using Last = At<TCont, Size<TCont> - 1>;
// =============================================================

} // namespace venus::Sequential