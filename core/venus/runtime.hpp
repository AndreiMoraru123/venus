#pragma once

#ifndef VENUS_MODULE
#include <concepts>
#include <cstddef>
#include <type_traits>
#endif

namespace venus {

// How a tensor executes and what it stores. Tensor's second template parameter
namespace Runtime {
// Computes on the host when an op is called
struct CPU {};
} // namespace Runtime

// A runtime opts in to each of these, so a new runtime starts with none

// Values can be read on the host: data(), value(), indexing and iteration
template <typename TRuntime> inline constexpr bool readable = false;
template <> inline constexpr bool readable<Runtime::CPU> = true;

// Values can be written in place
template <typename TRuntime> inline constexpr bool writable = false;
template <> inline constexpr bool writable<Runtime::CPU> = true;

// venus::eager accepts it: an op's result holds values when the call returns
template <typename TRuntime> inline constexpr bool eager_runtime = false;
template <> inline constexpr bool eager_runtime<Runtime::CPU> = true;

template <typename TRuntime>
concept Readable = readable<TRuntime>;
template <typename TRuntime>
concept Writable = writable<TRuntime>;
template <typename TRuntime>
concept EagerRuntime = eager_runtime<TRuntime>;

template <typename TElem, typename TRuntime, std::size_t Rank> class Tensor;

// The runtime of a scalar operand
struct NoRuntime {};

template <typename T> struct RuntimeOf {
  using type = NoRuntime;
};
template <typename TElem, typename TRuntime, std::size_t Rank>
struct RuntimeOf<Tensor<TElem, TRuntime, Rank>> {
  using type = TRuntime;
};
template <typename T>
using runtime_of_t = typename RuntimeOf<std::remove_cvref_t<T>>::type;

// The runtime an op runs on when its operands use A and B. A pair with no join
// gives NoJoin, and so does any join that includes NoJoin
struct NoJoin {};

template <typename A, typename B> struct Join {
  using type = NoJoin;
};
template <typename A> struct Join<A, A> {
  using type = A;
};
template <typename A> struct Join<NoRuntime, A> {
  using type = A;
};
template <typename A> struct Join<A, NoRuntime> {
  using type = A;
};
template <> struct Join<NoRuntime, NoRuntime> {
  using type = NoRuntime;
};

template <typename... TRuntimes> struct JoinAll;
template <typename A> struct JoinAll<A> {
  using type = A;
};
template <typename A, typename B, typename... Rest>
struct JoinAll<A, B, Rest...> : JoinAll<typename Join<A, B>::type, Rest...> {};

template <typename... Operands>
using join_t = typename JoinAll<runtime_of_t<Operands>...>::type;

} // namespace venus
