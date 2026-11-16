#pragma once
// venus::eager is venus's ops with one more constraint: the result holds
// values when the call returns. Each op repeats the constraint of its venus::
// op and adds EagerOperands, so with both namespaces in scope the eager op is
// the more constrained and an unqualified call on CPU resolves to it.

#include <venus/tensor/ops.hpp>

namespace venus::eager {

#define VENUS_EAGER_BINARY_OP(op_name)                                         \
  template <typename A, typename B>                                            \
    requires BinaryOperands<A, B> && EagerOperands<A, B>                       \
  auto op_name(const A &a, const B &b) {                                       \
    return venus::op_name(a, b);                                               \
  }

VENUS_EAGER_BINARY_OP(add)
VENUS_EAGER_BINARY_OP(sub)
VENUS_EAGER_BINARY_OP(mul)
VENUS_EAGER_BINARY_OP(div)
VENUS_EAGER_BINARY_OP(gt)
VENUS_EAGER_BINARY_OP(gte)
VENUS_EAGER_BINARY_OP(lt)
VENUS_EAGER_BINARY_OP(lte)
VENUS_EAGER_BINARY_OP(eq)
VENUS_EAGER_BINARY_OP(neq)

#undef VENUS_EAGER_BINARY_OP

template <typename P, typename A, typename B>
  requires VenusTensor<P> && Operand<A> && Operand<B> && EagerOperands<P, A, B>
auto where(const P &predicate, const A &a, const B &b) {
  return venus::where(predicate, a, b);
}

template <typename A, typename B>
  requires Matrices<A, B> && EagerOperands<A, B>
auto mm(const A &a, const B &b) {
  return venus::mm(a, b);
}

template <typename A, typename B>
  requires Tensors<A, B> && EagerOperands<A, B>
auto inner(const A &a, const B &b) {
  return venus::inner(a, b);
}

template <typename A, typename B>
  requires Vectors<A, B> && EagerOperands<A, B>
auto dot(const A &a, const B &b) {
  return venus::dot(a, b);
}

template <typename T>
  requires Tensors<T> && EagerOperands<T>
auto sort(const T &t) {
  return venus::sort(t);
}

template <typename T, typename V>
  requires Tensors<T> && Scalar<V> && EagerOperands<T>
auto fill(const T &t, V value) {
  return venus::fill(t, value);
}

template <typename T, typename V>
  requires Tensors<T> && Scalar<V> && EagerOperands<T>
auto iota(const T &t, V start) {
  return venus::iota(t, start);
}

template <typename T>
  requires AtLeastMatrix<T> && EagerOperands<T>
auto eye_like(const T &t) {
  return venus::eye_like(t);
}

template <std::size_t Dim, typename T>
  requires Tensors<T> && EagerOperands<T>
auto sum_dim(const T &t) {
  return venus::sum_dim<Dim>(t);
}

template <std::size_t... Dims, typename T>
  requires Tensors<T> && EagerOperands<T>
auto sum_dims(const T &t) {
  return venus::sum_dims<Dims...>(t);
}

template <std::size_t... SumDims, typename A, typename B>
  requires Tensors<A, B> && EagerOperands<A, B>
auto sumproduct_pair(const A &a, const B &b) {
  return venus::sumproduct_pair<SumDims...>(a, b);
}

template <ConstexprString Eqn, typename... Ts>
  requires Tensors<Ts...> && EagerOperands<Ts...>
auto einsum(const Ts &...tensors) {
  return venus::einsum<Eqn>(tensors...);
}

// These only run where values exist already
using venus::equal;
using venus::nonzero;
using venus::nonzero_flat;
using venus::transform;

template <typename T>
  requires ReadableTensor<T> && BoolTensor<T> && EagerOperands<T>
auto where(const T &predicate) {
  return venus::where(predicate);
}

} // namespace venus::eager
