#pragma once
// Runtime::CPU actions. Each allocates the result its op's rule describes and
// calls a kernel that writes into it.

#ifndef VENUS_MODULE
#include <cstddef>
#include <type_traits>
#endif
#include <venus/runtime.hpp>
#include <venus/str.hpp>
#include <venus/tensor/kernels.hpp>
#include <venus/tensor/ops.hpp>

namespace venus {

namespace detail {
template <typename T> auto input(const T &operand) {
  if constexpr (Scalar<T>) {
    return operand;
  } else {
    return kernel::Input<typename T::ElementType, T::rank>{operand.data(),
                                                           operand.shape()};
  }
}

template <typename TMeta> auto allocate(const TMeta &m) {
  return result_t<TMeta, Runtime::CPU>(m.shape);
}
} // namespace detail

template <typename Fn, ConstexprString Name>
struct Action<Elementwise<Fn, Name>, Runtime::CPU> {
  template <typename A, typename B> static auto run(const A &a, const B &b) {
    const auto m = Elementwise<Fn, Name>::rule(meta(a), meta(b));
    auto out = detail::allocate(m);
    kernel::elementwise(Fn{}, out.data(), m.shape, detail::input(a),
                        detail::input(b));
    return out;
  }
};

template <> struct Action<WhereOp, Runtime::CPU> {
  template <typename P, typename A, typename B>
  static auto run(const P &p, const A &a, const B &b) {
    const auto m = WhereOp::rule(meta(p), meta(a), meta(b));
    using Elem = typename decltype(m)::ElementType;
    auto out = detail::allocate(m);
    kernel::elementwise([](const auto &pred, const auto &x,
                           const auto &y) -> Elem { return pred ? x : y; },
                        out.data(), m.shape, detail::input(p), detail::input(a),
                        detail::input(b));
    return out;
  }
};

template <> struct Action<MmOp, Runtime::CPU> {
  template <typename A, typename B> static auto run(const A &a, const B &b) {
    const auto m = MmOp::rule(meta(a), meta(b));
    auto out = detail::allocate(m);
    kernel::mm(out.data(), a.data(), b.data(), m.shape[0], a.shape()[1],
               m.shape[1]);
    return out;
  }
};

template <> struct Action<InnerOp, Runtime::CPU> {
  template <typename A, typename B> static auto run(const A &a, const B &b) {
    const auto m = InnerOp::rule(meta(a), meta(b));
    auto out = detail::allocate(m);
    kernel::inner(out.data(), a.data(), b.data(), a.size());
    return out;
  }
};

template <> struct Action<SortOp, Runtime::CPU> {
  template <typename T> static auto run(const T &t) {
    auto out = detail::allocate(SortOp::rule(meta(t)));
    kernel::sort(out.data(), t.data(), t.size());
    return out;
  }
};

template <> struct Action<FillOp, Runtime::CPU> {
  template <typename T, typename V> static auto run(const T &t, V value) {
    auto out = detail::allocate(FillOp::rule(meta(t)));
    kernel::fill(out.data(), out.size(), value);
    return out;
  }
};

template <> struct Action<IotaOp, Runtime::CPU> {
  template <typename T, typename V> static auto run(const T &t, V start) {
    auto out = detail::allocate(IotaOp::rule(meta(t)));
    kernel::iota(out.data(), out.size(), start);
    return out;
  }
};

template <> struct Action<EyeLikeOp, Runtime::CPU> {
  template <typename T> static auto run(const T &t) {
    auto out = detail::allocate(EyeLikeOp::rule(meta(t)));
    kernel::eye(out.data(), t.shape());
    return out;
  }
};

template <std::size_t Dim> struct Action<SumDimOp<Dim>, Runtime::CPU> {
  template <typename T> static auto run(const T &t) {
    auto out = detail::allocate(SumDimOp<Dim>::rule(meta(t)));
    kernel::sum_dim<Dim>(out.data(), t.data(), t.shape());
    return out;
  }
};

// Shares the operand's memory, so it calls no kernel
template <std::size_t ToRank> struct Action<SqueezeOp<ToRank>, Runtime::CPU> {
  template <typename T> static auto run(const T &t) {
    const auto m = SqueezeOp<ToRank>::rule(meta(t));
    if constexpr (ToRank == 0) {
      return t.asScalar();
    } else {
      return t.template reshape<ToRank>(m.shape);
    }
  }
};

template <std::size_t OpIdx, ConstexprString Eqn>
struct Action<EinsumOperandOp<OpIdx, Eqn>, Runtime::CPU> {
  template <typename T> static auto run(const T &t) {
    using Op = EinsumOperandOp<OpIdx, Eqn>;
    const auto m = Op::rule(meta(t));
    auto out = detail::allocate(m);
    kernel::gather_axes(out.data(), m.shape, t.data(), t.shape(), Op::axis);
    return out;
  }
};

} // namespace venus
