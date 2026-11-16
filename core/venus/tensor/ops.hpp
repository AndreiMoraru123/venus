#pragma once
// Every op has a rule, written once for all runtimes, that computes the
// result's element type, rank and shape. A runtime does the work in its
// Action<Op, Runtime>. When a runtime has no action for an op, dispatch builds
// the op from other ops through Op::compose.

#ifndef VENUS_MODULE
#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <format>
#include <functional>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#endif
#include <venus/concepts.hpp>
#include <venus/runtime.hpp>
#include <venus/str.hpp>
#include <venus/tensor/einsum.hpp>
#include <venus/tensor/kernels.hpp>
#include <venus/tensor/shape.hpp>

namespace venus {

template <typename T>
concept Operand = Scalar<T> || VenusTensor<T>;

template <typename A, typename B>
concept BinaryOperands =
    Operand<A> && Operand<B> && (VenusTensor<A> || VenusTensor<B>);

template <typename... Ts>
concept Tensors = (VenusTensor<Ts> && ...);

// Named so that an eager op's constraint subsumes its venus op's. Two
// spellings of rank == 2 in two declarations are different constraints
template <typename... Ts>
concept Vectors = Tensors<Ts...> && ((Ts::rank == 1) && ...);
template <typename... Ts>
concept Matrices = Tensors<Ts...> && ((Ts::rank == 2) && ...);
template <typename T>
concept AtLeastMatrix = VenusTensor<T> && (T::rank >= 2);

// The extra constraint of venus::eager. Its ops repeat the constraint of the
// venus:: op and add this one, so the eager op is the more constrained and
// wins when both namespaces are in scope
template <typename... Operands>
concept EagerOperands = EagerRuntime<join_t<Operands...>>;

// What a rule sees of an operand. A scalar is rank 0
template <typename TElem, std::size_t Rank> struct Meta {
  using ElementType = TElem;
  static constexpr std::size_t rank = Rank;
  Shape<Rank> shape;
};

template <typename T> auto meta(const T &operand) {
  using Plain = std::remove_cvref_t<T>;
  if constexpr (Scalar<Plain>) {
    return Meta<Plain, 0>{Shape<0>()};
  } else {
    return Meta<typename Plain::ElementType, Plain::rank>{operand.shape()};
  }
}

// The tensor a rule describes, on a given runtime
template <typename TMeta, typename TRuntime>
using result_t = Tensor<typename TMeta::ElementType, TRuntime, TMeta::rank>;

// Primitive ops ================================================

template <typename Fn, ConstexprString Name> struct Elementwise {
  static constexpr std::string_view name = Name.view();

  // Comparisons give bool, arithmetic gives the common type
  template <typename E1, std::size_t R1, typename E2, std::size_t R2>
  static auto rule(Meta<E1, R1> a, Meta<E2, R2> b) {
    constexpr auto rank = std::max(R1, R2);
    using Elem = std::conditional_t<
        std::is_same_v<std::invoke_result_t<Fn, E1, E2>, bool>, bool,
        std::common_type_t<E1, E2>>;
    return Meta<Elem, rank>{broadcast<rank>(a.shape, b.shape)};
  }
};

using AddOp = Elementwise<std::plus<>, "add">;
using SubOp = Elementwise<std::minus<>, "sub">;
using MulOp = Elementwise<std::multiplies<>, "mul">;
using DivOp = Elementwise<std::divides<>, "div">;
using GtOp = Elementwise<std::greater<>, "gt">;
using GteOp = Elementwise<std::greater_equal<>, "gte">;
using LtOp = Elementwise<std::less<>, "lt">;
using LteOp = Elementwise<std::less_equal<>, "lte">;
using EqOp = Elementwise<std::equal_to<>, "eq">;
using NeqOp = Elementwise<std::not_equal_to<>, "neq">;

// where(predicate, a, b) picks from a where predicate holds, else from b
struct WhereOp {
  static constexpr std::string_view name = "where";

  template <typename EP, std::size_t RP, typename EA, std::size_t RA,
            typename EB, std::size_t RB>
  static auto rule(Meta<EP, RP> p, Meta<EA, RA> a, Meta<EB, RB> b) {
    constexpr auto rank = std::max({RP, RA, RB});
    return Meta<std::common_type_t<EA, EB>, rank>{
        broadcast<rank>(p.shape, a.shape, b.shape)};
  }
};

struct MmOp {
  static constexpr std::string_view name = "mm";

  template <typename E1, typename E2>
  static auto rule(Meta<E1, 2> a, Meta<E2, 2> b) {
    const auto [I, K] = a.shape;
    const auto [K2, J] = b.shape;
    if (K != K2) {
      throw std::invalid_argument(
          std::format("Shape mismatch between tensors in matrix mul: t1 has "
                      "shape {}, whereas t2 has shape {}.",
                      a.shape, b.shape));
    }
    return Meta<std::common_type_t<E1, E2>, 2>{Shape<2>(I, J)};
  }
};

// Sum of the products of all elements, in row-major order
struct InnerOp {
  static constexpr std::string_view name = "inner";

  template <typename E1, std::size_t R1, typename E2, std::size_t R2>
  static auto rule(Meta<E1, R1> a, Meta<E2, R2> b) {
    if (a.shape.count() != b.shape.count()) {
      throw std::invalid_argument(
          std::format("inner needs tensors of equal size, got shapes {} and {}",
                      a.shape, b.shape));
    }
    return Meta<std::common_type_t<E1, E2>, 0>{Shape<0>()};
  }
};

// The rule for ops whose result has the shape and element type of their
// first operand
struct SameAsFirst {
  template <typename E, std::size_t R, typename... Rest>
  static auto rule(Meta<E, R> t, Rest...) {
    return t;
  }
};

struct SortOp : SameAsFirst {
  static constexpr std::string_view name = "sort";
};
struct FillOp : SameAsFirst {
  static constexpr std::string_view name = "fill";
};
struct IotaOp : SameAsFirst {
  static constexpr std::string_view name = "iota";
};
struct EyeLikeOp : SameAsFirst {
  static constexpr std::string_view name = "eye_like";
};

template <std::size_t Dim> struct SumDimOp {
  static constexpr std::string_view name = "sum_dim";

  template <typename E, std::size_t R> static auto rule(Meta<E, R> t) {
    static_assert(Dim < R, "sum dimension cannot be higher than tensor rank");
    auto ext = kernel::extents(t.shape);
    ext[Dim] = 1;
    return Meta<E, R>{Shape<R>(ext)};
  }
};

// The leading ToRank extents of a tensor whose other extents are 1
template <std::size_t ToRank> struct SqueezeOp {
  static constexpr std::string_view name = "squeeze";

  template <typename E, std::size_t R> static auto rule(Meta<E, R> t) {
    if constexpr (ToRank == 0) {
      return Meta<E, 0>{Shape<0>()};
    } else {
      return Meta<E, ToRank>{t.shape.template slice<ToRank>()};
    }
  }
};

// Operand OpIdx of einsum Eqn, with one axis per index letter in the order
// the output uses them and extent 1 for letters it does not have
template <std::size_t OpIdx, ConstexprString Eqn> struct EinsumOperandOp {
  static constexpr std::string_view name = "einsum_operand";

  static constexpr auto eqn = Eqn.view();
  static constexpr auto total_dims = detail::count_total_dimensions(eqn);

  // axis[d] is the operand's axis that output axis d reads, or -1
  static constexpr auto axis = [] {
    constexpr auto occ = detail::compute_occurences(eqn);
    constexpr auto pos_labels = detail::compute_position_labels(
        detail::compute_sorted_position(eqn, occ));
    constexpr auto axes = detail::compute_axes_for_op(eqn, OpIdx);
    std::array<std::int64_t, total_dims> out{};
    for (std::size_t d = 0; d < total_dims; ++d) {
      out[d] = axes[pos_labels[d]];
    }
    return out;
  }();

  template <typename E, std::size_t R> static auto rule(Meta<E, R> t) {
    std::array<std::size_t, total_dims> ext{};
    for (std::size_t d = 0; d < total_dims; ++d) {
      ext[d] = axis[d] == -1 ? 1 : t.shape[static_cast<std::size_t>(axis[d])];
    }
    return Meta<E, total_dims>{Shape<total_dims>(ext)};
  }
};

// Dispatch =====================================================

// A runtime implements an op by specialising Action for it. This primary
// template means it does not, and dispatch composes the op instead. dispatch
// instantiates the primary, so an explicit specialisation declared after a
// call with the same op and runtime fails to compile. A partial
// specialisation declared that late is ill-formed with no diagnostic
template <typename Op, typename TRuntime> struct Action {
  using Unimplemented = void;
};

template <typename Op, typename TRuntime>
concept Implements =
    !requires { typename Action<Op, TRuntime>::Unimplemented; };

// Converts a tensor on runtime From into one on runtime To. other is an
// operand already on To, which carries To's context, such as its graph
template <typename From, typename To> struct Lift;

template <typename> inline constexpr bool dependent_false = false;

namespace detail {
template <typename TRuntime, typename T, typename... Rest>
auto first_on(const T &operand, const Rest &...rest) -> const auto & {
  if constexpr (std::is_same_v<runtime_of_t<T>, TRuntime>) {
    return operand;
  } else {
    return first_on<TRuntime>(rest...);
  }
}

template <typename TRuntime, typename T, typename... All>
auto lift(const T &operand, const All &...all) -> decltype(auto) {
  using From = runtime_of_t<T>;
  if constexpr (std::is_same_v<From, NoRuntime> ||
                std::is_same_v<From, TRuntime>) {
    return (operand);
  } else {
    return Lift<From, TRuntime>::run(operand, first_on<TRuntime>(all...));
  }
}
} // namespace detail

template <typename Op, typename... Operands>
auto dispatch(const Operands &...operands) {
  using Rt = join_t<Operands...>;
  if constexpr (std::is_same_v<Rt, NoJoin>) {
    static_assert(dependent_false<Op>,
                  "venus: the operands' runtimes do not join; move one "
                  "operand to the other's runtime first");
    return NoJoin{};
  } else if constexpr (Implements<Op, Rt>) {
    return Action<Op, Rt>::run(detail::lift<Rt>(operands, operands...)...);
  } else if constexpr (requires { Op::compose(operands...); }) {
    return Op::compose(operands...);
  } else {
    static_assert(dependent_false<Op>,
                  "venus: this runtime has no action for the op, and the op "
                  "has no composition");
  }
}

// Composite ops ================================================
// No runtime needs an action for these. They have no rule, since their result
// is whatever the ops they call return

struct Composite {};

template <typename Op>
concept CompositeOp = std::derived_from<Op, Composite>;

struct DotOp : Composite {
  template <typename A, typename B>
  static auto compose(const A &a, const B &b) {
    return dispatch<InnerOp>(a, b);
  }
};

template <std::size_t... Dims> struct SumDimsOp : Composite {
  template <typename T> static auto compose(const T &t) {
    if constexpr (sizeof...(Dims) == 0) {
      return t;
    } else {
      return []<std::size_t First, std::size_t... Rest>(
                 std::index_sequence<First, Rest...>, const T &tensor) {
        auto result = dispatch<SumDimOp<First>>(tensor);
        ((result = dispatch<SumDimOp<Rest>>(result)), ...);
        return result;
      }(std::index_sequence<Dims...>{}, t);
    }
  }
};

template <std::size_t... SumDims> struct SumproductPairOp : Composite {
  template <typename A, typename B>
  static auto compose(const A &a, const B &b) {
    return dispatch<SumDimsOp<SumDims...>>(dispatch<MulOp>(a, b));
  }
};

template <ConstexprString Eqn> struct EinsumOp : Composite {
  template <typename... Ts> static auto compose(const Ts &...tensors) {
    constexpr auto eqn = Eqn.view();
    static_assert(detail::count_operands(eqn) == sizeof...(Ts),
                  "operand count mismatch");
    constexpr auto num_out =
        detail::count_output_dimensions(eqn, detail::compute_occurences(eqn));

    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      return contract<num_out>(dispatch<EinsumOperandOp<Is, Eqn>>(tensors)...);
    }(std::make_index_sequence<sizeof...(Ts)>{});
  }

private:
  // Sums each index letter out after the last operand that uses it
  template <std::size_t NumOut, typename... Operands>
  static auto contract(const Operands &...operands) {
    const auto &first = std::get<0>(std::tie(operands...));
    constexpr auto first_dims = detail::compute_sum_dims_for_step<0, Eqn>();

    auto current = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      return dispatch<SumDimsOp<first_dims[Is]...>>(first);
    }(std::make_index_sequence<first_dims.size()>{});

    [&]<std::size_t... Steps>(std::index_sequence<Steps...>) {
      auto step = [&]<std::size_t OpIdx>() {
        constexpr auto dims = detail::compute_sum_dims_for_step<OpIdx, Eqn>();
        const auto &next = std::get<OpIdx>(std::tie(operands...));
        current = [&]<std::size_t... Js>(std::index_sequence<Js...>) {
          return dispatch<SumproductPairOp<dims[Js]...>>(current, next);
        }(std::make_index_sequence<dims.size()>{});
      };
      (step.template operator()<Steps + 1>(), ...);
    }(std::make_index_sequence<sizeof...(Operands) - 1>{});

    return dispatch<SqueezeOp<NumOut>>(current);
  }
};

// Public ops ===================================================

#define VENUS_BINARY_OP(op_name, op_type)                                      \
  template <typename A, typename B>                                            \
    requires BinaryOperands<A, B>                                              \
  auto op_name(const A &a, const B &b) {                                       \
    return dispatch<op_type>(a, b);                                            \
  }

VENUS_BINARY_OP(add, AddOp)
VENUS_BINARY_OP(sub, SubOp)
VENUS_BINARY_OP(mul, MulOp)
VENUS_BINARY_OP(div, DivOp)
VENUS_BINARY_OP(gt, GtOp)
VENUS_BINARY_OP(gte, GteOp)
VENUS_BINARY_OP(lt, LtOp)
VENUS_BINARY_OP(lte, LteOp)
VENUS_BINARY_OP(eq, EqOp)
VENUS_BINARY_OP(neq, NeqOp)

#undef VENUS_BINARY_OP

template <typename P, typename A, typename B>
  requires VenusTensor<P> && Operand<A> && Operand<B>
auto where(const P &predicate, const A &a, const B &b) {
  return dispatch<WhereOp>(predicate, a, b);
}

template <typename A, typename B>
  requires Matrices<A, B>
auto mm(const A &a, const B &b) {
  return dispatch<MmOp>(a, b);
}

template <typename A, typename B>
  requires Tensors<A, B>
auto inner(const A &a, const B &b) {
  return dispatch<InnerOp>(a, b);
}

template <typename A, typename B>
  requires Vectors<A, B>
auto dot(const A &a, const B &b) {
  return dispatch<DotOp>(a, b);
}

template <typename T>
  requires Tensors<T>
auto sort(const T &t) {
  return dispatch<SortOp>(t);
}

template <typename T, typename V>
  requires Tensors<T> && Scalar<V>
auto fill(const T &t, V value) {
  return dispatch<FillOp>(t, value);
}

template <typename T, typename V>
  requires Tensors<T> && Scalar<V>
auto iota(const T &t, V start) {
  return dispatch<IotaOp>(t, start);
}

template <typename T>
  requires AtLeastMatrix<T>
auto eye_like(const T &t) {
  return dispatch<EyeLikeOp>(t);
}

template <std::size_t Dim, typename T>
  requires Tensors<T>
auto sum_dim(const T &t) {
  return dispatch<SumDimOp<Dim>>(t);
}

template <std::size_t... Dims, typename T>
  requires Tensors<T>
auto sum_dims(const T &t) {
  return dispatch<SumDimsOp<Dims...>>(t);
}

template <std::size_t... SumDims, typename A, typename B>
  requires Tensors<A, B>
auto sumproduct_pair(const A &a, const B &b) {
  return dispatch<SumproductPairOp<SumDims...>>(a, b);
}

template <ConstexprString Eqn, typename... Ts>
  requires Tensors<Ts...>
auto einsum(const Ts &...tensors) {
  return dispatch<EinsumOp<Eqn>>(tensors...);
}

// Operators forward to the dispatching ops, so they work on every runtime
#define VENUS_OPERATOR(symbol, op_name)                                        \
  template <typename A, typename B>                                            \
    requires BinaryOperands<A, B>                                              \
  auto operator symbol(const A &a, const B &b) {                               \
    return op_name(a, b);                                                      \
  }

VENUS_OPERATOR(+, add)
VENUS_OPERATOR(-, sub)
VENUS_OPERATOR(*, mul)
VENUS_OPERATOR(/, div)
VENUS_OPERATOR(>, gt)
VENUS_OPERATOR(>=, gte)
VENUS_OPERATOR(<, lt)
VENUS_OPERATOR(<=, lte)
VENUS_OPERATOR(==, eq)
VENUS_OPERATOR(!=, neq)

#undef VENUS_OPERATOR

// Host-only functions ==========================================
// These read values, or produce a shape that depends on values, so no rule
// can describe their result. They need a runtime that holds values

template <typename T>
concept ReadableTensor = VenusTensor<T> && Readable<runtime_of_t<T>>;

// A lambda has no node to record and no gradient rule, so transform stays on
// Runtime::CPU
template <typename T, typename Fn>
  requires VenusTensor<T> && std::same_as<runtime_of_t<T>, Runtime::CPU>
auto transform(const T &tensor, Fn &&fn) {
  using Elem = std::invoke_result_t<Fn, typename T::ElementType>;
  if constexpr (T::rank == 0) {
    return Tensor<Elem, Runtime::CPU, 0>(fn(tensor.value()));
  } else {
    auto result = Tensor<Elem, Runtime::CPU, T::rank>(tensor.shape());
    std::ranges::transform(tensor, result.begin(), std::forward<Fn>(fn));
    return result;
  }
}

// True when both tensors have the same shape and elements
template <typename A, typename B>
  requires ReadableTensor<A> && ReadableTensor<B>
auto equal(const A &a, const B &b) -> bool {
  if constexpr (A::rank != B::rank) {
    return false;
  } else {
    if (a.shape() != b.shape()) {
      return false;
    }
    return std::ranges::equal(a, b);
  }
}

// Flat offsets of the elements that convert to true
template <typename T>
  requires ReadableTensor<T> && BoolTensor<T>
auto nonzero_flat(const T &condition) {
  const auto count = static_cast<std::size_t>(std::ranges::count_if(
      condition, [](auto v) { return static_cast<bool>(v); }));

  auto result = Tensor<std::size_t, runtime_of_t<T>, 1>(count);
  std::size_t pos = 0;
  for (std::size_t i = 0; i < condition.size(); ++i) {
    if (static_cast<bool>(condition.data()[i])) {
      result.data()[pos++] = i;
    }
  }
  return result;
}

// One row per element that converts to true, holding its index
template <typename T>
  requires ReadableTensor<T> && BoolTensor<T>
auto nonzero(const T &condition) {
  constexpr auto rank = T::rank;
  const auto count = static_cast<std::size_t>(std::ranges::count_if(
      condition, [](auto v) { return static_cast<bool>(v); }));

  auto result = Tensor<std::size_t, runtime_of_t<T>, 2>(count, rank);
  auto *out = result.data();
  std::size_t row = 0;
  for (std::size_t i = 0; i < condition.size(); ++i) {
    if (static_cast<bool>(condition.data()[i])) {
      const auto idx = condition.shape().offsetToIdx(i);
      for (std::size_t d = 0; d < rank; ++d) {
        out[(row * rank) + d] = idx[d];
      }
      ++row;
    }
  }
  return result;
}

template <typename T>
  requires ReadableTensor<T> && BoolTensor<T>
auto where(const T &predicate) {
  if constexpr (T::rank == 1) {
    return nonzero_flat(predicate);
  } else {
    return nonzero(predicate);
  }
}

} // namespace venus
