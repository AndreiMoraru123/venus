#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <venus/memory/device.hpp>
#include <venus/str.hpp>
#include <venus/tensor/shape.hpp>

constexpr std::size_t NUMBER_OF_LETTERS = 'z' - 'a' + 1;
using AlphabetArray = std::array<std::int64_t, NUMBER_OF_LETTERS>;
using PositionLabels = std::array<std::size_t, NUMBER_OF_LETTERS>;

namespace venus {
template <typename T>
concept VenusTensor = requires {
  typename std::remove_cvref_t<T>::ElementType;
  typename std::remove_cvref_t<T>::DeviceType;
  { std::remove_cvref_t<T>::rank } -> std::convertible_to<std::size_t>;
};

template <typename T>
concept Scalar = std::is_arithmetic_v<std::remove_cvref_t<T>>;

template <typename T>
concept ScalarTensor =
    VenusTensor<std::remove_cvref_t<T>> && (std::remove_cvref_t<T>::rank == 0);

template <typename T>
concept MDTensor =
    VenusTensor<std::remove_cvref_t<T>> && (std::remove_cvref_t<T>::rank > 0);

template <typename T>
concept BoolTensor =
    VenusTensor<std::remove_cvref_t<T>> &&
    std::is_convertible_v<typename std::remove_cvref_t<T>::ElementType, bool>;
} // namespace venus

#define REGISTER_BINARY_OP(op_name, std_op, op_symbol)                         \
  template <typename T1, typename T2>                                          \
    requires(Scalar<T1> || VenusTensor<T1>) && (Scalar<T2> || VenusTensor<T2>) \
  auto op_name(T1 &&t1, T2 &&t2) {                                             \
    /* Tensor op Tensor */                                                     \
    if constexpr (MDTensor<T1> && MDTensor<T2>) {                              \
      return detail::binary_elementwise_op(std::std_op{}, t1, t2);             \
    } /* Tensor/ScalarTensor op Scalar */                                      \
    else if constexpr (VenusTensor<T1> && Scalar<T2>) {                        \
      return transform(t1, [s = t2](auto &&t) { return t op_symbol s; });      \
    } /* Scalar op Tensor/ScalarTensor */                                      \
    else if constexpr (Scalar<T1> && VenusTensor<T2>) {                        \
      return transform(t2, [s = t1](auto &&t) { return s op_symbol t; });      \
    } /* Tensor op ScalarTensor */                                             \
    else if constexpr (MDTensor<T1> && ScalarTensor<T2>) {                     \
      return op_name(t1, t2.value());                                          \
    } /* ScalarTensor op Tensor */                                             \
    else if constexpr (ScalarTensor<T1> && MDTensor<T2>) {                     \
      return op_name(t1.value(), t2);                                          \
    } /* ScalarTensor op ScalarTensor */                                       \
    else if constexpr (ScalarTensor<T1> && ScalarTensor<T2>) {                 \
      return detail::binary_elementwise_op(std::std_op{}, t1, t2);             \
    }                                                                          \
  }

namespace venus::eager {

// Details =====================================================
namespace detail {

template <typename Op, template <typename, typename, std::size_t> class Tensor,
          typename Elem1, typename Dev1, std::size_t Rank1, typename Elem2,
          typename Dev2, std::size_t Rank2>
auto binary_elementwise_op(Op op, const Tensor<Elem1, Dev1, Rank1> &t1,
                           const Tensor<Elem2, Dev2, Rank2> &t2) {

  using ResultElementType = std::common_type_t<Elem1, Elem2>;

  if constexpr (Rank1 == 0 && Rank2 == 0) {
    return Tensor<ResultElementType, Dev1, 0>(op(t1.value(), t2.value()));
  } else {
    constexpr std::size_t RankOut = std::max(Rank1, Rank2);
    auto out_shape = broadcast<RankOut>(t1.shape(), t2.shape());

    auto result = Tensor<ResultElementType, Dev1, RankOut>(out_shape);
    auto out_ptr = result.data();

    // Does not need broadcasting
    if constexpr (Rank1 == Rank2) {
      if (t1.shape() == t2.shape()) {
        const auto *t1_ptr = t1.data();
        const auto *t2_ptr = t2.data();
        for (std::size_t flat = 0; flat < result.size(); ++flat) {
          out_ptr[flat] = op(t1_ptr[flat], t2_ptr[flat]);
        }
        return result;
      }
    }

    // Needs broadcasting
    for (std::size_t flat = 0; flat < result.size(); ++flat) {
      const auto out_idx = out_shape.offsetToIdx(flat);

      const auto idx1 = project_broadcast_idx(out_idx, t1.shape());
      const auto idx2 = project_broadcast_idx(out_idx, t2.shape());

      out_ptr[flat] = op(t1[idx1], t2[idx2]);
    }

    return result;
  }
}

template <typename Op, template <typename, typename, std::size_t> class Tensor,
          typename Elem1, typename Dev1, std::size_t Rank1, typename Elem2,
          typename Dev2, std::size_t Rank2, typename Elem3, typename Dev3,
          std::size_t Rank3>
auto ternary_elementwise_op(Op op, const Tensor<Elem1, Dev1, Rank1> &t1,
                            const Tensor<Elem2, Dev2, Rank2> &t2,
                            const Tensor<Elem3, Dev3, Rank3> &t3) {

  using ResultElementType = std::common_type_t<Elem1, Elem2, Elem3>;

  if constexpr (Rank1 == 0 && Rank2 == 0 && Rank3 == 0) {
    return Tensor<ResultElementType, Dev1, 0>(
        op(t1.value(), t2.value(), t3.value()));
  } else {
    constexpr std::size_t RankOut = std::max({Rank1, Rank2, Rank3});
    auto out_shape = broadcast<RankOut>(t1.shape(), t2.shape(), t3.shape());

    auto result = Tensor<ResultElementType, Dev1, RankOut>(out_shape);
    auto out_ptr = result.data();

    // Does not need broadcasting
    if constexpr (Rank1 == Rank2 && Rank2 == Rank3) {
      if (t1.shape() == t2.shape() && t2.shape() == t3.shape()) {
        const auto *t1_ptr = t1.data();
        const auto *t2_ptr = t2.data();
        const auto *t3_ptr = t3.data();
        for (std::size_t flat = 0; flat < result.size(); ++flat) {
          out_ptr[flat] = op(t1_ptr[flat], t2_ptr[flat], t3_ptr[flat]);
        }
        return result;
      }
    }

    // Needs broadcasting
    for (std::size_t flat = 0; flat < result.size(); ++flat) {
      const auto out_idx = out_shape.offsetToIdx(flat);

      const auto idx1 = project_broadcast_idx(out_idx, t1.shape());
      const auto idx2 = project_broadcast_idx(out_idx, t2.shape());
      const auto idx3 = project_broadcast_idx(out_idx, t3.shape());

      out_ptr[flat] = op(t1[idx1], t2[idx2], t3[idx3]);
    }

    return result;
  }
}

consteval auto count_operands(const std::string_view eqn) {
  auto lhs = eqn.substr(0, eqn.find("->"));
  return std::ranges::count(lhs, ',') + 1;
}

consteval auto compute_occurences(const std::string_view eqn) -> AlphabetArray {
  AlphabetArray occ{};
  const auto lhs = eqn.substr(0, eqn.find("->"));
  for (char c : lhs)
    if (c != ',')
      occ[c - 'a']++;

  return occ;
}

consteval auto compute_last_occurence(const std::string_view eqn)
    -> AlphabetArray {
  AlphabetArray last{};
  last.fill(-1);
  const auto lhs = eqn.substr(0, eqn.find("->"));
  std::int64_t operand = 0;
  for (char c : lhs) {
    if (c == ',') {
      operand++;
      continue;
    }
    last[c - 'a'] = operand;
  }
  return last;
}

consteval auto compute_sorted_position(const std::string_view eqn,
                                       const AlphabetArray &occ)
    -> AlphabetArray {
  AlphabetArray pos{};
  pos.fill(-1);
  const auto arrow = eqn.find("->");
  std::int64_t dim = 0;
  if (arrow != std::string_view::npos) {
    const auto rhs = eqn.substr(arrow + 2);
    for (char c : rhs)
      pos[c - 'a'] = dim++;
  } else {
    for (std::size_t i = 0; i < NUMBER_OF_LETTERS; i++)
      if (occ[i] == 1)
        pos[i] = dim++;
  }
  // summation indices
  for (std::size_t i = 0; i < NUMBER_OF_LETTERS; i++)
    if (occ[i] > 0 && pos[i] == -1)
      pos[i] = dim++;
  return pos;
}

consteval auto count_total_dimensions(std::string_view eqn) -> std::size_t {
  const auto occ = compute_occurences(eqn);
  return std::ranges::count_if(occ, [](auto &&val) { return val > 0; });
}

consteval auto count_output_dimensions(std::string_view eqn,
                                       const AlphabetArray &occ)
    -> std::size_t {
  const auto arrow = eqn.find("->");
  if (arrow != std::string_view::npos) {
    return eqn.size() - (arrow + 2);
  }
  return std::ranges::count(occ, 1);
}

consteval auto compute_position_labels(const AlphabetArray &sorted_pos)
    -> PositionLabels {
  PositionLabels pos_labels{};
  for (std::size_t letter = 0; letter < NUMBER_OF_LETTERS; letter++) {
    if (sorted_pos[letter] >= 0)
      pos_labels[static_cast<std::size_t>(sorted_pos[letter])] = letter;
  }
  return pos_labels;
}

consteval auto count_dims_for_op(std::string_view eqn, std::size_t op_idx) {
  const auto lhs = eqn.substr(0, eqn.find("->"));
  std::size_t op = 0, n = 0;
  for (char c : lhs) {
    if (c == ',') {
      ++op;
      continue;
    }
    if (op == op_idx)
      n++;
  }
  return n;
}

consteval auto compute_axes_for_op(std::string_view eqn, std::size_t op_idx)
    -> AlphabetArray {
  AlphabetArray axes{};
  axes.fill(-1);

  const auto lhs = eqn.substr(0, eqn.find("->"));
  std::size_t op = 0, dim = 0;
  for (char c : lhs) {
    if (c == ',') {
      ++op;
      dim = 0;
      continue;
    }
    if (op == op_idx) {
      const auto letter = static_cast<std::size_t>(c - 'a');
      if (axes[letter] != -1) {
        throw "einsum repeated label in one operand is diagonal, unsupported";
      }
      axes[letter] = static_cast<std::int64_t>(dim++);
    }
  }

  return axes;
}

consteval auto count_sum_dims_for_op(std::size_t op_idx,
                                     const AlphabetArray &last_occ,
                                     const AlphabetArray &sorted_pos,
                                     std::size_t num_output_dims)
    -> std::size_t {
  std::size_t n = 0;
  for (std::size_t letter = 0; letter < NUMBER_OF_LETTERS; ++letter) {
    if (last_occ[letter] == static_cast<std::int64_t>(op_idx) &&
        sorted_pos[letter] >= static_cast<std::int64_t>(num_output_dims)) {
      ++n;
    }
  }
  return n;
}

template <std::size_t OpIdx, ConstexprString Eqn>
consteval auto compute_sum_dims_for_step() {
  constexpr auto eqn = Eqn.view();
  constexpr auto occ = compute_occurences(eqn);
  constexpr auto last_occ = compute_last_occurence(eqn);
  constexpr auto sorted_pos = compute_sorted_position(eqn, occ);
  constexpr auto num_out = count_output_dimensions(eqn, occ);

  constexpr auto count =
      count_sum_dims_for_op(OpIdx, last_occ, sorted_pos, num_out);
  std::array<std::size_t, count> dims{};
  std::size_t n = 0;

  for (std::size_t letter = 0; letter < NUMBER_OF_LETTERS; ++letter) {
    if (last_occ[letter] == static_cast<std::int64_t>(OpIdx) &&
        sorted_pos[letter] >= static_cast<std::int64_t>(num_out)) {
      dims[n++] = static_cast<std::size_t>(sorted_pos[letter]);
    }
  }
  return dims;
}

template <std::size_t ToRank,
          template <typename, typename, std::size_t> class Tensor,
          typename Elem, typename Dev, std::size_t FromRank>
  requires VenusTensor<Tensor<Elem, Dev, FromRank>>
auto squeeze_to_rank(const Tensor<Elem, Dev, FromRank> &tensor) {
  if constexpr (ToRank == 0) {
    return tensor.toScalar();
  } else {
    return tensor.template reshape<ToRank>(
        tensor.shape().template slice<ToRank>());
  }
}

template <std::size_t OpIdx, ConstexprString Eqn,
          template <typename, typename, std::size_t> class Tensor, Scalar Elem,
          typename Dev, std::size_t Rank>
  requires VenusTensor<Tensor<Elem, Dev, Rank>>
auto homogenize_operand(const Tensor<Elem, Dev, Rank> &t) {
  constexpr auto eqn = Eqn.view();
  constexpr auto occ = detail::compute_occurences(eqn);
  constexpr auto sorted_pos = detail::compute_sorted_position(eqn, occ);
  constexpr auto total_dims = detail::count_total_dimensions(eqn);
  constexpr auto pos_labels = detail::compute_position_labels(sorted_pos);
  constexpr auto axes = detail::compute_axes_for_op(eqn, OpIdx);

  std::array<std::size_t, total_dims> homo_dims{};
  for (std::size_t i = 0; i < total_dims; ++i) {
    const auto letter = pos_labels[i];
    const auto axis = axes[letter];
    if (axis != -1) {
      homo_dims[i] = t.shape()[axis];
    } else {
      homo_dims[i] = 1;
    }
  }

  const auto homo_shape = Shape<total_dims>(homo_dims);

  auto project_homo_idx =
      [pos_labels, axes](const std::array<std::size_t, total_dims> &out_idx) {
        std::array<std::size_t, Rank> orig_idx{};
        for (std::size_t i = 0; i < total_dims; ++i) {
          const auto letter = pos_labels[i];
          const auto axis = axes[letter];
          if (axis != -1) {
            orig_idx[static_cast<std::size_t>(axis)] = out_idx[i];
          }
        }
        return orig_idx;
      };

  auto homogenized = Tensor<Elem, Dev, total_dims>(homo_shape);
  for (std::size_t flat = 0; flat < homogenized.size(); ++flat) {
    const auto out_idx = homo_shape.offsetToIdx(flat);
    const auto orig_idx = project_homo_idx(out_idx);
    homogenized.data()[flat] = t[orig_idx];
  }

  return homogenized;
}

} // namespace detail

// Copy Transform
template <template <typename, typename, std::size_t> class Tensor, Scalar Elem,
          typename Dev, std::size_t Rank, typename Fn>
  requires VenusTensor<Tensor<Elem, Dev, Rank>>
auto transform(const Tensor<Elem, Dev, Rank> &tensor, Fn &&fn) {
  static_assert(std::is_same_v<Dev, Device::CPU>,
                "Transform is currently only supported on CPU");

  using ResultElementType = std::invoke_result_t<Fn, Elem>;

  if constexpr (Rank == 0) {
    return Tensor<ResultElementType, Dev, 0>(fn(tensor.value()));
  } else {
    auto result = Tensor<ResultElementType, Dev, Rank>(tensor.shape());
    std::ranges::transform(tensor, result.begin(), std::forward<Fn>(fn));
    return result;
  }
}

REGISTER_BINARY_OP(add, plus, +)
REGISTER_BINARY_OP(sub, minus, -)
REGISTER_BINARY_OP(mul, multiplies, *)
REGISTER_BINARY_OP(div, divides, /)
REGISTER_BINARY_OP(gt, greater, >)
REGISTER_BINARY_OP(gte, greater_equal, >=)
REGISTER_BINARY_OP(lt, less, <)
REGISTER_BINARY_OP(lte, less_equal, <=)
REGISTER_BINARY_OP(eq, equal_to, ==)
REGISTER_BINARY_OP(neq, not_equal_to, !=)

// Copy Sort
template <template <typename, typename, std::size_t> class Tensor, Scalar Elem,
          typename Dev, std::size_t Rank>
  requires VenusTensor<Tensor<Elem, Dev, Rank>>
auto sort(const Tensor<Elem, Dev, Rank> &tensor) {
  static_assert(std::is_same_v<Dev, Device::CPU>,
                "Sort is currently only supported on CPU");
  auto copy = tensor.clone();
  std::ranges::sort(copy);
  return copy;
}

// All equal
template <template <typename, typename, std::size_t> class Tensor, Scalar Elem1,
          typename Dev1, std::size_t Rank1, Scalar Elem2, typename Dev2,
          std::size_t Rank2>
  requires VenusTensor<Tensor<Elem1, Dev1, Rank1>> &&
           VenusTensor<Tensor<Elem2, Dev2, Rank2>>
auto equal(const Tensor<Elem1, Dev1, Rank1> &t1,
           const Tensor<Elem2, Dev2, Rank2> &t2) -> bool {
  if (t1.shape() != t2.shape()) {
    return false;
  }
  return std::ranges::equal(t1, t2);
}

// Inner product
template <template <typename, typename, std::size_t> class Tensor, Scalar Elem1,
          typename Dev1, Scalar Elem2, typename Dev2, std::size_t Rank1,
          std::size_t Rank2>
  requires VenusTensor<Tensor<Elem1, Dev1, Rank1>> &&
           VenusTensor<Tensor<Elem2, Dev2, Rank2>>
auto inner(const Tensor<Elem1, Dev1, Rank1> &t1,
           const Tensor<Elem2, Dev2, Rank2> &t2) {
  using ResultElementType = std::common_type_t<Elem1, Elem2>;
  auto product =
      std::inner_product(t1.begin(), t1.end(), t2.begin(), ResultElementType{});
  return Tensor<ResultElementType, Dev1, 0>(product);
}

// Dot product
template <template <typename, typename, std::size_t> class Tensor, Scalar Elem1,
          typename Dev1, Scalar Elem2, typename Dev2>
  requires VenusTensor<Tensor<Elem1, Dev1, 1>> &&
           VenusTensor<Tensor<Elem2, Dev2, 1>>
auto dot(const Tensor<Elem1, Dev1, 1> &t1, const Tensor<Elem2, Dev2, 1> &t2) {
  return inner(t1, t2);
}

// Out-Of-Place Arange
template <template <typename, typename, std::size_t> class Tensor, Scalar Elem,
          Scalar Idx, typename Dev, std::size_t Rank>
  requires VenusTensor<Tensor<Elem, Dev, Rank>>
auto iota(const Tensor<Elem, Dev, Rank> &tensor, Idx i) {
  auto result = Tensor<Elem, Dev, Rank>(tensor.shape());
#if _cpp_lib_ranges >= 202110L
  std::ranges::iota(result, i);
#else
  std::iota(result.begin(), result.end(), i);
#endif
  return result;
}

// Out-Of-Place Identity
template <template <typename, typename, std::size_t> class Tensor, Scalar Elem,
          typename Dev, std::size_t Rank>
  requires VenusTensor<Tensor<Elem, Dev, Rank>> && (Rank > 2)
auto eye_like(const Tensor<Elem, Dev, Rank> &tensor) {
  auto result = Tensor<Elem, Dev, Rank>(tensor.shape());
  result.eye();
  return result;
}

// Out-Of-Place Fill
template <template <typename, typename, std::size_t> class Tensor, Scalar Elem,
          Scalar Idx, typename Dev, std::size_t Rank>
  requires VenusTensor<Tensor<Elem, Dev, Rank>>
auto fill(const Tensor<Elem, Dev, Rank> &tensor, Idx i) {
  auto result = Tensor<Elem, Dev, Rank>(tensor.shape());
#if _cpp_lib_ranges >= 202110L
  std::ranges::fill(result, i);
#else
  std::fill(result.begin(), result.end(), i);
#endif
  return result;
}

// Matrix Multiplication (2D)
template <template <typename, typename, std::size_t> class Tensor, Scalar Elem1,
          Scalar Elem2, typename Dev>
  requires VenusTensor<Tensor<Elem1, Dev, 2>> &&
           VenusTensor<Tensor<Elem2, Dev, 2>>
auto mm(const Tensor<Elem1, Dev, 2> &t1, const Tensor<Elem2, Dev, 2> &t2) {
  static_assert(std::is_same_v<Dev, Device::CPU>,
                "MatMul is currently only supported on CPU");

  using ResultElementType = std::common_type_t<Elem1, Elem2>;

  const auto [I, K] = t1.shape();
  const auto [K2, J] = t2.shape();

  if (K != K2) {
    throw std::invalid_argument(
        std::format("Shape mismatch between tensors in matrix mul: t1 has "
                    "shape {}, whereas t2 has shape {}.",
                    t1.shape(), t2.shape()));
  }

  auto t3 = Tensor<ResultElementType, Dev, 2>(I, J);

  // TODO: This is optimized for row major layout
  for (std::size_t i = 0; i < I; i++) {
    for (std::size_t k = 0; k < K; k++) {
      if (t1[i, k] == 0) {
        continue;
      }
      for (std::size_t j = 0; j < J; j++) {
        t3[i, j] += t1[i, k] * t2[k, j];
      }
    }
  }

  return t3;
}

template <template <typename, typename, std::size_t> class Tensor, Scalar Elem,
          typename Dev, std::size_t Rank>
  requires BoolTensor<Tensor<Elem, Dev, Rank>>
auto nonzero_flat(const Tensor<Elem, Dev, Rank> &condition) {
  const auto nz_count = static_cast<std::size_t>(std::ranges::count_if(
      condition, [](auto v) { return static_cast<bool>(v); }));

  auto result = Tensor<std::size_t, Dev, 1>(nz_count);
  auto out_ptr = result.data();

  std::size_t pos = 0;
  const auto indices = std::views::iota(std::size_t{0}, condition.size());
  std::ranges::for_each(std::views::zip(condition, indices),
                        [out_ptr, pos](auto &&pair) mutable {
                          const auto &[cond_val, idx] = pair;
                          if (static_cast<bool>(cond_val)) {
                            out_ptr[pos++] = idx;
                          }
                        });

  return result;
}

template <template <typename, typename, std::size_t> class Tensor, Scalar Elem,
          typename Dev, std::size_t Rank>
  requires BoolTensor<Tensor<Elem, Dev, Rank>>
auto nonzero(const Tensor<Elem, Dev, Rank> &condition) {
  const auto nz_count = static_cast<std::size_t>(std::ranges::count_if(
      condition, [](auto v) { return static_cast<bool>(v); }));

  auto result = Tensor<std::size_t, Dev, 2>(nz_count, Rank);
  auto out_ptr = result.data();

  std::size_t row = 0;
  for (std::size_t i = 0; i < condition.size(); ++i) {
    if (static_cast<bool>(condition.data()[i])) {
      const auto idx = condition.shape().offsetToIdx(i);
      for (std::size_t d = 0; d < Rank; ++d) {
        out_ptr[(row * Rank) + d] = idx[d];
      }
      ++row;
    }
  }

  return result;
}

template <template <typename, typename, std::size_t> class Tensor, Scalar Elem,
          typename Dev, std::size_t Rank>
  requires BoolTensor<Tensor<Elem, Dev, Rank>>
auto where(const Tensor<Elem, Dev, Rank> &predicate) {
  if constexpr (Rank == 1) {
    return nonzero_flat(predicate);
  } else {
    return nonzero(predicate);
  }
}

template <typename T1, typename T2, typename T3>
  requires VenusTensor<T1> && (VenusTensor<T2> || Scalar<T2>) &&
           (VenusTensor<T3> || Scalar<T3>)
auto where(T1 &&predicate, T2 &&true_tensor, T3 &&false_tensor) {
  const auto pred_val = [&] {
    if constexpr (ScalarTensor<T1>) {
      return predicate.value();
    } else {
      return std::forward<T1>(predicate);
    }
  }();

  const auto true_val = [&] {
    if constexpr (ScalarTensor<T2>) {
      return true_tensor.value();
    } else {
      return std::forward<T2>(true_tensor);
    }
  }();

  const auto false_val = [&] {
    if constexpr (ScalarTensor<T3>) {
      return false_tensor.value();
    } else {
      return std::forward<T3>(false_tensor);
    }
  }();

  // Tensor, Tensor, Tensor
  if constexpr (MDTensor<T1> && MDTensor<T2> && MDTensor<T3>) {
    return detail::ternary_elementwise_op(
        [](auto &&t1, auto &&t2, auto &&t3) { return t1 ? t2 : t3; }, pred_val,
        true_val, false_val);
  }

  // Tensor, Scalar, Scalar
  else if constexpr (MDTensor<T1> && Scalar<T2> && Scalar<T3>) {
    return transform(pred_val, [s2 = true_val, s3 = false_val](auto &&t1) {
      return t1 ? s2 : s3;
    });
  }

  // Tensor, Tensor, Scalar
  else if constexpr (MDTensor<T1> && MDTensor<T2> && Scalar<T3>) {
    return detail::binary_elementwise_op(
        [s3 = false_val](auto &&t1, auto &&t2) { return t1 ? t2 : s3; },
        pred_val, true_val);
  }

  // Tensor, Scalar, Tensor
  else if constexpr (MDTensor<T1> && Scalar<T2> && MDTensor<T3>) {
    return detail::binary_elementwise_op(
        [s2 = true_val](auto &&t1, auto &&t3) { return t1 ? s2 : t3; },
        pred_val, false_val);
  }
}

template <std::size_t Dim,
          template <typename, typename, std::size_t> class Tensor, Scalar Elem,
          typename Dev, std::size_t Rank>
  requires VenusTensor<Tensor<Elem, Dev, Rank>>
auto sum_dim(const Tensor<Elem, Dev, Rank> &t) -> Tensor<Elem, Dev, Rank> {
  static_assert(Dim < Rank, "sum dimension cannot be higher than tensor rank");

  const auto &in_shape = t.shape();
  std::array<std::size_t, Rank> out_ext;
  for (std::size_t i = 0; i < Rank; i++) {
    out_ext[i] = (i == Dim) ? 1 : in_shape[i];
  }

  auto out_shape = Shape<Rank>(out_ext);
  auto result = Tensor<Elem, Dev, Rank>(out_shape);

  for (auto [flat, val] :
       std::views::zip(std::views::iota(std::size_t{0}, t.size()), t)) {
    auto midx = in_shape.offsetToIdx(flat);
    midx[Dim] = 0;
    result[midx] += val;
  }

  return result;
}

template <std::size_t... Dims,
          template <typename, typename, std::size_t> class Tensor, Scalar Elem,
          typename Dev, std::size_t Rank>
  requires VenusTensor<Tensor<Elem, Dev, Rank>>
auto sum_dims(const Tensor<Elem, Dev, Rank> &t) -> Tensor<Elem, Dev, Rank> {
  if constexpr (sizeof...(Dims) == 0) {
    return t.clone();
  } else {
    return []<std::size_t First, std::size_t... Rest>(
               std::index_sequence<First, Rest...>, const auto &tensor) {
      auto result = sum_dim<First>(tensor);
      ((result = sum_dim<Rest>(result)), ...);
      return result;
    }(std::index_sequence<Dims...>{}, t);
  }
}

// Sumproduct pair
template <std::size_t... SumDims,
          template <typename, typename, std::size_t> class Tensor, Scalar Elem1,
          typename Dev1, Scalar Elem2, typename Dev2, std::size_t Rank1,
          std::size_t Rank2>
  requires VenusTensor<Tensor<Elem1, Dev1, Rank1>> &&
           VenusTensor<Tensor<Elem2, Dev2, Rank2>>
auto sumproduct_pair(const Tensor<Elem1, Dev1, Rank1> &t1,
                     const Tensor<Elem2, Dev2, Rank2> &t2) {
  auto product = t1 * t2;
  return sum_dims<SumDims...>(product);
}

template <ConstexprString Eqn, std::size_t NumOut,
          typename... HomogenizedTensors>
auto _einsum_contract(HomogenizedTensors... tensors) {
  const auto &t0 = tensors...[0];
  constexpr auto initial_sum_dims = detail::compute_sum_dims_for_step<0, Eqn>();

  const auto initial_result =
      [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        if constexpr (sizeof...(Is) > 0) {
          return sum_dims<initial_sum_dims[Is]...>(t0);
        } else {
          return t0;
        }
      }(std::make_index_sequence<initial_sum_dims.size()>{});

  const auto final_contracted = [&]<std::size_t... Is>(
                                    std::index_sequence<Is...>) {
    auto current = std::move(initial_result);
    auto step = [&]<std::size_t OpIdx>() {
      constexpr auto sum_dims = detail::compute_sum_dims_for_step<OpIdx, Eqn>();
      const auto &next_op = tensors...[OpIdx];
      current = [&]<std::size_t... Js>(std::index_sequence<Js...>) {
        return sumproduct_pair<sum_dims[Js]...>(current, next_op);
        ;
      }(std::make_index_sequence<sum_dims.size()>{});
    };
    (step.template operator()<Is + 1>(), ...);
    return current;
  }(std::make_index_sequence<sizeof...(HomogenizedTensors) - 1>{});

  return detail::squeeze_to_rank<NumOut>(final_contracted);
}

template <ConstexprString Eqn,
          template <typename, typename, std::size_t> class... Tensors,
          typename... Ts, typename... Devs, std::size_t... Ranks>
  requires(VenusTensor<Tensors<Ts, Devs, Ranks>> && ...)
auto einsum(const Tensors<Ts, Devs, Ranks> &...tensors) {
  static_assert((std::is_same_v<Devs, Device::CPU> && ...),
                "Einsum is currently only supported on CPU");
  constexpr auto eqn = Eqn.view();
  constexpr auto occ = detail::compute_occurences(eqn);
  constexpr auto num_out = detail::count_output_dimensions(eqn, occ);

  static_assert(detail::count_operands(eqn) == sizeof...(Tensors),
                "operand count mismatch");

  return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
    return _einsum_contract<Eqn, num_out>(
        detail::homogenize_operand<Is, Eqn>(tensors)...);
  }(std::make_index_sequence<sizeof...(Tensors)>{});
}

} // namespace venus::eager

#undef REGISTER_BINARY_OP
