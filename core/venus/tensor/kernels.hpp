#pragma once
// CPU kernels. Each writes into memory it is handed and never allocates, so
// an eager op, an executor and an arena can share one kernel.

#ifndef VENUS_MODULE
#include <algorithm>
#include <array>
#include <cstddef>
#include <type_traits>
#endif
#include <venus/concepts.hpp>
#include <venus/tensor/shape.hpp>

namespace venus::kernel {

// A tensor operand: contiguous row-major data and its shape
template <typename TElem, std::size_t Rank> struct Input {
  const TElem *data;
  Shape<Rank> shape;
};

template <std::size_t Rank> using Index = std::array<std::size_t, Rank>;

template <std::size_t Rank>
constexpr auto extents(const Shape<Rank> &shape) -> Index<Rank> {
  Index<Rank> out{};
  if constexpr (Rank > 0) {
    std::ranges::copy(shape, out.begin());
  }
  return out;
}

template <std::size_t Rank>
constexpr void next(Index<Rank> &idx, const Index<Rank> &ext) {
  for (std::size_t i = Rank; i-- > 0;) {
    if (++idx[i] < ext[i]) {
      return;
    }
    idx[i] = 0;
  }
}

// Row-major offset into an operand of rank RankIn broadcast against an index
// of rank Rank, aligned from the right
template <std::size_t Rank, std::size_t RankIn>
constexpr auto offset(const Index<Rank> &idx, const Index<RankIn> &ext) {
  std::size_t off = 0;
  for (std::size_t i = 0; i < RankIn; ++i) {
    off = (off * ext[i]) + (ext[i] == 1 ? 0 : idx[i + Rank - RankIn]);
  }
  return off;
}

template <typename T> struct IsInput : std::false_type {};
template <typename TElem, std::size_t Rank>
struct IsInput<Input<TElem, Rank>> : std::true_type {};

// out[i] = fn(in...[i]), each input a scalar or broadcast to shape. out may
// alias an input of the same shape
template <typename Fn, typename TOut, std::size_t Rank, typename... Ins>
void elementwise(Fn fn, TOut *out, const Shape<Rank> &shape, const Ins &...in) {
  const auto n = shape.count();
  auto flat = [](const auto &operand, std::size_t i) {
    if constexpr (IsInput<std::remove_cvref_t<decltype(operand)>>::value) {
      return operand.data[i];
    } else {
      return operand;
    }
  };
  auto same_shape = [&shape](const auto &operand) {
    if constexpr (IsInput<std::remove_cvref_t<decltype(operand)>>::value) {
      return operand.shape == shape;
    } else {
      return true;
    }
  };
  if ((same_shape(in) && ...)) {
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = fn(flat(in, i)...);
    }
    return;
  }

  const auto ext = extents(shape);
  auto at = [](const auto &operand, const Index<Rank> &idx) {
    if constexpr (IsInput<std::remove_cvref_t<decltype(operand)>>::value) {
      return operand.data[offset(idx, extents(operand.shape))];
    } else {
      return operand;
    }
  };
  Index<Rank> idx{};
  for (std::size_t i = 0; i < n; ++i, next(idx, ext)) {
    out[i] = fn(at(in, idx)...);
  }
}

// out (I x J) = a (I x K) * b (K x J)
template <typename TOut, typename A, typename B>
void mm(TOut *out, const A *a, const B *b, std::size_t I, std::size_t K,
        std::size_t J) {
  std::fill(out, out + (I * J), TOut{});
  for (std::size_t i = 0; i < I; ++i) {
    for (std::size_t k = 0; k < K; ++k) {
      const auto a_ik = a[(i * K) + k];
      if (a_ik == 0) {
        continue;
      }
      for (std::size_t j = 0; j < J; ++j) {
        out[(i * J) + j] += a_ik * b[(k * J) + j];
      }
    }
  }
}

template <typename TOut, typename A, typename B>
void inner(TOut *out, const A *a, const B *b, std::size_t n) {
  TOut acc{};
  for (std::size_t i = 0; i < n; ++i) {
    acc += a[i] * b[i];
  }
  *out = acc;
}

template <typename TOut, typename TIn>
void sort(TOut *out, const TIn *in, std::size_t n) {
  std::copy(in, in + n, out);
  std::sort(out, out + n);
}

template <typename TOut, typename TValue>
void fill(TOut *out, std::size_t n, TValue value) {
  std::fill(out, out + n, value);
}

template <typename TOut, typename TValue>
void iota(TOut *out, std::size_t n, TValue start) {
  for (std::size_t i = 0; i < n; ++i, ++start) {
    out[i] = start;
  }
}

// Ones where the last two indices are equal, zeros elsewhere
template <typename TOut, std::size_t Rank>
  requires(Rank >= 2)
void eye(TOut *out, const Shape<Rank> &shape) {
  const auto ext = extents(shape);
  Index<Rank> idx{};
  for (std::size_t i = 0; i < shape.count(); ++i, next(idx, ext)) {
    out[i] = idx[Rank - 2] == idx[Rank - 1] ? TOut{1} : TOut{0};
  }
}

// out has the shape of in with dimension Dim reduced to 1
template <std::size_t Dim, typename TOut, typename TIn, std::size_t Rank>
void sum_dim(TOut *out, const TIn *in, const Shape<Rank> &shape) {
  auto out_ext = extents(shape);
  out_ext[Dim] = 1;
  std::fill(out, out + Shape<Rank>(out_ext).count(), TOut{});

  const auto ext = extents(shape);
  Index<Rank> idx{};
  for (std::size_t i = 0; i < shape.count(); ++i, next(idx, ext)) {
    auto out_idx = idx;
    out_idx[Dim] = 0;
    out[offset(out_idx, out_ext)] += in[i];
  }
}

// Copies in into out, whose axes are a permutation of in's axes plus extents
// of 1. axis[i] is the axis of in that out's axis i reads, or -1
template <typename TOut, typename TIn, std::size_t RankOut, std::size_t RankIn,
          typename Axes>
void gather_axes(TOut *out, const Shape<RankOut> &out_shape, const TIn *in,
                 const Shape<RankIn> &in_shape, const Axes &axis) {
  const auto out_ext = extents(out_shape);
  const auto in_ext = extents(in_shape);
  Index<RankOut> idx{};
  for (std::size_t i = 0; i < out_shape.count(); ++i, next(idx, out_ext)) {
    Index<RankIn> in_idx{};
    for (std::size_t d = 0; d < RankOut; ++d) {
      if (axis[d] != -1) {
        in_idx[static_cast<std::size_t>(axis[d])] = idx[d];
      }
    }
    out[i] = in[offset(in_idx, in_ext)];
  }
}

} // namespace venus::kernel
