#pragma once
// Index bookkeeping for einsum, all of it evaluated at compile time

#ifndef VENUS_MODULE
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#endif
#include <venus/str.hpp>

inline constexpr std::size_t NUMBER_OF_LETTERS = 'z' - 'a' + 1;
using AlphabetArray = std::array<std::int64_t, NUMBER_OF_LETTERS>;
using PositionLabels = std::array<std::size_t, NUMBER_OF_LETTERS>;

namespace venus::detail {

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

} // namespace venus::detail
