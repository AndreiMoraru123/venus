#pragma once

#ifndef VENUS_MODULE
#include <concepts>
#include <cstddef>
#include <format>
#include <optional>
#include <ostream>
#include <print>
#endif
#include <venus/concepts.hpp>
#include <venus/tensor/tensor.hpp>

namespace venus {

// Read by every print; a format spec such as {:t4e1.0} overrides it for one
struct PrintOptions {
  std::size_t threshold = 1000; // past this many elements, print only the ends
  std::size_t edgeItems = 3;    // elements printed at each end
  int precision = 2;            // digits after the point for floating point
};

inline PrintOptions printOptions{};

template <typename TElem, typename TOut>
auto formatElement(TOut out, const TElem &elem, const PrintOptions &options)
    -> TOut {
  if constexpr (StringLike<TElem>) {
    return std::format_to(out, "\"{}\"", elem);
  } else if constexpr (CharLike<TElem>) {
    return std::format_to(out, "'{:c}'", elem);
  } else if constexpr (std::same_as<TElem, bool>) {
    return std::format_to(out, "{:d}", elem);
  } else if constexpr (std::floating_point<TElem>) {
    return std::format_to(out, "{:.{}f}", elem, options.precision);
  } else {
    return std::format_to(out, "{}", elem);
  }
}

template <VenusTensor TTensor, typename TOut>
auto formatTo(TOut out, const TTensor &tensor, const PrintOptions &options)
    -> TOut {
  using TElem = typename TTensor::ElementType;
  out = std::format_to(out, AnyTensorView<TTensor> ? "venus::TensorView("
                                                   : "venus::Tensor(");
  if constexpr (ScalarTensor<TTensor>) {
    out = formatElement(out, tensor.value(), options);
    return std::format_to(out, ")");
  } else {
    const auto formatElems = [&out, &options](auto first, auto last) {
      for (auto it = first; it != last; ++it) {
        if (it != first)
          out = std::format_to(out, ", ");
        out = formatElement(out, static_cast<TElem>(*it), options);
      }
    };

    const std::size_t count = tensor.shape().count();
    out = std::format_to(out, "[");
    if (count > options.threshold and count > 2 * options.edgeItems) {
      const auto edge = static_cast<std::ptrdiff_t>(options.edgeItems);
      formatElems(tensor.begin(), tensor.begin() + edge);
      out = std::format_to(out, ", ..., ");
      formatElems(tensor.end() - edge, tensor.end());
    } else {
      formatElems(tensor.begin(), tensor.end());
    }
    return std::format_to(out, "], shape={})", tensor.shape());
  }
}

// {:t<threshold>e<edgeItems>.<precision>}, in any order, each optional
struct TensorFormatter {
  constexpr auto parse(std::format_parse_context &ctx) {
    auto it = ctx.begin();
    const auto number = [&] {
      if (it == ctx.end() or *it < '0' or *it > '9')
        throw std::format_error("venus::Tensor: t, e and . take a number");
      std::size_t value = 0;
      for (; it != ctx.end() and *it >= '0' and *it <= '9'; ++it)
        value = value * 10 + static_cast<std::size_t>(*it - '0');
      return value;
    };
    while (it != ctx.end() and *it != '}') {
      switch (*it++) {
      case 't':
        threshold = number();
        break;
      case 'e':
        edgeItems = number();
        break;
      case '.':
        precision = static_cast<int>(number());
        break;
      default:
        throw std::format_error(
            "venus::Tensor: the format spec takes t, e and .");
      }
    }
    return it;
  }

  template <VenusTensor TTensor>
  auto format(const TTensor &tensor, std::format_context &ctx) const {
    venus::PrintOptions options = venus::printOptions;
    options.threshold = threshold.value_or(options.threshold);
    options.edgeItems = edgeItems.value_or(options.edgeItems);
    options.precision = precision.value_or(options.precision);
    return venus::formatTo(ctx.out(), tensor, options);
  }

private:
  std::optional<std::size_t> threshold;
  std::optional<std::size_t> edgeItems;
  std::optional<int> precision;
};
} // namespace venus

template <typename TElem, typename TDevice, std::size_t Rank>
struct std::formatter<venus::Tensor<TElem, TDevice, Rank>>
    : venus::TensorFormatter {};

template <typename TElem, typename TDevice, std::size_t Rank>
struct std::formatter<venus::TensorView<TElem, TDevice, Rank>>
    : venus::TensorFormatter {};

template <typename TElem, typename TDevice, std::size_t Rank>
struct std::formatter<venus::Selection<TElem, TDevice, Rank>>
    : venus::TensorFormatter {};

namespace venus {

template <typename TElem, typename TDevice, std::size_t Rank>
auto operator<<(std::ostream &os, const Tensor<TElem, TDevice, Rank> &tensor)
    -> std::ostream & {
  std::print(os, "{}", tensor);
  return os;
}

template <typename TElem, typename TDevice, std::size_t Rank>
auto operator<<(std::ostream &os, const TensorView<TElem, TDevice, Rank> &view)
    -> std::ostream & {
  std::print(os, "{}", view);
  return os;
}

} // namespace venus
