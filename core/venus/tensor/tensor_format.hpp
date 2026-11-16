#pragma once

#ifndef VENUS_MODULE
#include <concepts>
#include <cstddef>
#include <format>
#include <ostream>
#include <print>
#endif
#include <venus/concepts.hpp>
#include <venus/tensor/tensor.hpp>

namespace venus {

// Read by every print; with(tensor, options) replaces it for one print
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
  out = std::format_to(out, "venus::Tensor(");
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

template <VenusTensor TTensor> struct WithPrintOptions {
  const TTensor &tensor;
  PrintOptions options;
};

// Holds a reference to tensor, so print it in the same statement
template <VenusTensor TTensor>
auto with(const TTensor &tensor, PrintOptions options)
    -> WithPrintOptions<TTensor> {
  return {tensor, options};
}

} // namespace venus

template <typename TElem, typename TDevice, std::size_t Rank>
struct std::formatter<venus::Tensor<TElem, TDevice, Rank>> {
  constexpr auto parse(std::format_parse_context &ctx) { return ctx.begin(); }

  auto format(const venus::Tensor<TElem, TDevice, Rank> &tensor,
              std::format_context &ctx) const {
    return venus::formatTo(ctx.out(), tensor, venus::printOptions);
  }
};

template <venus::VenusTensor TTensor>
struct std::formatter<venus::WithPrintOptions<TTensor>> {
  constexpr auto parse(std::format_parse_context &ctx) { return ctx.begin(); }

  auto format(const venus::WithPrintOptions<TTensor> &printed,
              std::format_context &ctx) const {
    return venus::formatTo(ctx.out(), printed.tensor, printed.options);
  }
};

namespace venus {

template <typename TElem, typename TDevice, std::size_t Rank>
auto operator<<(std::ostream &os, const Tensor<TElem, TDevice, Rank> &tensor)
    -> std::ostream & {
  std::print(os, "{}", tensor);
  return os;
}

template <VenusTensor TTensor>
auto operator<<(std::ostream &os, const WithPrintOptions<TTensor> &printed)
    -> std::ostream & {
  std::print(os, "{}", printed);
  return os;
}

} // namespace venus
