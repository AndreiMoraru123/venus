#pragma once

namespace venus {

template <typename T> struct Identity_ {
  using type = T;
};

template <bool curr, typename TNext> static constexpr bool AndValue = false;

template <typename TNext>
static constexpr bool AndValue<true, TNext> = TNext::value;

template <bool curr, typename TNext> static constexpr bool OrValue = true;

template <typename TNext>
static constexpr bool OrValue<false, TNext> = TNext::value;
} // namespace venus