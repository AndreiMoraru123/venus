#pragma once

namespace venus {

template <typename T> struct Identity_ {
  using type = T;
};

template <bool curr, typename TNext> inline constexpr bool AndValue = false;

template <typename TNext>
inline constexpr bool AndValue<true, TNext> = TNext::value;

template <bool curr, typename TNext> inline constexpr bool OrValue = true;

template <typename TNext>
inline constexpr bool OrValue<false, TNext> = TNext::value;
} // namespace venus