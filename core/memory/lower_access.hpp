#pragma once

#include "../traits.hpp"
#include <utility>

namespace venus {
template <typename TData> struct LowLevelAccess;

template <typename TData> auto LowLevel(TData &&p) {
  using RawType = RemoveConstRef<TData>;
  return LowLevelAccess<RawType>(std::forward<TData>(p));
}
} // namespace venus