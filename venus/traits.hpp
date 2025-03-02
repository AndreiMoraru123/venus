
#include <type_traits>

namespace venus {

template <typename T>
using RemoveConstRef = std::remove_const_t<std::remove_reference_t<T>>;

template <typename T> struct Identity_ {
  using type = T;
};

template <bool curr, typename TNext> static constexpr bool AndValue = false;

template <typename TNext>
static constexpr bool AndValue<true, TNext> = TNext::value;
} // namespace venus