
#include <type_traits>

template <typename T>
using RemoveConstRef = std::remove_const_t<std::remove_reference_t<T>>;