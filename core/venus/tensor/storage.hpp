#pragma once

#ifndef VENUS_MODULE
#include <cstddef>
#endif
#include <venus/memory/contiguous_memory.hpp>
#include <venus/runtime.hpp>

namespace venus {

// What a tensor holds besides its shape, chosen by its runtime
template <typename TElem, typename TRuntime, std::size_t Rank>
struct StorageFor;

template <typename TElem, std::size_t Rank>
struct StorageFor<TElem, Runtime::CPU, Rank> {
  using type = ContiguousMemory<TElem, Runtime::CPU>;
};

template <typename TElem, typename TRuntime, std::size_t Rank>
using storage_t = typename StorageFor<TElem, TRuntime, Rank>::type;

} // namespace venus
