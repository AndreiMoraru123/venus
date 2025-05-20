#pragma once

#include "device.hpp"
#include <cstddef>
#include <memory>

namespace venus {
template <typename TDevice> struct Allocator;

template <> struct Allocator<Device::CPU> {
  // TODO: Replace with a memory pool (mempool)
  template <typename TElem>
  static std::shared_ptr<TElem> Allocate(std::size_t p_elemSize) {
    return std::shared_ptr<TElem>(new TElem[p_elemSize],
                                  [](TElem *ptr) { delete[] ptr; });
  }
};
}; // namespace venus
