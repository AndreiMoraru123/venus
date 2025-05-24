#pragma once

#include "device.hpp"
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace venus {
template <typename TDevice> struct Allocator;

template <> struct Allocator<Device::CPU> {

private:
  struct MemoryPool {
    std::unordered_map<std::size_t, std::deque<void *>> memBuffer;
    ~MemoryPool() {
      for (auto &pool : memBuffer) {
        auto &blocks = pool.second;
        for (const auto &block : blocks) {
          char *buf = (char *)(block);
          delete[] buf;
        }
        blocks.clear();
      }
    }
  };

  struct Deleter {
    Deleter(std::deque<void *> &p_refPool) : m_refPool(p_refPool) {}
    void operator()(void *p_val) const {
      std::lock_guard<std::mutex> guard(m_mutex);
      m_refPool.push_back(p_val);
    }

  private:
    std::deque<void *> &m_refPool;
  };

public:
  template <typename T, std::size_t BlockSize = 1024>
  static std::shared_ptr<T> Allocate(std::size_t p_elemSize) {
    static_assert((BlockSize & (BlockSize - 1)) == 0,
                  "BlockSize must be a power of 2");

    if (p_elemSize == 0)
      return nullptr;

    p_elemSize *= sizeof(T);
    if (p_elemSize & (BlockSize - 1)) {
      p_elemSize = ((p_elemSize / BlockSize) + 1) * BlockSize;
    }

    std::lock_guard<std::mutex> guard(m_mutex);

    auto &slot = m_pool.memBuffer[p_elemSize];
    if (slot.empty()) {
      auto raw_buf = (T *)new char[p_elemSize];
      return std::shared_ptr<T>(raw_buf, Deleter(slot));
    } else {
      void *mem = slot.back();
      slot.pop_back();
      return std::shared_ptr<T>((T *)mem, Deleter(slot));
    }
  }

private:
  inline static std::mutex m_mutex;
  inline static MemoryPool m_pool;
};
}; // namespace venus
