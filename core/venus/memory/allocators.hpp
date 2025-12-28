#pragma once

#include <cstddef>
#include <cstring>
#include <memory>
#include <venus/memory/device.hpp>

#ifdef VENUS_INTERPRETER
// Simple allocator for repl interpreter

namespace venus {
template <typename TDevice> struct Allocator;
template <>

struct Allocator<Device::CPU> {
  template <typename TElem>
  static std::shared_ptr<TElem> Allocate(std::size_t p_elemSize) {
    TElem *raw_buf = new TElem[p_elemSize];

    if constexpr (std::is_trivially_constructible_v<TElem>) {
      std::memset(raw_buf, 0, p_elemSize * sizeof(TElem));
    } else {
      for (std::size_t i = 0; i < p_elemSize; ++i) {
        new (raw_buf + i) TElem();
      }
    }
    return std::shared_ptr<TElem>(raw_buf, [](TElem *ptr) { delete[] ptr; });
  }
};
} // namespace venus
#else

constexpr auto BLOCK_SIZE = 1024;

// Memory pool allocator for compiled venus

#include <deque>
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
  template <typename T, std::size_t BlockSize = BLOCK_SIZE>
  static auto Allocate(std::size_t p_elemSize) -> std::shared_ptr<T> {
    static_assert((BlockSize & (BlockSize - 1)) == 0,
                  "BlockSize must be a power of 2");

    if (p_elemSize == 0) {
      return nullptr;
    }

    p_elemSize *= sizeof(T);
    if (p_elemSize & (BlockSize - 1)) {
      p_elemSize = ((p_elemSize / BlockSize) + 1) * BlockSize;
    }

    std::lock_guard<std::mutex> guard(m_mutex);

    T *raw_buf = nullptr;
    auto &slot = m_pool.memBuffer[p_elemSize];

    if (slot.empty()) {
      raw_buf = (T *)new char[p_elemSize];
    } else {
      void *mem = slot.back();
      slot.pop_back();
      raw_buf = (T *)mem;
    }

    if constexpr (std::is_trivially_constructible_v<T>) {
      std::memset(raw_buf, 0, p_elemSize);
    } else {
      std::size_t count = p_elemSize / sizeof(T);
      for (std::size_t i = 0; i < count; ++i) {
        new (raw_buf + i) T();
      }
    }
    return std::shared_ptr<T>(raw_buf, Deleter(slot));
  }

private:
  inline static std::mutex m_mutex;
  inline static MemoryPool m_pool;
};
}; // namespace venus

#endif