#pragma once

#include <cstddef>
#include <cstdlib>
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
  static std::shared_ptr<TElem> alloc(std::size_t p_elemSize) {
    // TElem *raw_buf =
    //     static_cast<TElem *>(::operator new(p_elemSize * sizeof(TElem)));
    TElem *raw_buf = std::allocator<TElem>{}.allocate(p_elemSize);

    for (std::size_t i = 0; i < p_elemSize; ++i) {
      new (raw_buf + i) TElem();
    }

    return std::shared_ptr<TElem>(raw_buf, [p_elemSize](TElem *ptr) {
      if constexpr (!std::is_trivially_destructible_v<TElem>) {
        for (std::size_t i = 0; i < p_elemSize; ++i) {
          ptr[i].~TElem();
        }
      }
      // ::operator delete(ptr);
      std::allocator<TElem>{}.deallocate(ptr, p_elemSize);
    });
  }
};
} // namespace venus
#else

constexpr auto BLOCK_SIZE = 1024;
constexpr auto ALIGN_SIZE = 64;

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
          std::free(block);
        }
        blocks.clear();
      }
    }
  };

  template <typename T> struct Deleter {
    Deleter(std::deque<void *> &p_refPool, std::size_t p_count)
        : m_refPool(p_refPool), m_count(p_count) {}
    void operator()(void *p_val) const {
      if constexpr (!std::is_trivially_destructible_v<T>) {
        T *typed = static_cast<T *>(p_val);
        for (std::size_t i = 0; i < m_count; i++) {
          typed[i].~T();
        }
      }
      std::scoped_lock<std::mutex> guard(m_mutex);
      m_refPool.push_back(p_val);
    }

  private:
    std::deque<void *> &m_refPool;
    std::size_t m_count;
  };

public:
  template <typename T, std::size_t BlockSize = BLOCK_SIZE,
            std::size_t AlignSize = ALIGN_SIZE>
  static auto alloc(std::size_t p_elemSize) -> std::shared_ptr<T> {
    static_assert((BlockSize & (BlockSize - 1)) == 0,
                  "BlockSize must be a power of 2");
    static_assert((AlignSize & (AlignSize - 1)) == 0,
                  "AlignSize must be a power of 2");
    static_assert((BlockSize & (AlignSize - 1)) == 0,
                  "BlockSize must be a multiple of AlignSize");

    if (p_elemSize == 0) {
      return nullptr;
    }

    p_elemSize *= sizeof(T);
    if (p_elemSize & (BlockSize - 1)) {
      p_elemSize = ((p_elemSize / BlockSize) + 1) * BlockSize;
    }

    std::scoped_lock<std::mutex> guard(m_mutex);

    T *raw_buf = nullptr;
    auto &slot = m_pool.memBuffer[p_elemSize];

    if (slot.empty()) {
      raw_buf = static_cast<T *>(std::aligned_alloc(AlignSize, p_elemSize));
    } else {
      void *mem = slot.back();
      slot.pop_back();
      raw_buf = (T *)mem;
    }

    std::size_t count = p_elemSize / sizeof(T);
    for (std::size_t i = 0; i < count; ++i) {
      new (raw_buf + i) T();
    }
    return std::shared_ptr<T>(raw_buf, Deleter<T>(slot, count));
  }

private:
  inline static std::mutex m_mutex;
  inline static MemoryPool m_pool;
};
}; // namespace venus

#endif