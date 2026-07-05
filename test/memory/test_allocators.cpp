#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <venus/memory/allocators.hpp>
#include <venus/tensor/tensor.hpp>

using namespace venus;

TEST_CASE("Memory Pool return strictly 64-byte aligned pointers",
          "[allocators][pool][simd]") {
  auto ptr1 = Allocator<Device::CPU>::alloc<float>(10);
  auto ptr2 = Allocator<Device::CPU>::alloc<int>(1000);
  auto ptr3 = Allocator<Device::CPU>::alloc<int8_t>(2000);

  auto addr1 = reinterpret_cast<std::uintptr_t>(ptr1.get());
  auto addr2 = reinterpret_cast<std::uintptr_t>(ptr2.get());
  auto addr3 = reinterpret_cast<std::uintptr_t>(ptr3.get());

  REQUIRE(addr1 % 64 == 0);
  REQUIRE(addr2 % 64 == 0);
  REQUIRE(addr3 % 64 == 0);
}

TEST_CASE("Allocations are corrently rounded up to BlockSize buckets",
          "[allocators][bucket]") {
  std::uintptr_t saved_addr = 0;

  {
    // Request 100 floats (400 bytes).
    // will be rounded to 1024-byte bucket
    auto ptr1 = Allocator<Device::CPU>::alloc<float>(100);
    saved_addr = reinterpret_cast<std::uintptr_t>(ptr1.get());
  } // memory goes back into the 1024-byte bucket in the deque

  {
    // Request 200 floats (800 bytes).
    // will be rounded to 1024-byte bucket
    auto ptr2 = Allocator<Device::CPU>::alloc<float>(200);
    auto new_addr = reinterpret_cast<std::uintptr_t>(ptr2.get());

    // It should reuse the exact same memory block
    REQUIRE(new_addr == saved_addr);
  }

  {
    // Request 300 floats (1200 bytes).
    // will be rounded to 2048-byte bucket
    auto ptr3 = Allocator<Device::CPU>::alloc<float>(300);
    auto overflow_addr = reinterpret_cast<std::uintptr_t>(ptr3.get());

    // It should NOT reuse the previous memory block
    REQUIRE(overflow_addr != saved_addr);
  }
}

TEST_CASE("Recycled memory blocks are zero-initialized",
          "[allocators][pool][tensor]") {
  std::uintptr_t first_addr = 0;

  {
    auto tensor = Tensor<float, Device::CPU, 1>(100);
    tensor.fill(99.0f);

    REQUIRE(tensor[0] == 99.0f);
    REQUIRE(tensor[99] == 99.0f);

    first_addr = reinterpret_cast<std::uintptr_t>(tensor.data());
  }

  {
    auto tensor = Tensor<float, Device::CPU, 1>(100);
    auto second_addr = reinterpret_cast<std::uintptr_t>(tensor.data());

    // It should reuse the exact same memory block
    REQUIRE(second_addr == first_addr);

    // Memory should be zero-initialized
    REQUIRE(tensor[0] == 0.0f);
    REQUIRE(tensor[99] == 0.0f);
  }
}
