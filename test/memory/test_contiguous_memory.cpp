#include <catch2/catch_test_macros.hpp>

#include <core/memory/contiguous_memory.hpp>

using namespace venus;

TEST_CASE("ContiguousMemory basics", "[memory]") {
  SECTION("Construction") {
    constexpr std::size_t size = 8;
    ContiguousMemory<float *, Device::CPU> memo(size);

    REQUIRE(memo.Size() == size);
    REQUIRE(memo.RawMemory() != nullptr);
    REQUIRE_FALSE(memo.IsShared());
  }

  SECTION("Equality Ops") {
    ContiguousMemory<float *, Device::CPU> memo1(10);
    ContiguousMemory<float *, Device::CPU> memo2(10);

    REQUIRE(memo1 == memo1);
    REQUIRE(memo1 != memo2);
  }

  SECTION("Shift Op") {
    ContiguousMemory<char, Device::CPU> memo(100);
    auto shifted = memo.Shift(20);

    REQUIRE(shifted.Size() == 80);
    REQUIRE(shifted.RawMemory() == memo.RawMemory() + 20);
    REQUIRE(shifted.IsShared());
    REQUIRE(memo.IsShared());
  }

  SECTION("Shared state") {
    ContiguousMemory<char, Device::CPU> memo(50);

    REQUIRE_FALSE(memo.IsShared());
    {
      auto copy = memo;
      REQUIRE(memo.IsShared());
      REQUIRE(copy.IsShared());
      REQUIRE(memo == copy);
    }
    REQUIRE_FALSE(memo.IsShared());
  }

  SECTION("Moved ownership") {
    ContiguousMemory<char, Device::CPU> memo(50);

    REQUIRE_FALSE(memo.IsShared());
    {
      auto copy = std::move(memo);
      REQUIRE(not memo.IsShared());
      REQUIRE(not copy.IsShared());
      REQUIRE(memo != copy);
    }
    REQUIRE_FALSE(memo.IsShared());
  }
}