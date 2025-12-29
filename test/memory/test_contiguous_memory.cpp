#include <catch2/catch_test_macros.hpp>

#include <venus/memory/contiguous_memory.hpp>

using namespace venus;

TEST_CASE("ContiguousMemory basics", "[memory]") {
  SECTION("Construction") {
    constexpr std::size_t size = 8;
    ContiguousMemory<float *, Device::CPU> memo(size);

    REQUIRE(memo.size() == size);
    REQUIRE(memo.rawMemory() != nullptr);
    REQUIRE_FALSE(memo.isShared());
  }

  SECTION("Equality Ops") {
    ContiguousMemory<float *, Device::CPU> memo1(10);
    ContiguousMemory<float *, Device::CPU> memo2(10);

    REQUIRE(memo1 == memo1);
    REQUIRE(memo1 != memo2);
  }

  SECTION("Shift Op") {
    ContiguousMemory<char, Device::CPU> memo(100);
    auto shifted = memo.shift(20);

    REQUIRE(shifted.size() == 80);
    REQUIRE(shifted.rawMemory() == memo.rawMemory() + 20);
    REQUIRE(shifted.isShared());
    REQUIRE(memo.isShared());
  }

  SECTION("Shared state") {
    ContiguousMemory<char, Device::CPU> memo(50);

    REQUIRE_FALSE(memo.isShared());
    {
      auto copy = memo;
      REQUIRE(memo.isShared());
      REQUIRE(copy.isShared());
      REQUIRE(memo == copy);
    }
    REQUIRE_FALSE(memo.isShared());
  }

  SECTION("Moved ownership") {
    ContiguousMemory<char, Device::CPU> memo(50);

    REQUIRE_FALSE(memo.isShared());
    {
      auto copy = std::move(memo);
      REQUIRE_FALSE(memo.isShared());
      REQUIRE_FALSE(copy.isShared());
      REQUIRE(memo != copy);
    }
    REQUIRE_FALSE(memo.isShared());
  }
}