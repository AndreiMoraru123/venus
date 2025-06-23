#include "core/memory/device.hpp"
#include <cassert>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <core/tensor/tensor.hpp>
#include <numeric>
#include <ranges>
#include <vector>

using namespace venus;

TEST_CASE("Tensor as Range", "[tensor][range]") {
  SECTION("Basic Range Requirements") {
    auto tensor = Tensor<float, Device::CPU, 2>(3, 2);

    STATIC_REQUIRE(std::ranges::range<decltype(tensor)>);
    STATIC_REQUIRE(std::ranges::sized_range<decltype(tensor)>);
    STATIC_REQUIRE(std::ranges::viewable_range<decltype(tensor)>);
    STATIC_REQUIRE(std::ranges::random_access_range<decltype(tensor)>);
  }

  SECTION("Random Access Iterators Ops") {
    const auto shape = Shape(2, 3);
    auto tensor = Tensor<int, Device::CPU, 2>(shape);
#if _cpp_lib_ranges >= 202110L
    std::ranges::iota(tensor, 10);
#else
    std::iota(tensor.begin(), tensor.end(), 10);
#endif

    auto it = tensor.begin();
    REQUIRE(*(it + 2) == 12);
    REQUIRE(it[3] == 13);

    auto end_it = tensor.end();
    REQUIRE(end_it - it == shape.Count());

    auto back_it = end_it - 1;
    REQUIRE(*back_it == 15);

    --back_it;
    REQUIRE(*back_it == 14);
  }

  SECTION("Iterate over Tensor") {
    auto tensor = Tensor<int, Device::CPU, 2>(2, 3);

    int value = 0;
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        tensor[i, j] = value++;
      }
    }

    std::vector<int> collected;
    for (auto element : tensor) {
      collected.push_back(element);
    }

    REQUIRE(collected == std::vector{0, 1, 2, 3, 4, 5});
  }

  SECTION("Iterate over const tensor") {
    auto tensor = Tensor<int, Device::CPU, 2>(2, 3);

    std::vector<int> collected;
    for (const auto &element : tensor) {
      collected.push_back(element);
    }

    REQUIRE(collected == std::vector{0, 0, 0, 0, 0, 0});
  }

  SECTION("Tensor Size") {
    auto tensor = Tensor<float, Device::CPU, 3>(2, 3, 4);
    REQUIRE(std::ranges::size(tensor) == 24);
    REQUIRE(tensor.size() == 24);
  }

  SECTION("Modify Tensor through ElementProxy") {
    auto tensor = Tensor<float, Device::CPU, 3>(2, 3, 4);
    auto iterator = tensor.begin();
    auto proxy = *iterator;

    float value = 25.0f;
    proxy = value;
    proxy = 30.0f;

    const float &const_ref = value;
    proxy = const_ref;

    const auto &const_proxy = proxy;
    auto &&moved_const_proxy = std::move(const_proxy);
    moved_const_proxy = 25.0f;

    REQUIRE(tensor[0, 0, 0] == 25.0f);
  }

  SECTION("Ranges Algos - Fill") {
    auto tensor = Tensor<float, Device::CPU, 3>(2, 3, 4);

    std::ranges::fill(tensor, 25.0f);
    for (auto element : tensor) {
      REQUIRE(element == 25.0f);
    }
  }

  SECTION("Range Algos - Accumulation") {
    auto tensor = Tensor<int, Device::CPU, 2>(2, 3);
#if _cpp_lib_ranges >= 202110L
    std::ranges::iota(tensor, 1);
#else
    std::iota(tensor.begin(), tensor.end(), 1);
#endif

    auto sum = std::ranges::fold_left(tensor, 0, std::plus<>{});
    REQUIRE(sum == 21);
  }

  SECTION("Ranges Algos - Transform and Copy") {
    auto tensor = Tensor<int, Device::CPU, 2>(2, 2);

#if _cpp_lib_ranges >= 202110L
    std::ranges::iota(tensor, 1);
#else
    std::iota(tensor.begin(), tensor.end(), 1);
#endif

    std::ranges::transform(tensor, tensor.begin(), [](auto proxy) {
      return static_cast<int>(proxy) * 2;
    });

    std::vector<int> expected = {2, 4, 6, 8};
    std::vector<int> actual;
    std::ranges::copy(tensor, std::back_inserter(actual));

    REQUIRE(actual == expected);
  }

  SECTION("Range Algos - Find and Count") {
    auto tensor = Tensor<int, Device::CPU, 2>(2, 3);
#if _cpp_lib_ranges >= 202110L
    std::ranges::iota(tensor, 1);
#else
    std::iota(tensor.begin(), tensor.end(), 1);
#endif

    auto it = std::ranges::find(tensor, 4);
    REQUIRE(it != tensor.end());
    REQUIRE(*it == 4);

    auto count = std::ranges::count_if(tensor, [](auto x) { return x > 3; });
    REQUIRE(count == 3);
  }

  SECTION("Ranges Views") {
    auto tensor = Tensor<int, Device::CPU, 2>(2, 2);

#if _cpp_lib_ranges >= 202110L
    std::ranges::iota(tensor, 1);
#else
    std::iota(tensor.begin(), tensor.end(), 1);
#endif

    auto result =
        tensor |
        std::views::transform([](auto x) { return static_cast<int>(x) * 2; }) |
        std::views::filter([](int x) { return x > 5; }) | std::views::take(1);

    auto it = result.begin();
    REQUIRE(*it == 6);
  }
}