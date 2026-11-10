#include <catch2/catch_test_macros.hpp>
#include <venus/memory/device.hpp>

#include <ranges>
#include <stdexcept>
#include <string>
#include <vector>
#include <venus/tensor/tensor.hpp>

using namespace venus;

namespace {
template <typename R> auto collect(R &&range) {
  return std::ranges::to<std::vector>(std::forward<R>(range));
}

template <typename T, typename... Selectors>
concept Sliceable =
    requires(Selectors... selectors) { std::declval<T>()[selectors...]; };
} // namespace

TEST_CASE("Tensor axis slicing", "[tensor][slice]") {
  auto tensor = Tensor<int, Device::CPU, 3>(2, 2, 3);
  tensor.iota(0); // [[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]]

  SECTION("Range Requirements") {
    using Slice = decltype(tensor[0, 0, all]);
    STATIC_REQUIRE(std::ranges::random_access_range<Slice>);
    STATIC_REQUIRE(std::ranges::sized_range<Slice>);
    STATIC_REQUIRE(std::ranges::view<Slice>);
  }

  SECTION("Last axis") {
    REQUIRE(collect(tensor[0, 1, all]) == std::vector{3, 4, 5});
    REQUIRE(collect(tensor[1, 0, all]) == std::vector{6, 7, 8});
  }

  SECTION("Middle axis") {
    REQUIRE(collect(tensor[0, all, 2]) == std::vector{2, 5});
  }

  SECTION("First axis") {
    REQUIRE(collect(tensor[all, 1, 2]) == std::vector{5, 11});
  }

  SECTION("Writes go through to the tensor") {
    for (auto &element : tensor[all, 0, 0]) {
      element = -1;
    }
    REQUIRE(tensor[0, 0, 0] == -1);
    REQUIRE(tensor[1, 0, 0] == -1);
    REQUIRE(tensor[0, 0, 1] == 1);
  }

  SECTION("Const tensor") {
    const auto &view = tensor;
    auto slice = view[1, all, 1];
    STATIC_REQUIRE(std::same_as<std::ranges::range_reference_t<decltype(slice)>,
                                const int &>);
    REQUIRE(collect(slice) == std::vector{7, 10});
  }

  SECTION("Out of bounds index throws") {
    REQUIRE_THROWS_AS((tensor[2, 0, all]), std::out_of_range);
  }

  SECTION("Element indexing still works") { REQUIRE(tensor[1, 1, 2] == 11); }

  SECTION("Rejected selectors") {
    using T = Tensor<int, Device::CPU, 3>;
    STATIC_REQUIRE(Sliceable<T &, int, int, all_t>);
    STATIC_REQUIRE_FALSE(Sliceable<T &, int, all_t, all_t>);
    STATIC_REQUIRE_FALSE(Sliceable<T &, int, all_t>);
    STATIC_REQUIRE_FALSE(Sliceable<T, int, int, all_t>);
  }
}

TEST_CASE("String tensor axis slicing", "[tensor][slice]") {
  auto tensor = Tensor<std::string, Device::CPU, 3>(1, 2, 3);
  std::ranges::fill(tensor, "x");
  tensor[0, 1, 2] = "y";

  auto row = tensor[0, 1, all];
  REQUIRE(std::ranges::size(row) == 3);
  REQUIRE(collect(row) == std::vector<std::string>{"x", "x", "y"});
}

TEST_CASE("Rank 1 tensor axis slicing", "[tensor][slice]") {
  auto tensor = Tensor<int, Device::CPU, 1>(4);
  tensor.iota(1);
  REQUIRE(collect(tensor[all]) == std::vector{1, 2, 3, 4});
}
