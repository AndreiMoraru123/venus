#include <catch2/catch_test_macros.hpp>

#include "flux/core/functional.hpp"
#include "sequential.h"
#include <cstddef>
#include <tuple>
#include <type_traits>

using namespace venus;

template <typename... Params> struct Vector;

TEST_CASE("Sequential::At type selection", "[sequential]") {
  using Check = Vector<int, short, double>;
  STATIC_REQUIRE(std::is_same_v<Sequential::At<Check, 0>, int>);
  STATIC_REQUIRE(std::is_same_v<Sequential::At<Check, 1>, short>);
  STATIC_REQUIRE(std::is_same_v<Sequential::At<Check, 2>, double>);
}
