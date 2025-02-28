#include <catch2/catch_test_macros.hpp>

#include "sequential.h"
#include <type_traits>

using namespace venus;

template <typename... Params> struct Vector;

TEST_CASE("Sequential::At type selection", "[sequential]") {
  using Check = Vector<short, short, double>;
  STATIC_REQUIRE(std::is_same_v<Sequential::At<Check, 0>, short>);
  STATIC_REQUIRE(std::is_same_v<Sequential::At<Check, 1>, short>);
  STATIC_REQUIRE(std::is_same_v<Sequential::At<Check, 2>, double>);
}

TEST_CASE("Sequential::Order type indexing", "[order]") {
  using Check = Vector<int, short, double>;
  STATIC_REQUIRE(Sequential::Order<Check, int> == 0);
  STATIC_REQUIRE(Sequential::Order<Check, short> == 1);
  STATIC_REQUIRE(Sequential::Order<Check, double> == 2);
}
