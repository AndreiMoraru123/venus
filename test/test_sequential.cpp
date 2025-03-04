#include <catch2/catch_test_macros.hpp>

#include <type_traits>
#include <venus/sequential.hpp>

using namespace venus;

template <typename... Params> struct Vector;

TEST_CASE("Sequential::At type selection", "[sequential]") {
  using Check = Vector<int, short, double>;
  STATIC_REQUIRE(std::is_same_v<Sequential::At<Check, 0>, int>);
  STATIC_REQUIRE(std::is_same_v<Sequential::At<Check, 1>, short>);
  STATIC_REQUIRE(std::is_same_v<Sequential::At<Check, 2>, double>);
}

TEST_CASE("Sequential::Order type indexing", "[sequential]") {
  // ! only works with all types heterogenous for now
  using Check = Vector<int, short, double>;
  STATIC_REQUIRE(Sequential::Order<Check, int> == 0);
  STATIC_REQUIRE(Sequential::Order<Check, short> == 1);
  STATIC_REQUIRE(Sequential::Order<Check, double> == 2);
}

TEST_CASE("Sequential::Set type setting", "[sequential]") {
  using Check = Vector<int, short, double, float>;

  using Res1 = Sequential::Set<Check, 0, bool>;
  STATIC_REQUIRE(std::is_same_v<Res1, Vector<bool, short, double, float>>);

  using Res2 = Sequential::Set<Check, 1, bool>;
  STATIC_REQUIRE(std::is_same_v<Res2, Vector<int, bool, double, float>>);

  using Res3 = Sequential::Set<Check, 2, bool>;
  STATIC_REQUIRE(std::is_same_v<Res3, Vector<int, short, bool, float>>);

  using Res4 = Sequential::Set<Check, 3, bool>;
  STATIC_REQUIRE(std::is_same_v<Res4, Vector<int, short, double, bool>>);
}

TEST_CASE("Sequential::PushBack type appending", "[sequential]") {
  using Check = Vector<int, short, double>;

  using Res1 = Sequential::PushBack<Check, float>;
  STATIC_REQUIRE(std::is_same_v<Sequential::At<Res1, 3>, float>);

  using Res2 = Sequential::PushBack<Check, float, char>;
  STATIC_REQUIRE(std::is_same_v<Sequential::At<Res2, 3>, float>);
  STATIC_REQUIRE(std::is_same_v<Sequential::At<Res2, 4>, char>);
}
