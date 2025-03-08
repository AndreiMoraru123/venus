#include <catch2/catch_test_macros.hpp>

#include <core/sequential.hpp>
#include <type_traits>

using namespace venus;

template <typename... Params> struct Vector;

template <typename T1, typename T2> struct Sum {
  using value_type = decltype(T1::value);
  static constexpr value_type value = T1::value + T2::value;
  using type = std::integral_constant<value_type, value>;
};

template <typename CurrMax, typename Curr> struct Max {
  using value_type = decltype(CurrMax::value);
  static constexpr value_type value =
      (CurrMax::value > Curr::value) ? CurrMax::value : Curr::value;
  using type = std::integral_constant<value_type, value>;
};

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

TEST_CASE("Sequential::Fold type reduction", "[sequential]") {
  using ConstVector =
      Vector<std::integral_constant<int, 1>, std::integral_constant<int, 2>,
             std::integral_constant<int, 3>>;

  using ZeroInit = std::integral_constant<int, 0>;

  SECTION("Sum Reduce") {
    using SumReduce = Sequential::Fold<ZeroInit, ConstVector, Sum>;
    STATIC_REQUIRE(SumReduce::value == 6);
  }

  SECTION("Max Reduce") {
    using MaxReduce = Sequential::Fold<ZeroInit, ConstVector, Max>;
    STATIC_REQUIRE(MaxReduce::value == 3);
  }
}
