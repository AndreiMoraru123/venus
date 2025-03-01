
#include <catch2/catch_test_macros.hpp>
#include <venus/var_type_dict.hpp>

using namespace venus;

struct A;
struct B;
struct Weight;

constexpr auto fn(const auto &in) -> float {
  auto a = in.template Get<A>();
  auto b = in.template Get<B>();
  auto weight = in.template Get<Weight>();

  return a * weight + b * (1 - weight);
}

TEST_CASE("VarTypeDict ops") {
  using Params = VarTypeDict<A, B, Weight>;
  auto params = Params::Create().Set<A>(2.5f).Set<B>(1.5f).Set<Weight>(0.5f);

  SECTION("Initial values") { REQUIRE(fn(params) == 2.0f); }

  SECTION("After update") {
    params.Update<A>(3.5f);
    REQUIRE(fn(params) == 2.5f);
  }

  SECTION("After two updates") {
    params.Update<A>(3.5f);
    params.Update<B>(2.0f);
    REQUIRE(fn(params) == 2.75f);
  }

  SECTION("After chained update") {
    params.ChainUpdate<A>(4.0f).ChainUpdate<B>(3.0f).ChainUpdate<Weight>(0.25f);
    REQUIRE(fn(params) == 3.25f);
  }
}