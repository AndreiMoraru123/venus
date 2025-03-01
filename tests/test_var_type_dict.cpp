
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

TEST_CASE("VarTypeDict setter and getter", "tags_ [set-get]") {
  using Params = VarTypeDict<A, B, Weight>;
  auto params = Params::Create().Set<A>(2.5f).Set<B>(1.5f).Set<Weight>(0.5f);
  REQUIRE(fn(params) == 2.0f);
}