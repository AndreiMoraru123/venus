
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <core/policies/policy_container.hpp>
#include <core/policies/policy_macro_begin.hpp>
#include <core/policies/policy_ops.hpp>

using namespace venus;

struct SomePolicy {
  using MajorClass = SomePolicy;
  struct ATypeCate {
    struct Option1;
    struct Option2;
  };
  using A = ATypeCate::Option1;

  struct BValueCate;
  static constexpr int B = 10;

  struct CValueCate;
  using C = float;
};

struct SomeLayer {};

ValuePolicyObj(SomePolicyB20, SomePolicy, B, 20);

TEST_CASE("HasNonTrivialPolicy correctly evaluates the result", "[policy]") {

  SECTION("emtpy container has a trivial policy") {
    using TestContainer = PolicyContainer<>;
    constexpr bool nonTrivial =
        HasNonTrivialPolicy<TestContainer, SomePolicy, SomePolicy::BValueCate>;
    STATIC_REQUIRE(nonTrivial == false);
  }

  SECTION("non-emtpy container has a non-trivial policy") {
    using TestContainer = PolicyContainer<SomePolicyB20>;
    constexpr bool nonTrivial =
        HasNonTrivialPolicy<TestContainer, SomePolicy, SomePolicy::BValueCate>;
    STATIC_REQUIRE(nonTrivial == true);
  }
}

#include <core/policies/policy_macro_end.hpp>