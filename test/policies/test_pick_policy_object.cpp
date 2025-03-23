
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <core/policies/policy_container.hpp>
#include <core/policies/policy_macro_begin.hpp>
#include <core/policies/policy_ops.hpp>
#include <type_traits>

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

TEST_CASE("PickPolicyObject correctly picks the policy", "[policy]") {

  SECTION("empty container defaults to major policy") {
    using TestContainer = PolicyContainer<>;
    using PickedPolicy =
        PickPolicyObject<TestContainer, SomePolicy, SomePolicy::BValueCate>;

    using PickedContainer = PolicyContainer<PickedPolicy>;
    using SelectedPolicy = PolicySelect<SomePolicy, PickedContainer>;
    STATIC_REQUIRE(SelectedPolicy::B == 10);
    STATIC_REQUIRE(
        std::is_same_v<SelectedPolicy::A, SomePolicy::ATypeCate::Option1>);
  }

  SECTION("single policy is correctly picked") {
    using TestContainer = PolicyContainer<SomePolicyB20>;
    using PickedPolicy =
        PickPolicyObject<TestContainer, SomePolicy, SomePolicy::BValueCate>;

    using PickedContainer = PolicyContainer<PickedPolicy>;
    using SelectedPolicy = PolicySelect<SomePolicy, PickedContainer>;
    STATIC_REQUIRE(SelectedPolicy::B == 20);
    STATIC_REQUIRE(
        std::is_same_v<SelectedPolicy::A, SomePolicy::ATypeCate::Option1>);
  }
}

#include <core/policies/policy_macro_end.hpp>