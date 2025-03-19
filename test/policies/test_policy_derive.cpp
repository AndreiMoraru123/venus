#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <core/policies/policy_container.hpp>
#include <core/policies/policy_macro_begin.hpp>
#include <core/policies/policy_ops.hpp>
#include <type_traits>

using namespace venus;

struct Parent {
  using MajorClass = Parent;
  struct ATypeCate {
    struct Option1;
    struct Option2;
  };
  using A = ATypeCate::Option1;

  struct ValueBValueCate;
  static constexpr int B = 10;

  struct CValueCate;
  using C = float;
};

// Define Parent Policy for Parent Container
EnumValuePolicyObj(ParentPolicyTypeA, Parent, A, Option1);

struct Child {
  using MajorClass = Child;

  struct XValueCate;
  static constexpr bool X = false;
};

// Define Child Policy for Child container
ValuePolicyObj(ChildPolicyX, Child, X, true);

// Define override Parent Policy to be used by Child Container
EnumValuePolicyObj(ParentOverridePolicyTypeA, Parent, A, Option2);

TEST_CASE("PolicyDerive derives the correct policies", "[policy]") {

  SECTION("Inherit both parent and child policies") {
    using ParentPolicies = PolicyContainer<ParentPolicyTypeA>;
    using ChildPolicies = PolicyContainer<ChildPolicyX>;
    using Combined = PolicyDerive<ChildPolicies, ParentPolicies>;

    // Policy Container is merged
    STATIC_REQUIRE(Sequential::Size<Combined> == 2);

    // Parent policies are inherited
    using ParentRes = PolicySelect<Parent, Combined>;
    STATIC_REQUIRE(std::is_same_v<ParentRes::A, Parent::ATypeCate::Option1>);
    STATIC_REQUIRE(ParentRes::B == 10);
    STATIC_REQUIRE(std::is_same_v<ParentRes::C, float>);

    // Child policies are also inherited
    using ChildRes = PolicySelect<Child, Combined>;
    STATIC_REQUIRE(ChildRes::X == true);
  }

  SECTION("Override parent policy") {
    using ParentPolicies = PolicyContainer<ParentPolicyTypeA>;
    using ChildPolicies =
        PolicyContainer<ChildPolicyX, ParentOverridePolicyTypeA>;
    using Combined = PolicyDerive<ChildPolicies, ParentPolicies>;

    // Policy Container is merged via overriding one of the policies
    STATIC_REQUIRE(Sequential::Size<Combined> == 2);

    // Parent policy is overriden
    using ParentRes = PolicySelect<Parent, Combined>;
    STATIC_REQUIRE(std::is_same_v<ParentRes::A, Parent::ATypeCate::Option2>);
  }
}

#include <core/policies/policy_macro_end.hpp>