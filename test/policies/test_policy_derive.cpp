#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <type_traits>
#include <venus/policies/policy_container.hpp>
#include <venus/policies/policy_macro_begin.hpp>
#include <venus/policies/policy_ops.hpp>

using namespace venus;

struct Parent {
  using MajorClass = Parent;
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

// Define Parent Policy for Parent Container
EnumValuePolicyObj(ParentPolicyOption1A, Parent, A, Option1);

struct Child {
  using MajorClass = Child;

  struct XValueCate;
  static constexpr bool X = false;

  struct YTypeCate;
  using Y = int;
};

// Define Children Policies for Child Container
ValuePolicyObj(ChildPolicyTrueX, Child, X, true);
TypePolicyObj(ChildPolicyDoubleY, Child, Y, double);

// Define override Parent Policies to be used by Child Container
EnumValuePolicyObj(ParentPolicyOption2A, Parent, A, Option2);
ValuePolicyObj(ParentPolicyValueB20, Parent, B, 20);

TEST_CASE("PolicyDerive derives the correct policies", "[policy]") {

  SECTION("Inherit both a parent and a child policy") {
    using ParentPolicies = PolicyContainer<ParentPolicyOption1A>;
    using ChildPolicies = PolicyContainer<ChildPolicyTrueX>;
    using Combined = PolicyDerive<ChildPolicies, ParentPolicies>;

    // Policy Container is merged
    STATIC_REQUIRE(Sequential::Size<Combined> == 2);

    // Parent policy is inherited
    using ParentRes = PolicySelect<Parent, Combined>;
    STATIC_REQUIRE(std::is_same_v<ParentRes::A, Parent::ATypeCate::Option1>);
    STATIC_REQUIRE(ParentRes::B == 10);
    STATIC_REQUIRE(std::is_same_v<ParentRes::C, float>);

    // Child policy is inherited
    using ChildRes = PolicySelect<Child, Combined>;
    STATIC_REQUIRE(ChildRes::X == true);
  }

  SECTION("No parent policy") {
    using ParentPolicies = PolicyContainer<>;
    using ChildPolicies = PolicyContainer<ChildPolicyTrueX, ChildPolicyDoubleY>;
    using Combined = PolicyDerive<ChildPolicies, ParentPolicies>;

    // Policy Container is just the child
    STATIC_REQUIRE(Sequential::Size<Combined> == 2);

    // Child policy is inherited
    using ChildRes = PolicySelect<Child, Combined>;
    STATIC_REQUIRE(ChildRes::X == true);
    STATIC_REQUIRE(std::is_same_v<ChildRes::Y, double>);
  }

  SECTION("No child policy") {
    using ParentPolicies = PolicyContainer<ParentPolicyOption1A>;
    using ChildPolicies = PolicyContainer<>;
    using Combined = PolicyDerive<ChildPolicies, ParentPolicies>;

    // Policy Container is just the parent
    STATIC_REQUIRE(Sequential::Size<Combined> == 1);

    // Parent policy is inherited
    using ParentRes = PolicySelect<Parent, Combined>;
    STATIC_REQUIRE(std::is_same_v<ParentRes::A, Parent::ATypeCate::Option1>);
    STATIC_REQUIRE(ParentRes::B == 10);
    STATIC_REQUIRE(std::is_same_v<ParentRes::C, float>);
  }

  SECTION("Override parent policies") {
    using ParentPolicies = PolicyContainer<ParentPolicyOption1A>;
    using ChildPolicies =
        PolicyContainer<ChildPolicyTrueX, ParentPolicyOption2A,
                        ParentPolicyValueB20>;
    using Combined = PolicyDerive<ChildPolicies, ParentPolicies>;

    // Policy Container is merged via overriding the parent policies
    STATIC_REQUIRE(Sequential::Size<Combined> == 3);

    // Parent policy is overriden
    using ParentRes = PolicySelect<Parent, Combined>;
    STATIC_REQUIRE(std::is_same_v<ParentRes::A, Parent::ATypeCate::Option2>);
    STATIC_REQUIRE(ParentRes::B == 20);
  }
}

#include <venus/policies/policy_macro_end.hpp>