
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

EnumValuePolicyObj(SomePolicyOption1A, SomePolicy, A, Option1);
ValuePolicyObj(SomePolicyB20, SomePolicy, B, 20);
ValuePolicyObj(SomePolicyB10, SomePolicy, B, 10);

TEST_CASE("ChangePolicy correctly changes policy", "[policy]") {

  SECTION("Single policy is added to an empty container") {
    using TestContainer = PolicyContainer<>;

    using NewContainer = ChangePolicy<SomePolicyOption1A, TestContainer>;
    STATIC_REQUIRE(Sequential::Size<NewContainer> == 1);

    using NewPolicy = PolicySelect<SomePolicy, NewContainer>;
    STATIC_REQUIRE(
        std::is_same_v<NewPolicy::A, SomePolicy::ATypeCate::Option1>);
  }

  SECTION("Single non-conflicting policy is added to an non-empty container") {
    using TestContainer = PolicyContainer<SomePolicyB10>;

    using NewContainer = ChangePolicy<SomePolicyOption1A, TestContainer>;
    STATIC_REQUIRE(Sequential::Size<NewContainer> == 2);

    using NewPolicy = PolicySelect<SomePolicy, NewContainer>;
    STATIC_REQUIRE(
        std::is_same_v<NewPolicy::A, SomePolicy::ATypeCate::Option1>);
    STATIC_REQUIRE(NewPolicy::B == 10);
  }

  SECTION("Conflicting policy is changed") {
    using TestContainer = PolicyContainer<SomePolicyB10>;

    using NewContainer = ChangePolicy<SomePolicyB20, TestContainer>;
    STATIC_REQUIRE(Sequential::Size<NewContainer> == 1);

    using NewPolicy = PolicySelect<SomePolicy, NewContainer>;
    STATIC_REQUIRE(NewPolicy::B == 20);
  }

  SECTION("All SubPolicyContainers are preserved regardless of layer") {
    using MixedContainer =
        PolicyContainer<SomePolicyB10,
                        SubPolicyContainer<SomeLayer, SomePolicyB10>>;

    using NewContainer = ChangePolicy<SomePolicyB20, MixedContainer>;
    STATIC_REQUIRE(Sequential::Size<NewContainer> == 2);

    using FirstPolicy = Sequential::At<NewContainer, 0>;
    STATIC_REQUIRE(
        std::is_same_v<FirstPolicy,
                       SubPolicyContainer<SomeLayer, SomePolicyB10>>);

    using SecondPolicy = Sequential::At<NewContainer, 1>;
    STATIC_REQUIRE(std::is_same_v<SecondPolicy, SomePolicyB20>);
    STATIC_REQUIRE(SecondPolicy::B == 20);
  }
}

#include <core/policies/policy_macro_end.hpp>