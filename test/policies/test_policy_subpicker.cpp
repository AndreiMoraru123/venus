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
struct OtherLayer {};

EnumValuePolicyObj(SomePolicyOption1A, SomePolicy, A, Option1);
ValuePolicyObj(SomePolicyB20, SomePolicy, B, 20);
ValuePolicyObj(SomePolicyB10, SomePolicy, B, 10);

TEST_CASE("SubPolicyPicker correctly picks policies", "[policy]") {

  SECTION("Empty policy container returns an empty result") {
    using TestPolicy = PolicyContainer<>;

    using SubContainer = SubPolicyPicker<TestPolicy, SomeLayer>;
    STATIC_REQUIRE(Sequential::Size<SubContainer> == 0);
  }

  SECTION("Regular policies are included in the result") {
    using TestPolicy = PolicyContainer<SomePolicyOption1A>;

    using SubContainer = SubPolicyPicker<TestPolicy, SomeLayer>;
    STATIC_REQUIRE(Sequential::Size<SubContainer> == 1);

    using ExtractedPolicy = Sequential::At<SubContainer, 0>;
    STATIC_REQUIRE(std::is_same_v<ExtractedPolicy, SomePolicyOption1A>);
  }

  SECTION("Sub policy container with matching Layer type is included") {
    using TestPolicy =
        PolicyContainer<SomePolicyB20,
                        SubPolicyContainer<SomeLayer, SomePolicyOption1A>>;

    using SubPolicy = SubPolicyPicker<TestPolicy, SomeLayer>;
    STATIC_REQUIRE(Sequential::Size<SubPolicy> == 2);

    using FirstPolicy = Sequential::At<SubPolicy, 0>;
    STATIC_REQUIRE(std::is_same_v<FirstPolicy, SomePolicyOption1A>);

    using SecondPolicy = Sequential::At<SubPolicy, 1>;
    STATIC_REQUIRE(std::is_same_v<SecondPolicy, SomePolicyB20>);
  }

  SECTION("Sub-policy container with non-matching Layer type is not included") {
    using TestPolicy =
        PolicyContainer<SomePolicyB20,
                        SubPolicyContainer<SomeLayer, SomePolicyOption1A>>;

    using SubPolicy = SubPolicyPicker<TestPolicy, OtherLayer>;
    STATIC_REQUIRE(Sequential::Size<SubPolicy> == 1);

    using ExtractedPolicy = Sequential::At<SubPolicy, 0>;
    STATIC_REQUIRE(std::is_same_v<ExtractedPolicy, SomePolicyB20>);
  }

  SECTION("Sub-policy overrides regular policy") {
    using TestPolicy =
        PolicyContainer<SomePolicyB10,
                        SubPolicyContainer<SomeLayer, SomePolicyB20>>;

    using SubPolicy = SubPolicyPicker<TestPolicy, SomeLayer>;
    STATIC_REQUIRE(Sequential::Size<SubPolicy> == 1);

    using Result = PolicySelect<SomePolicy, SubPolicy>;
    STATIC_REQUIRE(Result::B == 20);
  }
}

#include <core/policies/policy_macro_end.hpp>