#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <core/policies/policy_container.hpp>
#include <core/policies/policy_macro_begin.hpp>
#include <core/policies/policy_ops.hpp>
#include <type_traits>

using namespace venus;

struct TestPolicy {
  using MajorClass = TestPolicy;
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

EnumValuePolicyObj(TestPolicyOption1A, TestPolicy, A, Option1);

TEST_CASE("PlainPolicy correctly parses policies", "[policy]") {

  SECTION("Empty policy is copied") {
    using TestPolicies = PolicyContainer<>;
    using PlainRes = PlainPolicy<TestPolicies>;

    STATIC_REQUIRE(Sequential::Size<PlainRes> == 0);
  }

  SECTION("Regular policy is copied") {
    using TestPolicies = PolicyContainer<TestPolicy>;

    using PlainRes = PlainPolicy<TestPolicies>;
    STATIC_REQUIRE(Sequential::Size<PlainRes> == 1);

    using Extracted = Sequential::At<PlainRes, 0>;
    STATIC_REQUIRE(
        std::is_same_v<Extracted::A, TestPolicy::ATypeCate::Option1>);
  }

  SECTION("SubPolicyContainer is ignored") {
    using MixedPolicies =
        PolicyContainer<TestPolicy,
                        SubPolicyContainer<struct SomeLayer, TestPolicy>>;
    using PlainRes = PlainPolicy<MixedPolicies>;

    // Plain Result contains just then non-subcontainer policy
    STATIC_REQUIRE(Sequential::Size<PlainRes> == 1);

    using FirstPolicy = Sequential::Head<PlainRes>;
    using LastPolicy = Sequential::Last<PlainRes>;

    STATIC_REQUIRE(
        std::is_same_v<FirstPolicy::A, TestPolicy::ATypeCate::Option1>);
    STATIC_REQUIRE(std::is_same_v<FirstPolicy, LastPolicy>);
  }
}