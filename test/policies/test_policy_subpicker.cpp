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

struct SomeLayer {};
struct OtherLayer {};

EnumValuePolicyObj(TestPolicyOption1A, TestPolicy, A, Option1);
ValuePolicyObj(TestPolicyValueB20, TestPolicy, B, 20);
ValuePolicyObj(TestPolicyValueB10, TestPolicy, B, 10);

TEST_CASE("SubPolicyPicker correctly picks policies", "[policy]") {

  SECTION("Empty policy container returns an empty result") {
    using TestPolicy = PolicyContainer<>;

    using SubContainer = SubPolicyPicker<TestPolicy, SomeLayer>;
    STATIC_REQUIRE(Sequential::Size<SubContainer> == 0);
  }

  SECTION("Regular policies are included in the result") {
    using TestPolicy = PolicyContainer<TestPolicyOption1A>;

    using SubContainer = SubPolicyPicker<TestPolicy, SomeLayer>;
    STATIC_REQUIRE(Sequential::Size<SubContainer> == 1);

    using ExtractedPolicy = Sequential::At<SubContainer, 0>;
    STATIC_REQUIRE(std::is_same_v<ExtractedPolicy, TestPolicyOption1A>);
  }

  SECTION("Sub policy container with matching Layer type is included") {
    using TestContainer =
        PolicyContainer<TestPolicyValueB20,
                        SubPolicyContainer<SomeLayer, TestPolicyOption1A>>;

    using SubContainer = SubPolicyPicker<TestContainer, SomeLayer>;
    STATIC_REQUIRE(Sequential::Size<SubContainer> == 2);

    using FirstPolicy = Sequential::At<SubContainer, 0>;
    STATIC_REQUIRE(std::is_same_v<FirstPolicy, TestPolicyOption1A>);

    using SecondPolicy = Sequential::At<SubContainer, 1>;
    STATIC_REQUIRE(std::is_same_v<SecondPolicy, TestPolicyValueB20>);
  }

  SECTION("Sub-policy container with non-matching Layer type is not included") {
    using TestContainer =
        PolicyContainer<TestPolicyValueB20,
                        SubPolicyContainer<SomeLayer, TestPolicyOption1A>>;

    using SubContainer = SubPolicyPicker<TestContainer, OtherLayer>;
    STATIC_REQUIRE(Sequential::Size<SubContainer> == 1);

    using ExtractedPolicy = Sequential::At<SubContainer, 0>;
    STATIC_REQUIRE(std::is_same_v<ExtractedPolicy, TestPolicyValueB20>);
  }

  SECTION("Sub-policy overrides regular policy") {
    using TestContainer =
        PolicyContainer<TestPolicyValueB10,
                        SubPolicyContainer<SomeLayer, TestPolicyValueB20>>;

    using SubContainer = SubPolicyPicker<TestContainer, SomeLayer>;
    STATIC_REQUIRE(Sequential::Size<SubContainer> == 1);

    using Result = PolicySelect<TestPolicy, SubContainer>;
    STATIC_REQUIRE(Result::B == 20);
  }
}