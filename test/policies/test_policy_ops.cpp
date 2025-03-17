#include <catch2/catch_test_macros.hpp>

#include <core/policies/policy_container.hpp>
#include <core/policies/policy_ops.hpp>
#include <type_traits>

using namespace venus;

struct AccPolicy {
  struct AccuTypeCate {
    struct Add;
    struct Mul;
  };
  using Accu = AccuTypeCate::Add;

  struct IsAveValueCate;
  static constexpr bool IsAve = false;

  struct ValueTypeCate;
  using Value = float;
};

struct PAddAccu {
  using TMajorClass = AccPolicy;
  using MinorClass = AccPolicy::AccuTypeCate;
  using Accu = MinorClass::Add;
};

struct PMulAccu {
  using TMajorClass = AccPolicy;
  using MinorClass = AccPolicy::AccuTypeCate;
  using Accu = MinorClass::Mul;
};

struct PAve {
  using TMajorClass = AccPolicy;
  using MinorClass = AccPolicy::IsAveValueCate;
  static constexpr bool IsAve = true;
};

template <typename T> struct PValueTypeIs {
  using TMajorClass = AccPolicy;
  using MinorClass = AccPolicy::ValueTypeCate;
  using Value = T;
};

TEST_CASE("PolicySelect selects correct policies", "[policy]") {
  SECTION("Default values when no policies provided") {
    using Result = PolicySelect<AccPolicy, PolicyContainer<>>;

    STATIC_REQUIRE(std::is_same_v<Result::Accu, AccPolicy::AccuTypeCate::Add>);
    STATIC_REQUIRE(Result::IsAve == false);
    STATIC_REQUIRE(std::is_same_v<Result::Value, float>);
  }
}