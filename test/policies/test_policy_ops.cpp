#include "core/traits.hpp"
#include <array>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <core/policies/policy_container.hpp>
#include <core/policies/policy_ops.hpp>
#include <type_traits>

using namespace venus;

struct AccPolicy {
  using MajorClass = AccPolicy;
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

struct PAddAccu : virtual AccPolicy {
  using MinorClass = AccPolicy::AccuTypeCate;
  using Accu = MinorClass::Add;
};

struct PMulAccu : virtual AccPolicy {
  using MinorClass = AccPolicy::AccuTypeCate;
  using Accu = MinorClass::Mul;
};

struct PAve : virtual AccPolicy {
  using MinorClass = AccPolicy::IsAveValueCate;
  static constexpr bool IsAve = true;
};

template <typename T> struct PValueTypeIs : virtual AccPolicy {
  using MajorClass = AccPolicy;
  using Value = T;
};

template <typename... TPolicies> struct Accumulator {
  using Container = PolicyContainer<TPolicies...>;
  using PolicyRes = PolicySelect<AccPolicy, Container>;

  using ValueType = typename PolicyRes::Value;
  static constexpr bool isAve = PolicyRes::IsAve;
  using AccuType = typename PolicyRes::Accu;

  template <typename TIn> static constexpr auto Eval(const TIn &in) {
    if constexpr (std::is_same_v<AccuType, AccPolicy::AccuTypeCate::Add>) {
      ValueType count = 0;
      ValueType res = 0;

      for (const auto &x : in) {
        res += x;
        count += 1;
      }

      if constexpr (isAve) {
        return res / count;
      }
      return res;
    } else if constexpr (std::is_same_v<AccuType,
                                        AccPolicy::AccuTypeCate::Mul>) {
      ValueType count = 0;
      ValueType res = 1;

      for (const auto &x : in) {
        res *= x;
        count += 1;
      }

      if constexpr (isAve) {
        return std::pow(res, 1.0f / count);
      }
      return res;
    }
  }
};

TEST_CASE("PolicySelect selects correct policies", "[policy]") {
  SECTION("Default values when no policies provided") {
    using Result = PolicySelect<AccPolicy, PolicyContainer<>>;

    STATIC_REQUIRE(std::is_same_v<Result::Accu, AccPolicy::AccuTypeCate::Add>);
    STATIC_REQUIRE(Result::IsAve == false);
    STATIC_REQUIRE(std::is_same_v<Result::Value, float>);
  }

  SECTION("Override single policy") {
    using Result = PolicySelect<AccPolicy, PolicyContainer<PMulAccu>>;

    STATIC_REQUIRE(std::is_same_v<Result::Accu, AccPolicy::AccuTypeCate::Mul>);
    STATIC_REQUIRE(Result::IsAve == false);
    STATIC_REQUIRE(std::is_same_v<Result::Value, float>);
  }
}

TEST_CASE("Accumulator with different policies", "[policy]") {
  constexpr std::array a = {1, 2, 3, 4, 5};

  SECTION("Default policies") {
    constexpr auto result = Accumulator<>::Eval(a);
    STATIC_REQUIRE(result == 15.0f);
    STATIC_REQUIRE(std::is_same_v<std::remove_cv_t<decltype(result)>, float>);
  }

  SECTION("Double value alias") {
    using PDoubleValue = PValueTypeIs<double>;
    constexpr auto result = Accumulator<PDoubleValue>::Eval(a);
    STATIC_REQUIRE(result == 15.0);
    STATIC_REQUIRE(std::is_same_v<std::remove_cv_t<decltype(result)>, double>);
  }

  SECTION("Multiply accumulation") {
    constexpr auto result = Accumulator<PMulAccu>::Eval(a);
    STATIC_REQUIRE(result == 120.0f);
    STATIC_REQUIRE(std::is_same_v<std::remove_cv_t<decltype(result)>, float>);
  }

  SECTION("Add and average") {
    constexpr auto result = Accumulator<PAddAccu, PAve>::Eval(a);
    STATIC_REQUIRE(result == 3.0f);
    STATIC_REQUIRE(std::is_same_v<std::remove_cv_t<decltype(result)>, float>);
  }

  SECTION("Policy Order does not affect result") {
    constexpr auto res1 = Accumulator<PAddAccu, PAve>::Eval(a);
    constexpr auto res2 = Accumulator<PAve, PAddAccu>::Eval(a);
    STATIC_REQUIRE(res1 == res2);
  }
}