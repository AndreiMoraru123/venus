#pragma once
#include "../sequential.hpp"
#include "../traits.hpp"
#include "policy_container.hpp"
#include <type_traits>

namespace venus {

template <typename T>
concept Policy = requires {
  typename T::MajorClass;
  typename T::MinorClass;
};

template <typename T, typename MajorClass>
concept SameMajorClass = requires {
  Policy<T> and std::is_same_v<typename T::MajorClass, MajorClass>;
};

template <typename T, typename U>
concept SameMinorClass = requires {
  typename T::MinorClass;
  typename U::MinorClass;
} and std::is_same_v<typename T::MinorClass, typename U::MinorClass>;

// Details =====================================================
namespace detail {

// Policy Select ===============================================
template <typename TPolicyCont> struct PolicySelectionRes;

template <typename TCurrPolicy, typename... TOtherPolicies>
struct PolicySelectionRes<PolicyContainer<TCurrPolicy, TOtherPolicies...>>
    : TCurrPolicy, TOtherPolicies... {};

template <typename TMajorClass> struct MajorFilter_ {
  template <typename TState, typename TInput>
    requires SameMajorClass<TInput, TMajorClass>
  using apply = std::conditional_t<
      std::is_same_v<typename TInput::MajorClass, TMajorClass>,
      Sequential::PushBack_<TState, TInput>, Identity_<TState>>;
};

template <typename TPolicyCont> struct MinorCheck_ {
  static constexpr bool value = true;
};

template <typename TCurrPolicy, typename... TP>
struct MinorCheck_<PolicyContainer<TCurrPolicy, TP...>> {
  static constexpr bool currCheck =
      ((not SameMinorClass<TCurrPolicy, TP>) and ...);
  static constexpr bool value =
      AndValue<currCheck, MinorCheck_<PolicyContainer<TP...>>>;
};

template <typename TMajorClass, typename TPolicyContainer> struct Selector_ {
  using MajFilt = Sequential::Fold<PolicyContainer<>, TPolicyContainer,
                                   MajorFilter_<TMajorClass>::template apply>;
  static_assert(MinorCheck_<MajFilt>::value, "Minor class set conflict!");
  using type = std::conditional_t<Sequential::Size<MajFilt> == 0, TMajorClass,
                                  PolicySelectionRes<MajFilt>>;
};
// =============================================================

// Policy Derive ===============================================
template <typename TPolicy>
  requires Policy<TPolicy>
struct PolicyExistsKV {
  static std::true_type apply(typename TPolicy::MajorClass *,
                              typename TPolicy::MinorClass *);
};

template <typename TLayer, typename... TParams>
struct PolicyExistsKV<SubPolicyContainer<TLayer, TParams...>> {
  static void apply(SubPolicyContainer<TLayer, TParams...> *);
};

template <typename TSubPolicies> struct PolicyExist;

template <typename... TPolicies>
struct PolicyExist<PolicyContainer<TPolicies...>>
    : PolicyExistsKV<TPolicies>... {
  static std::false_type apply(...);
  using PolicyExistsKV<TPolicies>::apply...;
};

template <typename TSubPolicies> struct Filter_ {
  template <typename TState, typename TInput>
  using apply =
      std::conditional_t<decltype(PolicyExist<TSubPolicies>::apply(
                             (typename TInput::MajorClass *)nullptr,
                             (typename TInput::MinorClass *)nullptr))::value,
                         Identity_<TState>,
                         Sequential::PushBack_<TState, TInput>>;
};
// =============================================================

} // namespace detail

// Policy Select ===============================================
template <typename TMajorClass, typename TPolicyContainer>
using PolicySelect =
    typename detail::Selector_<TMajorClass, TPolicyContainer>::type;
// =============================================================

// Policy Derive ===============================================
template <typename TSubPolicies, typename TParentPolicies>
using PolicyDerive =
    Sequential::Fold<TSubPolicies, TParentPolicies,
                     detail::Filter_<TSubPolicies>::template apply>;
// =============================================================
} // namespace venus