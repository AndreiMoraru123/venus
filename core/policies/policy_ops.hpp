#pragma once
#include "../sequential.hpp"
#include "../traits.hpp"
#include "policy_concepts.hpp"
#include "policy_container.hpp"
#include <type_traits>

namespace venus {

// Details =====================================================
namespace detail {

// Policy Select ===============================================
template <typename TPolicyCont> struct PolicySelectionRes;

template <typename TCurrPolicy, typename... TOtherPolicies>
struct PolicySelectionRes<PolicyContainer<TCurrPolicy, TOtherPolicies...>>
    : TCurrPolicy, TOtherPolicies... {};

template <typename TMajorClass> struct MajorFilter_ {
  template <typename TState, typename TInput>
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
template <typename ParentPolicy, typename ChildPoliciesContainer>
struct PolicyConflict;

template <typename ParentPolicy, typename... ChildPolicies>
struct PolicyConflict<ParentPolicy, PolicyContainer<ChildPolicies...>> {
  static constexpr bool value =
      (HasSameClassTags<ParentPolicy, ChildPolicies> || ...);
};

template <typename... ChildPolicies> struct Filter_ {
  template <typename TState, typename TParentPolicy>
  using apply =
      std::conditional_t<PolicyConflict<TParentPolicy, ChildPolicies...>::value,
                         Identity_<TState>,
                         Sequential::PushBack_<TState, TParentPolicy>>;
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