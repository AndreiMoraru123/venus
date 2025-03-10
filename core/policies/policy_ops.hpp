#pragma once
#include "../sequential.hpp"
#include "../traits.hpp"
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
      std::is_same_v<typename TInput::TMajorClass, TMajorClass>,
      Sequential::PushBack_<TState, TInput>, Identity_<TState>>;
};

template <typename TPolicyCont> struct MinorCheck_ {
  static constexpr bool value = true;
};

template <typename TCurrPolicy, typename... TP>
struct MinorCheck_<PolicyContainer<TCurrPolicy, TP...>> {
  static constexpr bool currCheck =
      ((not std::is_same_v<typename TCurrPolicy::MinorClass,
                           typename TP::MinorClass>) and
       ...); // checks for uniqueness in minor class
  static constexpr bool value =
      AndValue<currCheck, MinorCheck_<PolicyContainer<TP...>>>;
};

template <typename TMajorClass, typename TPolicyContainer> struct Selector_ {};

} // namespace detail

// =============================================================

} // namespace venus