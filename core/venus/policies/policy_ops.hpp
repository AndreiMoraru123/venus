#pragma once
#include <type_traits>
#include <venus/policies/policy_concepts.hpp>
#include <venus/policies/policy_container.hpp>
#include <venus/sequential.hpp>
#include <venus/traits.hpp>

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
      (SameClassTags<ParentPolicy, ChildPolicies> or ...);
};

template <typename... ChildPolicies> struct DeriveFilter {
  template <typename TState, typename TParentPolicy>
  using apply =
      std::conditional_t<PolicyConflict<TParentPolicy, ChildPolicies...>::value,
                         Identity_<TState>,
                         Sequential::PushBack_<TState, TParentPolicy>>;
};
// =============================================================

// Plain Policy ================================================
struct Plain {
  template <typename TState, typename TInput> struct apply {
    using type = Sequential::PushBack<TState, TInput>;
  };

  template <typename TState, typename TLayerName, typename... TAdded>
  struct apply<TState, SubPolicyContainer<TLayerName, TAdded...>> {
    using type = TState;
  };
};
// =============================================================

// Sub Policy Picker ===========================================
template <typename TLayerName> struct PolicySubPicker {
  template <typename TState, typename TInput> struct apply {
    using type = TState;
  };

  template <typename... TProcessed, typename... TAdded>
  struct apply<PolicyContainer<TProcessed...>,
               SubPolicyContainer<TLayerName, TAdded...>> {
    using type = PolicyContainer<TProcessed..., TAdded...>;
  };
};
// =============================================================

// Change Policy ===============================================
template <Policy NewPolicy> struct ChangeFilter {
  template <typename TState, typename TInput>
  struct apply
      : std::conditional_t<SameClassTags<TInput, NewPolicy>, Identity_<TState>,
                           Sequential::PushBack_<TState, TInput>> {};

  template <typename TState, typename TLayer, typename... TParams>
  struct apply<TState, SubPolicyContainer<TLayer, TParams...>> {
    using type =
        Sequential::PushBack<TState, SubPolicyContainer<TLayer, TParams...>>;
  };
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
                     detail::DeriveFilter<TSubPolicies>::template apply>;

// Plain Policy ================================================
template <typename TPolicyContainer>
using PlainPolicy = Sequential::Fold<PolicyContainer<>, TPolicyContainer,
                                     detail::Plain::template apply>;
// =============================================================

// Sub Policy Picker ===========================================
template <typename TPolicyContainer, typename TLayerName>
struct SubPolicyPicker_ {
  using SubPolicies =
      Sequential::Fold<PolicyContainer<>, TPolicyContainer,
                       detail::PolicySubPicker<TLayerName>::template apply>;
  using type = PolicyDerive<SubPolicies, PlainPolicy<TPolicyContainer>>;
  static_assert(AllPolicies<type>,
                "SubPolicyPicker must return only policy types");
};

template <typename TPolicyContainer, typename TLayerName>
using SubPolicyPicker = SubPolicyPicker_<TPolicyContainer, TLayerName>::type;
// =============================================================

// Change Policy ===============================================
template <typename NewPolicy, typename SourceContainer> struct ChangePolicy_ {
  using type = Sequential::PushBack<
      Sequential::Fold<PolicyContainer<>, SourceContainer,
                       detail::ChangeFilter<NewPolicy>::template apply>,
      NewPolicy>;
};

template <typename NewPolicy, typename SourceContainer>
using ChangePolicy = ChangePolicy_<NewPolicy, SourceContainer>::type;
// =============================================================

// Pick Policy Object ==========================================
template <typename TPolicyContainer, typename TMajorClass, typename TMinorClass>
struct PickPolicyObject_;

template <typename TMajorClass, typename TMinorClass, Policy... TPolicies>
struct PickPolicyObject_<PolicyContainer<TPolicies...>, TMajorClass,
                         TMinorClass> {
  using type = TMajorClass;
};

template <typename TMajorClass, typename TMinorClass, typename TCurrPolicy,
          Policy... TPolicies>
struct PickPolicyObject_<PolicyContainer<TCurrPolicy, TPolicies...>,
                         TMajorClass, TMinorClass> {
  using type = std::conditional_t<
      SameTargetClasses<TCurrPolicy, TMajorClass, TMinorClass>,
      Identity_<TCurrPolicy>,
      PickPolicyObject_<PolicyContainer<TPolicies...>, TMajorClass,
                        TMinorClass>>::type;
  static_assert(Policy<type>, "PickPolicyObject must return a policy");
};

template <typename TPolicyContainer, typename TMajorClass, typename TMinorClass>
using PickPolicyObject =
    PickPolicyObject_<TPolicyContainer, TMajorClass, TMinorClass>::type;
// =============================================================

// Has Non Trivial Policy ======================================
template <typename TPolicyContainer, typename TMajorClass, typename TMinorClass>
struct HasNonTrivialPolicy_;
template <typename TMajorClass, typename TMinorClass, Policy... TPolicies>
struct HasNonTrivialPolicy_<PolicyContainer<TPolicies...>, TMajorClass,
                            TMinorClass> {
  static constexpr bool value = false;
};

template <typename TMajorClass, typename TMinorClass, typename TCurrPolicy,
          Policy... TPolicies>
struct HasNonTrivialPolicy_<PolicyContainer<TCurrPolicy, TPolicies...>,
                            TMajorClass, TMinorClass> {
  static constexpr bool value =
      OrValue<SameTargetClasses<TCurrPolicy, TMajorClass, TMinorClass>,
              HasNonTrivialPolicy_<PolicyContainer<TPolicies...>, TMajorClass,
                                   TMinorClass>>;
};

template <typename TPolicyContainer, typename TMajorClass, typename TMinorClass>
static constexpr bool HasNonTrivialPolicy =
    HasNonTrivialPolicy_<TPolicyContainer, TMajorClass, TMinorClass>::value;

// =============================================================
} // namespace venus