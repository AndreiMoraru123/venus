#include <type_traits>

namespace venus {

template <typename T>
concept Policy = requires {
  typename T::MajorClass;
  typename T::MinorClass;
};

template <typename T, typename U>
concept SameMajorClass =
    Policy<T> and Policy<U> and
    std::is_same_v<typename T::MajorClass, typename U::MajorClass>;

template <typename T, typename U>
concept SameMinorClass =
    Policy<T> and Policy<U> and
    std::is_same_v<typename T::MinorClass, typename U::MinorClass>;

template <typename P1, typename P2>
concept SameClassTags = Policy<P1> and Policy<P2> and SameMajorClass<P1, P2> and
                        SameMinorClass<P1, P2>;

template <typename Container> struct AllPolicies_;

template <typename P, typename TargetMajorClass, typename TargetMinorClass>
concept SameTargetClasses =
    Policy<P> and std::is_same_v<typename P::MajorClass, TargetMajorClass> and
    std::is_same_v<typename P::MinorClass, TargetMinorClass>;

template <template <typename...> typename Container, typename... Ts>
struct AllPolicies_<Container<Ts...>> {
  static constexpr bool value = (Policy<Ts> and ...);
};

template <typename Container>
static constexpr bool AllPolicies = AllPolicies_<Container>::value;
} // namespace venus