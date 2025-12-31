#pragma once

// Auto-generated main header

#include <cstddef>
#include <cstring>
#include <memory>
namespace venus::Device {
struct CPU;
}


#ifdef VENUS_INTERPRETER
// Simple allocator for repl interpreter

namespace venus {
template <typename TDevice> struct Allocator;
template <>

struct Allocator<Device::CPU> {
  template <typename TElem>
  static std::shared_ptr<TElem> alloc(std::size_t p_elemSize) {
    TElem *raw_buf = new TElem[p_elemSize];

    if constexpr (std::is_trivially_constructible_v<TElem>) {
      std::memset(raw_buf, 0, p_elemSize * sizeof(TElem));
    } else {
      for (std::size_t i = 0; i < p_elemSize; ++i) {
        new (raw_buf + i) TElem();
      }
    }
    return std::shared_ptr<TElem>(raw_buf, [](TElem *ptr) { delete[] ptr; });
  }
};
} // namespace venus
#else

constexpr auto BLOCK_SIZE = 1024;

// Memory pool allocator for compiled venus

#include <deque>
#include <mutex>
#include <unordered_map>

namespace venus {
template <typename TDevice> struct Allocator;

template <> struct Allocator<Device::CPU> {

private:
  struct MemoryPool {
    std::unordered_map<std::size_t, std::deque<void *>> memBuffer;
    ~MemoryPool() {
      for (auto &pool : memBuffer) {
        auto &blocks = pool.second;
        for (const auto &block : blocks) {
          char *buf = (char *)(block);
          delete[] buf;
        }
        blocks.clear();
      }
    }
  };

  struct Deleter {
    Deleter(std::deque<void *> &p_refPool) : m_refPool(p_refPool) {}
    void operator()(void *p_val) const {
      std::lock_guard<std::mutex> guard(m_mutex);
      m_refPool.push_back(p_val);
    }

  private:
    std::deque<void *> &m_refPool;
  };

public:
  template <typename T, std::size_t BlockSize = BLOCK_SIZE>
  static auto alloc(std::size_t p_elemSize) -> std::shared_ptr<T> {
    static_assert((BlockSize & (BlockSize - 1)) == 0,
                  "BlockSize must be a power of 2");

    if (p_elemSize == 0) {
      return nullptr;
    }

    p_elemSize *= sizeof(T);
    if (p_elemSize & (BlockSize - 1)) {
      p_elemSize = ((p_elemSize / BlockSize) + 1) * BlockSize;
    }

    std::lock_guard<std::mutex> guard(m_mutex);

    T *raw_buf = nullptr;
    auto &slot = m_pool.memBuffer[p_elemSize];

    if (slot.empty()) {
      raw_buf = (T *)new char[p_elemSize];
    } else {
      void *mem = slot.back();
      slot.pop_back();
      raw_buf = (T *)mem;
    }

    if constexpr (std::is_trivially_constructible_v<T>) {
      std::memset(raw_buf, 0, p_elemSize);
    } else {
      std::size_t count = p_elemSize / sizeof(T);
      for (std::size_t i = 0; i < count; ++i) {
        new (raw_buf + i) T();
      }
    }
    return std::shared_ptr<T>(raw_buf, Deleter(slot));
  }

private:
  inline static std::mutex m_mutex;
  inline static MemoryPool m_pool;
};
}; // namespace venus

#endif
#include <cassert>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>


namespace venus {

template <typename TElem, typename TDevice> class ContiguousMemory {
  static_assert(std::is_same_v<std::remove_cvref_t<TElem>, TElem>);
  using ElementType = TElem;

#ifdef VENUS_INTERPRETER
  template <typename, typename, std::size_t> friend class Tensor;

private:
#else
public:
#endif
  explicit ContiguousMemory(std::size_t p_size)
      : m_mem(Allocator<TDevice>::template alloc<ElementType>(p_size)),
        m_size(p_size) {
    if (p_size == 0) {
      throw std::invalid_argument("Cannot allocate zero-sized memory.");
    }
  }

  auto shift(size_t pos) const {
    assert(pos < m_size);
    return ContiguousMemory(
        std::shared_ptr<ElementType>(m_mem, m_mem.get() + pos), m_size - pos);
  }

public:
  auto rawMemory() -> ElementType * { return m_mem.get(); }
  auto rawMemory() const -> const ElementType * { return m_mem.get(); }
  [[nodiscard]] auto isShared() const -> bool { return m_mem.use_count() > 1; }
  [[nodiscard]] auto size() const -> std::size_t { return m_size; }

  auto operator==(const ContiguousMemory &val) const -> bool {
    return (m_mem == val.m_mem) and (m_size == val.m_size);
  }

private:
  ContiguousMemory(std::shared_ptr<ElementType> ptr, std::size_t size)
      : m_mem(std::move(ptr)), m_size(size) {}
  std::shared_ptr<ElementType> m_mem;
  std::size_t m_size;
};

} // namespace venus

namespace venus {
template <typename TData> struct LowLevelAccess;
} // namespace venus
#include <cstddef>
#include <initializer_list>

template <class T, std::size_t Depth> struct nested_initializer_list {
  using type = std::initializer_list<
      typename nested_initializer_list<T, Depth - 1>::type>;
};

template <class T> struct nested_initializer_list<T, 1> {
  using type = std::initializer_list<T>;
};

template <class T, std::size_t Depth>
using nested_initializer_list_t =
    typename nested_initializer_list<T, Depth>::type;

namespace venus {
struct NullParameter {};
} // namespace venus

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
namespace venus {
template <typename... TPolicies> struct PolicyContainer;

template <typename TLayerName, typename... TPolicies> struct SubPolicyContainer;
} // namespace venus
#define EnumValuePolicyObj(PolicyName, Ma, Mi, Val)                            \
  struct PolicyName : virtual public Ma {                                      \
    using MinorClass = Ma::Mi##TypeCate;                                       \
    using Mi = Ma::Mi##TypeCate::Val;                                          \
  }

#define TypePolicyObj(PolicyName, Ma, Mi, Val)                                 \
  struct PolicyName : virtual public Ma {                                      \
    using MinorClass = Ma::Mi##TypeCate;                                       \
    using Mi = Val;                                                            \
  }

#define ValuePolicyObj(PolicyName, Ma, Mi, Val)                                \
  struct PolicyName : virtual public Ma {                                      \
    using MinorClass = Ma::Mi##ValueCate;                                      \
                                                                               \
  private:                                                                     \
    using type1 = decltype(Ma::Mi);                                            \
    using type2 = std::remove_cvref_t<type1>;                                  \
                                                                               \
  public:                                                                      \
    static constexpr type2 Mi = static_cast<type2>(Val);                       \
  }

#define TypePolicyTemplate(PolicyName, Ma, Mi)                                 \
  template <typename T> struct PolicyName : virtual public Ma {                \
    using MinorClass = Ma::Mi##TypeCate;                                       \
    using Mi = T;                                                              \
  }

#undef EnumTypePolicyObj
#undef ValuePolicyObj
#undef TypePolicyTemplate
#undef TypePolicyObj
#include <type_traits>


#include <cstddef>
#include <type_traits>


namespace venus::Sequential {

// Details =====================================================
namespace detail {

// At details ==================================================
template <std::size_t N, typename T, typename... Rest> struct TypeAt {
  using type = TypeAt<N - 1, Rest...>::type;
};
template <typename T, typename... Rest> struct TypeAt<0, T, Rest...> {
  using type = T;
};
// =============================================================

// Order details ===============================================
template <typename T, typename... Types> struct FindTypeIndex;

template <typename T, typename U, typename... Rest>
struct FindTypeIndex<T, U, Rest...> {
  constexpr static std::size_t value = 1 + FindTypeIndex<T, Rest...>::value;
};

template <typename T, typename... Rest> struct FindTypeIndex<T, T, Rest...> {
  constexpr static std::size_t value = 0;
};
// =============================================================

// Set details =================================================
template <typename TCon, std::size_t N, typename TValue, typename Processed,
          typename Remain>
struct SetImpl;

// Base case: Processed N elements and now I want to insert TValue
template <template <typename...> typename TCon, typename TValue,
          typename... Processed, typename Current, typename... Remaining>
struct SetImpl<TCon<>, 0, TValue, TCon<Processed...>,
               TCon<Current, Remaining...>> {
  using type = TCon<Processed..., TValue, Remaining...>;
};

// Recursive case: Move elements from Remaining to Processed until I reach Nth
template <template <typename...> typename TCon, std::size_t N, typename TValue,
          typename... Processed, typename Current, typename... Remaining>
struct SetImpl<TCon<>, N, TValue, TCon<Processed...>,
               TCon<Current, Remaining...>> {
  using type = SetImpl<TCon<>, N - 1, TValue, TCon<Processed..., Current>,
                       TCon<Remaining...>>::type;
};
// =============================================================

// Fold details ================================================
template <typename TState, template <typename, typename> typename Fn,
          typename... TRemain>
struct FoldImpl {
  using type = TState;
};

template <typename TState, template <typename, typename> typename Fn,
          typename T0, typename... TRemain>
struct FoldImpl<TState, Fn, T0, TRemain...> {
  using type = FoldImpl<Fn<TState, T0>, Fn, TRemain...>::type;
};
// =============================================================

} // namespace detail

// Create ======================================================
// TODO: Might have to make this binary if recursion gets too deep
template <std::size_t N, template <typename...> typename TCont, typename... T>
struct Create_ {
  using type = Create_<N - 1, TCont, NullParameter, T...>::type;
};

template <template <typename...> class TCont, typename... T>
struct Create_<0, TCont, T...> {
  using type = TCont<T...>;
};

template <std::size_t N, template <typename...> typename TCon, typename... T>
using Create = Create_<N, TCon, T...>::type;
// =============================================================

// At ==========================================================
template <typename TCon, std::size_t N> struct At_;

template <template <typename...> typename TCon, typename... TParams,
          std::size_t N>
struct At_<TCon<TParams...>, N> {
  static_assert(N < sizeof...(TParams), "index out of bounds");
  using type = detail::TypeAt<N, TParams...>::type;
};

template <typename TCon, std::size_t N> using At = At_<TCon, N>::type;
// =============================================================

// Order =======================================================
template <typename TCon, typename TReq> struct Order_ {};

template <template <typename...> typename TCon, typename... TParams,
          typename TReq>
struct Order_<TCon<TParams...>, TReq> {
  static constexpr std::size_t value =
      detail::FindTypeIndex<TReq, TParams...>::value;
};

template <typename TCon, typename TReq>
static constexpr std::size_t Order = Order_<TCon, TReq>::value;
// =============================================================

// Set =========================================================
template <typename TCon, std::size_t N, typename TValue> struct Set_;

template <template <typename...> typename TCont, std::size_t N, typename TValue,
          typename... TParams>
struct Set_<TCont<TParams...>, N, TValue> {
  static_assert(N < sizeof...(TParams), "index out of bounds");
  using type =
      detail::SetImpl<TCont<>, N, TValue, TCont<>, TCont<TParams...>>::type;
};

template <typename TCon, std::size_t N, typename TValue>
using Set = Set_<TCon, N, TValue>::type;
// =============================================================

// PushBack ====================================================
template <typename TCon, typename... TValue> struct PushBack_;

template <template <typename...> typename TCon, typename... TParams,
          typename... TValue>
struct PushBack_<TCon<TParams...>, TValue...> {
  using type = TCon<TParams..., TValue...>;
};

template <typename TCon, typename... TValue>
using PushBack = PushBack_<TCon, TValue...>::type;
// =============================================================

// Fold ========================================================
template <typename TInitState, typename TInputCont,
          template <typename, typename> typename Fn>
struct Fold_;

template <typename TInitState, template <typename...> typename TCont,
          typename... TParams, template <typename, typename> typename Fn>
struct Fold_<TInitState, TCont<TParams...>, Fn> {
  template <typename S, typename I> using Fun = Fn<S, I>::type;
  using type = detail::FoldImpl<TInitState, Fun, TParams...>::type;
};

template <typename TInitState, typename TInputCont,
          template <typename, typename> typename Fn>
using Fold = Fold_<TInitState, TInputCont, Fn>::type;
// =============================================================

// Size ========================================================
template <typename T> struct Size_;

template <template <typename...> typename TCont, typename... T>
struct Size_<TCont<T...>> {
  static constexpr size_t value = sizeof...(T);
};

template <typename T>
static constexpr size_t Size = Size_<std::remove_cvref_t<T>>::value;
// =============================================================

// Head ========================================================
template <typename TCont> using Head = At<TCont, 0>;
// =============================================================

// Tail ========================================================
template <typename TCont> struct Tail_;

template <template <typename...> typename TCont, typename H,
          typename... TRemain>
struct Tail_<TCont<H, TRemain...>> {
  using type = TCont<TRemain...>;
};

template <typename TCont> using Tail = Tail_<TCont>::type;
// =============================================================

// Last ========================================================
template <typename TCont> using Last = At<TCont, Size<TCont> - 1>;
// =============================================================

} // namespace venus::Sequential
namespace venus {

template <typename T> struct Identity_ {
  using type = T;
};

template <bool curr, typename TNext> static constexpr bool AndValue = false;

template <typename TNext>
static constexpr bool AndValue<true, TNext> = TNext::value;

template <bool curr, typename TNext> static constexpr bool OrValue = true;

template <typename TNext>
static constexpr bool OrValue<false, TNext> = TNext::value;
} // namespace venus

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

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <numeric>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>


namespace venus {
template <typename T>
concept VenusTensor = requires {
  typename std::remove_cvref_t<T>::ElementType;
  typename std::remove_cvref_t<T>::DeviceType;
  { std::remove_cvref_t<T>::Dimension } -> std::convertible_to<std::size_t>;
};

template <typename T>
concept Scalar = std::is_arithmetic_v<std::remove_cvref_t<T>>;

template <typename T>
concept ScalarTensor = VenusTensor<std::remove_cvref_t<T>> &&
                       (std::remove_cvref_t<T>::Dimension == 0);

template <typename T>
concept MDTensor = VenusTensor<std::remove_cvref_t<T>> &&
                   (std::remove_cvref_t<T>::Dimension > 0);

template <typename T>
concept BoolTensor =
    VenusTensor<std::remove_cvref_t<T>> &&
    std::is_convertible_v<typename std::remove_cvref_t<T>::ElementType, bool>;
} // namespace venus

#define REGISTER_BINARY_OP(op_name, std_op, op_symbol)                         \
  template <typename T1, typename T2>                                          \
    requires(Scalar<T1> || VenusTensor<T1>) && (Scalar<T2> || VenusTensor<T2>) \
  auto op_name(T1 &&t1, T2 &&t2) {                                             \
    /* Tensor op Tensor */                                                     \
    if constexpr (MDTensor<T1> && MDTensor<T2>) {                              \
      return detail::binary_elementwise_op(std::std_op{}, t1, t2);             \
    } /* Tensor op Scalar */                                                   \
    else if constexpr (MDTensor<T1> && Scalar<T2>) {                           \
      return transform(t1, [s = t2](auto &&t) { return t op_symbol s; });      \
    } /* Scalar op Tensor */                                                   \
    else if constexpr (Scalar<T1> && MDTensor<T2>) {                           \
      return transform(t2, [s = t1](auto &&t) { return s op_symbol t; });      \
    } /* Tensor op ScalarTensor */                                             \
    else if constexpr (MDTensor<T1> && ScalarTensor<T2>) {                     \
      return op_name(t1, t2.value());                                          \
    } /* ScalarTensor op Tensor */                                             \
    else if constexpr (ScalarTensor<T1> && MDTensor<T2>) {                     \
      return op_name(t1.value(), t2);                                          \
    } /* ScalarTensor op ScalarTensor */                                       \
    else if constexpr (ScalarTensor<T1> && ScalarTensor<T2>) {                 \
      return detail::binary_elementwise_op(std::std_op{}, t1, t2);             \
    }                                                                          \
  }

namespace venus::ops {

// Details =====================================================
namespace detail {

template <template <typename, typename, std::size_t> class Tensor,
          typename Elem1, typename Dev1, std::size_t Dim1, typename Elem2,
          typename Dev2, std::size_t Dim2>
void validate_binary_op(const Tensor<Elem1, Dev1, Dim1> &t1,
                        const Tensor<Elem2, Dev2, Dim2> &t2) {
  static_assert(Dim1 == Dim2, "Tensor dimensions must match");
  static_assert(std::is_same_v<Dev1, Dev2>,
                "Tensors must be on the same device");
  static_assert(std::is_same_v<Dev1, Device::CPU>,
                "Operation is currently only supported on CPU");

  if constexpr (Dim1 > 0) {
    if (t1.shape() != t2.shape()) {
      throw std::invalid_argument("Tensor shapes must match");
    }
  }
}

template <typename Op, template <typename, typename, std::size_t> class Tensor,
          typename Elem1, typename Dev1, std::size_t Dim1, typename Elem2,
          typename Dev2, std::size_t Dim2>
auto binary_elementwise_op(Op op, const Tensor<Elem1, Dev1, Dim1> &t1,
                           const Tensor<Elem2, Dev2, Dim2> &t2) {

  validate_binary_op(t1, t2);
  using ResultElementType = std::common_type_t<Elem1, Elem2>;

  if constexpr (Dim1 == 0 && Dim2 == 0) {
    return Tensor<ResultElementType, Dev1, 0>(op(t1.value(), t2.value()));
  } else {
    if (t1.shape() != t2.shape()) {
      throw std::invalid_argument("Tensor shapes must match");
    }

    using ResultTensor = Tensor<ResultElementType, Dev1, Dim1>;
    ResultTensor result(t1.shape());
    auto computation =
        std::views::zip(t1, t2) | std::views::transform([op](auto &&tuple) {
          return std::apply(op, tuple);
        });
    std::ranges::copy(computation, result.begin());
    return result;
  }
}

template <typename Op, template <typename, typename, std::size_t> class Tensor,
          typename Elem1, typename Dev1, std::size_t Dim1, typename Elem2,
          typename Dev2, std::size_t Dim2, typename Elem3, typename Dev3,
          std::size_t Dim3>
auto ternary_elementwise_op(Op op, const Tensor<Elem1, Dev1, Dim1> &t1,
                            const Tensor<Elem2, Dev2, Dim2> &t2,
                            const Tensor<Elem3, Dev3, Dim3> &t3) {

  validate_binary_op(t1, t2);
  validate_binary_op(t2, t3);
  using ResultElementType = std::common_type_t<Elem1, Elem2, Elem3>;

  if constexpr (Dim1 == 0 && Dim2 == 0 && Dim3 == 0) {
    return Tensor<ResultElementType, Dev1, 0>(
        op(t1.value(), t2.value(), t3.value()));
  } else {
    if (t1.shape() != t2.shape() || t2.shape() != t3.shape()) {
      throw std::invalid_argument("Tensor shapes must match");
    }

    using ResultTensor = Tensor<ResultElementType, Dev1, Dim1>;
    ResultTensor result(t1.shape());
    auto computation =
        std::views::zip(t1, t2, t3) | std::views::transform([op](auto &&tuple) {
          return std::apply(op, tuple);
        });
    std::ranges::copy(computation, result.begin());
    return result;
  }
}

} // namespace detail

// Transform
template <template <typename, typename, std::size_t> class Tensor, Scalar Elem,
          typename Dev, std::size_t Dim, typename Fn>
  requires VenusTensor<Tensor<Elem, Dev, Dim>>
auto transform(const Tensor<Elem, Dev, Dim> &tensor, Fn &&fn) {
  static_assert(std::is_same_v<Dev, Device::CPU>,
                "Transform is currently only supported on CPU");

  using ResultElementType = std::invoke_result_t<Fn, Elem>;

  if constexpr (Dim == 0) {
    return Tensor<ResultElementType, Dev, 0>(fn(tensor.value()));
  } else {
    using ResultTensor = Tensor<ResultElementType, Dev, Dim>;
    ResultTensor result(tensor.shape());
    auto computation =
        tensor | std::views::transform(
                     [f = std::forward<Fn>(fn)](auto &&t) { return f(t); });
    std::ranges::copy(computation, result.begin());
    return result;
  }
}

REGISTER_BINARY_OP(add, plus, +)
REGISTER_BINARY_OP(sub, minus, -)
REGISTER_BINARY_OP(mul, multiplies, *)
REGISTER_BINARY_OP(div, divides, /)
REGISTER_BINARY_OP(gt, greater, >)
REGISTER_BINARY_OP(gte, greater_equal, >=)
REGISTER_BINARY_OP(lt, less, <)
REGISTER_BINARY_OP(lte, less_equal, <=)
REGISTER_BINARY_OP(eq, equal_to, ==)
REGISTER_BINARY_OP(neq, not_equal_to, !=)

// Sort
template <template <typename, typename, std::size_t> class Tensor, Scalar Elem,
          typename Dev, std::size_t Dim>
  requires VenusTensor<Tensor<Elem, Dev, Dim>>
auto sort(const Tensor<Elem, Dev, Dim> &tensor) {
  static_assert(std::is_same_v<Dev, Device::CPU>,
                "Sort is currently only supported on CPU");
  auto copy = tensor.clone();
  std::ranges::sort(copy);
  return copy;
}

// All equal
template <template <typename, typename, std::size_t> class Tensor, Scalar Elem1,
          typename Dev1, std::size_t Dim1, Scalar Elem2, typename Dev2,
          std::size_t Dim2>
  requires VenusTensor<Tensor<Elem1, Dev1, Dim1>> &&
           VenusTensor<Tensor<Elem2, Dev2, Dim2>>
auto equal(const Tensor<Elem1, Dev1, Dim1> &t1,
           const Tensor<Elem2, Dev2, Dim2> &t2) -> bool {
  detail::validate_binary_op(t1, t2);
  if (t1.shape() != t2.shape()) {
    return false;
  }
  return std::ranges::equal(t1, t2);
}

// Dot product
template <template <typename, typename, std::size_t> class Tensor, Scalar Elem1,
          typename Dev1, Scalar Elem2, typename Dev2, std::size_t Dim1,
          std::size_t Dim2>
  requires VenusTensor<Tensor<Elem1, Dev1, Dim1>> &&
           VenusTensor<Tensor<Elem2, Dev2, Dim2>>
auto dot(const Tensor<Elem1, Dev1, Dim1> &t1,
         const Tensor<Elem2, Dev2, Dim2> &t2) {
  detail::validate_binary_op(t1, t2);
  using ResultElementType = std::common_type_t<Elem1, Elem2>;
  auto product =
      std::inner_product(t1.begin(), t1.end(), t2.begin(), ResultElementType{});
  return Tensor<ResultElementType, Dev1, 0>(product);
}

template <template <typename, typename, std::size_t> class Tensor, Scalar Elem,
          Scalar Idx, typename Dev, std::size_t Dim>
  requires VenusTensor<Tensor<Elem, Dev, Dim>>
auto iota(Tensor<Elem, Dev, Dim> &tensor, Idx i) {
#if _cpp_lib_ranges >= 202110L
  std::ranges::iota(tensor, i);
#else
  std::iota(tensor.begin(), tensor.end(), i);
#endif
}

template <template <typename, typename, std::size_t> class Tensor, Scalar Elem,
          Scalar Idx, typename Dev, std::size_t Dim>
  requires VenusTensor<Tensor<Elem, Dev, Dim>>
auto fill(Tensor<Elem, Dev, Dim> &tensor, Idx i) {
#if _cpp_lib_ranges >= 202110L
  std::ranges::fill(tensor, i);
#else
  std::fill(tensor.begin(), tensor.end(), i);
#endif
}

template <template <typename, typename, std::size_t> class Tensor, Scalar Elem,
          typename Dev, std::size_t Dim>
  requires BoolTensor<Tensor<Elem, Dev, Dim>>
auto where(const Tensor<Elem, Dev, Dim> &condition) {
  using ResultTensor = Tensor<std::size_t, Dev, Dim>;
  ResultTensor result(condition.shape());

  auto result_ptr = std::ranges::data(result);
  auto indices = std::views::iota(std::size_t{0}, condition.size());
  std::ranges::for_each(std::views::zip(condition, indices),
                        [result_ptr](auto &&pair) {
                          const auto &[cond_val, idx] = pair;
                          if (static_cast<bool>(cond_val)) {
                            result_ptr[idx] = idx;
                          }
                        });

  return result;
}

template <typename T1, typename T2, typename T3>
  requires VenusTensor<T1> && (VenusTensor<T2> || Scalar<T2>) &&
           (VenusTensor<T3> || Scalar<T3>)
auto where(T1 &&t1, T2 &&t2, T3 &&t3) {
  auto v1 = [&] {
    if constexpr (ScalarTensor<T1>) {
      return t1.value();
    } else {
      return std::forward<T1>(t1);
    }
  }();

  auto v2 = [&] {
    if constexpr (ScalarTensor<T2>) {
      return t2.value();
    } else {
      return std::forward<T2>(t2);
    }
  }();

  auto v3 = [&] {
    if constexpr (ScalarTensor<T3>) {
      return t3.value();
    } else {
      return std::forward<T3>(t3);
    }
  }();

  // Tensor, Tensor, Tensor
  if constexpr (MDTensor<T1> && MDTensor<T2> && MDTensor<T3>) {
    return detail::ternary_elementwise_op(
        [](auto &&a, auto &&b, auto &&c) { return a ? b : c; }, v1, v2, v3);
  }

  // Tensor, Scalar, Scalar
  else if constexpr (MDTensor<T1> && Scalar<T2> && Scalar<T3>) {
    return transform(v1, [s2 = v2, s3 = v3](auto &&a) { return a ? s2 : s3; });
  }

  // Tensor, Tensor, Scalar
  else if constexpr (MDTensor<T1> && MDTensor<T2> && Scalar<T3>) {
    return detail::binary_elementwise_op(
        [s3 = v3](auto &&a, auto &&b) { return a ? b : s3; }, v1, v2);
  }

  // Tensor, Scalar, Tensor
  else if constexpr (MDTensor<T1> && Scalar<T2> && MDTensor<T3>) {
    return detail::binary_elementwise_op(
        [s2 = v2](auto &&a, auto &&c) { return a ? s2 : c; }, v1, v3);
  }
}

} // namespace venus::ops

#undef REGISTER_BINARY_OP
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <experimental/mdspan>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace venus {

template <typename... TIntTypes>
concept SizeTLike = (std::is_convertible_v<TIntTypes, std::size_t> and ...);

template <std::size_t Dim> class Shape {
  static_assert(Dim > 0);

public:
  static constexpr std::size_t dimNum = Dim;

  constexpr explicit Shape() = default;

  template <SizeTLike... TIntTypes>
    requires(sizeof...(TIntTypes) == Dim)
  constexpr explicit Shape(TIntTypes... shapes)
      : m_dims({static_cast<std::size_t>(shapes)...}) {}

  template <SizeTLike... TIntTypes>
    requires(sizeof...(TIntTypes) != Dim)
  constexpr explicit Shape(TIntTypes...) = delete;

  constexpr auto operator==(const Shape &val) const -> bool {
    return m_dims == val.m_dims;
  }

  template <size_t otherDim>
  auto constexpr operator==(const Shape<otherDim> &) const -> bool {
    return false;
  }

  [[nodiscard]] constexpr auto count() const -> std::size_t {
    return std::ranges::fold_left(m_dims, static_cast<std::size_t>(1),
                                  std::multiplies<>());
  }

  constexpr auto operator[](size_t idx) const -> std::size_t {
    if (std::is_constant_evaluated()) {
      if (idx >= dimNum) {
        // TODO: This won't actually throw, do I really need comptime? (shape)
        throw std::out_of_range("Index out of bounds for Shape");
      }
    } else {
      assert(idx < dimNum);
    }
    return m_dims[idx];
  }

  constexpr auto offsetToIdx(std::size_t offset) const
      -> std::array<std::size_t, dimNum> {
    std::array<std::size_t, dimNum> result{};
    for (int i = (int)dimNum - 1; i >= 0 && offset > 0; --i) {
      result[i] = offset % m_dims[i];
      offset /= m_dims[i];
    }
    if (offset != 0) {
      throw std::runtime_error("Offset out of bounds!");
    }
    return result;
  }

  template <SizeTLike... TIntTypes>
  constexpr auto idxToOffset(TIntTypes... indices) const -> std::size_t {
    static_assert(sizeof...(TIntTypes) == dimNum, "Wrong number of indices");

    // TODO: The accessor policy in mdspan should be able to perform this (???)
    // bounds checking
    const std::array<std::size_t, dimNum> idx_array = {
        static_cast<std::size_t>(indices)...};
    for (std::size_t i = 0; i < dimNum; ++i) {
      if (idx_array[i] >= m_dims[i]) {
        throw std::out_of_range("Index out of bounds in Shape::IndexToOffset");
      }
    }

    auto span = createSpan(std::make_index_sequence<dimNum>{});
    return span.mapping()(indices...);
  }

  constexpr static auto fromNestedInitializerList(auto nested_init_list)
      -> Shape<dimNum> {
    Shape<dimNum> shape;

    auto extract = [](const auto &list, std::size_t level,
                      std::array<std::size_t, dimNum> &dims,
                      const auto &self_ref) -> void {
      if constexpr (requires { list.size(); }) {
        dims[level] = list.size();

        if (level + 1 < dimNum && list.size() > 0) {
          if constexpr (requires { (*list.begin()).size(); }) {
            const auto expected_size = (*list.begin()).size();

            // Horizontal: Check all sibling lists at this level
            for (const auto &sublist : list) {
              if (sublist.size() != expected_size) {
                throw std::invalid_argument(
                    std::format("Inconsistent dimensions at dimension {}: "
                                "expected size {}, got {}",
                                level + 2, expected_size, sublist.size()));
              }
            }
            // Vertical: Go deeper into each sublist
            for (const auto &sublist : list) {
              self_ref(sublist, level + 1, dims, self_ref);
            }
          }
        }
      }
    };

    extract(nested_init_list, 0, shape.m_dims, extract);
    return shape;
  }

  // Range Ops
  constexpr auto begin() { return m_dims.begin(); }
  constexpr auto end() { return m_dims.end(); }

  constexpr auto begin() const { return m_dims.begin(); }
  constexpr auto end() const { return m_dims.end(); }

  constexpr auto cbegin() const { return m_dims.begin(); }
  constexpr auto cend() const { return m_dims.end(); }

  constexpr auto size() const { return m_dims.size(); }

private:
  std::array<std::size_t, Dim> m_dims{};

  template <std::size_t... Is>
  constexpr auto createSpan(std::index_sequence<Is...> /*unused*/) const {
    return std::mdspan<int, std::dextents<std::size_t, dimNum>>(0,
                                                                m_dims[Is]...);
  }
};

template <> class Shape<0> {
public:
  static constexpr std::size_t dimNum = 0;

  explicit Shape() = default;

  static constexpr auto count() -> std::size_t { return 1; }

  constexpr auto operator==(const Shape &val) const -> bool { return true; }

  template <size_t otherDim>
  auto constexpr operator==(const Shape<otherDim> &) const -> bool {
    return false;
  }
};

template <std::size_t Dim>
auto operator<<(std::ostream &os, const Shape<Dim> &shape) -> std::ostream & {
  os << "(";
  std::size_t count = 0;
  for (auto dim : shape) {
    if (count > 0)
      os << ", ";
    count++;
    os << dim;
  }
  return os << ")";
}

template <std::size_t Dim>
auto operator<<(std::ostream &os, const Shape<0> &shape) -> std::ostream & {
  return os << "()";
}

template <SizeTLike... TShapeParameter>
explicit Shape(TShapeParameter...) -> Shape<sizeof...(TShapeParameter)>;

} // namespace venus

template <std::size_t Dim> struct std::formatter<venus::Shape<Dim>> {
  constexpr auto parse(std::format_parse_context &ctx) { return ctx.begin(); }

  auto format(const venus::Shape<Dim> &shape, std::format_context &ctx) const {
    ostringstream oss;
    oss << shape;
    return std::format_to(ctx.out(), "{}", oss.str());
  }
};
#include <algorithm>
#include <cassert>
#include <compare>
#include <concepts>
#include <cstddef>
#include <format>
#include <initializer_list>
#include <iomanip>
#include <ios>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <utility>







#define REGISTER_SCALAR_OP(op)                                                 \
  auto operator op(const ElementType &element) const noexcept                  \
      -> Tensor<bool, DeviceType, 0> {                                         \
    return Tensor<bool, DeviceType, 0>(value() op element);                    \
  }

#define REGISTER_PRE_OPERATOR(op)                                              \
  auto operator op()->ElementProxy & {                                         \
    if (not m_tensor.unique()) {                                               \
      throw std::runtime_error("Cannot write to shared tensor");               \
    }                                                                          \
    op m_element;                                                              \
    return *this;                                                              \
  }

#define REGISTER_POST_OPERATOR(op)                                             \
  auto operator op(int)->ElementType {                                         \
    if (not m_tensor.unique()) {                                               \
      throw std::runtime_error("Cannot write to shared tensor");               \
    }                                                                          \
    ElementType old_value = m_element;                                         \
    m_element op;                                                              \
    return old_value;                                                          \
  }

#define REGISTER_OPERATOR_EQUAL(op)                                            \
  auto operator op## = (const ElementType &value)->ElementProxy & {            \
    if (not m_tensor.unique()) {                                               \
      throw std::runtime_error("Cannot write to shared tensor");               \
    }                                                                          \
    m_element op## = value;                                                    \
    return *this;                                                              \
  }

namespace venus {

template <typename T> class tensor_iterator {
public:
  using iterator_category = std::contiguous_iterator_tag;
  using value_type = T::ElementType;
  using difference_type = std::ptrdiff_t;
  using pointer =
      std::conditional_t<std::is_const_v<T>, const value_type *, value_type *>;
  using reference =
      std::conditional_t<std::is_const_v<T>, const value_type &, value_type &>;

private:
  T *m_tensor;
  std::size_t m_offset;

public:
  constexpr tensor_iterator() : m_tensor(nullptr), m_offset(0) {}
  constexpr tensor_iterator(T *tensor, std::size_t offset)
      : m_tensor(tensor), m_offset(offset) {}

  constexpr auto operator*() const -> reference {
    return m_tensor->data()[m_offset];
  };

  constexpr auto operator->() const -> pointer {
    return &(m_tensor->data()[m_offset]);
  }

  constexpr auto operator++() -> tensor_iterator & {
    ++m_offset;
    return *this;
  }

  constexpr auto operator++(int) -> tensor_iterator {
    auto temp = *this;
    ++m_offset;
    return temp;
  }

  constexpr auto operator--() -> tensor_iterator & {
    --m_offset;
    return *this;
  }

  constexpr auto operator--(int) -> tensor_iterator {
    auto temp = *this;
    --m_offset;
    return temp;
  }

  constexpr auto operator+=(difference_type n) -> tensor_iterator & {
    m_offset += n;
    return *this;
  }

  constexpr auto operator-=(difference_type n) -> tensor_iterator & {
    m_offset -= n;
    return *this;
  }

  constexpr auto operator+(difference_type n) -> tensor_iterator {
    return tensor_iterator(m_tensor, m_offset + n);
  }

  constexpr auto operator+(difference_type n) const -> tensor_iterator {
    return tensor_iterator(m_tensor, m_offset + n);
  }

  constexpr auto operator-(difference_type n) -> tensor_iterator {
    return tensor_iterator(m_tensor, m_offset - n);
  }

  constexpr auto operator-(difference_type n) const -> tensor_iterator {
    return tensor_iterator(m_tensor, m_offset - n);
  }

  constexpr auto operator-(const tensor_iterator &other) const
      -> difference_type {
    return static_cast<difference_type>(m_offset) -
           static_cast<difference_type>(other.m_offset);
  }

  constexpr auto operator==(const tensor_iterator &other) const -> bool {
    return m_tensor == other.m_tensor && m_offset == other.m_offset;
  }

  constexpr auto operator<=>(const tensor_iterator &other) const {
    if (m_tensor != other.m_tensor) {
      // comparing different tensors alltogether
      return std::compare_three_way{}(m_tensor, other.m_tensor);
    }
    // comparing offsets on the same tensor
    return m_offset <=> other.m_offset;
  }

  constexpr auto operator[](difference_type n) const -> reference {
    return *(*this + n);
  }
};

template <typename TElem, typename TDevice, std::size_t Dim> class Tensor {
  static_assert(std::is_same_v<std::remove_cvref_t<TElem>, TElem>);
  static_assert(Dim > 0);

public:
  using ElementType = TElem;
  using DeviceType = TDevice;
  static constexpr std::size_t Dimension = Dim;

  friend struct LowLevelAccess<Tensor>;
  friend struct LowLevelAccess<const Tensor>;

  explicit Tensor(Shape<Dim> shape)
      : m_shape(std::move(shape)), m_mem(shape.count()) {}

  explicit Tensor(ContiguousMemory<ElementType, DeviceType> p_mem,
                  Shape<Dim> p_shape)
      : m_shape(std::move(p_shape)), m_mem(std::move(p_mem)) {
    if (m_mem.size() < m_shape.count()) {
      throw std::invalid_argument(
          std::format("Insufficient memory for tensor shape: need {} elements, "
                      "but only {} provided",
                      m_shape.count(), m_mem.size()));
    }
  }

  explicit Tensor(nested_initializer_list_t<ElementType, Dim> init_list)
      : m_shape(Shape<Dim>::fromNestedInitializerList(init_list)),
        m_mem(m_shape.count()) {

    auto flatten = [](const auto &list, ElementType *output_ptr,
                      const auto &self_ref) -> ElementType * {
      if constexpr (std::same_as<std::decay_t<decltype(*list.begin())>,
                                 ElementType>) {
        return std::ranges::copy(list, output_ptr).out;
      } else {
        for (const auto &inner_list : list) {
          output_ptr = self_ref(inner_list, output_ptr, self_ref);
        }
        return output_ptr;
      }
    };

    flatten(init_list, data(), flatten);
  }

  template <std::size_t D = Dim>
    requires(D == 1)
  explicit Tensor(std::initializer_list<ElementType> init_list)
      : m_shape(init_list.size()), m_mem(init_list.size()) {
    std::ranges::copy(init_list, data());
  }

  template <std::size_t D = Dim>
    requires(D != 1)
  explicit Tensor(std::initializer_list<ElementType>) = delete;

  template <typename... Dims>
    requires(sizeof...(Dims) == Dim) &&
            (std::is_convertible_v<Dims, std::size_t> && ...)
  explicit Tensor(Dims &&...dimensions)
      : Tensor(Shape<Dim>(std::forward<Dims>(dimensions)...)) {}

  template <typename... Dims>
    requires(sizeof...(Dims) != Dim) &&
                (std::is_convertible_v<Dims, std::size_t> && ...)
  explicit Tensor(Dims &&...) = delete;

  auto operator=(const Tensor &other) -> Tensor & {
    if (this != &other) {
      m_shape = other.m_shape;
      m_mem = ContiguousMemory<ElementType, DeviceType>(m_shape.count());
      std::ranges::copy(other, this->begin());
    }
    return *this;
  }

  auto operator=(Tensor &&other) noexcept -> Tensor & {
    if (this != &other) {
      m_shape = other.m_shape;
      m_mem = std::move(other.m_mem);
    }
    return *this;
  }

  Tensor(const Tensor &other) : m_shape(other.m_shape), m_mem(m_shape.count()) {
    std::ranges::copy(other, this->begin());
  }

  Tensor(Tensor &&other) noexcept
      : m_shape(other.m_shape), m_mem(std::move(other.m_mem)) {}

  ~Tensor() = default; // give back to mem-pool

  auto shape() const noexcept -> const Shape<Dim> & { return m_shape; }

  [[nodiscard]] auto unique() const -> bool { return not m_mem.isShared(); }

  auto clone() const -> Tensor { return Tensor(*this); }

  auto toScalar() const -> Tensor<TElem, TDevice, 0> {
    static_assert(Dim == 1,
                  "ToScalar can only be called on 1D tensors with 1 element.");
    if (size() != 1) {
      throw std::runtime_error(
          std::format("Cannot convert non-scalar tensor to scalar tensor: "
                      "Tensor size is {}, while the size of a scalar is 1.",
                      size()));
    }
    return Tensor<TElem, TDevice, 0>(*std::ranges::data(*this));
  }

  // Addition
  template <typename OtherType> auto operator+(OtherType &&other) const {
    return venus::ops::add(*this, std::forward<OtherType>(other));
  }

  // Subtraction
  template <typename OtherType> auto operator-(OtherType &&other) const {
    return venus::ops::sub(*this, std::forward<OtherType>(other));
  }

  // Multiplication
  template <typename OtherType> auto operator*(OtherType &&other) const {
    return venus::ops::mul(*this, std::forward<OtherType>(other));
  }

  // Division
  template <typename OtherType> auto operator/(OtherType &&other) const {
    return venus::ops::div(*this, std::forward<OtherType>(other));
  }

  // Greater than
  template <typename OtherType> auto operator>(OtherType &&other) const {
    return venus::ops::gt(*this, std::forward<OtherType>(other));
  }

  // Greater or equal
  template <typename OtherType> auto operator>=(OtherType &&other) const {
    return venus::ops::gte(*this, std::forward<OtherType>(other));
  }

  // Less than
  template <typename OtherType> auto operator<(OtherType &&other) const {
    return venus::ops::lt(*this, std::forward<OtherType>(other));
  }

  // Less or equal
  template <typename OtherType> auto operator<=(OtherType &&other) const {
    return venus::ops::lte(*this, std::forward<OtherType>(other));
  }

  // Equal
  template <typename OtherType> auto operator==(OtherType &&other) const {
    return venus::ops::eq(*this, std::forward<OtherType>(other));
  }

  // Not equal
  template <typename OtherType> auto operator!=(OtherType &&other) const {
    return venus::ops::neq(*this, std::forward<OtherType>(other));
  }

  // Dot product
  template <typename OtherElementType>
  auto dot(const Tensor<OtherElementType, DeviceType, Dim> &other) const {
    return venus::ops::dot(*this, other);
  }

  //* Proxy pattern for indexing elements (know when I'm reading vs writing)
  //? Price to pay: have to specify all possible operator overloads that I want
  class ElementProxy {
  private:
    Tensor &m_tensor;
    ElementType &m_element;

  public:
    ElementProxy(Tensor &tensor, ElementType &element)
        : m_tensor(tensor), m_element(element) {}

    // reading
    operator ElementType() const { return m_element; }

    // writing
    auto operator=(const ElementType &value) -> ElementProxy & {
      if (not m_tensor.unique()) {
        // TODO: Do I want to throw here or do I want copy on write (cow)
        throw std::runtime_error("Cannot write to shared tensor");
        //     const std::size_t offset = &m_element -
        //     m_tensor.m_mem.RawMemory();
        // auto new_mem =
        //     ContiguousMemory<ElementType, DeviceType>(m_tensor.m_mem.Size());
        // std::copy_n(m_tensor.m_mem.RawMemory(), m_tensor.m_mem.Size(),
        //             new_mem.RawMemory());
        // m_tensor.m_mem = std::move(new_mem);
        // m_tensor.m_mem.RawMemory()[offset] = value;
        // return *this;
      }
      m_element = value;
      return *this;
    }

    auto operator=(ElementType &&value) -> ElementProxy & {
      if (not m_tensor.unique()) {
        throw std::runtime_error("Cannot write to shared tensor");
      }
      m_element = std::move(value);
      return *this;
    }

    // Required for modifying the tensor through range algos
    template <typename U>
      requires std::convertible_to<U, ElementType>
    auto operator=(U &&value) -> ElementProxy & {
      if (not m_tensor.unique()) {
        throw std::runtime_error("Cannot write to shared tensor");
      }
      m_element = std::forward<U>(value);
      return *this;
    }

    // explicit conversion (to extract the element by type)
    template <typename U> explicit operator U() const {
      return static_cast<U>(m_element);
    }

    REGISTER_OPERATOR_EQUAL(+)
    REGISTER_OPERATOR_EQUAL(-)
    REGISTER_OPERATOR_EQUAL(*)
    REGISTER_OPERATOR_EQUAL(/)
    REGISTER_OPERATOR_EQUAL(%)

    REGISTER_PRE_OPERATOR(++)
    REGISTER_PRE_OPERATOR(--)
    REGISTER_POST_OPERATOR(++)
    REGISTER_POST_OPERATOR(--)
  };

#ifdef VENUS_INTERPRETER
  // Simple direct indexing for interpreter mode (no proxy)
  template <typename... Indices>
    requires(sizeof...(Indices) == Dim)
  constexpr auto operator[](Indices... indices) -> ElementType {
    static_assert(std::is_same_v<DeviceType, Device::CPU>,
                  "Indexing is currently only supported on CPU");
    const auto offset =
        m_shape.IndexToOffset(static_cast<std::size_t>(indices)...);
    return data()[offset];
  }

  template <typename... Indices>
    requires(sizeof...(Indices) == Dim)
  constexpr auto operator[](Indices... indices) const -> ElementType {
    static_assert(std::is_same_v<DeviceType, Device::CPU>,
                  "Indexing is currently only supported on CPU");
    const auto offset =
        m_shape.IndexToOffset(static_cast<std::size_t>(indices)...);
    return data()[offset];
  }
#else
  // Tensor indexing
  template <typename... Indices>
    requires(sizeof...(Indices) == Dim)
  auto operator[](Indices... indices) -> ElementProxy {
    static_assert(std::is_same_v<DeviceType, Device::CPU>,
                  "Indexing is currently only supported on CPU");
    const auto offset =
        m_shape.idxToOffset(static_cast<std::size_t>(indices)...);
    return ElementProxy(*this, data()[offset]);
  }

  template <typename... Indices>
    requires(sizeof...(Indices) == Dim)
  auto operator[](Indices... indices) const -> ElementType {
    static_assert(std::is_same_v<DeviceType, Device::CPU>,
                  "Indexing is currently only supported on CPU");
    const auto offset =
        m_shape.idxToOffset(static_cast<std::size_t>(indices)...);
    return data()[offset];
  }
#endif

  auto evalRegister() const;

  auto lowLevel() { return LowLevelAccess<Tensor>(*this); }
  auto lowLevel() const { return LowLevelAccess<const Tensor>(*this); }

private:
  Shape<Dim> m_shape;
  ContiguousMemory<ElementType, DeviceType> m_mem;

public:
  constexpr auto begin() {
    static_assert(std::is_same_v<DeviceType, Device::CPU>,
                  "Range iteration is currently only supported on CPU");
    return tensor_iterator<Tensor>(this, 0);
  }

  constexpr auto end() {
    static_assert(std::is_same_v<DeviceType, Device::CPU>,
                  "Range iteration is currently only supported on CPU");
    return tensor_iterator<Tensor>(this, m_shape.count());
  }

  constexpr auto begin() const {
    static_assert(std::is_same_v<DeviceType, Device::CPU>,
                  "Range iteration is currently only supported on CPU");
    return tensor_iterator<const Tensor>(this, 0);
  }

  constexpr auto end() const {
    static_assert(std::is_same_v<DeviceType, Device::CPU>,
                  "Range iteration is currently only supported on CPU");
    return tensor_iterator<const Tensor>(this, m_shape.count());
  }

  constexpr auto cbegin() const { return begin(); }
  constexpr auto cend() const { return end(); }

  [[nodiscard]] constexpr auto size() const -> std::size_t {
    return m_shape.count();
  }

  auto data() -> ElementType * { return m_mem.rawMemory(); }
  auto data() const -> const ElementType * { return m_mem.rawMemory(); }
};

// Scalar Tensor ===============================================
template <typename TElem, typename TDevice> class Tensor<TElem, TDevice, 0> {
  static_assert(std::is_same_v<std::remove_cvref_t<TElem>, TElem>);

public:
  using ElementType = TElem;
  using DeviceType = TDevice;
  static constexpr std::size_t Dimension = 0;

  friend struct LowLevelAccess<Tensor>;
  friend struct LowLevelAccess<const Tensor>;

  explicit Tensor(ElementType value = ElementType()) : m_mem(1) {
    setValue(value);
  }

  explicit Tensor(Shape<0> /*unused*/) : Tensor() {};

  explicit Tensor(ContiguousMemory<ElementType, DeviceType> p_mem)
      : m_mem(std::move(p_mem)) {}

  auto operator=(const Tensor &other) -> Tensor & {
    if (this != &other) {
      m_mem = ContiguousMemory<ElementType, DeviceType>(1);
      setValue(other.value());
    }
    return *this;
  }

  auto operator=(Tensor &&other) noexcept -> Tensor & {
    if (this != &other) {
      m_mem = std::move(other.m_mem);
    }
    return *this;
  }

  Tensor(const Tensor &other) : m_mem(1) { setValue(other.value()); }

  Tensor(Tensor &&other) noexcept : m_mem(std::move(other.m_mem)) {}

  ~Tensor() = default; // give back to mem-pool

  auto shape() const noexcept -> const auto & {
    static const Shape<Dimension> shape;
    return shape;
  }

  [[nodiscard]] auto unique() const -> bool { return not m_mem.isShared(); }

  void setValue(ElementType value) const = delete;

  void setValue(ElementType value) {
    if (not unique()) {
      throw std::runtime_error("Cannot write to shared scalar tensor.");
    }
    data()[0] = value;
  }

  auto value() const noexcept { return data()[0]; }

  auto operator==(const Tensor &tensor) const noexcept -> bool {
    return value() == tensor.value();
  }

  // Addition
  template <typename OtherType> auto operator+(OtherType &&other) const {
    return venus::ops::add(*this, std::forward<OtherType>(other));
  }

  // Subtraction
  template <typename OtherType> auto operator-(OtherType &&other) const {
    return venus::ops::sub(*this, std::forward<OtherType>(other));
  }

  // Multiplication
  template <typename OtherType> auto operator*(OtherType &&other) const {
    return venus::ops::mul(*this, std::forward<OtherType>(other));
  }

  // Division
  template <typename OtherType> auto operator/(OtherType &&other) const {
    return venus::ops::div(*this, std::forward<OtherType>(other));
  }

  // Dot product
  template <typename OtherElementType>
  auto dot(const Tensor<OtherElementType, DeviceType, Dimension> &other) const {
    return venus::ops::dot(*this, other);
  }

  operator bool() const noexcept {
    if constexpr (std::is_same_v<ElementType, bool>) {
      return value();
    } else {
      return value() != ElementType{};
    }
  }

  REGISTER_SCALAR_OP(==)
  REGISTER_SCALAR_OP(!=)
  REGISTER_SCALAR_OP(<)
  REGISTER_SCALAR_OP(<=)
  REGISTER_SCALAR_OP(>)
  REGISTER_SCALAR_OP(>=)

  auto evalRegister() const;

  auto lowLevel() { return LowLevelAccess<Tensor>(*this); }
  auto lowLevel() const { return LowLevelAccess<const Tensor>(*this); }

  [[nodiscard]] constexpr auto size() const -> std::size_t { return 1; }

  auto data() -> ElementType * { return m_mem.rawMemory(); }
  auto data() const -> const ElementType * { return m_mem.rawMemory(); }

private:
  ContiguousMemory<ElementType, DeviceType> m_mem;
};

template <typename TElem, typename TDevice, std::size_t Dim>
struct LowLevelAccess<Tensor<TElem, TDevice, Dim>> {
  LowLevelAccess(Tensor<TElem, TDevice, Dim> &tensor) : m_tensor(tensor) {}
  auto rawMemory() -> TElem * { return m_tensor.m_mem.rawMemory(); }
  auto sharedMemory() const { return m_tensor.m_mem; }

private:
  Tensor<TElem, TDevice, Dim> &m_tensor;
};

template <typename TElem, typename TDevice, std::size_t Dim>
struct LowLevelAccess<const Tensor<TElem, TDevice, Dim>> {
  LowLevelAccess(const Tensor<TElem, TDevice, Dim> &tensor)
      : m_tensor(tensor) {}
  auto rawMemory() const -> const TElem * { return m_tensor.m_mem.rawMemory(); }
  auto sharedMemory() const { return m_tensor.m_mem; }

private:
  const Tensor<TElem, TDevice, Dim> &m_tensor;
};

// difference_type + iterator (for random_access_range | addition commutative)
template <typename T>
constexpr auto operator+(typename tensor_iterator<T>::difference_type n,
                         const tensor_iterator<T> &it) -> tensor_iterator<T> {
  return it + n;
}

// Print Tensor ================================================
template <typename T>
concept StringLike = requires(T str) {
  { str.c_str() } -> std::convertible_to<const char *>;
} or std::convertible_to<T, const char *>;

template <typename T>
concept CharLike = std::same_as<T, char> || std::same_as<T, signed char> ||
                   std::same_as<T, unsigned char> || std::same_as<T, wchar_t> ||
                   std::same_as<T, char8_t> || std::same_as<T, char16_t> ||
                   std::same_as<T, char32_t>;

template <typename TElem, typename TDevice, std::size_t Dim>
auto operator<<(std::ostream &os, const Tensor<TElem, TDevice, Dim> &tensor)
    -> std::ostream & {
  os << "venus::Tensor([";

  std::size_t count = 0;
  for (auto elem : tensor) {
    if (count > 0)
      os << ", ";
    count++;
    if constexpr (StringLike<TElem>) {
      os << "\"" << static_cast<TElem>(elem) << "\"";
    } else if constexpr (CharLike<TElem>) {
      os << "'" << static_cast<TElem>(elem) << "'";
    } else if constexpr (std::floating_point<TElem>) {
      os << std::fixed << std::setprecision(2) << static_cast<TElem>(elem);
    } else {
      os << static_cast<TElem>(elem);
    }
  }

  return os << "], " << "shape=" << tensor.shape() << ")";
}

template <typename TElem, typename TDevice>
auto operator<<(std::ostream &os, const Tensor<TElem, TDevice, 0> &tensor)
    -> std::ostream & {
  if constexpr (StringLike<TElem>) {
    return os << "venus::Tensor(" << "\"" << tensor.value() << "\"" << ")";
  } else if constexpr (CharLike<TElem>) {
    return os << "venus::Tensor(" << "'" << tensor.value() << "'" << ")";
  } else if constexpr (std::floating_point<TElem>) {
    return os << std::fixed << std::setprecision(2) << "venus::Tensor("
              << tensor.value() << ")";
  } else {
    return os << "venus::Tensor(" << tensor.value() << ")";
  }
}

} // namespace venus

template <typename TElem, typename TDevice, std::size_t Dim>
struct std::formatter<venus::Tensor<TElem, TDevice, Dim>> {
  constexpr auto parse(std::format_parse_context &ctx) { return ctx.begin(); }

  auto format(const venus::Tensor<TElem, TDevice, Dim> &tensor,
              std::format_context &ctx) const {
    ostringstream oss;
    oss << tensor;
    return std::format_to(ctx.out(), "{}", oss.str());
  }
};

// Scalar-first Addition
template <venus::Scalar Scalar, venus::Scalar TElem, typename TDevice,
          std::size_t Dim>
auto operator+(const Scalar &scalar,
               const venus::Tensor<TElem, TDevice, Dim> &tensor) {
  return venus::ops::add(scalar, tensor);
}

// Scalar-first Substraction
template <venus::Scalar Scalar, venus::Scalar TElem, typename TDevice,
          std::size_t Dim>
auto operator-(const Scalar &scalar,
               const venus::Tensor<TElem, TDevice, Dim> &tensor) {
  return venus::ops::sub(scalar, tensor);
}

// Scalar-first Multiplication
template <venus::Scalar Scalar, venus::Scalar TElem, typename TDevice,
          std::size_t Dim>
auto operator*(const Scalar &scalar,
               const venus::Tensor<TElem, TDevice, Dim> &tensor) {
  return venus::ops::mul(scalar, tensor);
}

// Scalar-fist Division
template <venus::Scalar Scalar, venus::Scalar TElem, typename TDevice,
          std::size_t Dim>
auto operator/(const Scalar &scalar,
               const venus::Tensor<TElem, TDevice, Dim> &tensor) {
  return venus::ops::div(scalar, tensor);
}

#undef REGISTER_OPERATOR_EQUAL
#undef REGISTER_POST_OPERATOR
#undef REGISTER_PRE_OPERATOR
#undef REGISTER_SCALAR_OP

#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>



namespace venus {

template <typename... TParameters> struct VarTypeDict {

  template <typename... Types> struct Values {

    static constexpr std::size_t TupleSize =
        sizeof...(Types) == 0 ? 1 : sizeof...(Types);

    using Tuple = std::array<std::shared_ptr<void>, TupleSize>;

    template <typename TKey>
    using ValueType =
        Sequential::At<Values, Sequential::Order<VarTypeDict, TKey>>;

    template <typename TKey>
    static constexpr bool IsValueEmpty =
        std::is_same_v<ValueType<TKey>, NullParameter>;

    Values() = default;

    Values(Values &&val) noexcept {
      for (size_t i = 0; i < sizeof...(Types); ++i) {
        m_tuple[i] = std::move(val.m_tuple[i]);
      }
    }

    Values(const Values &) = default;
    auto operator=(const Values &) -> Values & = default;
    auto operator=(Values &&) -> Values & = default;
    ~Values() = default;

    Values(Tuple &&input) : m_tuple(std::move(input)) {}

    template <typename TTag, typename... TParams>
    void Update(TParams &&...params) {
      static constexpr auto idx = Sequential::Order<VarTypeDict, TTag>;
      using RawType = Sequential::At<Values, idx>;

      auto *tmp = new RawType(std::forward<TParams>(params)...);
      m_tuple[idx] = std::shared_ptr<void>(tmp, [](void *ptr) {
        auto *nptr = static_cast<RawType *>(ptr);
        delete nptr;
      });
    }

    template <typename TTag, typename... TParams>
    auto ChainUpdate(TParams &&...params) -> Values & {
      static constexpr auto idx = Sequential::Order<VarTypeDict, TTag>;
      using RawType = Sequential::At<Values, idx>;

      auto *tmp = new RawType(std::forward<TParams>(params)...);
      m_tuple[idx] = std::shared_ptr<void>(tmp, [](void *ptr) {
        auto *nptr = static_cast<RawType *>(ptr);
        delete nptr;
      });

      return *this;
    }

    template <typename TTag, typename TVal> auto Set(TVal &&val) && {
      static constexpr auto idx = Sequential::Order<VarTypeDict, TTag>;
      using RawType = std::remove_cvref_t<TVal>;

      auto *tmp = new RawType(std::forward<TVal>(val));
      m_tuple[idx] = std::shared_ptr<void>(tmp, [](void *ptr) {
        auto *nptr = static_cast<RawType *>(ptr);
        delete nptr;
      });

      if constexpr (std::is_same_v<RawType, Sequential::At<Values, idx>>) {
        return *this;
      } else {
        using NewType = Sequential::Set<Values, idx, RawType>;
        return NewType(std::move(m_tuple));
      }
    }

    template <typename TTag> auto Get() const -> const auto & {
      static constexpr auto idx = Sequential::Order<VarTypeDict, TTag>;
      using AimType = Sequential::At<Values, idx>;

      void *tmp = m_tuple[idx].get();
      if (!tmp) {
        throw std::runtime_error("Empty Value.");
      }
      auto *res = static_cast<AimType *>(tmp);
      return *res;
    }

    template <typename TTag> auto Get() -> auto & {
      static constexpr auto idx = Sequential::Order<VarTypeDict, TTag>;
      using AimType = Sequential::At<Values, idx>;

      void *tmp = m_tuple[idx].get();
      if (!tmp) {
        throw std::runtime_error("Empty Value.");
      }
      auto *res = static_cast<AimType *>(tmp);
      return *res;
    }

  private:
    Tuple m_tuple{};
  };

  static auto Create() {
    using type = Sequential::Create<sizeof...(TParameters), Values>;
    return type{};
  }
};

}; // namespace venus
