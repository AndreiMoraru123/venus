#include <cstddef>
#include <memory>

struct NullParameter;

namespace venus {

// TODO: Rename TParameters to TParams
// TODO: Rename TTag to TKey

namespace NSVarTypeDict {

/**
 * @brief
 * @tparam N = number of elements to be created
 * @tparam TCont = array of types that stores the final results
 * @tparam T = sequence of types that have already been generated
 */
template <size_t N, template <typename...> class TCont, typename... T>
struct Create_ {
  using type = Create_<N - 1, TCont, NullParameter, T...>::type;
};

// N = 0 base case => return the array of types directly
template <template <typename...> class TCont, typename... T>
struct Create_<0, TCont, T...> {
  using type = TCont<T...>;
};

/**
 * @brief
 *
 * @tparam TVal = target data for replacement
 * @tparam N = index of the target type in the array of types
 * @tparam M = number of types that have been scanned
 * @tparam TProcessedTypes = array containers for the scanned part
 * @tparam TRemainTypes = the part to be scanned and replaced
 */
template <typename TVal, size_t N, size_t M, typename TProcessedTypes,
          typename... TRemainTypes>
struct NewTupleType_;

template <typename TVal, size_t N, size_t M,
          template <typename...> typename TCont, typename... TModifiedTypes,
          typename TCurType, typename... TRemainTypes>
struct NewTupleType_<TVal, N, M, TCont<TModifiedTypes...>, TCurType,
                     TRemainTypes...> {
  using type =
      typename NewTupleType_<TVal, N, M + 1, TCont<TModifiedTypes..., TCurType>,
                             TRemainTypes...>::type;
};

}; // namespace NSVarTypeDict

template <typename... TParameters> struct VarTypeDict {

  template <typename... TTypes> struct Values {

    Values() = default;

    Values(std::shared_ptr<void> (&&input)[sizeof...(TTypes)]) {
      for (size_t i = 0; i < sizeof...(TTypes); ++i) {
        m_tuple[i] = std::move(input[i]);
      }
    }

    // template <typename TTag, typename TVal> auto Set(TVal &&val) && {
    //   using namespace NSMultiTypeDict;
    //   constexpr static size_t TagPos = Tag2ID<TTag, TParameters...>;

    //   using rawVal = std::decay_t<TVal>;
    //   auto tmp = new rawVal(std::forward<TVal>(val));
    //   m_tuple[TagPos] = std::shared_ptr<void>(tmp, [](void *ptr) {
    //     rawVal *nptr = static_cast<rawVal *>(ptr);
    //     delete nptr;
    //   });

    //   using new_type = NewTupleType<rawVal, TagPos, Values<>, TTypes...>;
    //   return new_type(std::move(m_tuple));
    // }

    template <typename TTag> const auto Get() const;

  private:
    // data storage array
    std::shared_ptr<void> m_tuple[sizeof...(TTypes)];
  };

  static auto Create() {
    using namespace NSVarTypeDict;
    using type = typename Create_<sizeof...(TParameters), Values>::type;
    return type{};
  }
};

}; // namespace venus