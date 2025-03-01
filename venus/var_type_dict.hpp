#include "sequential.hpp"
#include "traits.hpp"
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>

struct NullParameter;

namespace venus {

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

}; // namespace NSVarTypeDict

template <typename... TParameters> struct VarTypeDict {

  template <typename... Types> struct Values {

    using Keys = VarTypeDict;

    template <typename TKey>
    using ValueType =
        Sequential::At<Values, Sequential::Order<VarTypeDict, TKey>>;

    template <typename TKey>
    static constexpr bool IsValueEmpty =
        std::is_same_v<ValueType<TKey>, NullParameter>;

    Values() = default;

    Values(Values &&val) {
      for (size_t i = 0; i < sizeof...(Types); ++i) {
        m_tuple[i] = std::move(val.m_tuple[i]);
      }
    }

    Values(const Values &) = default;
    Values &operator=(const Values &) = default;
    Values &operator=(Values &&) = default;

    Values(std::shared_ptr<void> (&&input)[sizeof...(Types)]) {
      for (size_t i = 0; i < sizeof...(Types); ++i) {
        m_tuple[i] = std::move(input[i]);
      }
    }

    template <typename TTag, typename... TParams>
    void Update(TParams &&...params) {
      static constexpr auto TagPos = Sequential::Order<VarTypeDict, TTag>;
      using rawType = Sequential::At<Values, TagPos>;

      rawType *tmp = new rawType(std::forward<TParams>(params)...);
      m_tuple[TagPos] = std::shared_ptr<void>(tmp, [](void *ptr) {
        rawType *nptr = static_cast<rawType *>(ptr);
        delete nptr;
      });
    }

    template <typename TTag, typename TVal> auto Set(TVal &&val) && {
      static constexpr auto TagPos = Sequential::Order<VarTypeDict, TTag>;
      using rawType = RemoveConstRef<TVal>;

      rawType *tmp = new rawType(std::forward<TVal>(val));
      m_tuple[TagPos] = std::shared_ptr<void>(tmp, [](void *ptr) {
        rawType *nptr = static_cast<rawType *>(ptr);
        delete nptr;
      });

      if constexpr (std::is_same_v<rawType, Sequential::At<Values, TagPos>>) {
        return *this;
      } else {
        using newType = Sequential::Set<Values, TagPos, rawType>;
        return newType(std::move(m_tuple));
      }
    }

    template <typename TTag> const auto &Get() const {
      static constexpr auto TagPos = Sequential::Order<VarTypeDict, TTag>;
      using AimType = Sequential::At<Values, TagPos>;

      void *tmp = m_tuple[TagPos].get();
      if (!tmp)
        throw std::runtime_error("Empty Value.");
      AimType *res = static_cast<AimType *>(tmp);
      return *res;
    }

    template <typename TTag> auto &Get() {
      static constexpr auto TagPos = Sequential::Order<VarTypeDict, TTag>;
      using AimType = Sequential::At<Values, TagPos>;

      void *tmp = m_tuple[TagPos].get();
      if (!tmp)
        throw std::runtime_error("Empty Value.");
      AimType *res = static_cast<AimType *>(tmp);
      return *res;
    }

  private:
    std::shared_ptr<void> m_tuple[sizeof...(Types) == 0 ? 1 : sizeof...(Types)];
  };

  static auto Create() {
    using namespace NSVarTypeDict;
    using type = typename Create_<sizeof...(TParameters), Values>::type;
    return type{};
  }
};

}; // namespace venus