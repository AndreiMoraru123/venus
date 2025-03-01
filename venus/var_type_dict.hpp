#include "null_param.hpp"
#include "sequential.hpp"
#include "traits.hpp"
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace venus {

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

    // chained update alternative
    template <typename TTag, typename... TParams>
    Values &ChainUpdate(TParams &&...params) {
      static constexpr auto TagPos = Sequential::Order<VarTypeDict, TTag>;
      using rawType = Sequential::At<Values, TagPos>;

      rawType *tmp = new rawType(std::forward<TParams>(params)...);
      m_tuple[TagPos] = std::shared_ptr<void>(tmp, [](void *ptr) {
        rawType *nptr = static_cast<rawType *>(ptr);
        delete nptr;
      });

      return *this;
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
    using type = typename Sequential::Create<sizeof...(TParameters), Values>;
    return type{};
  }
};

}; // namespace venus