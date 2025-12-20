#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <venus/null_param.hpp>
#include <venus/sequential.hpp>

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
      static constexpr auto idx = Sequential::Order<VarTypeDict, TTag>;
      using RawType = Sequential::At<Values, idx>;

      RawType *tmp = new RawType(std::forward<TParams>(params)...);
      m_tuple[idx] = std::shared_ptr<void>(tmp, [](void *ptr) {
        RawType *nptr = static_cast<RawType *>(ptr);
        delete nptr;
      });
    }

    template <typename TTag, typename... TParams>
    auto ChainUpdate(TParams &&...params) -> Values & {
      static constexpr auto idx = Sequential::Order<VarTypeDict, TTag>;
      using RawType = Sequential::At<Values, idx>;

      RawType *tmp = new RawType(std::forward<TParams>(params)...);
      m_tuple[idx] = std::shared_ptr<void>(tmp, [](void *ptr) {
        RawType *nptr = static_cast<RawType *>(ptr);
        delete nptr;
      });

      return *this;
    }

    template <typename TTag, typename TVal> auto Set(TVal &&val) && {
      static constexpr auto idx = Sequential::Order<VarTypeDict, TTag>;
      using RawType = std::remove_cvref_t<TVal>;

      RawType *tmp = new RawType(std::forward<TVal>(val));
      m_tuple[idx] = std::shared_ptr<void>(tmp, [](void *ptr) {
        RawType *nptr = static_cast<RawType *>(ptr);
        delete nptr;
      });

      if constexpr (std::is_same_v<RawType, Sequential::At<Values, idx>>) {
        return *this;
      } else {
        using NewType = Sequential::Set<Values, idx, RawType>;
        return NewType(std::move(m_tuple));
      }
    }

    template <typename TTag> const auto &Get() const {
      static constexpr auto idx = Sequential::Order<VarTypeDict, TTag>;
      using AimType = Sequential::At<Values, idx>;

      void *tmp = m_tuple[idx].get();
      if (!tmp)
        throw std::runtime_error("Empty Value.");
      AimType *res = static_cast<AimType *>(tmp);
      return *res;
    }

    template <typename TTag> auto &Get() {
      static constexpr auto idx = Sequential::Order<VarTypeDict, TTag>;
      using AimType = Sequential::At<Values, idx>;

      void *tmp = m_tuple[idx].get();
      if (!tmp)
        throw std::runtime_error("Empty Value.");
      AimType *res = static_cast<AimType *>(tmp);
      return *res;
    }

  private:
    std::shared_ptr<void> m_tuple[sizeof...(Types) == 0 ? 1 : sizeof...(Types)];
  };

  static auto Create() {
    using type = Sequential::Create<sizeof...(TParameters), Values>;
    return type{};
  }
};

}; // namespace venus