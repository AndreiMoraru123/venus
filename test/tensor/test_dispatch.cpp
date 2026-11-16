#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#ifdef VENUS_IMPORT
import venus;
#else
#include <venus/tensor/tensor.hpp>
#endif

using namespace venus;

// A runtime that records the name of each op it runs and holds no values
namespace {
struct Recorder {};

struct Node {
  std::vector<std::string> *log;
};

template <typename Elem, std::size_t Rank>
auto input(std::vector<std::string> &log, Shape<Rank> shape) {
  return Tensor<Elem, Recorder, Rank>(Node{&log}, shape);
}

template <typename T, typename... Rest>
auto log_of(const T &first, const Rest &...rest) {
  if constexpr (VenusTensor<T>) {
    return first.storage().log;
  } else {
    return log_of(rest...);
  }
}
} // namespace

template <typename Elem, std::size_t Rank>
struct venus::StorageFor<Elem, Recorder, Rank> {
  using type = Node;
};

// A CPU operand joins a recorded one and becomes a recorded constant
template <> struct venus::Join<Runtime::CPU, Recorder> {
  using type = Recorder;
};
template <> struct venus::Join<Recorder, Runtime::CPU> {
  using type = Recorder;
};

template <> struct venus::Lift<Runtime::CPU, Recorder> {
  template <typename T, typename Other>
  static auto run(const T &tensor, const Other &other) {
    other.storage().log->emplace_back("const");
    return Tensor<typename T::ElementType, Recorder, T::rank>(other.storage(),
                                                              tensor.shape());
  }
};

// Composite ops get no action, so dispatch composes them
template <typename Op>
  requires(!CompositeOp<Op>)
struct venus::Action<Op, Recorder> {
  template <typename... Operands> static auto run(const Operands &...operands) {
    const auto m = Op::rule(meta(operands)...);
    auto *log = log_of(operands...);
    log->emplace_back(Op::name);
    return result_t<decltype(m), Recorder>(Node{log}, m.shape);
  }
};

namespace {
template <typename T>
concept HasValue = requires(const T &t) { t.value(); };
template <typename T>
concept Indexable = requires(const T &t) { t[0, 0]; };
template <typename T>
concept Iterable = requires(const T &t) { t.begin(); };
template <typename T>
concept EagerAddable = requires(const T &t) { eager::add(t, t); };

using Names = std::vector<std::string>;
} // namespace

TEST_CASE("Dispatch on a recording runtime", "[tensor][dispatch]") {
  Names log;
  const auto a = input<float>(log, Shape(2, 3));
  const auto row = input<int>(log, Shape(3));

  SECTION("Operators record one node per primitive op") {
    auto c = a + row;
    auto d = c * 2;

    STATIC_REQUIRE(std::same_as<decltype(d), Tensor<float, Recorder, 2>>);
    REQUIRE(d.shape() == Shape(2, 3));
    REQUIRE(log == Names{"add", "mul"});
  }

  SECTION("A comparison gives bool") {
    auto mask = a > 0;
    STATIC_REQUIRE(std::same_as<decltype(mask), Tensor<bool, Recorder, 2>>);
    REQUIRE(log == Names{"gt"});
  }

  SECTION("The rule checks shapes on every runtime") {
    const auto b = input<float>(log, Shape(2, 2));
    REQUIRE_THROWS_AS(a + b, std::invalid_argument);
    REQUIRE_THROWS_AS(mm(a, b), std::invalid_argument);
  }

  SECTION("A CPU operand is lifted to a constant") {
    const auto weights = Tensor<float, Runtime::CPU, 2>(3, 4);
    STATIC_REQUIRE(
        std::same_as<join_t<decltype(a), decltype(weights)>, Recorder>);

    auto c = mm(a, weights);
    REQUIRE(c.shape() == Shape(2, 4));
    REQUIRE(log == Names{"const", "mm"});
  }

  SECTION("Composite ops fall back to their parts") {
    const auto v = input<float>(log, Shape(3));
    auto d = dot(v, v);
    STATIC_REQUIRE(std::same_as<decltype(d), Tensor<float, Recorder, 0>>);
    REQUIRE(log == Names{"inner"});

    log.clear();
    auto s = sum_dims<0, 1>(a);
    REQUIRE(s.shape() == Shape(1, 1));
    REQUIRE(log == Names{"sum_dim", "sum_dim"});

    log.clear();
    const auto b = input<float>(log, Shape(3, 4));
    auto e = einsum<"ij,jk->ik">(a, b);
    REQUIRE(e.shape() == Shape(2, 4));
    REQUIRE(log == Names{"einsum_operand", "einsum_operand", "mul", "sum_dim",
                         "squeeze"});
  }

  SECTION("Values and eager ops are not available") {
    using Scalar = Tensor<float, Recorder, 0>;
    using Matrix = Tensor<float, Recorder, 2>;
    STATIC_REQUIRE(!HasValue<Scalar>);
    STATIC_REQUIRE(!Indexable<Matrix>);
    STATIC_REQUIRE(!Iterable<Matrix>);
    STATIC_REQUIRE(!EagerAddable<Matrix>);
    STATIC_REQUIRE(EagerAddable<Tensor<float, Runtime::CPU, 2>>);
  }
}

TEST_CASE("Unqualified calls with both namespaces in scope",
          "[tensor][dispatch]") {
  using namespace venus::eager;
  const auto ints = Tensor<int, Runtime::CPU, 1>{5, 2, 4, 3, 1};

  auto changed = where(sort(ints) != ints);
  REQUIRE(equal(changed, Tensor<std::size_t, Runtime::CPU, 1>{0, 2, 3, 4}));
  REQUIRE(equal(add(ints, 1), ints + 1));
  REQUIRE(dot(ints, ints) == 55);

  auto square = Tensor<int, Runtime::CPU, 2>(2, 2);
  square.iota(1);
  REQUIRE(equal(mm(square, eye_like(square)), square));
  REQUIRE(equal(where(square > 2, square, 0), square * (square > 2)));
  REQUIRE(sum_dims<0, 1>(square).asScalar() == 10);
  REQUIRE(einsum<"ij,ij->">(square, square) == 30);
}
