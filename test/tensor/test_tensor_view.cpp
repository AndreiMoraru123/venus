#include <catch2/catch_test_macros.hpp>

#include <format>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#ifdef VENUS_IMPORT
import venus;
#else
#include <venus/memory/device.hpp>
#include <venus/tensor/tensor.hpp>
#include <venus/tensor/tensor_format.hpp>
#endif

using namespace venus;

namespace {
template <typename R> auto collect(R &&range) {
  return std::ranges::to<std::vector>(std::forward<R>(range));
}

template <typename T, typename... Selectors>
concept Sliceable =
    requires(Selectors... selectors) { std::declval<T>()[selectors...]; };

template <typename S, typename T>
concept Assignable = requires(const T &value) { std::declval<S>() = value; };

// The view starts inside the tensor's memory, so making it allocated nothing
template <typename V, typename T>
auto inside(const V &view, const T &tensor) -> bool {
  const auto *first = std::as_const(tensor).data();
  return view.data() >= first && view.data() < first + tensor.size();
}

using Ints1 = Tensor<int, Device::CPU, 1>;
using Ints2 = Tensor<int, Device::CPU, 2>;
} // namespace

TEST_CASE("Tensor slicing", "[tensor][view][slice]") {
  auto tensor = Tensor<int, Device::CPU, 3>(2, 2, 3);
  tensor.iota(0); // [[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]]
  const auto &c = tensor;

  SECTION("An integer drops its dimension and a range keeps it") {
    STATIC_REQUIRE(
        std::same_as<decltype(c[0, 1, all]), TensorView<int, Device::CPU, 1>>);
    STATIC_REQUIRE(std::same_as<decltype(c[range(0, 2), all, 1]),
                                TensorView<int, Device::CPU, 2>>);
    STATIC_REQUIRE(std::same_as<decltype(c[all, all, all]),
                                TensorView<int, Device::CPU, 3>>);
  }

  SECTION("A non-const tensor gives a selection") {
    STATIC_REQUIRE(std::same_as<decltype(tensor[0, 1, all]),
                                Selection<int, Device::CPU, 1>>);
  }

  SECTION("Last axis") {
    REQUIRE(collect(c[0, 1, all]) == std::vector{3, 4, 5});
    REQUIRE(collect(c[1, 0, all]) == std::vector{6, 7, 8});
  }

  SECTION("Middle axis") {
    REQUIRE(collect(c[0, all, 2]) == std::vector{2, 5});
  }

  SECTION("First axis") {
    REQUIRE(collect(c[all, 1, 2]) == std::vector{5, 11});
  }

  SECTION("Ranges") {
    auto view = c[range(0, 2), 1, range(1, 3)];
    REQUIRE(view.shape() == Shape(2, 2));
    REQUIRE(collect(view) == std::vector{4, 5, 10, 11});
  }

  SECTION("Slicing allocates nothing") {
    auto view = c[1, range(0, 2), all];
    REQUIRE(inside(view, tensor));
    REQUIRE(view.data() == c.data() + 6);
    REQUIRE_FALSE(tensor.unique());
  }

  SECTION("Const tensor") {
    auto column = c[1, all, 1];
    STATIC_REQUIRE(
        std::same_as<std::ranges::range_reference_t<decltype(column)>,
                     const int &>);
    REQUIRE(collect(column) == std::vector{7, 10});
  }

  SECTION("Out of bounds selectors throw") {
    REQUIRE_THROWS_AS((c[2, 0, all]), std::out_of_range);
    REQUIRE_THROWS_AS((c[0, range(0, 3), all]), std::out_of_range);
    REQUIRE_THROWS_AS((c[0, range(1, 1), all]), std::invalid_argument);
  }

  SECTION("Element indexing still works") { REQUIRE(tensor[1, 1, 2] == 11); }

  SECTION("Element indexing on a view") {
    auto view = c[1, all, all];
    REQUIRE(view[1, 2] == 11);
    REQUIRE(view[std::array<std::size_t, 2>{0, 1}] == 7);
    REQUIRE_THROWS_AS((view[2, 0]), std::out_of_range);
  }

  SECTION("Accepted and rejected selectors") {
    using T = Tensor<int, Device::CPU, 3>;
    STATIC_REQUIRE(Sliceable<T &, int, int, all_t>);
    STATIC_REQUIRE(Sliceable<T &, int, all_t, all_t>);
    STATIC_REQUIRE(Sliceable<const T &, Range, int, int>);
    STATIC_REQUIRE(Sliceable<T, int, int, all_t>);
    STATIC_REQUIRE_FALSE(Sliceable<T &, int, all_t>);
    STATIC_REQUIRE_FALSE(Sliceable<T &, int, all_t, all_t, all_t>);
  }

  SECTION("Slicing a temporary gives a view that keeps the memory alive") {
    auto row = Ints2{{1, 2}, {3, 4}}[1, all];
    STATIC_REQUIRE(
        std::same_as<decltype(row), TensorView<int, Device::CPU, 1>>);
    REQUIRE(row.unique());
    REQUIRE(collect(row) == std::vector{3, 4});
  }

  SECTION("Slicing a view") {
    auto rows = c[all, 1, all];
    REQUIRE(collect(rows[1, all]) == std::vector{9, 10, 11});
  }
}

TEST_CASE("String tensor slicing", "[tensor][view][slice]") {
  auto tensor = Tensor<std::string, Device::CPU, 3>(1, 2, 3);
  std::ranges::fill(tensor, "x");
  tensor[0, 1, 2] = "y";

  auto row = std::as_const(tensor)[0, 1, all];
  REQUIRE(std::ranges::size(row) == 3);
  REQUIRE(collect(row) == std::vector<std::string>{"x", "x", "y"});
}

TEST_CASE("Rank 1 tensor slicing", "[tensor][view][slice]") {
  auto tensor = Ints1(4);
  tensor.iota(1);
  REQUIRE(collect(std::as_const(tensor)[all]) == std::vector{1, 2, 3, 4});
}

TEST_CASE("View ops", "[tensor][view]") {
  auto tensor = Ints2{{0, 1, 2}, {3, 4, 5}};
  const auto &c = tensor;

  SECTION("view() shares the memory with the same shape") {
    auto view = c.view();
    STATIC_REQUIRE(
        std::same_as<decltype(view), TensorView<int, Device::CPU, 2>>);
    REQUIRE(view.shape() == Shape(2, 3));
    REQUIRE(view.data() == c.data());
    REQUIRE(view.isContiguous());
    REQUIRE_FALSE(tensor.unique());
  }

  SECTION("view() on a scalar") {
    const auto scalar = Tensor<float, Device::CPU, 0>(2.0f);
    auto view = scalar.view();
    STATIC_REQUIRE(
        std::same_as<decltype(view), TensorView<float, Device::CPU, 0>>);
    REQUIRE(view.value() == 2.0f);
    REQUIRE(view.data() == scalar.data());
  }

  SECTION("permute") {
    auto view = c.permute(1, 0);
    REQUIRE(view.shape() == Shape(3, 2));
    REQUIRE(collect(view) == std::vector{0, 3, 1, 4, 2, 5});
    REQUIRE(view[2, 1] == 5);
    REQUIRE(view.data() == c.data());
    REQUIRE_FALSE(view.isContiguous());
    REQUIRE_THROWS_AS(c.permute(0, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(c.permute(0, 2), std::invalid_argument);
  }

  SECTION("expand") {
    const auto row = Ints1{1, 2, 3};
    auto rows = row.expand(2, 3);
    REQUIRE(rows.shape() == Shape(2, 3));
    REQUIRE(rows.strides() == std::array<std::size_t, 2>{0, 1});
    REQUIRE(collect(rows) == std::vector{1, 2, 3, 1, 2, 3});
    REQUIRE(rows.data() == row.data());
    REQUIRE_FALSE(rows.isContiguous());

    const auto column = Ints2{{1}, {2}};
    REQUIRE(collect(column.expand(Shape(2, 3))) ==
            std::vector{1, 1, 1, 2, 2, 2});
    REQUIRE_THROWS_AS(row.expand(2, 4), std::invalid_argument);
  }

  SECTION("squeeze and unsqueeze") {
    auto wide = c.unsqueeze<1>();
    REQUIRE(wide.shape() == Shape(2, 1, 3));
    REQUIRE(wide.isContiguous());
    REQUIRE(collect(wide) == collect(c));
    REQUIRE(wide.data() == c.data());

    auto narrow = wide.squeeze<1>();
    REQUIRE(narrow.shape() == Shape(2, 3));
    REQUIRE(collect(narrow) == collect(c));
    REQUIRE_THROWS_AS(c.squeeze<0>(), std::invalid_argument);

    auto scalar = c[0, range(1, 2)].squeeze<0>();
    STATIC_REQUIRE(
        std::same_as<decltype(scalar), TensorView<int, Device::CPU, 0>>);
    REQUIRE(scalar.value() == 1);
  }

  SECTION("Copying a view shares the memory") {
    auto view = c.permute(1, 0);
    auto copy = view;
    REQUIRE(copy.data() == view.data());
    REQUIRE(collect(copy) == collect(view));
  }

  SECTION("clone() copies a view into a tensor") {
    auto copy = c.permute(1, 0).clone();
    STATIC_REQUIRE(std::same_as<decltype(copy), Ints2>);
    REQUIRE(copy.unique());
    REQUIRE(copy.data() != c.data());
    REQUIRE(collect(copy) == std::vector{0, 3, 1, 4, 2, 5});
  }

  SECTION("A view keeps the memory alive after its tensor") {
    auto view = Ints2{{7, 8}, {9, 10}}.permute(1, 0);
    REQUIRE(view.unique());
    REQUIRE(collect(view) == std::vector{7, 9, 8, 10});
  }
}

TEST_CASE("Reshape", "[tensor][view][reshape]") {
  auto tensor = Ints2{{0, 1, 2}, {3, 4, 5}};

  SECTION("On a named tensor it returns a view") {
    auto view = tensor.reshape(3, 2);
    STATIC_REQUIRE(
        std::same_as<decltype(view), TensorView<int, Device::CPU, 2>>);
    REQUIRE(view.data() == tensor.data());
    REQUIRE(view[2, 1] == 5);
    REQUIRE_FALSE(tensor.unique());
  }

  SECTION("On a moved tensor it takes over the memory") {
    const auto *memory = tensor.data();
    auto flat = std::move(tensor).reshape(6);
    STATIC_REQUIRE(std::same_as<decltype(flat), Ints1>);
    REQUIRE(flat.data() == memory);
    REQUIRE(flat.unique());
    REQUIRE(collect(flat) == std::vector{0, 1, 2, 3, 4, 5});
  }

  SECTION("On a temporary it returns a tensor") {
    auto flat = Ints2{{1, 2}, {3, 4}}.reshape(Shape(4));
    STATIC_REQUIRE(std::same_as<decltype(flat), Ints1>);
    REQUIRE(flat.unique());
  }

  SECTION("A contiguous view reshapes without a copy") {
    const auto &c = tensor;
    auto rows = c[range(0, 2), all].reshape(6);
    REQUIRE(rows.data() == c.data());
    REQUIRE(collect(rows) == std::vector{0, 1, 2, 3, 4, 5});
  }

  SECTION("Any other view is copied first") {
    auto flat = std::as_const(tensor).permute(1, 0).reshape(6);
    STATIC_REQUIRE(
        std::same_as<decltype(flat), TensorView<int, Device::CPU, 1>>);
    REQUIRE(flat.data() != tensor.data());
    REQUIRE(flat.unique());
    REQUIRE(collect(flat) == std::vector{0, 3, 1, 4, 2, 5});
  }

  SECTION("A size mismatch throws") {
    REQUIRE_THROWS_AS(tensor.reshape(4), std::invalid_argument);
    REQUIRE_THROWS_AS(tensor.view().reshape(4), std::invalid_argument);
  }
}

TEST_CASE("Writes while a view is alive", "[tensor][view][ownership]") {
  auto tensor = Ints2(2, 2);
  auto view = std::as_const(tensor).view();

  SECTION("fill throws") {
    REQUIRE_THROWS_AS(tensor.fill(1), std::runtime_error);
  }
  SECTION("iota throws") {
    REQUIRE_THROWS_AS(tensor.iota(1), std::runtime_error);
  }
  SECTION("transform throws") {
    REQUIRE_THROWS_AS(tensor.transform([](int x) { return x + 1; }),
                      std::runtime_error);
  }
  SECTION("sort throws") {
    REQUIRE_THROWS_AS(tensor.sort(), std::runtime_error);
  }
  SECTION("eye throws") { REQUIRE_THROWS_AS(tensor.eye(), std::runtime_error); }
  SECTION("An element write throws") {
    REQUIRE_THROWS_AS((tensor[0, 0] = 1), std::runtime_error);
  }

  SECTION("A held selection counts as a view") {
    auto row = tensor[0, all];
    REQUIRE_THROWS_AS(tensor.fill(1), std::runtime_error);
  }

  REQUIRE(collect(view) == std::vector{0, 0, 0, 0});
}

TEST_CASE("Writes once the views are gone", "[tensor][view][ownership]") {
  auto tensor = Ints2(2, 2);
  {
    auto view = tensor.view();
  }

  REQUIRE(tensor.unique());
  tensor.fill(1);
  tensor.iota(1);
  tensor.transform([](int x) { return x * 2; });
  tensor.sort();
  REQUIRE(collect(tensor) == std::vector{2, 4, 6, 8});
  tensor.eye();
  REQUIRE(collect(tensor) == std::vector{1, 0, 0, 1});
}

TEST_CASE("Selection assignment", "[tensor][view][selection]") {
  auto cache = Ints2(3, 2);
  const auto *memory = std::as_const(cache).data();

  SECTION("A row is written in place") {
    cache[1, all] = Ints1{1, 2};
    REQUIRE(std::as_const(cache).data() == memory);
    REQUIRE(cache.unique());
    REQUIRE(collect(cache) == std::vector{0, 0, 1, 2, 0, 0});
  }

  SECTION("A scalar fills the selection") {
    cache[all, 0] = 7;
    REQUIRE(collect(cache) == std::vector{7, 0, 7, 0, 7, 0});
  }

  SECTION("The right side broadcasts") {
    cache[range(0, 2), all] = Ints1{5, 6};
    REQUIRE(collect(cache) == std::vector{5, 6, 5, 6, 0, 0});
  }

  SECTION("The right side may be a view of another tensor") {
    const auto other = Ints2{{1, 2}, {3, 4}};
    cache[2, all] = other[all, 1];
    REQUIRE(collect(cache) == std::vector{0, 0, 0, 0, 2, 4});
  }

  SECTION("A shape mismatch throws before writing") {
    REQUIRE_THROWS_AS((cache[0, all] = Ints1{1, 2, 3}), std::invalid_argument);
    REQUIRE(collect(cache) == std::vector{0, 0, 0, 0, 0, 0});
  }

  SECTION("It throws while a view is alive") {
    auto view = std::as_const(cache).view();
    REQUIRE_THROWS_AS((cache[0, all] = Ints1{1, 2}), std::runtime_error);
    REQUIRE_THROWS_AS((cache[0, all] = 1), std::runtime_error);
    REQUIRE(collect(view) == std::vector{0, 0, 0, 0, 0, 0});
  }

  SECTION("A selection of the same tensor on the right is a live view") {
    cache[0, all] = Ints1{1, 2};
    REQUIRE_THROWS_AS((cache[1, all] = cache[0, all]), std::runtime_error);

    const auto row = std::as_const(cache)[0, all].clone();
    cache[1, all] = row;
    REQUIRE(collect(cache) == std::vector{1, 2, 1, 2, 0, 0});
  }

  SECTION("A copy of a selection counts as a view") {
    auto row = cache[0, all];
    const TensorView<int, Device::CPU, 1> copy = row;
    REQUIRE_THROWS_AS((std::move(row) = 1), std::runtime_error);
    REQUIRE(collect(copy) == std::vector{0, 0});
  }

  SECTION("Only a temporary selection can be assigned to") {
    using Sel = decltype(cache[0, all]);
    STATIC_REQUIRE(Assignable<Sel, Ints1>);
    STATIC_REQUIRE(Assignable<Sel, int>);
    STATIC_REQUIRE(Assignable<Sel, Sel>);
    STATIC_REQUIRE_FALSE(Assignable<Sel &, Ints1>);
    STATIC_REQUIRE_FALSE(Assignable<Sel &, int>);
    STATIC_REQUIRE_FALSE(Assignable<Sel &, Sel>);
    STATIC_REQUIRE_FALSE(Assignable<TensorView<int, Device::CPU, 1>, Ints1>);
    STATIC_REQUIRE_FALSE(Assignable<TensorView<int, Device::CPU, 1>, int>);
    STATIC_REQUIRE_FALSE(Assignable<decltype(Ints2(3, 2)[0, all]), int>);
  }
}

TEST_CASE("KV cache", "[tensor][view][selection]") {
  constexpr std::size_t steps = 4;
  constexpr std::size_t dim = 3;
  auto cache = Tensor<float, Device::CPU, 2>(steps, dim);
  const auto *memory = std::as_const(cache).data();

  for (std::size_t t = 0; t < steps; ++t) {
    auto row = Tensor<float, Device::CPU, 1>(dim);
    row.iota(static_cast<float>(t * dim));
    cache[t, all] = row;
    REQUIRE(std::as_const(cache).data() == memory);

    const auto seen = cache[range(0, t + 1), all];
    REQUIRE(seen.shape() == Shape(t + 1, dim));
    REQUIRE(seen.isContiguous());
    REQUIRE(seen.data() == memory);

    const auto expected = std::views::iota(std::size_t{0}, (t + 1) * dim) |
                          std::views::transform([](std::size_t i) {
                            return static_cast<float>(i);
                          });
    REQUIRE(std::ranges::equal(seen, expected));
  }
}

TEST_CASE("Ops on views return tensors", "[tensor][view][ops]") {
  const auto a = Ints2{{1, 2, 3}, {4, 5, 6}};
  const auto at = a.permute(1, 0); // [[1, 4], [2, 5], [3, 6]]
  const auto at_copy = at.clone();

  SECTION("Elementwise ops read strides") {
    auto sum = at + at_copy;
    STATIC_REQUIRE(std::same_as<decltype(sum), Ints2>);
    REQUIRE(sum.unique());
    REQUIRE(eager::equal(sum, Ints2{{2, 8}, {4, 10}, {6, 12}}));
    REQUIRE(eager::equal(at * 2, at_copy * 2));
    REQUIRE(eager::equal(2 * at, at_copy * 2));
    REQUIRE(eager::equal(at > 2, at_copy > 2));
    REQUIRE(eager::equal(at - at, Ints2(3, 2)));
  }

  SECTION("Broadcasting with a view") {
    auto column = a[all, 0].unsqueeze<1>(); // [[1], [4]]
    REQUIRE(eager::equal(a + column, Ints2{{2, 3, 4}, {8, 9, 10}}));
    REQUIRE(eager::equal(column + a, Ints2{{2, 3, 4}, {8, 9, 10}}));
  }

  SECTION("mm reads a transpose without a copy and copies other layouts") {
    REQUIRE(eager::equal(eager::mm(at, a), eager::mm(at_copy, a)));
    REQUIRE(eager::equal(eager::mm(a, at), eager::mm(a, at_copy)));
    auto left = a[all, range(0, 2)]; // [[1, 2], [4, 5]], not contiguous
    REQUIRE_FALSE(left.isContiguous());
    REQUIRE(eager::equal(eager::mm(left, a), Ints2{{9, 12, 15}, {24, 33, 42}}));
  }

  SECTION("Reductions and sorting") {
    REQUIRE(eager::equal(eager::sum_dim<0>(at), Ints2{{6, 15}}));
    REQUIRE(eager::equal(eager::sum_dims<0, 1>(at), Ints2{{21}}));
    auto sorted = eager::sort(a[all, 1]);
    STATIC_REQUIRE(std::same_as<decltype(sorted), Ints1>);
    REQUIRE(collect(sorted) == std::vector{2, 5});
    REQUIRE(eager::dot(a[0, all], a[1, all]).value() == 32);
    REQUIRE(eager::dot(a[all, 0], a[all, 1]).value() == 22);
  }

  SECTION("transform, where and einsum") {
    auto doubled = eager::transform(at, [](int x) { return x * 2; });
    STATIC_REQUIRE(std::same_as<decltype(doubled), Ints2>);
    REQUIRE(eager::equal(doubled, at_copy * 2));
    REQUIRE(eager::equal(eager::where(at > 3), eager::where(at_copy > 3)));
    const auto mask = Ints2{{0, 1, 0}, {1, 0, 1}}.permute(1, 0);
    REQUIRE(eager::equal(eager::nonzero(mask), eager::nonzero(mask.clone())));
    REQUIRE(eager::equal(eager::nonzero_flat(mask[all, 1]),
                         Tensor<std::size_t, Device::CPU, 1>{0, 2}));
    auto product = eager::einsum<"ij,jk->ik">(at, a);
    STATIC_REQUIRE(std::same_as<decltype(product), Ints2>);
    REQUIRE(eager::equal(product, eager::mm(at_copy, a)));
  }
}

TEST_CASE("Tensor view printing", "[tensor][view][print]") {
  auto a = Ints2{{1, 2, 3}, {4, 5, 6}};
  CHECK(std::format("{}", std::as_const(a).permute(1, 0)) ==
        "venus::TensorView([1, 4, 2, 5, 3, 6], shape=(3, 2))");
  CHECK(std::format("{}", a[0, all]) ==
        "venus::TensorView([1, 2, 3], shape=(3))");
}
