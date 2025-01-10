#include <concepts>
#include <print>
#include <tuple>

template <typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

consteval auto Vector(Numeric auto x, Numeric auto y) {
  auto getX = [x] consteval { return x; };
  auto getY = [y] consteval { return y; };

  auto add = [x, y](auto other) consteval {
    const auto [otherX, otherY, _] = other;
    return Vector(x + otherX(), y + otherY());
  };

  return std::make_tuple(getX, getY, add);
}

auto main() -> int {
  constexpr auto vector1 = Vector(1, 2);
  constexpr auto vector2 = Vector(2.5, 4.5);

  constexpr auto v1Add = std::get<2>(vector1);

  constexpr auto vector3 = v1Add(vector2);
  constexpr auto X3 = std::get<0>(vector3);
  constexpr auto Y3 = std::get<1>(vector3);
  std::println("{}", X3()); // 3
  std::println("{}", Y3()); // 6
}
