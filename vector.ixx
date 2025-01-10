module;
#include <tuple>

export module vector;

export consteval auto Vector(auto x, auto y) {
  auto getX = [x] consteval { return x; };
  auto getY = [y] consteval { return y; };

  auto add = [x, y](auto other) consteval {
    const auto [otherX, otherY, _] = other;
    return Vector(x + otherX(), y + otherY());
  };

  return std::make_tuple(getX, getY, add);
}
