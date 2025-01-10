import vector;
#include <print>

auto main() -> int {
  constexpr auto vector1 = Vector(1, 2);
  constexpr auto vector2 = Vector(2.5, 3.5);

  constexpr auto v1Add = std::get<2>(vector1);

  constexpr auto vector3 = v1Add(vector2);
  constexpr auto X3 = std::get<0>(vector3);
  constexpr auto Y3 = std::get<1>(vector3);
  std::println("{}", X3());  // 3
  std::println("{}", Y3());  // 6
}
