import venus;

auto main() -> int {
  auto x = Tensor<float, Runtime::CPU, 1>(7);
  auto y = Tensor<float, Runtime::CPU, 1>(7);

  x.iota(1);
  y.iota(1);

  auto z = x.dot(y);
  std::println("{}", z); // venus::Tensor(140.00)

  std::println("{}", z == y.dot(x));         // true
  std::println("{}", z == eager::dot(x, y)); // true

  std::println("{}", eager::equal(x + y, eager::add(x, y))); // true
  std::println("{}", eager::equal(x - y, eager::sub(x, y))); // true
  std::println("{}", eager::equal(x * y, eager::mul(x, y))); // true
  std::println("{}", eager::equal(x / y, eager::div(x, y))); // true
}