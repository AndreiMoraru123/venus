import venus;

auto main() -> int {
  auto tensor = Tensor<float, Device::CPU, 0>(10.0f);
  std::println("{}", tensor.value()); // 10

  tensor.assign(12.0f);

  auto good = (tensor == 12.0f);
  std::println("{}", good); // venus::Tensor(1)

  auto bad = (tensor != 12.0f);
  std::println("{}", bad); // venus::Tensor(0)

  static_assert(tensor.shape() == Shape<0>{});
}