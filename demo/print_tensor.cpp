import venus;

auto main() -> int {
  auto scalar = Tensor<int, Device::CPU, 0>(10);
  auto float_scalar = Tensor<float, Device::CPU, 0>(10.0f);
  auto double_scalar = Tensor<double, Device::CPU, 0>(10.0);
  auto tensor = Tensor<int, Device::CPU, 3>(1, 2, 3);
  auto float_tensor = Tensor<float, Device::CPU, 3>(1, 2, 3);
  auto ch_tensor = Tensor<char, Device::CPU, 3>(1, 2, 3);
  auto str_tensor = Tensor<std::string, Device::CPU, 3>(1, 2, 3);

  // clang-format off
  std::println("{}", scalar);            // venus::Tensor(10)
  std::println("{}", float_scalar);      // venus::Tensor(10.00)
  std::println("{}", double_scalar);     // venus::Tensor(10.00)
  std::println("{}", tensor);            // venus::Tensor([0, 0, 0, 0, 0, 0], shape=(1, 2, 3))
  std::println("{}", float_tensor);      // venus::Tensor([0.00, 0.00, 0.00, 0.00, 0.00, 0.00], shape=(1, 2, 3))
  std::println("{}", ch_tensor);         // venus::Tensor(['', '', '', '', '', ''], shape=(1, 2, 3))
  std::println("{}", str_tensor);        // venus::Tensor(["", "", "", "", "", ""], shape=(1, 2, 3))

  auto big_tensor = Tensor<int, Device::CPU, 2>(1000, 1000);
  big_tensor.iota(0);
  std::println("{}", big_tensor);        // venus::Tensor([0, 1, 2, ..., 999997, 999998, 999999], shape=(1000, 1000))

  auto big_float_tensor = Tensor<float, Device::CPU, 1>(2000);
  big_float_tensor.iota(0);
  std::println("{}", big_float_tensor);  // venus::Tensor([0.00, 1.00, 2.00, ..., 1997.00, 1998.00, 1999.00], shape=(2000))
}