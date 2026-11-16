import venus;

auto main() -> int {
  auto scalar = Tensor<int, Device::CPU, 0>(10);
  auto float_scalar = Tensor<float, Device::CPU, 0>(10.0f);
  auto double_scalar = Tensor<double, Device::CPU, 0>(10.0);
  auto tensor = Tensor<int, Device::CPU, 3>(1, 2, 3);
  auto float_tensor = Tensor<float, Device::CPU, 3>(1, 2, 3);
  auto ch_tensor = Tensor<char, Device::CPU, 3>(1, 2, 3);
  auto str_tensor = Tensor<std::string, Device::CPU, 3>(1, 2, 3);
  auto range_tensor = Tensor<float, Device::CPU, 1>(10);
  auto big_tensor = Tensor<int, Device::CPU, 2>(1000, 1000);
  auto big_float_tensor = Tensor<float, Device::CPU, 1>(2000);
  range_tensor.iota(0.3f);
  big_tensor.iota(0);
  big_float_tensor.iota(0);

  // clang-format off
  std::println("{}", scalar);               // venus::Tensor(10)
  std::println("{}", float_scalar);         // venus::Tensor(10.00)
  std::println("{}", double_scalar);        // venus::Tensor(10.00)
  std::println("{}", tensor);               // venus::Tensor([0, 0, 0, 0, 0, 0], shape=(1, 2, 3))
  std::println("{}", float_tensor);         // venus::Tensor([0.00, 0.00, 0.00, 0.00, 0.00, 0.00], shape=(1, 2, 3))
  std::println("{}", ch_tensor);            // venus::Tensor(['', '', '', '', '', ''], shape=(1, 2, 3))
  std::println("{}", str_tensor);           // venus::Tensor(["", "", "", "", "", ""], shape=(1, 2, 3))
  std::println("{}", range_tensor);         // venus::Tensor([0.30, 1.30, 2.30, 3.30, 4.30, 5.30, 6.30, 7.30, 8.30, 9.30], shape=(10))

  // Past 1000 elements, only the first and last three
  std::println("{}", big_tensor);           // venus::Tensor([0, 1, 2, ..., 999997, 999998, 999999], shape=(1000, 1000))
  std::println("{}", big_float_tensor);     // venus::Tensor([0.00, 1.00, 2.00, ..., 1997.00, 1998.00, 1999.00], shape=(2000))

  // A format spec changes one print: t<threshold>, e<edgeItems>, .<precision>
  std::println("{:e1}", big_tensor);        // venus::Tensor([0, ..., 999999], shape=(1000, 1000))
  std::println("{:e5}", big_tensor);        // venus::Tensor([0, 1, 2, 3, 4, ..., 999995, 999996, 999997, 999998, 999999], shape=(1000, 1000))
  std::println("{:t5}", range_tensor);      // venus::Tensor([0.30, 1.30, 2.30, ..., 7.30, 8.30, 9.30], shape=(10))
  std::println("{:t5e1.0}", range_tensor);  // venus::Tensor([0, ..., 9], shape=(10))
  std::println("{:.0}", float_scalar);      // venus::Tensor(10)
  std::println("{:e1t5.8}", range_tensor);  // venus::Tensor([0.30000001, ..., 9.30000019], shape=(10))

  // printOptions changes every print after it, and a format spec still wins
  printOptions.precision = 4;
  printOptions.edgeItems = 2;
  std::println("{}", float_scalar);         // venus::Tensor(10.0000)
  std::println("{}", big_float_tensor);     // venus::Tensor([0.0000, 1.0000, ..., 1998.0000, 1999.0000], shape=(2000))
  printOptions.threshold = 5;
  std::println("{}", range_tensor);         // venus::Tensor([0.3000, 1.3000, ..., 8.3000, 9.3000], shape=(10))
  std::println("{:.1}", range_tensor);      // venus::Tensor([0.3, 1.3, ..., 8.3, 9.3], shape=(10))

  printOptions = {};                        // threshold 1000, edgeItems 3, precision 2
  std::println("{}", range_tensor);         // venus::Tensor([0.30, 1.30, 2.30, 3.30, 4.30, 5.30, 6.30, 7.30, 8.30, 9.30], shape=(10))
}
