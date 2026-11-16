// <ostream> or <iostream> here hides the misprint test_print_tensor.cpp catches
#include <format>
#include <string>
import venus;

template <typename T> auto toText(const T &tensor) -> std::string {
  return std::format("{}", tensor);
}

template auto toText(const Tensor<int, Device::CPU, 0> &) -> std::string;
template auto toText(const Tensor<int, Device::CPU, 2> &) -> std::string;
template auto toText(const Tensor<float, Device::CPU, 1> &) -> std::string;
template auto toText(const Tensor<char, Device::CPU, 1> &) -> std::string;
template auto toText(const Tensor<std::string, Device::CPU, 1> &)
    -> std::string;
