// <ostream> or <iostream> here hides the misprint this test catches
#include <cstdio>
#include <format>
#include <print>
#include <string>
#include <string_view>
import venus;

using namespace venus;

namespace {
auto failures = 0;

void expect(const std::string &got, std::string_view want) {
  if (got != want) {
    std::println(stderr, "got  {}\nwant {}", got, want);
    ++failures;
  }
}
} // namespace

auto main() -> int {
  expect(std::format("{}", Tensor<int, Device::CPU, 0>(10)),
         "venus::Tensor(10)");
  expect(std::format("{}", Tensor<int, Device::CPU, 2>{{1, 2}, {3, 4}}),
         "venus::Tensor([1, 2, 3, 4], shape=(2, 2))");
  expect(std::format("{}", Tensor<float, Device::CPU, 1>{1.0f, 2.5f}),
         "venus::Tensor([1.00, 2.50], shape=(2))");
  expect(std::format("{}", Tensor<char, Device::CPU, 1>{'a', 'b'}),
         "venus::Tensor(['a', 'b'], shape=(2))");
  expect(std::format("{}", Tensor<std::string, Device::CPU, 1>{"hi", "hi"}),
         R"(venus::Tensor(["hi", "hi"], shape=(2)))");
  return failures;
}
