#include <catch2/catch_test_macros.hpp>
#include <string>
import venus;

// Instantiated in format_tensors.cpp, away from Catch2's <ostream>
template <typename T> auto toText(const T &tensor) -> std::string;

TEST_CASE("Tensor prints through import venus", "[module][print]") {
  CHECK(toText(Tensor<int, Device::CPU, 0>(10)) == "venus::Tensor(10)");
  CHECK(toText(Tensor<int, Device::CPU, 2>{{1, 2}, {3, 4}}) ==
        "venus::Tensor([1, 2, 3, 4], shape=(2, 2))");
  CHECK(toText(Tensor<float, Device::CPU, 1>{1.0f, 2.5f}) ==
        "venus::Tensor([1.00, 2.50], shape=(2))");
  CHECK(toText(Tensor<char, Device::CPU, 1>{'a', 'b'}) ==
        "venus::Tensor(['a', 'b'], shape=(2))");
  CHECK(toText(Tensor<std::string, Device::CPU, 1>{"hi", "hi"}) ==
        R"(venus::Tensor(["hi", "hi"], shape=(2)))");
}
