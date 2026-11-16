import venus;

// clang-format off
auto main() -> int {
  {
    auto tensor = Tensor<float, Runtime::CPU, 2>(2, 3);
    std::println("{}", tensor.unique()); // true

    std::ranges::fill(tensor, 10.0f);
    std::println("{}", tensor); // venus::Tensor([10.00, 10.00, 10.00, 10.00, 10.00, 10.00], shape=(2, 3))

    auto tensor2 = Tensor<float, Runtime::CPU, 2>(2, 3);
    std::println("{}", tensor2.unique()); // true

    std::println("{}", tensor2); // venus::Tensor([0.00, 0.00, 0.00, 0.00, 0.00, 0.00], shape=(2, 3))
  }
  {
    const auto memo = ContiguousMemory<float, Runtime::CPU>(6);

    auto tensor = Tensor<float, Runtime::CPU, 2>(memo, Shape(2, 3));
    std::println("{}", tensor.unique()); // false

    std::ranges::fill(tensor, 5.0f);
    std::println("{}", tensor); // venus::Tensor([5.00, 5.00, 5.00, 5.00, 5.00, 5.00], shape=(2, 3))

    auto tensor2 = Tensor<float, Runtime::CPU, 2>(memo, Shape(2, 3));
    std::println("{}", tensor2.unique()); // false

    std::println("{}", tensor2); // venus::Tensor([5.00, 5.00, 5.00, 5.00, 5.00, 5.00], shape=(2, 3))
  }
}