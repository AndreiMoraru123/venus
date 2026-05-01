#include <print>
#include <venus/memory/device.hpp>
#include <venus/tensor/tensor.hpp>

using namespace venus;

template <typename T, std::size_t Rank>
  requires(Rank == 2)
auto conv2d(const Tensor<T, Device::CPU, Rank> &input,
            const Tensor<T, Device::CPU, Rank> &kernel, std::size_t stride = 1)
    -> Tensor<T, Device::CPU, Rank> {
  const auto [w, h] = input.shape();
  const auto [kW, kH] = kernel.shape();

  const auto outW = (w - kW) / stride + 1;
  const auto outH = (h - kH) / stride + 1;

  auto out = Tensor<T, Device::CPU, Rank>(outW, outH);

  for (std::size_t i{}; i < outW; ++i) {
    for (std::size_t j{}; j < outH; ++j) {
      for (std::size_t k1{}; k1 < kW; ++k1) {
        for (std::size_t k2{}; k2 < kH; ++k2) {
          out[i, j] += kernel[kW - 1 - k1, kH - 1 - k2] *
                       input[i * stride + k1, j * stride + k2];
        }
      }
    }
  }

  return out;
}

auto main() -> int {
  auto img = Tensor<float, Device::CPU, 2>(5, 5);
  img.fill(1.0F);

  auto kernel =
      Tensor<float, Device::CPU, 2>{{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};

  auto res = conv2d(img, kernel);
  assert(res.shape() == Shape(3, 3));

  std::println("{}", res);
}
