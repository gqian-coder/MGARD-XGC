#include <cmath>
#include <cstddef>

#include <array>
#include <iostream>
#include <stdexcept>

#include "mgard/TensorQuantityOfInterest.hpp"
#include "mgard/mgard_api.h"

class AverageFunctional {
public:
  AverageFunctional(const std::array<std::size_t, 2> lower_left,
                    const std::array<std::size_t, 2> upper_right)
      : lower_left(lower_left), upper_right(upper_right) {
    for (std::size_t i = 0; i < 2; ++i) {
      if (upper_right.at(i) <= lower_left.at(i)) {
        throw std::invalid_argument("invalid region");
      }
    }
  }

  float operator()(const mgard::TensorMeshHierarchy<2, float> &hierarchy,
                   float const *const u) const {
    const std::array<std::size_t, 2> shape = hierarchy.shapes.back();
    const std::size_t n = shape.at(0);
    const std::size_t m = shape.at(1);
    if (upper_right.at(0) > n || upper_right.at(1) > m) {
      throw std::invalid_argument("region isn't contained in domain");
    }
    float total = 0;
    std::size_t count = 0;
    for (std::size_t i = lower_left.at(0); i < upper_right.at(0); ++i) {
      for (std::size_t j = lower_left.at(1); j < upper_right.at(1); ++j) {
        total += u[n * i + j];
        ++count;
      }
    }
    return total / count;
  }

private:
  std::array<std::size_t, 2> lower_left;
  std::array<std::size_t, 2> upper_right;
};

int main() {
  const AverageFunctional average({10, 15}, {20, 35});
  const mgard::TensorMeshHierarchy<2, float> hierarchy({50, 60});

  const mgard::TensorQuantityOfInterest<2, float> Q(hierarchy, average);
  std::cout << "norm of the average as a functional on L^2: " << Q.norm(0)
            << std::endl;

  float *const u =
      static_cast<float *>(std::malloc(hierarchy.ndof() * sizeof(*u)));
  {
    float *p = u;
    for (std::size_t i = 0; i < 50; ++i) {
      const float x = static_cast<float>(i) / 50;
      for (std::size_t j = 0; j < 60; ++j) {
        const float y = 2.5 + static_cast<float>(j) / 60;
        *p++ = 12 + std::sin(2.1 * x - 1.3 * y);
      }
    }
  }

  std::cout << "average using original data: " << average(hierarchy, u)
            << std::endl;

  const float s = 0;
  const float tolerance = 0.1;
  const mgard::CompressedDataset<2, float> compressed =
      mgard::compress(hierarchy, u, s, tolerance);
  const mgard::DecompressedDataset<2, float> decompressed =
      mgard::decompress(compressed);

  std::cout << "average using decompressed data: "
            << average(hierarchy, decompressed.data()) << std::endl;

  std::free(u);
  return 0;
}
