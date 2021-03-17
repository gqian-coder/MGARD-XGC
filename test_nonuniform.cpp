#include <cstddef>
#include <cstdlib>

#include <array>
#include <vector>
#include <iostream>

#include "mgard/mgard_api.h"

int main() {
  const std::vector<float> coords_x = {0, 1, 2, 2.5, 4};
  const std::vector<float> coords_y = {-4.5, -4.4, -4.3, -4.2, -3.2};
  const std::array<std::vector<float>, 2> coords = {coords_x, coords_y};
  const mgard::TensorMeshHierarchy<2, float> hierarchy({5, 5}, coords);

  float * const u = static_cast<float *>(std::malloc(25 * sizeof(float)));
  {
    float * p = u;
    for (std::size_t i = 0; i < 5; ++i) {
        const float x = coords_x.at(i);
        for (std::size_t j = 0; j < 5; ++j) {
          const float y = coords_y.at(j);
          *p++ = 1.2 * x * x - 2.1 * x * y + 0.4 * y * y;
        }
    }
  }

  const float s = 0;
  const float tolerance = 0.1;
  const mgard::CompressedDataset<2, float> compressed =
    mgard::compress(hierarchy, u, s, tolerance);

  std::cout << "compression ratio is " <<
    static_cast<float>(25 * sizeof(float)) / compressed.size() << std::endl;

  std::free(u);
  return 0;
}
