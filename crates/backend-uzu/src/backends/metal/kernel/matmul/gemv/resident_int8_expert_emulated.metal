#include "../../common/dsl.h"
#include "../../common/defines.h"

using namespace metal;

// Portable numerical twin of the resident TensorOps projection. Each lane
// owns one output row and preserves the TensorOps K32 quantization boundary.
template <typename ST, typename BT, typename DT>
VARIANTS(ST, bfloat, float)
VARIANTS(BT, bfloat, float)
VARIANTS(DT, bfloat, float)
KERNEL(ResidentInt8ExpertEmulated)(
    const device int8_t* weight_codes,
    const device ST* weight_scales,
    const device int8_t* activation_codes,
    const device float* activation_scales,
    device DT* d,
    const device BT* expert_biases OPTIONAL(expert_bias),
    const device int* expert_ids,
    const constant uint& in_vec_size,
    const constant uint& out_vec_size,
    const constant uint& route_count,
    const constant uint& routes_per_token,
    const constant uint& expert_count,
    const constant bool& input_is_route_major,
    const constant uint& output_tile_count,
    const bool expert_bias SPECIALIZE,
    const uint route_index GROUPS(route_count),
    const uint output_tile_index GROUPS(output_tile_count),
    const uint simd_lane THREADS(METAL_SIMD_SIZE)
) {
  constexpr uint GROUP_SIZE = 32;
  const uint output_row = output_tile_index * METAL_SIMD_SIZE + simd_lane;
  if (output_row >= out_vec_size) {
    return;
  }

  const int expert = expert_ids[route_index];
  if (expert < 0 || uint(expert) >= expert_count) {
    d[size_t(route_index) * size_t(out_vec_size) + size_t(output_row)] = DT(0);
    return;
  }

  const uint matrix = uint(expert);
  const uint activation_row = input_is_route_major ? route_index : route_index / routes_per_token;
  const uint group_count = in_vec_size / GROUP_SIZE;
  const size_t weight_row =
      (size_t(matrix) * size_t(out_vec_size) + size_t(output_row)) * size_t(in_vec_size);
  const size_t scale_row =
      (size_t(matrix) * size_t(out_vec_size) + size_t(output_row)) * size_t(group_count);
  const size_t activation_base = size_t(activation_row) * size_t(in_vec_size);
  const size_t activation_scale_base = size_t(activation_row) * size_t(group_count);

  float total = 0.0f;
  METAL_PRAGMA_NO_UNROLL
  for (uint group = 0; group < group_count; ++group) {
    const uint k_base = group * GROUP_SIZE;
    int integer_product = 0;
    METAL_PRAGMA_UNROLL
    for (uint offset = 0; offset < GROUP_SIZE; ++offset) {
      const uint column = k_base + offset;
      integer_product +=
          int(weight_codes[weight_row + size_t(column)]) *
          int(activation_codes[activation_base + size_t(column)]);
    }
    total += float(integer_product) *
        float(weight_scales[scale_row + size_t(group)]) *
        activation_scales[activation_scale_base + size_t(group)];
  }

  if (expert_bias) {
    total += float(expert_biases[size_t(matrix) * size_t(out_vec_size) + size_t(output_row)]);
  }
  d[size_t(route_index) * size_t(out_vec_size) + size_t(output_row)] = DT(total);
}
