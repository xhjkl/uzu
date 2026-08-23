#include "../../common/dsl.h"
#include "../common/fragment.h"
#include "../common/mxu_fragment/ops.h"

using namespace metal;

// Resident signed-INT8 path for decode-shaped expert projections. One
// simdgroup owns 32 output rows and evaluates [32,K] x [K,1]. Activation and
// weight scales are applied to each INT32 K32 partial before FP32 accumulation.
template <typename ST, typename BT, typename DT>
VARIANTS(ST, bfloat, float)
VARIANTS(BT, bfloat, float)
VARIANTS(DT, bfloat, float)
KERNEL(ResidentInt8ExpertTensorOps)(
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
  constexpr uint OUTPUT_ROWS = 32;
  using Ops = uzu::matmul::MxuFragmentOps<>;
  using WeightFragment = uzu::matmul::OperandFragment<int8_t, 2, 2, Ops>;
  using ActivationFragment = uzu::matmul::OperandFragment<int8_t, 2, 1, Ops>;
  using ProductFragment = uzu::matmul::Fragment<int, 2, 1, Ops>;
  using AccumulatorFragment = uzu::matmul::Fragment<float, 2, 1, Ops>;

  const uint output_row_base = output_tile_index * OUTPUT_ROWS;
  const uint output_rows = min(OUTPUT_ROWS, out_vec_size - output_row_base);
  const int expert = expert_ids[route_index];
  const uint activation_row = input_is_route_major ? route_index : route_index / routes_per_token;
  device DT* output = d + size_t(route_index) * size_t(out_vec_size) + size_t(output_row_base);

  AccumulatorFragment accumulated;
  accumulated.clear();

  if (expert >= 0 && uint(expert) < expert_count) {
    const uint matrix = uint(expert);
    const uint group_count = in_vec_size / GROUP_SIZE;
    const size_t matrix_code_stride = size_t(out_vec_size) * size_t(in_vec_size);
    const size_t matrix_scale_stride = size_t(out_vec_size) * size_t(group_count);
    const device int8_t* matrix_codes = weight_codes + size_t(matrix) * matrix_code_stride;
    const device ST* matrix_scales = weight_scales + size_t(matrix) * matrix_scale_stride;
    const device int8_t* activation =
        activation_codes + size_t(activation_row) * size_t(in_vec_size);
    const device float* scales = activation_scales + size_t(activation_row) * size_t(group_count);

    METAL_PRAGMA_NO_UNROLL
    for (uint group = 0; group < group_count; ++group) {
      const uint k_base = group * GROUP_SIZE;
      WeightFragment weights;
      weights.load_from(
          simd_lane,
          uzu::matmul::fragment_source(
              matrix_codes + size_t(output_row_base) * size_t(in_vec_size) + size_t(k_base),
              int(in_vec_size)
          )
              .bounded(short(output_rows), short(GROUP_SIZE))
      );

      ActivationFragment activation_group;
      activation_group.load_from(
          simd_lane,
          uzu::matmul::fragment_source(activation + k_base, 1).bounded(short(GROUP_SIZE), 1)
      );

      ProductFragment product;
      product.clear();
      uzu::matmul::fragment_mm(product, weights, activation_group);

      const float activation_scale = scales[group];
      AccumulatorFragment::zip_for_each_coord(
          simd_lane,
          [&](short row, short column, thread float& total, thread int& integer_product) {
            if (uint(row) < output_rows && column == 0) {
              const float weight_scale = float(
                  matrix_scales[(size_t(output_row_base) + size_t(row)) * size_t(group_count) + size_t(group)]
              );
              total += float(integer_product) * activation_scale * weight_scale;
            }
          },
          accumulated,
          product
      );
    }

    if (expert_bias) {
      const device BT* biases = expert_biases + size_t(matrix) * size_t(out_vec_size) + size_t(output_row_base);
      accumulated.map_coords(simd_lane, [&](short row, short column, float value) {
        return uint(row) < output_rows && column == 0 ? value + float(biases[row]) : value;
      });
    }
  }

  accumulated.store_safe(simd_lane, output, 1, short2(1, short(output_rows)));
}
