#pragma once

#include <metal_stdlib>

#include "../../common/defines.h"

namespace uzu {
namespace gemm {

template <typename AT_, typename BT_, typename DT_>
struct GemvOperands {
  using AT = AT_;
  using BT = BT_;
  using DT = DT_;
  const device uint32_t* b;
  const device BT* scales;
  const device uint8_t* zero_points;
  const device BT* biases;
  const device BT* outer_scales;
  const device AT* a;
  device DT* d;
  const device BT* output_bias;
  const device int32_t* hadamard_factors;
  const device uint* gather_indices;
  const device int* expert_ids;
  const device BT* expert_biases;
};

struct GemvParams {
  uint in_vec_size;
  uint out_vec_size;
  uint batch_size;
  float ab_scale;
  float soft_cap;
  GemmDTransform output_transform;
  bool gathered;
  bool expert_routed;
  bool expert_bias;
  bool signed_codes;
  uint routes_per_token;
  uint expert_count;
  bool input_is_route_major;
};

template <typename AT, typename BT, typename DT>
METAL_FUNC bool route_valid(
    const thread GemvOperands<AT, BT, DT>& ops,
    const thread GemvParams& params,
    uint route
) {
  return !params.expert_routed || (ops.expert_ids[route] >= 0 && uint(ops.expert_ids[route]) < params.expert_count);
}

template <typename AT, typename BT, typename DT>
METAL_FUNC uint matrix_index(
    const thread GemvOperands<AT, BT, DT>& ops,
    const thread GemvParams& params,
    uint route
) {
  return route_valid(ops, params, route) && params.expert_routed ? uint(ops.expert_ids[route]) : 0;
}

METAL_FUNC uint input_row(const thread GemvParams& params, uint route) {
  return params.expert_routed && !params.input_is_route_major ? route / params.routes_per_token : route;
}

} // namespace gemm
} // namespace uzu
