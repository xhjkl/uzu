#include <metal_stdlib>

#include "../../common/dsl.h"
#include "../../common/soft_cap.h"
#include "../../generated/gemm.h"
#include "../common/microfloat.h"

using namespace metal;
using namespace uzu::gemm;

#define MXFP4_GEMM_COLUMNS_PER_SIMDGROUP 4
#define MXFP4_GEMM_SIMDGROUPS 4
#define MXFP4_GEMM_COLUMNS 16
#define MXFP4_GEMM_ROWS 4

template <typename T>
METAL_FUNC float4 load_float4(const device T* values) {
  return float4(float(values[0]), float(values[1]), float(values[2]), float(values[3]));
}

template <typename AT, typename BT, typename DT, uint GROUP_SIZE>
VARIANTS(AT, bfloat, float)
VARIANTS(BT, bfloat, float)
VARIANTS(DT, bfloat, float)
VARIANTS(GROUP_SIZE, 16, 32)
CONSTRAINT(BT != "float" || (AT == "float" && DT == "float"))
KERNEL(MicrofloatGemm)(
    device const uchar* codes,
    device const uchar* scales,
    device const BT* outer_scales,
    device const AT* a,
    device DT* d,
    device const BT* output_bias
        OPTIONAL(output_transform.contains(GemmDTransform::BIAS)),
    constant uint& m,
    constant uint& n,
    constant uint& k,
    constant float& ab_scale,
    constant float& soft_cap
        OPTIONAL(output_transform.contains(GemmDTransform::SOFT_CAP)),
    const GemmDTransform output_transform SPECIALIZE,
    const uint column_tile GROUPS(n.div_ceil(MXFP4_GEMM_COLUMNS)),
    const uint row_tile GROUPS(m.div_ceil(MXFP4_GEMM_ROWS)),
    const uint simd_lane THREADS(32),
    const uint simd_group THREADS(MXFP4_GEMM_SIMDGROUPS)
) {
  const uint column_base = column_tile * MXFP4_GEMM_COLUMNS + simd_group * MXFP4_GEMM_COLUMNS_PER_SIMDGROUP;
  const uint row_base = row_tile * MXFP4_GEMM_ROWS;
  const float outer_scale = float(outer_scales[0]);
  float values[MXFP4_GEMM_ROWS][MXFP4_GEMM_COLUMNS_PER_SIMDGROUP] = {
      {0.0f, 0.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, 0.0f, 0.0f},
  };

  for (uint inner = simd_lane * 4; inner < k; inner += 32 * 4) {
    float4 input_values[MXFP4_GEMM_ROWS];
#pragma clang loop unroll(full)
    for (uint row = 0; row < MXFP4_GEMM_ROWS; ++row) {
      input_values[row] = row_base + row < m ? load_float4(a + ulong(row_base + row) * k + inner) : float4(0.0f);
    }

#pragma clang loop unroll(full)
    for (uint output = 0; output < MXFP4_GEMM_COLUMNS_PER_SIMDGROUP; ++output) {
      const uint column = column_base + output;
      if (column >= n) {
        continue;
      }
      const ulong code_offset = ulong(column) * (k / 2) + inner / 2;
      const ushort packed = *reinterpret_cast<const device ushort*>(codes + code_offset);
      const uint exponent = scales[ulong(column) * (k / GROUP_SIZE) + inner / GROUP_SIZE];
      const float scale = decode_e8m0(exponent) * outer_scale;
      const float4 weights = scale * float4(
                                         decode_e2m1(packed & 0x0fu),
                                         decode_e2m1((packed >> 4u) & 0x0fu),
                                         decode_e2m1((packed >> 8u) & 0x0fu),
                                         decode_e2m1(packed >> 12u)
                                     );
#pragma clang loop unroll(full)
      for (uint row = 0; row < MXFP4_GEMM_ROWS; ++row) {
        values[row][output] += dot(input_values[row], weights);
      }
    }
  }

#pragma clang loop unroll(full)
  for (uint row = 0; row < MXFP4_GEMM_ROWS; ++row) {
    if (row_base + row >= m) {
      continue;
    }
#pragma clang loop unroll(full)
    for (uint output = 0; output < MXFP4_GEMM_COLUMNS_PER_SIMDGROUP; ++output) {
      const uint column = column_base + output;
      float value = simd_sum(values[row][output]);
      if (simd_lane != 0 || column >= n) {
        continue;
      }
      const ulong output_offset = ulong(row_base + row) * n + column;
      if (output_transform.contains(GemmDTransform::SCALE)) {
        value *= ab_scale;
      }
      if (output_transform.contains(GemmDTransform::ACCUMULATE)) {
        value += float(d[output_offset]);
      }
      if (output_transform.contains(GemmDTransform::BIAS)) {
        value += float(output_bias[column]);
      }
      if (output_transform.contains(GemmDTransform::SOFT_CAP)) {
        value = uzu::apply_soft_cap(value, soft_cap);
      }
      d[output_offset] = DT(value);
    }
  }
}
