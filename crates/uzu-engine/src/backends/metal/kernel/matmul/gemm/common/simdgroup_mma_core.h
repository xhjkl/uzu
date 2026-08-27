#pragma once

#include "../../../common/integral_constant.h"
#include "../../../common/soft_cap.h"
#include "../../../common/thread_context.h"
#include "../../common/defines.h"
#include "../../common/loader.h"
#include "../../common/threadgroup_tile.h"
#include "../../../generated/matmul.h"
#include "../generated/gemm.h"
#include "../../../hadamard_transform/hadamard_transform.h"
#include "block_geometry.h"
#include "gemm_rht.h"
#include "gemm_alignment.h"
#include "gemm_tiling.h"
#include "quant_scale_bias.h"
#include "quant_scale_zero_point.h"
#include "operands.h"
#include "schedules/staged.h"

using namespace metal;

namespace uzu {
namespace gemm {

template <
    typename T,
    ushort ROWS,
    ushort COLS,
    ushort SHARED_LEADING_DIMENSION,
    ushort THREADGROUP_SIZE,
    ushort READS_PER_THREAD = (ROWS * COLS) / THREADGROUP_SIZE,
    ushort THREAD_COLS = COLS / READS_PER_THREAD,
    ushort THREAD_ROWS = THREADGROUP_SIZE / THREAD_COLS>
struct RoutedALoader {
  static_assert((ROWS * COLS) % THREADGROUP_SIZE == 0, "routed A tile must divide evenly across threads");
  static_assert(COLS % READS_PER_THREAD == 0, "routed A reads must divide the K tile");
  static_assert(THREADGROUP_SIZE % THREAD_COLS == 0, "routed A rows must divide evenly across threads");

  const device T* source;
  const device uint* grouped_routes;
  threadgroup T* destination;
  uint grouped_base;
  uint routes_per_token;
  uint source_leading_dimension;
  uint k_offset;
  bool input_is_route_major;
  ushort tile_row_index;
  ushort tile_col_index;

  METAL_FUNC RoutedALoader(
      const device T* source_,
      const device uint* grouped_routes_,
      threadgroup T* destination_,
      uint grouped_base_,
      uint routes_per_token_,
      uint source_leading_dimension_,
      bool input_is_route_major_,
      const thread ThreadContext& thread_context
  )
      : source(source_), grouped_routes(grouped_routes_), destination(destination_), grouped_base(grouped_base_),
        routes_per_token(routes_per_token_), source_leading_dimension(source_leading_dimension_), k_offset(0),
        input_is_route_major(input_is_route_major_) {
    const ushort thread_index = thread_context.simdgroup_index * METAL_SIMD_SIZE + thread_context.simd_lane_id;
    tile_row_index = thread_index / THREAD_COLS;
    tile_col_index = READS_PER_THREAD * (thread_index % THREAD_COLS);
  }

  METAL_FUNC void load_unsafe() const {
    METAL_PRAGMA_UNROLL
    for (ushort row = tile_row_index; row < ROWS; row += THREAD_ROWS) {
      const uint route = grouped_routes[grouped_base + row];
      const uint input_row = input_is_route_major ? route : route / routes_per_token;
      METAL_PRAGMA_UNROLL
      for (ushort column = 0; column < READS_PER_THREAD; ++column) {
        destination[row * SHARED_LEADING_DIMENSION + tile_col_index + column] =
            source[size_t(input_row) * source_leading_dimension + k_offset + tile_col_index + column];
      }
    }
  }

  METAL_FUNC void load_safe(short2 source_tile_dimensions) const {
    METAL_PRAGMA_UNROLL
    for (ushort row = tile_row_index; row < ROWS; row += THREAD_ROWS) {
      const bool valid_row = row < source_tile_dimensions.y;
      const uint route = valid_row ? grouped_routes[grouped_base + row] : 0;
      const uint input_row = input_is_route_major ? route : route / routes_per_token;
      METAL_PRAGMA_UNROLL
      for (ushort column = 0; column < READS_PER_THREAD; ++column) {
        const ushort local_column = tile_col_index + column;
        const bool valid = valid_row && local_column < source_tile_dimensions.x;
        destination[row * SHARED_LEADING_DIMENSION + local_column] =
            valid ? source[size_t(input_row) * source_leading_dimension + k_offset + local_column] : T(0);
      }
    }
  }

  METAL_FUNC void next() { k_offset += COLS; }
};

template <
    typename OutputElementType_,
    GemmTiling GEMM_TILING,
    bool TRANSPOSE_B,
    typename LeftOperand,
    typename RightOperand>
struct SimdgroupMmaCore {
  using OutputElementType = OutputElementType_;
  using Left = LeftOperand;
  using Right = RightOperand;
  using LeftElementType = typename Left::ElementType;
  using RightElementType = typename Right::ElementType;
  using LeftStorage = operands::LeftStorage<Left>;
  using RightStorage = operands::RightStorage<Right>;
  static_assert(!Left::QUANTIZED, "simdgroup MMA stages full-precision left operands only");
  UZU_CONST bool TRANSPOSE_RIGHT = TRANSPOSE_B;
  UZU_CONST int THREADGROUP_BLOCK_M = gemm_tiling_block_m(GEMM_TILING);
  UZU_CONST int THREADGROUP_BLOCK_N = gemm_tiling_block_n(GEMM_TILING);
  UZU_CONST int THREADGROUP_BLOCK_K = gemm_tiling_block_k(GEMM_TILING);
  UZU_CONST int SIMDGROUPS_PER_ROW = gemm_tiling_simdgroups_per_row(GEMM_TILING);
  UZU_CONST int SIMDGROUPS_PER_COLUMN = gemm_tiling_simdgroups_per_column(GEMM_TILING);
  UZU_CONST ushort PADDING_A = 16 / sizeof(LeftElementType);
  UZU_CONST ushort PADDING_B = 16 / sizeof(RightElementType);
  UZU_CONST ushort SHARED_STRIDE_A = THREADGROUP_BLOCK_K + PADDING_A;
  UZU_CONST ushort SHARED_STRIDE_B = (TRANSPOSE_B ? THREADGROUP_BLOCK_K : THREADGROUP_BLOCK_N) + PADDING_B;
  UZU_CONST ushort THREADGROUP_THREADS = SIMDGROUPS_PER_ROW * SIMDGROUPS_PER_COLUMN * METAL_SIMD_SIZE;

  using ALoader = uzu::matmul::ThreadgroupLoader<
      LeftElementType,
      THREADGROUP_BLOCK_M,
      THREADGROUP_BLOCK_K,
      SHARED_STRIDE_A,
      true,
      THREADGROUP_THREADS>;
  using FullPrecisionRightLoader = uzu::matmul::ThreadgroupLoader<
      RightElementType,
      TRANSPOSE_B ? THREADGROUP_BLOCK_N : THREADGROUP_BLOCK_K,
      TRANSPOSE_B ? THREADGROUP_BLOCK_K : THREADGROUP_BLOCK_N,
      SHARED_STRIDE_B,
      TRANSPOSE_B,
      THREADGROUP_THREADS>;
  using TileAccumulator = uzu::matmul::ThreadgroupTile<
      LeftElementType,
      RightElementType,
      OutputElementType,
      THREADGROUP_BLOCK_M,
      THREADGROUP_BLOCK_N,
      THREADGROUP_BLOCK_K,
      SIMDGROUPS_PER_ROW,
      SIMDGROUPS_PER_COLUMN,
      false,
      TRANSPOSE_B,
      SHARED_STRIDE_A,
      SHARED_STRIDE_B,
      float,
      uzu::matmul::TransformNone<OutputElementType, float>>;

  template <uint GEMM_ALIGNMENT_RAW, typename ALoaderType, typename BLoader>
  static METAL_FUNC void k_loop(
      threadgroup LeftElementType* a_shared,
      threadgroup RightElementType* b_shared,
      const int aligned_k_iterations,
      thread ALoaderType& loader_a,
      thread BLoader& loader_b,
      thread TileAccumulator& accumulator,
      thread const ushort& tile_block_rows,
      thread const ushort& tile_block_cols,
      thread const ushort& leftover_block_depth
  ) {
    constexpr GemmAlignment gemm_alignment{GEMM_ALIGNMENT_RAW};
    short2 tile_dimensions_a = short2(THREADGROUP_BLOCK_K, tile_block_rows);
    short2 tile_dimensions_b =
        TRANSPOSE_B ? short2(THREADGROUP_BLOCK_K, tile_block_cols) : short2(tile_block_cols, THREADGROUP_BLOCK_K);

    for (int k = 0; k < aligned_k_iterations; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if constexpr (gemm_alignment.contains(GemmAlignment::M)) {
        loader_a.load_unsafe();
      } else {
        loader_a.load_safe(tile_dimensions_a);
      }
      if constexpr (gemm_alignment.contains(GemmAlignment::N)) {
        loader_b.load_unsafe();
      } else {
        loader_b.load_safe(tile_dimensions_b);
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);
      accumulator.matmul(a_shared, b_shared);

      loader_a.next();
      loader_b.next();
    }

    if constexpr (!gemm_alignment.contains(GemmAlignment::K)) {
      threadgroup_barrier(mem_flags::mem_threadgroup);

      short2 last_tile_dimensions_a = short2(leftover_block_depth, tile_block_rows);
      short2 last_tile_dimensions_b =
          TRANSPOSE_B ? short2(leftover_block_depth, tile_block_cols) : short2(tile_block_cols, leftover_block_depth);

      loader_a.load_safe(last_tile_dimensions_a);
      loader_b.load_safe(last_tile_dimensions_b);

      threadgroup_barrier(mem_flags::mem_threadgroup);
      accumulator.matmul(a_shared, b_shared);
    }
  }

  template <uint GEMM_ALIGNMENT_RAW>
  static METAL_FUNC void finalize(
      thread TileAccumulator& accumulator,
      device OutputElementType* d,
      const constant uzu::matmul::GemmParams* params,
      const thread ushort& tile_block_rows,
      const thread ushort& tile_block_cols,
      const bool needs_epilogue,
      const thread uzu::matmul::TransformScaleAccumulate<float, float>& epilogue,
      const device RightElementType* bias_block,
      const bool needs_bias,
      const bool needs_soft_cap,
      const device int32_t* rht_factors_block,
      const bool needs_rht,
      const thread ThreadContext& thread_context
  ) {
    constexpr GemmAlignment gemm_alignment{GEMM_ALIGNMENT_RAW};
    if constexpr (gemm_alignment.contains(GemmAlignment::M) && gemm_alignment.contains(GemmAlignment::N)) {
      if (needs_epilogue) {
        accumulator.apply_epilogue(d, params->leading_dimension_d, 1, epilogue);
      }
      if (needs_bias) {
        accumulator.apply_bias(bias_block);
      }
      if constexpr (Right::MICROFLOAT) {
        if (needs_soft_cap) {
          accumulator.c_fragment.map([&](float value) { return uzu::apply_soft_cap(value, params->soft_cap); });
        }
      }
      accumulator.store_result(d, params->leading_dimension_d);
    } else {
      if (needs_epilogue) {
        accumulator
            .apply_epilogue_safe(d, params->leading_dimension_d, 1, short2(tile_block_cols, tile_block_rows), epilogue);
      }
      if (needs_bias) {
        accumulator.apply_bias_safe(bias_block, short2(tile_block_cols, tile_block_rows));
      }
      if constexpr (Right::MICROFLOAT) {
        if (needs_soft_cap) {
          accumulator.c_fragment.map([&](float value) { return uzu::apply_soft_cap(value, params->soft_cap); });
        }
      }
      accumulator.store_result_safe(d, params->leading_dimension_d, short2(tile_block_cols, tile_block_rows));
    }

    if (needs_rht) {
      threadgroup_barrier(mem_flags::mem_device);
      apply_output_random_hadamard_transform(
          d,
          rht_factors_block,
          tile_block_rows,
          tile_block_cols,
          params->leading_dimension_d,
          ushort(SIMDGROUPS_PER_ROW * SIMDGROUPS_PER_COLUMN),
          thread_context
      );
    }
  }

  static METAL_FUNC void finalize_routed(
      thread TileAccumulator& accumulator,
      device OutputElementType* d,
      const constant uzu::matmul::GemmParams* params,
      const device uint* grouped_routes,
      size_t grouped_base,
      ushort tile_block_rows,
      size_t block_col,
      ushort tile_block_cols,
      uint expert,
      GemmDTransform output_transform,
      const device RightElementType* output_bias,
      const device RightElementType* expert_biases
  ) {
    accumulator.for_each_output([&](ushort row_offset, ushort col_offset, ushort lane_offset, thread float& value) {
      const ushort row = accumulator.simdgroup_row_offset + row_offset;
      const ushort column = accumulator.simdgroup_col_offset + col_offset + lane_offset;
      if (row >= tile_block_rows || column >= tile_block_cols) {
        return;
      }

      if (output_transform.contains(GemmDTransform::SCALE)) {
        value *= params->ab_scale;
      }
      const size_t global_column = block_col + column;
      if (output_transform.contains(GemmDTransform::BIAS)) {
        value += float(output_bias[global_column]);
      }
      if (expert_biases != nullptr) {
        value += float(expert_biases[size_t(expert) * params->N + global_column]);
      }
      if (output_transform.contains(GemmDTransform::SOFT_CAP)) {
        value = uzu::apply_soft_cap(value, params->soft_cap);
      }

      const uint route = grouped_routes[grouped_base + row];
      d[size_t(route) * params->leading_dimension_d + global_column] = OutputElementType(value);
    });
  }

  static METAL_FUNC void run(
      LeftStorage left,
      RightStorage right,
      device OutputElementType* d,
      const constant uzu::matmul::GemmParams* params,
      GemmAlignment alignment,
      GemmDTransform output_transform,
      const device RightElementType* output_bias,
      const device int32_t* rht_factors,
      threadgroup LeftElementType* a_shared,
      threadgroup RightElementType* b_shared,
      const thread ThreadContext& thread_context
  ) {
    const uint partition = thread_context.threadgroup_position.z;
    const uint tile_row = thread_context.threadgroup_position.y;
    const uint2 tile = tile_id(uint2(thread_context.threadgroup_position.x, tile_row), params);
    const auto geometry = ThreadgroupTileGeometry<THREADGROUP_BLOCK_M, THREADGROUP_BLOCK_N>::compute(tile, params);
    if (geometry.out_of_bounds) {
      return;
    }

    const uint k_offset = partition * params->aligned_inner_iterations * THREADGROUP_BLOCK_K;

    threadgroup_barrier(mem_flags::mem_none);

    const size_t block_row = size_t(geometry.block_row_start);
    const size_t block_col = size_t(geometry.block_col_start);

    d +=
        size_t(partition) * size_t(params->M) * size_t(params->N) + block_row * params->leading_dimension_d + block_col;

    thread ALoader loader_a(
        left.values + block_row * params->leading_dimension_a + k_offset,
        params->leading_dimension_a,
        a_shared,
        thread_context
    );
    thread TileAccumulator accumulator(thread_context);

    const ushort tile_block_rows =
        min(THREADGROUP_BLOCK_M, static_cast<int>(params->M) - static_cast<int>(geometry.block_row_start));
    const ushort tile_block_cols =
        min(THREADGROUP_BLOCK_N, static_cast<int>(params->N) - static_cast<int>(geometry.block_col_start));
    const ushort leftover_block_depth = params->K - params->aligned_inner_iterations * THREADGROUP_BLOCK_K;

    const bool needs_scale = output_transform.contains(GemmDTransform::SCALE);
    const bool needs_accumulate = output_transform.contains(GemmDTransform::ACCUMULATE);
    const bool needs_bias = output_transform.contains(GemmDTransform::BIAS);
    bool needs_soft_cap = false;
    if constexpr (Right::MICROFLOAT) {
      needs_soft_cap = output_transform.contains(GemmDTransform::SOFT_CAP);
    }
    const bool needs_rht = output_transform.contains(GemmDTransform::RHT);
    const bool needs_epilogue = needs_scale || needs_accumulate;
    const float alpha = needs_scale ? params->ab_scale : 1.0f;
    const float beta = needs_accumulate ? 1.0f : 0.0f;
    uzu::matmul::TransformScaleAccumulate<float, float> epilogue(alpha, beta);
    const device RightElementType* bias_block = output_bias + block_col;
    const device int32_t* rht_factors_block = rht_factors + block_col;

    const bool all_aligned = ((alignment.contains(GemmAlignment::M)) || (tile_block_rows == THREADGROUP_BLOCK_M)) &&
                             ((alignment.contains(GemmAlignment::N)) || (tile_block_cols == THREADGROUP_BLOCK_N)) &&
                             alignment.contains(GemmAlignment::K);
    constexpr uint MASK_ALL =
        static_cast<uint>(GemmAlignment::M) | static_cast<uint>(GemmAlignment::N) | static_cast<uint>(GemmAlignment::K);
    const uint dynamic_alignment_mask =
        alignment.raw_value | ((tile_block_rows == THREADGROUP_BLOCK_M) ? static_cast<uint>(GemmAlignment::M) : 0u) |
        ((tile_block_cols == THREADGROUP_BLOCK_N) ? static_cast<uint>(GemmAlignment::N) : 0u);

    auto run_with_loader = [&](auto loader_b) {
      auto kernel_invoke = [&](auto gemm_alignment_mask) {
        constexpr uint gemm_alignment = gemm_alignment_mask.value;
        k_loop<gemm_alignment>(
            a_shared,
            b_shared,
            params->aligned_inner_iterations,
            loader_a,
            loader_b,
            accumulator,
            tile_block_rows,
            tile_block_cols,
            leftover_block_depth
        );
        if constexpr (Right::MICROFLOAT) {
          // Keep the outer scale in f32; E2M1 x E8M0 is exact in BF16 staging.
          const float outer_scale = right.mxfp4_outer_scale(0);
          accumulator.c_fragment.map([&](float value) { return value * outer_scale; });
        }
        finalize<gemm_alignment>(
            accumulator,
            d,
            params,
            tile_block_rows,
            tile_block_cols,
            needs_epilogue,
            epilogue,
            bias_block,
            needs_bias,
            needs_soft_cap,
            rht_factors_block,
            needs_rht,
            thread_context
        );
      };

      if (all_aligned) {
        kernel_invoke(integral_constant<uint, MASK_ALL>{});
      } else {
        dispatch_gemm_alignment(dynamic_alignment_mask, kernel_invoke);
      }
    };

    if constexpr (Right::DENSE) {
      auto loader_b = schedules::make_full_precision_loader<SimdgroupMmaCore>(
          right,
          params,
          block_col,
          k_offset,
          b_shared,
          thread_context
      );
      run_with_loader(loader_b);
    } else if constexpr (Right::MICROFLOAT) {
      auto loader_b = schedules::make_mxfp4_loader<SimdgroupMmaCore, RightOperand>(
          right,
          params,
          block_col,
          k_offset,
          b_shared,
          thread_context
      );
      run_with_loader(loader_b);
    } else {
      auto loader_b = schedules::make_staged_loader<SimdgroupMmaCore, RightOperand>(
          right,
          params,
          block_col,
          k_offset,
          b_shared,
          thread_context
      );
      run_with_loader(loader_b);
    }
  }

  static METAL_FUNC void run_routed(
      LeftStorage left,
      RightStorage right,
      device OutputElementType* d,
      const constant uzu::matmul::GemmParams* params,
      GemmAlignment alignment,
      GemmDTransform output_transform,
      const device RightElementType* output_bias,
      const device RightElementType* expert_biases,
      const device uint* route_offsets,
      const device uint* grouped_routes,
      uint routes_per_token,
      bool input_is_route_major,
      uint row_partitions,
      threadgroup LeftElementType* a_shared,
      threadgroup RightElementType* b_shared,
      const thread ThreadContext& thread_context
  ) {
    const uint expert = thread_context.threadgroup_position.y;
    const uint partition = thread_context.threadgroup_position.z;
    const uint begin = route_offsets[expert];
    const uint end = route_offsets[expert + 1];
    const size_t block_col = size_t(thread_context.threadgroup_position.x) * THREADGROUP_BLOCK_N;
    if (block_col >= params->N) {
      return;
    }

    const ushort tile_block_cols = min(THREADGROUP_BLOCK_N, int(params->N - block_col));
    const ushort leftover_block_depth = params->K - params->aligned_inner_iterations * THREADGROUP_BLOCK_K;
    const size_t bank_block_col = size_t(expert) * params->N + block_col;
    const uint grouped_stride = row_partitions * THREADGROUP_BLOCK_M;

    for (size_t grouped_base = size_t(begin) + partition * THREADGROUP_BLOCK_M; grouped_base < end;
         grouped_base += grouped_stride) {
      const ushort tile_block_rows = min(THREADGROUP_BLOCK_M, int(size_t(end) - grouped_base));
      thread RoutedALoader<
          LeftElementType,
          THREADGROUP_BLOCK_M,
          THREADGROUP_BLOCK_K,
          SHARED_STRIDE_A,
          THREADGROUP_THREADS>
          loader_a(
              left.values,
              grouped_routes,
              a_shared,
              uint(grouped_base),
              routes_per_token,
              params->leading_dimension_a,
              input_is_route_major,
              thread_context
          );
      thread TileAccumulator accumulator(thread_context);

      const bool all_aligned =
          (alignment.contains(GemmAlignment::N) || tile_block_cols == THREADGROUP_BLOCK_N) &&
          alignment.contains(GemmAlignment::K);
      constexpr uint MASK_NK = static_cast<uint>(GemmAlignment::N) | static_cast<uint>(GemmAlignment::K);
      const uint dynamic_alignment_mask =
          (tile_block_cols == THREADGROUP_BLOCK_N ? static_cast<uint>(GemmAlignment::N) : 0u) |
          (alignment.contains(GemmAlignment::K) ? static_cast<uint>(GemmAlignment::K) : 0u);

      auto run_with_loader = [&](auto loader_b) {
        auto kernel_invoke = [&](auto gemm_alignment_mask) {
          constexpr uint gemm_alignment = gemm_alignment_mask.value;
          k_loop<gemm_alignment>(
              a_shared,
              b_shared,
              params->aligned_inner_iterations,
              loader_a,
              loader_b,
              accumulator,
              tile_block_rows,
              tile_block_cols,
              leftover_block_depth
          );
          if constexpr (Right::MICROFLOAT) {
            const float outer_scale = right.mxfp4_outer_scale(expert);
            accumulator.c_fragment.map([&](float value) { return value * outer_scale; });
          }
          finalize_routed(
              accumulator,
              d,
              params,
              grouped_routes,
              grouped_base,
              tile_block_rows,
              block_col,
              tile_block_cols,
              expert,
              output_transform,
              output_bias,
              expert_biases
          );
        };

        if (all_aligned) {
          kernel_invoke(integral_constant<uint, MASK_NK>{});
        } else {
          dispatch_gemm_alignment(dynamic_alignment_mask, kernel_invoke);
        }
      };

      if constexpr (Right::DENSE) {
        auto loader_b = schedules::make_full_precision_loader<SimdgroupMmaCore>(
            right,
            params,
            bank_block_col,
            0,
            b_shared,
            thread_context
        );
        run_with_loader(loader_b);
      } else if constexpr (Right::MICROFLOAT) {
        auto loader_b = schedules::make_mxfp4_loader<SimdgroupMmaCore, RightOperand>(
            right,
            params,
            bank_block_col,
            0,
            b_shared,
            thread_context
        );
        run_with_loader(loader_b);
      } else {
        auto loader_b = schedules::make_staged_loader<SimdgroupMmaCore, RightOperand>(
            right,
            params,
            bank_block_col,
            0,
            b_shared,
            thread_context
        );
        run_with_loader(loader_b);
      }
    }
  }
};

} // namespace gemm
} // namespace uzu
