#include <metal_atomic>
#include <metal_stdlib>

#include "../../common/dsl.h"

using namespace metal;

KERNEL(ExpertRouteClearCounts)(
    device _atomic<uint>* counts,
    constant uint& expert_count,
    const uint slot AXIS(expert_count + 1, 256)
) {
  atomic_store_explicit(&counts[slot], 0u, memory_order_relaxed);
}

KERNEL(ExpertRouteCount)(
    device const int* expert_ids,
    device _atomic<uint>* counts,
    constant uint& route_count,
    constant uint& expert_count,
    const uint route AXIS(route_count, 256)
) {
  const int expert = expert_ids[route];
  if (expert < 0 || uint(expert) >= expert_count) {
    return;
  }
  atomic_fetch_add_explicit(&counts[uint(expert) + 1], 1u, memory_order_relaxed);
}

KERNEL(ExpertRoutePrefix)(
    device _atomic<uint>* counts_and_offsets,
    device uint* route_tiles,
    device _atomic<uint>* cursors,
    device uint* tile_count,
    constant uint& expert_count,
    constant uint& block_m,
    const uint lid THREADS(1),
    const uint group GROUPS(1)
) {
  (void)lid;
  (void)group;

  uint prefix = 0;
  uint tile_prefix = 0;
  for (uint expert = 0; expert < expert_count; ++expert) {
    const uint count = atomic_load_explicit(&counts_and_offsets[expert + 1], memory_order_relaxed);
    atomic_store_explicit(&counts_and_offsets[expert], prefix, memory_order_relaxed);
    atomic_store_explicit(&cursors[expert], prefix, memory_order_relaxed);
    for (uint row = 0; row < count; row += block_m) {
      const uint tile = 3 * tile_prefix++;
      route_tiles[tile] = expert;
      route_tiles[tile + 1] = prefix + row;
      route_tiles[tile + 2] = min(block_m, count - row);
    }
    prefix += count;
  }
  atomic_store_explicit(&counts_and_offsets[expert_count], prefix, memory_order_relaxed);
  if (tile_prefix == 0) {
    route_tiles[0] = 0;
    route_tiles[1] = 0;
    route_tiles[2] = 0;
    tile_prefix = 1;
  }
  tile_count[0] = tile_prefix;
}

KERNEL(ExpertRouteDispatchArguments)(
    const device uint* tile_count,
    device uint* dispatch_arguments,
    constant uint& column_tiles,
    const uint lid THREADS(1),
    const uint group GROUPS(1)
) {
  (void)lid;
  (void)group;
  dispatch_arguments[0] = column_tiles;
  dispatch_arguments[1] = tile_count[0];
  dispatch_arguments[2] = 1;
}

KERNEL(ExpertRouteScatter)(
    device const int* expert_ids,
    device _atomic<uint>* cursors,
    device uint* grouped_routes,
    constant uint& route_count,
    constant uint& expert_count,
    const uint route AXIS(route_count, 256)
) {
  const int expert = expert_ids[route];
  if (expert < 0 || uint(expert) >= expert_count) {
    return;
  }
  const uint destination = atomic_fetch_add_explicit(&cursors[uint(expert)], 1u, memory_order_relaxed);
  grouped_routes[destination] = route;
}

template <typename DT>
VARIANTS(DT, half, bfloat, float)
KERNEL(ExpertRouteZeroInvalid)(
    device const int* expert_ids,
    device DT* output,
    constant uint& route_count,
    constant uint& output_width,
    constant uint& expert_count,
    const uint route GROUPS(route_count),
    const uint lane THREADS(32)
) {
  const int expert = expert_ids[route];
  if (expert >= 0 && uint(expert) < expert_count) {
    return;
  }
  for (uint column = lane; column < output_width; column += 32) {
    output[ulong(route) * output_width + column] = DT(0);
  }
}
