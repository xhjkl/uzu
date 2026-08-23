#pragma once

#include "defines.h"

namespace uzu {
namespace gemm {

METAL_FUNC float decode_e2m1(uint code) {
  constexpr float magnitudes[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const uint magnitude_bits = as_type<uint>(magnitudes[code & 0x7u]);
  const uint sign_bit = (code & 0x8u) << 28u;
  return as_type<float>(magnitude_bits | sign_bit);
}

METAL_FUNC float decode_e8m0(uint exponent) {
  if (exponent == 0u) {
    return as_type<float>(0x00400000u);
  }
  if (exponent == 255u) {
    return as_type<float>(0x7fc00000u);
  }
  return as_type<float>(exponent << 23u);
}

METAL_FUNC float decode_e4m3(uint bits) {
  const uint sign = (bits & 0x80u) << 24u;
  const uint exponent = (bits >> 3u) & 0xfu;
  const uint mantissa = bits & 0x7u;
  if (exponent == 0u) {
    if (mantissa == 0u) {
      return as_type<float>(sign);
    }
    const float value = float(mantissa) / 512.0f;
    return as_type<float>(sign | as_type<uint>(value));
  }
  if (exponent == 15u && mantissa == 7u) {
    return as_type<float>(sign | 0x7fc00000u);
  }
  return as_type<float>(sign | ((exponent + 120u) << 23u) | (mantissa << 20u));
}

template <bool SCALE_E4M3>
METAL_FUNC float decode_group_scale(uint scale) {
  if constexpr (SCALE_E4M3) {
    return decode_e4m3(scale);
  }
  return decode_e8m0(scale);
}

} // namespace gemm
} // namespace uzu
