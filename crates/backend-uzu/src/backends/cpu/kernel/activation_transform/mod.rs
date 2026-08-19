pub mod activation_transform;

use crate::backends::common::gpu_types::HADAMARD_TRANSFORM_BLOCK_SIZE;

pub const INT8_SYMMETRIC_QUANTIZATION_MAXIMUM: f32 = 127.0;

pub use activation_transform::quantize_transformed_row;

pub fn min_max_symmetric_divisor(values: &[f32]) -> f32 {
    let magnitude =
        values.iter().filter(|value| value.is_finite()).fold(0.0f32, |maximum, &value| maximum.max(value.abs()));
    if magnitude > 0.0 {
        magnitude / INT8_SYMMETRIC_QUANTIZATION_MAXIMUM
    } else {
        1.0
    }
}

pub fn quantize_symmetric_i8(
    value: f32,
    divisor: f32,
) -> i8 {
    if !value.is_finite() {
        return 0;
    }
    (value / divisor).round().clamp(-INT8_SYMMETRIC_QUANTIZATION_MAXIMUM, INT8_SYMMETRIC_QUANTIZATION_MAXIMUM) as i8
}

pub(crate) fn hadamard_transform(values: &mut [f32; HADAMARD_TRANSFORM_BLOCK_SIZE as usize]) {
    let mut stride = 1;
    while stride < HADAMARD_TRANSFORM_BLOCK_SIZE as usize {
        for lane in 0..HADAMARD_TRANSFORM_BLOCK_SIZE as usize {
            if lane & stride == 0 {
                let a = values[lane];
                let b = values[lane | stride];
                values[lane] = a + b;
                values[lane | stride] = a - b;
            }
        }
        stride <<= 1;
    }
    let scale = 1.0 / (HADAMARD_TRANSFORM_BLOCK_SIZE as f32).sqrt();
    for v in values.iter_mut() {
        *v *= scale;
    }
}
