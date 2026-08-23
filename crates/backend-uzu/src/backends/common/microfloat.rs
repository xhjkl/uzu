use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MicrofloatFormat {
    Mxfp4,
    Nvfp4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MicrofloatLayout {
    OutputInput,
    InputOutput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum MicrofloatError {
    #[error("unsupported microfloat format: {0:?}")]
    UnsupportedFormat(MicrofloatFormat),
    #[error("unsupported {format:?} bit width: {bits}")]
    UnsupportedBits {
        format: MicrofloatFormat,
        bits: u32,
    },
    #[error("unsupported {format:?} group size: {group_size}")]
    UnsupportedGroupSize {
        format: MicrofloatFormat,
        group_size: u32,
    },
    #[error("unsupported microfloat matrix layout: {0:?}")]
    UnsupportedLayout(MicrofloatLayout),
    #[error("microfloat matrix count, rows, and columns must be nonzero")]
    EmptyShape,
    #[error("microfloat columns {columns} are not divisible by group size {group_size}")]
    MisalignedColumns {
        columns: u32,
        group_size: u32,
    },
    #[error("microfloat storage size overflows usize")]
    SizeOverflow,
    #[error("E8M0 exponent 255 is invalid")]
    InvalidE8M0Exponent,
    #[error("INT32 accumulator bound exceeds INT32_MAX: {left_max_abs} * {right_max_abs} * {length} = {bound}")]
    Int32AccumulatorOverflow {
        left_max_abs: u32,
        right_max_abs: u32,
        length: u32,
        bound: u128,
    },
}

/// Packed E2M1 values with per-group E8M0 scales.
///
/// Group 32 is the OCP MXFP4 block size. Group 16 is the converter-defined
/// GPT-OSS layout, which retains the same E2M1/E8M0 encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MicrofloatMetadata {
    format: MicrofloatFormat,
    group_size: u32,
    layout: MicrofloatLayout,
    matrix_count: u32,
    rows: u32,
    columns: u32,
}

impl MicrofloatMetadata {
    pub fn new(
        format: MicrofloatFormat,
        bits: u32,
        group_size: u32,
        layout: MicrofloatLayout,
        matrix_count: u32,
        rows: u32,
        columns: u32,
    ) -> Result<Self, MicrofloatError> {
        if format != MicrofloatFormat::Mxfp4 {
            return Err(MicrofloatError::UnsupportedFormat(format));
        }
        if bits != 4 {
            return Err(MicrofloatError::UnsupportedBits {
                format,
                bits,
            });
        }
        if !matches!(group_size, 16 | 32) {
            return Err(MicrofloatError::UnsupportedGroupSize {
                format,
                group_size,
            });
        }
        if layout != MicrofloatLayout::OutputInput {
            return Err(MicrofloatError::UnsupportedLayout(layout));
        }
        if matrix_count == 0 || rows == 0 || columns == 0 {
            return Err(MicrofloatError::EmptyShape);
        }
        if !columns.is_multiple_of(group_size) {
            return Err(MicrofloatError::MisalignedColumns {
                columns,
                group_size,
            });
        }
        let metadata = Self {
            format,
            group_size,
            layout,
            matrix_count,
            rows,
            columns,
        };
        metadata.checked_required_code_bytes().ok_or(MicrofloatError::SizeOverflow)?;
        metadata.checked_required_scale_bytes().ok_or(MicrofloatError::SizeOverflow)?;
        Ok(metadata)
    }

    pub fn format(self) -> MicrofloatFormat {
        self.format
    }

    pub fn bits(self) -> u32 {
        4
    }

    pub fn group_size(self) -> u32 {
        self.group_size
    }

    pub fn layout(self) -> MicrofloatLayout {
        self.layout
    }

    pub fn matrix_count(self) -> u32 {
        self.matrix_count
    }

    pub fn rows(self) -> u32 {
        self.rows
    }

    pub fn columns(self) -> u32 {
        self.columns
    }

    pub fn code_row_stride(self) -> usize {
        self.columns as usize / 2
    }

    pub fn scale_row_stride(self) -> usize {
        self.columns as usize / self.group_size as usize
    }

    pub fn code_matrix_stride(self) -> usize {
        self.checked_code_matrix_stride().expect("MicrofloatMetadata validates code storage size")
    }

    pub fn scale_matrix_stride(self) -> usize {
        self.checked_scale_matrix_stride().expect("MicrofloatMetadata validates scale storage size")
    }

    pub fn required_code_bytes(self) -> usize {
        self.checked_required_code_bytes().expect("MicrofloatMetadata validates code storage size")
    }

    pub fn required_scale_bytes(self) -> usize {
        self.checked_required_scale_bytes().expect("MicrofloatMetadata validates scale storage size")
    }

    fn checked_code_matrix_stride(self) -> Option<usize> {
        (self.rows as usize).checked_mul(self.code_row_stride())
    }

    fn checked_scale_matrix_stride(self) -> Option<usize> {
        (self.rows as usize).checked_mul(self.scale_row_stride())
    }

    fn checked_required_code_bytes(self) -> Option<usize> {
        self.checked_code_matrix_stride()?.checked_mul(self.matrix_count as usize)
    }

    fn checked_required_scale_bytes(self) -> Option<usize> {
        self.checked_scale_matrix_stride()?.checked_mul(self.matrix_count as usize)
    }
}

#[inline]
pub fn decode_e2m1(code: u8) -> f32 {
    const VALUES: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];
    VALUES[usize::from(code & 0x0f)]
}

/// Exact signed-integer representation of one E2M1 nibble at scale 0.5.
#[inline]
pub fn e2m1_to_exact_i8(code: u8) -> i8 {
    const VALUES: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
    VALUES[usize::from(code & 0x0f)]
}

#[inline]
pub fn decode_e8m0(exponent: u8) -> f32 {
    match exponent {
        0 => f32::from_bits(0x0040_0000),
        255 => f32::NAN,
        exponent => f32::from_bits(u32::from(exponent) << 23),
    }
}

#[inline]
pub fn decode_mxfp4(
    code: u8,
    exponent: u8,
    outer_scale: f32,
) -> f32 {
    decode_e2m1(code) * decode_e8m0(exponent) * outer_scale
}

/// FP32 scale paired with [`e2m1_to_exact_i8`] for one MXFP4 block.
#[inline]
pub fn mxfp4_exact_int8_scale(
    exponent: u8,
    outer_scale: f32,
) -> f32 {
    decode_e8m0(exponent) * outer_scale * 0.5
}

/// Proves that one signed dot-product partial fits its INT32 accumulator.
pub fn check_int32_accumulator_bound(
    left_max_abs: u32,
    right_max_abs: u32,
    length: u32,
) -> Result<u32, MicrofloatError> {
    let bound = u128::from(left_max_abs) * u128::from(right_max_abs) * u128::from(length);
    if bound > i32::MAX as u128 {
        return Err(MicrofloatError::Int32AccumulatorOverflow {
            left_max_abs,
            right_max_abs,
            length,
            bound,
        });
    }
    Ok(bound as u32)
}

#[cfg(test)]
mod tests {
    use backend_uzu_macros::uzu_test;

    use super::*;

    #[uzu_test]
    fn decodes_e2m1_and_e8m0_edges() {
        let expected = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for (code, value) in expected.into_iter().enumerate() {
            assert_eq!(decode_e2m1(code as u8), value);
            assert_eq!(decode_e2m1(code as u8 | 8), -value);
        }
        assert_eq!(decode_e8m0(0).to_bits(), 0x0040_0000);
        assert_eq!(decode_e8m0(127), 1.0);
        assert!(decode_e8m0(255).is_nan());
    }

    #[uzu_test]
    fn derives_exact_mxfp4_int8_representation() {
        let expected = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
        for (code, expected) in expected.into_iter().enumerate() {
            assert_eq!(e2m1_to_exact_i8(code as u8), expected);
        }

        for (exponent, outer_scales) in [
            (0, &[1.0, -1.0][..]),
            (1, &[1.0, -1.0, 0.75, -0.75][..]),
            (127, &[1.0, -1.0, 0.75, -0.75][..]),
            (254, &[1.0, -1.0][..]),
        ] {
            for &outer_scale in outer_scales {
                for code in 0..16 {
                    let original = decode_mxfp4(code, exponent, outer_scale);
                    let derived = f32::from(e2m1_to_exact_i8(code)) * mxfp4_exact_int8_scale(exponent, outer_scale);
                    if code == 8 {
                        assert_eq!(original, derived);
                    } else {
                        assert_eq!(original.to_bits(), derived.to_bits(), "code={code}, exponent={exponent}");
                    }
                }
            }
        }
        assert!(mxfp4_exact_int8_scale(255, 1.0).is_nan());
    }

    #[uzu_test]
    fn checks_int32_accumulator_bounds() {
        assert_eq!(check_int32_accumulator_bound(127, 12, 32), Ok(48_768));
        assert!(matches!(
            check_int32_accumulator_bound(127, 127, 1_000_000),
            Err(MicrofloatError::Int32AccumulatorOverflow { .. })
        ));
    }

    #[uzu_test]
    fn validates_supported_mxfp4_layouts() {
        for group_size in [16, 32] {
            let metadata = MicrofloatMetadata::new(
                MicrofloatFormat::Mxfp4,
                4,
                group_size,
                MicrofloatLayout::OutputInput,
                2,
                3,
                32,
            )
            .expect("supported MXFP4 metadata");
            assert_eq!(metadata.required_code_bytes(), 96);
            assert_eq!(metadata.required_scale_bytes(), 2 * 3 * 32 / group_size as usize);
        }
        assert!(matches!(
            MicrofloatMetadata::new(MicrofloatFormat::Nvfp4, 4, 16, MicrofloatLayout::OutputInput, 1, 1, 16),
            Err(MicrofloatError::UnsupportedFormat(MicrofloatFormat::Nvfp4))
        ));
    }
}
