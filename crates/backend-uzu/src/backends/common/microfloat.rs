use thiserror::Error;

/// Encoding of the one-byte scale applied to each packed E2M1 group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MicrofloatScaleFormat {
    E8m0,
    E4m3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MicrofloatLayout {
    OutputInput,
    InputOutput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum MicrofloatError {
    #[error("unsupported microfloat bit width: {0}")]
    UnsupportedBits(u32),
    #[error("unsupported microfloat group size: {0}")]
    UnsupportedGroupSize(u32),
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
}

/// Packed E2M1 values with per-group scales.
///
/// Scale encoding and group size are independent storage dimensions. OCP
/// MXFP4 is group-32 E8M0 and NVFP4 is group-16 E4M3, while converters may
/// intentionally emit combinations such as group-16 E8M0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MicrofloatMetadata {
    scale_format: MicrofloatScaleFormat,
    group_size: u32,
    layout: MicrofloatLayout,
    matrix_count: u32,
    rows: u32,
    columns: u32,
}

impl MicrofloatMetadata {
    pub fn new(
        scale_format: MicrofloatScaleFormat,
        bits: u32,
        group_size: u32,
        layout: MicrofloatLayout,
        matrix_count: u32,
        rows: u32,
        columns: u32,
    ) -> Result<Self, MicrofloatError> {
        if bits != 4 {
            return Err(MicrofloatError::UnsupportedBits(bits));
        }
        if !matches!(group_size, 16 | 32) {
            return Err(MicrofloatError::UnsupportedGroupSize(group_size));
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
            scale_format,
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

    pub fn scale_format(self) -> MicrofloatScaleFormat {
        self.scale_format
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

#[inline]
pub fn decode_e8m0(exponent: u8) -> f32 {
    match exponent {
        0 => f32::from_bits(0x0040_0000),
        255 => f32::NAN,
        exponent => f32::from_bits(u32::from(exponent) << 23),
    }
}

/// NVIDIA E4M3: 1 sign, 4 exponent (bias 7), 3 mantissa. No infinities;
/// exponent 15 with mantissa 7 is NaN, and the finite maximum is 448.
#[inline]
pub fn decode_e4m3(bits: u8) -> f32 {
    let sign = u32::from(bits >> 7) << 31;
    let exponent = (bits >> 3) & 0x0f;
    let mantissa = bits & 0x07;
    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign);
        }
        let value = f32::from(mantissa) / 512.0;
        return f32::from_bits(sign | value.to_bits());
    }
    if exponent == 15 && mantissa == 7 {
        return f32::from_bits(sign | 0x7fc0_0000);
    }
    f32::from_bits(sign | ((u32::from(exponent) + 120) << 23) | (u32::from(mantissa) << 20))
}

impl MicrofloatScaleFormat {
    /// Decode one stored group scale.
    #[inline]
    pub fn decode(
        self,
        scale: u8,
    ) -> f32 {
        match self {
            Self::E8m0 => decode_e8m0(scale),
            Self::E4m3 => decode_e4m3(scale),
        }
    }
}

#[cfg(test)]
mod tests {
    use proc_macros::uzu_test;

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
    fn decodes_e4m3_edges() {
        assert_eq!(decode_e4m3(0x00).to_bits(), 0.0f32.to_bits());
        assert_eq!(decode_e4m3(0x80).to_bits(), (-0.0f32).to_bits());
        assert_eq!(decode_e4m3(0x01), 1.0 / 512.0);
        assert_eq!(decode_e4m3(0x38), 1.0);
        assert_eq!(decode_e4m3(0x3c), 1.5);
        assert_eq!(decode_e4m3(0x40), 2.0);
        assert_eq!(decode_e4m3(0xb8), -1.0);
        assert_eq!(decode_e4m3(0x7e), 448.0);
        assert!(decode_e4m3(0x7f).is_nan());
        assert_ne!(MicrofloatScaleFormat::E4m3.decode(0x38), MicrofloatScaleFormat::E8m0.decode(0x38));
    }

    #[uzu_test]
    fn validates_supported_microfloat_layouts() {
        for group_size in [16, 32] {
            let metadata = MicrofloatMetadata::new(
                MicrofloatScaleFormat::E8m0,
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
        for group_size in [16, 32] {
            let e4m3 = MicrofloatMetadata::new(
                MicrofloatScaleFormat::E4m3,
                4,
                group_size,
                MicrofloatLayout::OutputInput,
                2,
                3,
                32,
            )
            .expect("supported E4M3-scaled metadata");
            assert_eq!(e4m3.required_code_bytes(), 96);
            assert_eq!(e4m3.required_scale_bytes(), 2 * 3 * 32 / group_size as usize);
        }
    }
}
