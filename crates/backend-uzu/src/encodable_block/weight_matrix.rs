use std::mem::size_of;

use half::{bf16, f16};
use thiserror::Error;

use crate::{
    backends::common::{
        Allocation, AllocationType, Backend, Context,
        gpu_types::{QuantizationMethod, QuantizationMode},
        kernel::matmul::MatmulB,
        microfloat::{
            MicrofloatFormat, MicrofloatLayout, MicrofloatMetadata, check_int32_accumulator_bound, e2m1_to_exact_i8,
            mxfp4_exact_int8_scale,
        },
    },
    config::weight_matrix::{
        AnyWeightMatrixSpec, Layout,
        microfloat_spec::{MicrofloatScaleMode, MicrofloatSpec},
    },
    data_type::DataType,
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum WeightMatrixError<B: Backend> {
    #[error("Parameter loading error: {0}")]
    ParameterError(#[from] ParameterLoaderError<B>),
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Unsupported weight matrix configuration: {0}")]
    UnsupportedConfiguration(String),
}

#[derive(Clone, Copy)]
pub struct QuantizationInfo {
    pub mode: QuantizationMode,
    pub method: QuantizationMethod,
    pub group_size: u32,
}

#[derive(Clone, Copy)]
pub struct MicrofloatInfo {
    pub format: MicrofloatFormat,
    pub bits: u32,
    pub group_size: u32,
}

pub struct ParsedWeightSpec {
    pub layout: Layout,
    pub quantization: Option<QuantizationInfo>,
    pub microfloat: Option<MicrofloatInfo>,
}

pub fn parse_spec<B: Backend>(spec: &AnyWeightMatrixSpec) -> Result<ParsedWeightSpec, WeightMatrixError<B>> {
    let (layout, quantized, microfloat) = match spec {
        AnyWeightMatrixSpec::FullPrecisionSpec(spec) => (spec.layout.clone(), None, None),
        AnyWeightMatrixSpec::MLXSpec(spec) => {
            (spec.layout.clone(), Some((spec.bits, spec.group_size, QuantizationMethod::ScaleBias)), None)
        },
        AnyWeightMatrixSpec::IntSpec(spec) => (
            spec.layout.clone(),
            Some((
                spec.bits,
                spec.group_size,
                if spec.is_symmetric {
                    QuantizationMethod::ScaleSymmetric
                } else {
                    QuantizationMethod::ScaleZeroPoint
                },
            )),
            None,
        ),
        AnyWeightMatrixSpec::MicrofloatSpec(MicrofloatSpec {
            bits,
            group_size,
            scale_mode,
            layout,
            ..
        }) => {
            let format = match scale_mode {
                MicrofloatScaleMode::Mxfp4 => MicrofloatFormat::Mxfp4,
                MicrofloatScaleMode::Nvfp4 => {
                    return Err(WeightMatrixError::UnsupportedConfiguration(
                        "NVFP4 runtime storage is not supported".into(),
                    ));
                },
            };
            let group_size = u32::try_from(*group_size).map_err(|_| {
                WeightMatrixError::UnsupportedConfiguration(format!("microfloat group size {group_size} exceeds u32"))
            })?;
            let runtime_layout = match layout {
                Layout::OutputInput => MicrofloatLayout::OutputInput,
                Layout::InputOutput => MicrofloatLayout::InputOutput,
            };
            MicrofloatMetadata::new(format, *bits, group_size, runtime_layout, 1, 1, group_size)
                .map_err(|error| WeightMatrixError::UnsupportedConfiguration(error.to_string()))?;
            (
                layout.clone(),
                None,
                Some(MicrofloatInfo {
                    format,
                    bits: *bits,
                    group_size,
                }),
            )
        },
        spec => return Err(WeightMatrixError::UnsupportedConfiguration(format!("{spec:?}"))),
    };
    let quantization = match quantized {
        None => None,
        Some((bits, group_size, method)) => {
            let mode = match bits {
                4 => QuantizationMode::U4,
                8 => QuantizationMode::U8,
                _ => {
                    return Err(WeightMatrixError::UnsupportedConfiguration(format!(
                        "{method} bits={bits}, group_size={group_size}"
                    )));
                },
            };
            if group_size == 0 {
                return Err(WeightMatrixError::UnsupportedConfiguration("group size must be non-zero".into()));
            }
            Some(QuantizationInfo {
                mode,
                method,
                group_size,
            })
        },
    };
    Ok(ParsedWeightSpec {
        layout,
        quantization,
        microfloat,
    })
}

enum QuantizedCorrection<B: Backend> {
    Symmetric,
    Biases(Allocation<B>),
    ZeroPoints(Allocation<B>),
}

struct Quantized<B: Backend> {
    scales: Allocation<B>,
    correction: QuantizedCorrection<B>,
    info: QuantizationInfo,
    signed_codes: bool,
}

struct Microfloat<B: Backend> {
    scales: Allocation<B>,
    outer_scales: Allocation<B>,
    #[allow(dead_code)]
    outer_scale_data_type: DataType,
    metadata: MicrofloatMetadata,
}

/// Storage expansion incurred by the resident INT8 TensorOps representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub struct DerivedInt8BankStatistics {
    pub source_code_bytes: usize,
    pub source_scale_bytes: usize,
    pub derived_code_bytes: usize,
    pub derived_scale_bytes: usize,
    pub group_count: usize,
}

/// Exact group-32 INT8 representation derived from canonical MXFP4 storage.
#[allow(dead_code)]
pub struct DerivedInt8Bank<B: Backend> {
    codes: Allocation<B>,
    scales: Allocation<B>,
    group_size: u32,
    statistics: DerivedInt8BankStatistics,
}

#[allow(dead_code)]
impl<B: Backend> DerivedInt8Bank<B> {
    pub fn codes(&self) -> &Allocation<B> {
        &self.codes
    }

    pub fn scales(&self) -> &Allocation<B> {
        &self.scales
    }

    pub fn group_size(&self) -> u32 {
        self.group_size
    }

    pub fn statistics(&self) -> DerivedInt8BankStatistics {
        self.statistics
    }
}

pub struct WeightMatrix<B: Backend> {
    values: Allocation<B>,
    quantized: Option<Quantized<B>>,
    microfloat: Option<Microfloat<B>>,
}

impl<B: Backend> WeightMatrix<B> {
    pub fn load(
        tree: &ParameterTree<B>,
        spec: AnyWeightMatrixSpec,
        required_layout: Layout,
        output_dim: u32,
        input_dim: u32,
        data_type: DataType,
    ) -> Result<Self, WeightMatrixError<B>> {
        Self::load_impl(tree, spec, required_layout, 1, output_dim, input_dim, data_type, false)
    }

    pub fn load_bank(
        tree: &ParameterTree<B>,
        spec: AnyWeightMatrixSpec,
        required_layout: Layout,
        matrix_count: u32,
        output_dim: u32,
        input_dim: u32,
        data_type: DataType,
    ) -> Result<Self, WeightMatrixError<B>> {
        Self::load_impl(tree, spec, required_layout, matrix_count, output_dim, input_dim, data_type, true)
    }

    #[allow(clippy::too_many_arguments)]
    fn load_impl(
        tree: &ParameterTree<B>,
        spec: AnyWeightMatrixSpec,
        required_layout: Layout,
        matrix_count: u32,
        output_dim: u32,
        input_dim: u32,
        data_type: DataType,
        banked: bool,
    ) -> Result<Self, WeightMatrixError<B>> {
        let ParsedWeightSpec {
            layout,
            quantization,
            microfloat,
        } = parse_spec(&spec)?;
        if layout != required_layout {
            return Err(WeightMatrixError::UnsupportedConfiguration(format!(
                "expected {required_layout:?} layout, got {layout:?}"
            )));
        }
        let (rows, columns) = physical_shape(&layout, output_dim, input_dim);

        if let Some(info) = microfloat {
            let runtime_layout = match layout {
                Layout::OutputInput => MicrofloatLayout::OutputInput,
                Layout::InputOutput => MicrofloatLayout::InputOutput,
            };
            let metadata = MicrofloatMetadata::new(
                info.format,
                info.bits,
                info.group_size,
                runtime_layout,
                matrix_count,
                rows,
                columns,
            )
            .map_err(|error| WeightMatrixError::UnsupportedConfiguration(error.to_string()))?;
            let values = tree
                .leaf("weights")?
                .validate(&bank_shape(matrix_count, &[rows, columns / 2], banked), DataType::U8)?
                .read_allocation()?;
            let scales = tree
                .leaf("scales")?
                .validate(&bank_shape(matrix_count, &[rows, columns / info.group_size], banked), DataType::U8)?
                .read_allocation()?;
            let outer_scale_shape = if banked {
                vec![matrix_count]
            } else {
                vec![1]
            };
            // Preserve the converter-facing `global_scale` tensor name while
            // describing its runtime role as one outer scale per matrix.
            let outer_scales = tree.leaf("global_scale")?.validate(&outer_scale_shape, data_type)?.read_allocation()?;
            return Ok(Self {
                values,
                quantized: None,
                microfloat: Some(Microfloat {
                    scales,
                    outer_scales,
                    outer_scale_data_type: data_type,
                    metadata,
                }),
            });
        }

        let Some(info) = quantization else {
            let values = tree
                .leaf("weights")?
                .validate(&bank_shape(matrix_count, &[rows, columns], banked), data_type)?
                .read_allocation()?;
            return Ok(Self {
                values,
                quantized: None,
                microfloat: None,
            });
        };

        let group_size = info.group_size;
        let packing_divisor = info.mode.packing_divisor();
        let storage_data_type = info.mode.storage_type();
        if !columns.is_multiple_of(packing_divisor) {
            return Err(WeightMatrixError::UnsupportedConfiguration(format!(
                "stored columns {columns} are not divisible by packing divisor {packing_divisor}"
            )));
        }
        let groups = columns.div_ceil(group_size);
        let values = tree
            .leaf("weights")?
            .validate(&bank_shape(matrix_count, &[rows, columns / packing_divisor], banked), storage_data_type)?
            .read_allocation()?;
        let scales = tree
            .leaf("scales")?
            .validate(&bank_shape(matrix_count, &[rows, groups], banked), data_type)?
            .read_allocation()?;
        let correction = match info.method {
            QuantizationMethod::ScaleBias => QuantizedCorrection::Biases(
                tree.leaf("biases")?
                    .validate(&bank_shape(matrix_count, &[rows, groups], banked), data_type)?
                    .read_allocation()?,
            ),
            QuantizationMethod::ScaleZeroPoint => QuantizedCorrection::ZeroPoints(
                tree.leaf("zero_points")?
                    .validate(
                        &bank_shape(matrix_count, &[rows, groups.div_ceil(packing_divisor)], banked),
                        storage_data_type,
                    )?
                    .read_allocation()?,
            ),
            QuantizationMethod::ScaleSymmetric => QuantizedCorrection::Symmetric,
        };
        Ok(Self {
            values,
            quantized: Some(Quantized {
                scales,
                correction,
                info,
                signed_codes: false,
            }),
            microfloat: None,
        })
    }

    pub fn values(&self) -> &Allocation<B> {
        &self.values
    }

    pub fn quantization(&self) -> Option<QuantizationInfo> {
        self.quantized.as_ref().map(|quantized| quantized.info)
    }

    pub fn scales(&self) -> Option<&Allocation<B>> {
        self.microfloat
            .as_ref()
            .map(|microfloat| &microfloat.scales)
            .or_else(|| self.quantized.as_ref().map(|quantized| &quantized.scales))
    }

    pub fn zero_points(&self) -> Option<&Allocation<B>> {
        match &self.quantized.as_ref()?.correction {
            QuantizedCorrection::ZeroPoints(zero_points) => Some(zero_points),
            QuantizedCorrection::Biases(_) | QuantizedCorrection::Symmetric => None,
        }
    }

    pub fn biases(&self) -> Option<&Allocation<B>> {
        match &self.quantized.as_ref()?.correction {
            QuantizedCorrection::Biases(biases) => Some(biases),
            QuantizedCorrection::ZeroPoints(_) | QuantizedCorrection::Symmetric => None,
        }
    }

    pub fn matmul_b(&self) -> MatmulB<'_, B> {
        if let Some(microfloat) = self.microfloat.as_ref() {
            return MatmulB::Microfloat {
                codes: &self.values,
                scales: &microfloat.scales,
                outer_scales: &microfloat.outer_scales,
                metadata: microfloat.metadata,
            };
        }
        let Some(quantized) = self.quantized.as_ref() else {
            return MatmulB::FullPrecision {
                b: &self.values,
            };
        };
        let mode = quantized.info.mode;
        let group_size = quantized.info.group_size;
        let signed_codes = quantized.signed_codes;
        match &quantized.correction {
            QuantizedCorrection::Biases(biases) => MatmulB::ScaleBiasDequant {
                b: &self.values,
                scales: &quantized.scales,
                biases,
                mode,
                group_size,
                signed_codes,
            },
            QuantizedCorrection::ZeroPoints(zero_points) => MatmulB::ScaleZeroPointDequant {
                b: &self.values,
                scales: &quantized.scales,
                zero_points,
                mode,
                group_size,
                signed_codes,
            },
            QuantizedCorrection::Symmetric => MatmulB::ScaleSymmetricDequant {
                b: &self.values,
                scales: &quantized.scales,
                mode,
                group_size,
                signed_codes,
            },
        }
    }

    /// Expand canonical group-32 MXFP4 into the resident TensorOps layout.
    #[allow(dead_code)]
    pub fn materialize_mxfp4_int8_bank(
        &self,
        context: &B::Context,
    ) -> Result<DerivedInt8Bank<B>, WeightMatrixError<B>> {
        let Some(microfloat) = self.microfloat.as_ref() else {
            return Err(WeightMatrixError::UnsupportedConfiguration(
                "derived INT8 banks require canonical MXFP4 storage".into(),
            ));
        };
        let metadata = microfloat.metadata;
        if metadata.format() != MicrofloatFormat::Mxfp4 || metadata.group_size() != 32 {
            return Err(WeightMatrixError::UnsupportedConfiguration(
                "resident INT8 TensorOps require group-32 E8M0 MXFP4 weights".into(),
            ));
        }
        check_int32_accumulator_bound(127, 12, 32)
            .map_err(|error| WeightMatrixError::UnsupportedConfiguration(error.to_string()))?;
        let source_codes = self.values.as_slice::<u8>();
        let source_scales = microfloat.scales.as_slice::<u8>();
        if source_scales.contains(&255) {
            return Err(WeightMatrixError::UnsupportedConfiguration(
                "cannot materialize an INT8 bank containing invalid E8M0 exponent 255".into(),
            ));
        }

        let outer_scale = |matrix: usize| -> Result<f32, WeightMatrixError<B>> {
            let scale = match microfloat.outer_scale_data_type {
                DataType::F16 => microfloat.outer_scales.as_slice::<f16>()[matrix].to_f32(),
                DataType::BF16 => microfloat.outer_scales.as_slice::<bf16>()[matrix].to_f32(),
                DataType::F32 => microfloat.outer_scales.as_slice::<f32>()[matrix],
                data_type => {
                    return Err(WeightMatrixError::UnsupportedConfiguration(format!(
                        "MXFP4 outer scale type {data_type:?} cannot be converted to an INT8 bank"
                    )));
                },
            };
            if !scale.is_finite() {
                return Err(WeightMatrixError::UnsupportedConfiguration(
                    "MXFP4 outer scales must be finite for an INT8 bank".into(),
                ));
            }
            Ok(scale)
        };

        let code_count = metadata
            .required_code_bytes()
            .checked_mul(2)
            .ok_or_else(|| WeightMatrixError::UnsupportedConfiguration("derived code size overflows usize".into()))?;
        let scale_count = metadata.required_scale_bytes();
        let mut codes =
            context.create_allocation(code_count, AllocationType::Global).map_err(WeightMatrixError::BackendError)?;
        let scale_bytes = scale_count
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| WeightMatrixError::UnsupportedConfiguration("derived scale size overflows usize".into()))?;
        let mut scales =
            context.create_allocation(scale_bytes, AllocationType::Global).map_err(WeightMatrixError::BackendError)?;

        let codes_out = codes.as_slice_mut::<i8>();
        let scales_out = scales.as_slice_mut::<f32>();
        let rows = metadata.rows() as usize;
        let columns = metadata.columns() as usize;
        let groups_per_row = metadata.scale_row_stride();
        for matrix in 0..metadata.matrix_count() as usize {
            let outer_scale = outer_scale(matrix)?;
            for row in 0..rows {
                let code_row = matrix * metadata.code_matrix_stride() + row * metadata.code_row_stride();
                let output_row = (matrix * rows + row) * columns;
                for column in 0..columns {
                    let packed = source_codes[code_row + column / 2];
                    let code = if column.is_multiple_of(2) {
                        packed & 0x0f
                    } else {
                        packed >> 4
                    };
                    codes_out[output_row + column] = e2m1_to_exact_i8(code);
                }

                let scale_row = matrix * metadata.scale_matrix_stride() + row * groups_per_row;
                let output_scale_row = (matrix * rows + row) * groups_per_row;
                for group in 0..groups_per_row {
                    scales_out[output_scale_row + group] =
                        mxfp4_exact_int8_scale(source_scales[scale_row + group], outer_scale);
                }
            }
        }

        Ok(DerivedInt8Bank {
            codes,
            scales,
            group_size: metadata.group_size(),
            statistics: DerivedInt8BankStatistics {
                source_code_bytes: metadata.required_code_bytes(),
                source_scale_bytes: metadata.required_scale_bytes()
                    + metadata.matrix_count() as usize * microfloat.outer_scale_data_type.size_in_bytes(),
                derived_code_bytes: code_count,
                derived_scale_bytes: scale_bytes,
                group_count: scale_count,
            },
        })
    }

    pub fn make_codes_signed(&mut self) {
        let Some(quantized) = self.quantized.as_mut() else {
            return;
        };
        if quantized.signed_codes {
            return;
        }
        let Some(sign_flip_mask) = quantized.info.mode.weight_codes_sign_flip_mask() else {
            return;
        };
        let broadcast_mask = u64::from(sign_flip_mask) * 0x0101_0101_0101_0101;
        let (prefix, words, suffix) = bytemuck::pod_align_to_mut::<u8, u64>(self.values.as_slice_mut());
        words.iter_mut().for_each(|word| *word ^= broadcast_mask);
        prefix.iter_mut().chain(suffix.iter_mut()).for_each(|code| *code ^= sign_flip_mask);
        quantized.signed_codes = true;
    }
}

fn physical_shape(
    layout: &Layout,
    output_dim: u32,
    input_dim: u32,
) -> (u32, u32) {
    match layout {
        Layout::OutputInput => (output_dim, input_dim),
        Layout::InputOutput => (input_dim, output_dim),
    }
}

fn bank_shape(
    matrix_count: u32,
    matrix_shape: &[u32],
    banked: bool,
) -> Vec<u32> {
    if !banked {
        return matrix_shape.to_vec();
    }
    std::iter::once(matrix_count).chain(matrix_shape.iter().copied()).collect()
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use half::bf16;
    use proc_macros::uzu_test;
    use serde_json::{Map, Value, json};
    use tempfile::NamedTempFile;

    use super::*;
    use crate::{
        backends::{common::Context, cpu::Cpu},
        parameters::ParameterLoader,
    };

    fn add_tensor(
        header: &mut Map<String, Value>,
        payload: &mut Vec<u8>,
        name: &str,
        shape: &[u32],
        data_type: &str,
        data: &[u8],
    ) {
        let begin = payload.len();
        payload.extend_from_slice(data);
        header.insert(
            name.into(),
            json!({
                "dtype": data_type,
                "shape": shape,
                "data_offsets": [begin, payload.len()]
            }),
        );
    }

    /// Pristine dense MXFP4 artifact at the `WeightMatrix` loading boundary.
    fn dense_microfloat_parameter_file() -> NamedTempFile {
        const ROWS: u32 = 2;
        const COLUMNS: u32 = 32;
        const GROUP_SIZE: u32 = 16;

        let mut header = Map::new();
        header.insert(
            "__metadata__".into(),
            json!({
                "spec": json!({
                    "type": "MicrofloatSpec",
                    "bits": 4,
                    "group_size": GROUP_SIZE,
                    "scale_mode": "mxfp4",
                    "layout": "output_input"
                }).to_string()
            }),
        );
        let mut payload = Vec::new();
        let codes: Vec<u8> = (0..ROWS * COLUMNS / 2).map(|index| 0x10 | (index % 8) as u8).collect();
        let scales: Vec<u8> = (0..ROWS * COLUMNS / GROUP_SIZE).map(|index| 126 + (index % 3) as u8).collect();
        let global_scale = bf16::from_f32(0.75).to_le_bytes();
        add_tensor(&mut header, &mut payload, "weights", &[ROWS, COLUMNS / 2], "U8", &codes);
        add_tensor(&mut header, &mut payload, "scales", &[ROWS, COLUMNS / GROUP_SIZE], "U8", &scales);
        add_tensor(&mut header, &mut payload, "global_scale", &[1], "BF16", &global_scale);

        let mut header = serde_json::to_vec(&Value::Object(header)).expect("serialize safetensors header");
        header.extend(std::iter::repeat_n(b' ', (8 - header.len() % 8) % 8));
        let mut file = NamedTempFile::new().expect("create dense MXFP4 fixture");
        file.write_all(&(header.len() as u64).to_le_bytes()).expect("write safetensors header length");
        file.write_all(&header).expect("write safetensors header");
        file.write_all(&payload).expect("write safetensors payload");
        file
    }

    #[uzu_test]
    fn loads_dense_microfloat_without_expert_routes() {
        let context = <Cpu as Backend>::Context::new().expect("create CPU context");
        let file = dense_microfloat_parameter_file();
        let loader = ParameterLoader::<Cpu>::new(file.as_file(), context.as_ref()).expect("load dense MXFP4 fixture");
        let tree = loader.tree();
        let spec = tree.metadata::<AnyWeightMatrixSpec>("spec").expect("load dense MXFP4 spec");

        let matrix = WeightMatrix::load(&tree, spec, Layout::OutputInput, 2, 32, DataType::BF16)
            .expect("load dense MXFP4 matrix");
        let MatmulB::Microfloat {
            metadata,
            ..
        } = matrix.matmul_b()
        else {
            panic!("dense MXFP4 matrix did not preserve its operand format");
        };
        assert_eq!(metadata.matrix_count(), 1);
        assert_eq!(metadata.rows(), 2);
        assert_eq!(metadata.columns(), 32);
        tree.assert_all_tensors_validated().expect("validate dense MXFP4 tensors");
    }
}

#[cfg(test)]
#[path = "../../tests/unit/encodable_block/weight_matrix_int8_test.rs"]
mod int8_tests;
