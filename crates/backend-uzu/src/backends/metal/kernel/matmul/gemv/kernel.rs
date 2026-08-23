use std::collections::{HashMap, hash_map::Entry};

use super::{
    mxfp4_expert_decode::{Mxfp4ExpertDecodeGemvDispatch, Mxfp4ExpertDecodeGemvSpec},
    policy::{self, DEFAULT_RESULTS_PER_SIMDGROUP, FP_K_BLOCK},
    resident_int8_expert_tensorops::ResidentInt8ExpertTensorOpsDispatch,
};
use crate::{
    backends::{
        common::{
            Allocation, BufferArg, Encoder,
            gpu_types::{
                HADAMARD_TRANSFORM_BLOCK_SIZE, QuantizationMode,
                gemm::{GemmBPrologueKind, GemmDTransform},
            },
            kernel::{
                ActivationTransform,
                matmul::{ExpertInput, GateActMulDOps, MatmulA, MatmulArguments, MatmulB, MatmulError, MatmulShape},
            },
        },
        metal::{Int8Execution, Metal, context::MetalContext, device_profile::DeviceProfile, kernel::GemvMetalKernel},
    },
    data_type::DataType,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GemvSpecialization {
    b_prologue: GemmBPrologueKind,
    group_size: u32,
    bits: u32,
    output_transform: GemmDTransform,
    input_aligned: bool,
    k_split: u32,
    results_per_simdgroup: u32,
    num_simdgroups: u32,
    microfloat: bool,
    gathered: bool,
    expert_routed: bool,
    expert_bias: bool,
    signed_codes: bool,
}

impl GemvSpecialization {
    pub(crate) fn select_shape(
        shape: &MatmulShape,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        device_profile: DeviceProfile,
        max_batch: u32,
    ) -> Option<GemvSpecialization> {
        if !shape.b_transpose || !shape.a_full_precision {
            return None;
        }
        let is_quant = shape.is_integer_quantized();
        let microfloat = shape.b_microfloat.is_some();
        let bad_leading_dimension = if is_quant {
            shape.b_leading_dimension.is_some()
        } else {
            shape.b_leading_dimension.is_some_and(|ld| ld != shape.k)
        };
        if bad_leading_dimension {
            return None;
        }
        if shape.d_transform.contains(GemmDTransform::ACCUMULATE) && !shape.n.is_multiple_of(32) {
            return None;
        }
        if shape.d_transform.contains(GemmDTransform::RHT) && !shape.n.is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE) {
            return None;
        }
        // Sparse MXFP4 has no gathered GEMM arm, including one-to-three-row readouts.
        let sparse_microfloat_readout = shape.sparse_readout && microfloat;
        if !shape.expert_routed && shape.n < DEFAULT_RESULTS_PER_SIMDGROUP && !sparse_microfloat_readout {
            return None;
        }
        // Integer expert banks have no grouped Metal arm yet. Preserve the
        // independent storage/routing contract with a correct GEMV fallback,
        // even for prefill-sized M; this is not a claim of prefill parity.
        let long_integer_expert_bank = shape.expert_routed && is_quant;
        // Sparse MXFP4 readout has no gathered GEMM arm. Keep it on the only
        // implementation that honors its physical B-row map at every M.
        if shape.m > max_batch && !long_integer_expert_bank && !sparse_microfloat_readout {
            return None;
        }
        if !is_quant {
            let mixed_precision = weights_data_type == DataType::F32
                && (input_data_type != DataType::F32 || output_data_type != DataType::F32);
            if mixed_precision {
                return None;
            }
        }

        let bits = shape.b_bits.unwrap_or(0);
        let block_size = if !is_quant {
            FP_K_BLOCK
        } else if bits == 4 {
            512
        } else {
            256
        };
        let input_aligned = shape.k.is_multiple_of(block_size);
        let has_rht = shape.d_transform.contains(GemmDTransform::RHT);
        let bf16_io = input_data_type == DataType::BF16 && output_data_type == DataType::BF16;
        let tile = if is_quant && bf16_io {
            policy::quant_tile(shape.m, shape.n, shape.k, bits, has_rht, device_profile)
        } else if is_quant || has_rht {
            // Non-bf16 quant IO and fp+RHT keep the default tile (the only
            // one instantiated for those modes).
            policy::DEFAULT_TILE
        } else {
            policy::fp_tile(shape.m, shape.n, shape.k, input_aligned, device_profile)
        };
        Some(Self {
            b_prologue: shape.b_prologue,
            group_size: shape.b_group_size.unwrap_or(0),
            bits,
            output_transform: shape.d_transform,
            input_aligned,
            k_split: tile.k_split,
            results_per_simdgroup: tile.results_per_simdgroup,
            num_simdgroups: tile.num_simdgroups,
            microfloat,
            gathered: shape.sparse_readout,
            expert_routed: shape.expert_routed,
            expert_bias: shape.expert_bias,
            signed_codes: shape.signed_codes,
        })
    }
}

fn rows_per_threadgroup(
    k_split: u32,
    results_per_simdgroup: u32,
    num_simdgroups: u32,
) -> u32 {
    (num_simdgroups / k_split) * results_per_simdgroup
}

pub(crate) struct GemvDispatch {
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
    pipelines: HashMap<GemvSpecialization, GemvMetalKernel>,
    mxfp4_expert_decode: Mxfp4ExpertDecodeGemvDispatch,
    resident_int8: ResidentInt8ExpertTensorOpsDispatch,
    resident_int8_quantizer: ActivationTransform<Metal>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GemvPlan {
    Generic(GemvSpecialization),
    Mxfp4ExpertDecode(Mxfp4ExpertDecodeGemvSpec),
    ResidentInt8(Int8Execution),
}

impl From<GemvSpecialization> for GemvPlan {
    fn from(specialization: GemvSpecialization) -> Self {
        Self::Generic(specialization)
    }
}

impl GemvDispatch {
    pub(crate) fn new(
        context: &MetalContext,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Result<Self, MatmulError<Metal>> {
        let resident_int8_quantizer = ActivationTransform::quantize_symmetric_plain(context, input_data_type, 32)
            .map_err(MatmulError::BackendError)?;
        Ok(Self {
            weights_data_type,
            input_data_type,
            output_data_type,
            pipelines: HashMap::new(),
            mxfp4_expert_decode: Mxfp4ExpertDecodeGemvDispatch::new(
                weights_data_type,
                input_data_type,
                output_data_type,
            ),
            resident_int8: ResidentInt8ExpertTensorOpsDispatch::new(
                weights_data_type,
                weights_data_type,
                output_data_type,
            ),
            resident_int8_quantizer,
        })
    }

    pub(crate) fn selects_resident_int8(
        shape: &MatmulShape,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        ab_scale: f32,
        gate_act: Option<GateActMulDOps>,
        max_batch: u32,
    ) -> bool {
        let supported_transform = shape.d_transform.difference(GemmDTransform::BIAS).is_empty();
        shape.expert_routed
            && !shape.sparse_readout
            && shape.a_full_precision
            && shape.b_transpose
            && shape.b_leading_dimension.is_none()
            && shape.b_prologue == GemmBPrologueKind::ScaleSymmetricDequant
            && shape.b_bits == Some(8)
            && shape.b_group_size == Some(32)
            && shape.signed_codes
            && shape.m <= max_batch
            && shape.n.is_multiple_of(32)
            && shape.k.is_multiple_of(32)
            && matches!(weights_data_type, DataType::BF16 | DataType::F32)
            && matches!(input_data_type, DataType::BF16 | DataType::F32)
            && matches!(output_data_type, DataType::BF16 | DataType::F32)
            && ab_scale == 1.0
            && gate_act.is_none()
            && supported_transform
    }

    /// Selects one implementation inside the GEMV family.
    pub(crate) fn select(
        shape: &MatmulShape,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        ab_scale: f32,
        gate_act: Option<GateActMulDOps>,
        device_profile: DeviceProfile,
        max_batch: u32,
        int8_execution: Int8Execution,
    ) -> Option<GemvPlan> {
        if Self::selects_resident_int8(
            shape,
            weights_data_type,
            input_data_type,
            output_data_type,
            ab_scale,
            gate_act,
            max_batch,
        ) {
            return Some(GemvPlan::ResidentInt8(int8_execution));
        }
        if let Some(spec) = Mxfp4ExpertDecodeGemvSpec::select(
            shape,
            weights_data_type,
            input_data_type,
            output_data_type,
            ab_scale,
            gate_act,
            device_profile,
        ) {
            return Some(GemvPlan::Mxfp4ExpertDecode(spec));
        }
        GemvSpecialization::select_shape(
            shape,
            weights_data_type,
            input_data_type,
            output_data_type,
            device_profile,
            max_batch,
        )
        .map(GemvPlan::Generic)
    }

    pub(crate) fn supports_fused_gate_act(
        shape: &MatmulShape,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        device_profile: DeviceProfile,
    ) -> bool {
        Mxfp4ExpertDecodeGemvSpec::select(
            shape,
            weights_data_type,
            input_data_type,
            output_data_type,
            1.0,
            Some(GateActMulDOps {
                activation_alpha: None,
                gate_clipping: None,
                value_clipping: None,
            }),
            device_profile,
        )
        .is_some()
    }

    pub(crate) fn encode<'a, 'b, 'd, TB: BufferArg<'b, Metal>>(
        &mut self,
        arguments: MatmulArguments<'a, 'b, 'd, Metal, TB>,
        plan: impl Into<GemvPlan>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        match plan.into() {
            GemvPlan::Mxfp4ExpertDecode(spec) => self.mxfp4_expert_decode.encode(arguments, spec, encoder),
            GemvPlan::ResidentInt8(execution) => self.encode_resident_int8(arguments, execution, encoder),
            GemvPlan::Generic(specialization) => {
                if arguments.d_transform.gate_act.is_some() {
                    return Err(MatmulError::UnsupportedDOp {
                        bit: GemmDTransform::GATE_ACT_MUL,
                        path: "Gemv",
                    });
                }
                self.encode_generic(arguments, specialization, encoder)
            },
        }
    }

    fn encode_resident_int8<'a, 'b, 'd, TB: BufferArg<'b, Metal>>(
        &mut self,
        arguments: MatmulArguments<'a, 'b, 'd, Metal, TB>,
        execution: Int8Execution,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        let MatmulArguments {
            a,
            b,
            d,
            routing,
            m,
            n,
            k,
            d_transform,
            ..
        } = arguments;
        let MatmulA::FullPrecision {
            values: activations,
            offset: 0,
        } = a
        else {
            return Err(MatmulError::IncompatibleA {
                path: "ResidentInt8TensorOps",
                reason: "resident INT8 requires unshifted full-precision activations",
            });
        };
        let MatmulB::ScaleSymmetricDequant {
            b: weight_codes,
            scales: weight_scales,
            mode: QuantizationMode::I8,
            group_size: 32,
            signed_codes: true,
        } = b
        else {
            return Err(MatmulError::UnsupportedRouting {
                path: "ResidentInt8TensorOps",
                reason: "resident INT8 requires native signed symmetric I8 group-32 expert weights",
            });
        };
        let Some(routes) = routing.expert_routes() else {
            return Err(MatmulError::UnsupportedRouting {
                path: "ResidentInt8TensorOps",
                reason: "resident INT8 requires direct expert routes",
            });
        };
        let activation_rows = match routes.input {
            ExpertInput::Routes => m,
            ExpertInput::Tokens => m / routes.routes_per_token.get(),
        };
        let mut activation_codes = encoder
            .allocate_scratch_for_shape(&[activation_rows, k], DataType::I8)
            .map_err(MatmulError::BackendError)?;
        let mut activation_scales = encoder
            .allocate_scratch_for_shape(&[activation_rows, k / 32], DataType::F32)
            .map_err(MatmulError::BackendError)?;
        self.resident_int8_quantizer.encode_quantize_symmetric_plain(
            activations,
            &mut activation_codes,
            &mut activation_scales,
            activation_rows,
            k,
            encoder,
        );
        self.resident_int8
            .encode(
                weight_codes,
                weight_scales,
                &activation_codes,
                &activation_scales,
                &mut *d,
                d_transform.per_matrix_bias,
                routes.expert_ids,
                k,
                n,
                m,
                routes.routes_per_token,
                routes.expert_count,
                routes.input,
                execution,
                encoder,
            )
            .map_err(MatmulError::BackendError)
    }

    fn get_or_create(
        &mut self,
        context: &MetalContext,
        specialization: GemvSpecialization,
    ) -> Result<&GemvMetalKernel, MatmulError<Metal>> {
        match self.pipelines.entry(specialization) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let kernel = GemvMetalKernel::new(
                    context,
                    self.input_data_type,
                    self.weights_data_type,
                    self.output_data_type,
                    specialization.b_prologue,
                    specialization.group_size,
                    specialization.bits,
                    specialization.k_split,
                    specialization.input_aligned,
                    specialization.results_per_simdgroup,
                    specialization.num_simdgroups,
                    specialization.microfloat,
                    specialization.output_transform,
                    specialization.gathered,
                    specialization.expert_routed,
                    specialization.expert_bias,
                    specialization.signed_codes,
                )
                .map_err(MatmulError::BackendError)?;
                Ok(entry.insert(kernel))
            },
        }
    }

    fn encode_generic<'a, 'b, 'd, TB: BufferArg<'b, Metal>>(
        &mut self,
        arguments: MatmulArguments<'a, 'b, 'd, Metal, TB>,
        specialization: GemvSpecialization,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        let ab_scale = arguments.d_transform.ab_scale;
        let output_bias = arguments.d_transform.bias;
        let per_matrix_bias = arguments.d_transform.per_matrix_bias;
        let rht_factors = arguments.d_transform.rht_factors;
        let soft_cap = arguments.d_transform.soft_cap;

        let MatmulArguments {
            a,
            b,
            d,
            m,
            n,
            k,
            routing,
            ..
        } = arguments;
        let MatmulA::FullPrecision {
            values: a,
            offset: a_offset,
        } = a
        else {
            return Err(MatmulError::IncompatibleA {
                path: "Gemv",
                reason: "prepared int8 activations require GEMM",
            });
        };

        let group_count_x = n.div_ceil(rows_per_threadgroup(
            specialization.k_split,
            specialization.results_per_simdgroup,
            specialization.num_simdgroups,
        ));

        let context = encoder.context();
        let pipeline = self.get_or_create(context, specialization)?;
        let gather_indices = routing.sparse_readout_rows();
        let (expert_ids, routes_per_token, expert_count, input_is_route_major) = match routing.expert_routes() {
            Some(routes) => (
                Some(routes.expert_ids),
                routes.routes_per_token.get(),
                routes.expert_count.get(),
                routes.input == ExpertInput::Routes,
            ),
            None => (None, 1, 1, false),
        };

        match b {
            MatmulB::FullPrecision {
                b: weights,
            } => {
                pipeline.encode(
                    weights,
                    None::<&Allocation<Metal>>,
                    None::<&Allocation<Metal>>,
                    None::<&Allocation<Metal>>,
                    None::<&Allocation<Metal>>,
                    (a, a_offset),
                    &mut *d,
                    output_bias,
                    rht_factors,
                    gather_indices,
                    expert_ids,
                    per_matrix_bias,
                    k,
                    n,
                    m,
                    ab_scale,
                    group_count_x,
                    routes_per_token,
                    expert_count,
                    input_is_route_major,
                    soft_cap,
                    encoder,
                );
            },
            MatmulB::Microfloat {
                codes,
                scales,
                outer_scales,
                ..
            } => {
                pipeline.encode(
                    codes,
                    Some(scales),
                    None::<&Allocation<Metal>>,
                    None::<&Allocation<Metal>>,
                    Some(outer_scales),
                    (a, a_offset),
                    &mut *d,
                    output_bias,
                    rht_factors,
                    gather_indices,
                    expert_ids,
                    per_matrix_bias,
                    k,
                    n,
                    m,
                    ab_scale,
                    group_count_x,
                    routes_per_token,
                    expert_count,
                    input_is_route_major,
                    soft_cap,
                    encoder,
                );
            },
            quant_b @ (MatmulB::ScaleBiasDequant {
                ..
            }
            | MatmulB::ScaleZeroPointDequant {
                ..
            }
            | MatmulB::ScaleSymmetricDequant {
                ..
            }) => {
                let (weights, scales, zero_points, biases) = match quant_b {
                    MatmulB::ScaleBiasDequant {
                        b: w,
                        scales,
                        biases,
                        ..
                    } => (w, scales, None, Some(biases)),
                    MatmulB::ScaleZeroPointDequant {
                        b: w,
                        scales,
                        zero_points,
                        ..
                    } => (w, scales, Some(zero_points), None),
                    MatmulB::ScaleSymmetricDequant {
                        b: w,
                        scales,
                        ..
                    } => (w, scales, None, None),
                    MatmulB::FullPrecision {
                        ..
                    }
                    | MatmulB::Microfloat {
                        ..
                    } => unreachable!(),
                };
                pipeline.encode(
                    weights,
                    Some(scales),
                    zero_points,
                    biases,
                    None::<&Allocation<Metal>>,
                    (a, a_offset),
                    &mut *d,
                    output_bias,
                    rht_factors,
                    gather_indices,
                    expert_ids,
                    per_matrix_bias,
                    k,
                    n,
                    m,
                    ab_scale,
                    group_count_x,
                    routes_per_token,
                    expert_count,
                    input_is_route_major,
                    soft_cap,
                    encoder,
                );
            },
        }

        Ok(())
    }
}
