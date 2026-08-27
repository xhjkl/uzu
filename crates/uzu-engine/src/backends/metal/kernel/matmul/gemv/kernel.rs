use super::policy::{self, DEFAULT_RESULTS_PER_SIMDGROUP, FP_K_BLOCK};
use crate::{
    backends::{
        common::{
            Buffer, BufferArg, Encoder,
            gpu_types::{
                HADAMARD_TRANSFORM_BLOCK_SIZE,
                gemm::{GemmBPrologueKind, GemmDTransform},
            },
            kernel::matmul::{ExpertInput, MatmulA, MatmulArguments, MatmulB, MatmulError, MatmulShape},
        },
        metal::{context::MetalContext, device_profile::DeviceProfile, error::MetalError, kernel::GemvMetalKernel},
    },
    data_type::DataType,
};

const GEMV_MAX_BATCH: u32 = 8;

#[derive(Clone, Copy)]
struct GemvBufferSlice<'a> {
    buffer: &'a dyn Buffer<Backend = Metal>,
    offset: usize,
    length: usize,
}

impl<'a> GemvBufferSlice<'a> {
    fn new(argument: impl BufferArg<'a, Metal>) -> Self {
        let (buffer, offset, length) = argument.into_parts();
        Self {
            buffer,
            offset,
            length,
        }
    }
}

impl<'a> BufferArg<'a, Metal> for GemvBufferSlice<'a> {
    fn into_parts(self) -> (&'a dyn Buffer<Backend = Metal>, usize, usize) {
        (self.buffer, self.offset, self.length)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GemvSpecialization {
    b_prologue: GemmBPrologueKind,
    group_size: u32,
    bits: u32,
    output_transform: GemmDTransform,
    input_aligned: bool,
    k_split: u32,
    output_row_tile: u32,
    num_simdgroups: u32,
    input_row_tile: u32,
    reduction_lanes: u32,
    group_lanes: u32,
    microfloat: bool,
    gathered: bool,
    expert_routed: bool,
    expert_bias: bool,
    signed_codes: bool,
    full_tile: bool,
}

impl GemvSpecialization {
    #[cfg(test)]
    pub fn tile(self) -> policy::GemvTile {
        policy::GemvTile {
            num_simdgroups: self.num_simdgroups,
            k_split: self.k_split,
            results_per_simdgroup: self.output_row_tile / (self.num_simdgroups / self.k_split),
            input_row_tile: self.input_row_tile,
            reduction_lanes: self.reduction_lanes,
            group_lanes: self.group_lanes,
        }
    }

    pub fn select_shape(
        shape: &MatmulShape,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        device_profile: DeviceProfile,
    ) -> Option<Self> {
        if shape.expert_routed && shape.m > GEMV_MAX_BATCH {
            return None;
        }
        let integer_quantized = shape.is_integer_quantized();
        let bits = shape.b_bits.unwrap_or(0);
        let bf16_io = input_data_type == DataType::BF16 && output_data_type == DataType::BF16;
        let tile = if integer_quantized && shape.sparse_readout {
            policy::gathered_tile(bits, shape.b_group_size.unwrap_or(0), shape.m, shape.n)
        } else if integer_quantized {
            policy::quantized_tile(
                device_profile,
                bits,
                shape.b_group_size.unwrap_or(0),
                shape.m,
                shape.n,
                shape.k,
                shape.d_transform,
                bf16_io,
            )
        } else {
            let mixed_precision = weights_data_type == DataType::F32
                && (input_data_type != DataType::F32 || output_data_type != DataType::F32);
            if mixed_precision || shape.n < DEFAULT_RESULTS_PER_SIMDGROUP || shape.m > GEMV_MAX_BATCH {
                return None;
            }
            let input_aligned = shape.k.is_multiple_of(FP_K_BLOCK);
            if shape.d_transform.contains(GemmDTransform::RHT) {
                Some(policy::DEFAULT_TILE)
            } else {
                Some(policy::fp_tile(shape.m, shape.n, shape.k, input_aligned, device_profile))
            }
        };
        let mut tile = tile?;
        if shape.expert_routed {
            tile.input_row_tile = 1;
        }
        Self::select_tile(shape, weights_data_type, input_data_type, output_data_type, tile)
    }

    pub fn select_tile(
        shape: &MatmulShape,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        tile: policy::GemvTile,
    ) -> Option<Self> {
        Self::select_tile_for_format(shape, weights_data_type, input_data_type, output_data_type, tile, false)
    }

    pub fn select_microfloat(
        shape: &MatmulShape,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Option<Self> {
        if shape.m > GEMV_MAX_BATCH && !shape.sparse_readout {
            return None;
        }
        Self::select_tile_for_format(
            shape,
            weights_data_type,
            input_data_type,
            output_data_type,
            policy::DEFAULT_TILE,
            true,
        )
    }

    fn select_tile_for_format(
        shape: &MatmulShape,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        tile: policy::GemvTile,
        microfloat: bool,
    ) -> Option<Self> {
        if !shape.b_transpose || !shape.a_full_precision {
            return None;
        }
        let integer_quantized = shape.is_integer_quantized();
        let bad_leading_dimension = if integer_quantized || microfloat {
            shape.b_leading_dimension.is_some()
        } else {
            shape.b_leading_dimension.is_some_and(|ld| ld != shape.k)
        };
        if bad_leading_dimension {
            return None;
        }
        if shape.d_transform.contains(GemmDTransform::RHT) && !shape.n.is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE) {
            return None;
        }
        if shape.d_transform.contains(GemmDTransform::ACCUMULATE) && !shape.n.is_multiple_of(32) {
            return None;
        }
        let bits = shape.b_bits.unwrap_or(0);
        if !integer_quantized && !microfloat {
            let mixed_precision = weights_data_type == DataType::F32
                && (input_data_type != DataType::F32 || output_data_type != DataType::F32);
            if mixed_precision || shape.n < DEFAULT_RESULTS_PER_SIMDGROUP || shape.m > GEMV_MAX_BATCH {
                return None;
            }
        }
        let block_size = if microfloat {
            shape.b_group_size.unwrap_or(FP_K_BLOCK)
        } else if !integer_quantized {
            FP_K_BLOCK
        } else if bits == 4 {
            512
        } else {
            256
        };
        let input_aligned = shape.k.is_multiple_of(block_size);
        if (shape.sparse_readout || shape.expert_routed || microfloat) && tile.input_row_tile > 1 {
            return None;
        }
        let specialization = Self {
            b_prologue: shape.b_prologue,
            group_size: shape.b_group_size.unwrap_or(0),
            bits,
            output_transform: shape.d_transform,
            input_aligned,
            k_split: tile.k_split,
            output_row_tile: tile.output_row_tile(),
            num_simdgroups: tile.num_simdgroups,
            input_row_tile: tile.input_row_tile,
            reduction_lanes: tile.reduction_lanes,
            group_lanes: tile.group_lanes,
            microfloat,
            gathered: shape.sparse_readout,
            expert_routed: shape.expert_routed,
            expert_bias: shape.expert_bias,
            signed_codes: shape.signed_codes,
            full_tile: full_tile(shape, tile),
        };
        Some(specialization)
    }

    pub fn output_row_tile(&self) -> u32 {
        self.output_row_tile
    }

    fn create_pipeline(
        &self,
        context: &MetalContext,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Result<GemvMetalKernel, MetalError> {
        GemvMetalKernel::new(
            context,
            input_data_type,
            weights_data_type,
            output_data_type,
            self.b_prologue,
            self.group_size,
            self.bits,
            self.k_split,
            self.input_aligned,
            self.input_row_tile,
            self.output_row_tile(),
            self.reduction_lanes,
            self.group_lanes,
            self.num_simdgroups,
            self.microfloat,
            self.output_transform,
            self.gathered,
            self.expert_routed,
            self.expert_bias,
            self.signed_codes,
            self.full_tile,
        )
    }
}

fn full_tile(
    shape: &MatmulShape,
    tile: policy::GemvTile,
) -> bool {
    shape.m.is_multiple_of(tile.input_row_tile) && shape.n.is_multiple_of(tile.output_row_tile())
}

use std::collections::{HashMap, hash_map::Entry};

/// GEMV pipelines compiled on first use.
pub struct GemvKernel {
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
    pipelines: HashMap<GemvSpecialization, GemvMetalKernel>,
}

impl GemvKernel {
    pub fn new(
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Self {
        Self {
            weights_data_type,
            input_data_type,
            output_data_type,
            pipelines: HashMap::new(),
        }
    }

    fn get_or_create(
        &mut self,
        context: &MetalContext,
        specialization: GemvSpecialization,
    ) -> Result<&GemvMetalKernel, MatmulError<Metal>> {
        match self.pipelines.entry(specialization) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let kernel = specialization
                    .create_pipeline(context, self.weights_data_type, self.input_data_type, self.output_data_type)
                    .map_err(MatmulError::BackendError)?;
                Ok(entry.insert(kernel))
            },
        }
    }

    pub fn encode<'a, 'b, 'd, TB: BufferArg<'b, Metal>>(
        &mut self,
        arguments: MatmulArguments<'a, 'b, 'd, Metal, TB>,
        specialization: GemvSpecialization,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        if arguments.d_transform.gate_act.is_some() {
            return Err(MatmulError::UnsupportedDOp {
                bit: GemmDTransform::GATE_ACT_MUL,
                path: "Gemv",
            });
        }
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
        let (weights, scales, zero_points, biases, outer_scales) = match b {
            MatmulB::FullPrecision {
                b,
            } => (GemvBufferSlice::new(b), None, None, None, None),
            MatmulB::Microfloat {
                codes,
                scales,
                outer_scales,
                ..
            } => (
                GemvBufferSlice::new(codes),
                Some(GemvBufferSlice::new(scales)),
                None,
                None,
                Some(GemvBufferSlice::new(outer_scales)),
            ),
            MatmulB::ScaleBiasDequant {
                b,
                scales,
                biases,
                ..
            } => (
                GemvBufferSlice::new(b),
                Some(GemvBufferSlice::new(scales)),
                None,
                Some(GemvBufferSlice::new(biases)),
                None,
            ),
            MatmulB::ScaleZeroPointDequant {
                b,
                scales,
                zero_points,
                ..
            } => (
                GemvBufferSlice::new(b),
                Some(GemvBufferSlice::new(scales)),
                Some(GemvBufferSlice::new(zero_points)),
                None,
                None,
            ),
            MatmulB::ScaleSymmetricDequant {
                b,
                scales,
                ..
            } => (GemvBufferSlice::new(b), Some(GemvBufferSlice::new(scales)), None, None, None),
        };

        let output_group_count = n.div_ceil(specialization.output_row_tile());
        let pipeline = self.get_or_create(encoder.context(), specialization)?;
        pipeline.encode(
            weights,
            scales,
            zero_points,
            biases,
            outer_scales,
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
            output_group_count,
            routes_per_token,
            expert_count,
            input_is_route_major,
            soft_cap,
            encoder,
        );
        Ok(())
    }
}
