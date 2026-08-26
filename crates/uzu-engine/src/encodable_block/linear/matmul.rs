use std::num::NonZeroU32;

use parking_lot::Mutex;
use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder,
        kernel::{
            Kernels,
            matmul::{
                A8ActivationPlan, ActivationFormat, ExpertRoutes, GateActMulDOps, MatmulA, MatmulArguments, MatmulB,
                MatmulDOps, MatmulKernel, MatmulRouting, MatmulShape,
            },
        },
    },
    config::weight_matrix::{AnyWeightMatrixSpec, Layout},
    data_type::DataType,
    encodable_block::{
        linear::{Linear, LinearInput},
        weight_matrix::{WeightMatrix, WeightMatrixError},
    },
    parameters::{ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum LinearMatmulError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Parameter loading error: {0}")]
    ParameterError(#[from] ParameterLoaderError<B>),
    #[error("Weight matrix error: {0}")]
    WeightMatrix(#[from] WeightMatrixError<B>),
    #[error("Unsupported data type: {0:?}")]
    UnsupportedDataType(DataType),
    #[error("Unsupported linear matmul configuration: {0}")]
    UnsupportedConfiguration(String),
}

pub struct LinearMatmul<B: Backend> {
    kernel: Mutex<<B::Kernels as Kernels>::MatmulKernel>,
    matrix: WeightMatrix<B>,
    biases: Option<Allocation<B>>,
    output_hadamard_factors: Option<Allocation<B>>,
    input_dim: u32,
    output_dim: u32,
    output_data_type: DataType,
    matrix_count: Option<NonZeroU32>,
}

fn load_biases<B: Backend>(
    weights_data_type: DataType,
    output_data_type: DataType,
    output_dim: u32,
    matrix_count: Option<NonZeroU32>,
    parameter_tree: Option<&ParameterTree<B>>,
) -> Result<Option<Allocation<B>>, LinearMatmulError<B>> {
    if matrix_count.is_none() && parameter_tree.is_some() && weights_data_type != output_data_type {
        return Err(LinearMatmulError::UnsupportedConfiguration(format!(
            "mixed precision linear with biases is not supported: weights={weights_data_type:?}, output={output_data_type:?}",
        )));
    }
    let shape = match matrix_count {
        Some(matrix_count) => vec![matrix_count.get(), output_dim],
        None => vec![output_dim],
    };
    Ok(parameter_tree
        .map(|tree| tree.leaf("biases")?.validate(&shape, weights_data_type)?.read_allocation())
        .transpose()?)
}

impl<B: Backend> LinearMatmul<B> {
    pub fn load(
        context: &B::Context,
        spec: AnyWeightMatrixSpec,
        input_dim: u32,
        output_dim: u32,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        weights_tree: &ParameterTree<B>,
        bias_tree: Option<&ParameterTree<B>>,
        output_hadamard_factors: Option<Allocation<B>>,
    ) -> Result<Self, LinearMatmulError<B>> {
        Self::load_impl(
            context,
            spec,
            input_dim,
            output_dim,
            None,
            weights_data_type,
            input_data_type,
            output_data_type,
            weights_tree,
            bias_tree,
            output_hadamard_factors,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load_bank(
        context: &B::Context,
        spec: AnyWeightMatrixSpec,
        input_dim: u32,
        output_dim: u32,
        matrix_count: NonZeroU32,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        weights_tree: &ParameterTree<B>,
        bias_tree: Option<&ParameterTree<B>>,
    ) -> Result<Self, LinearMatmulError<B>> {
        Self::load_impl(
            context,
            spec,
            input_dim,
            output_dim,
            Some(matrix_count),
            weights_data_type,
            input_data_type,
            output_data_type,
            weights_tree,
            bias_tree,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn load_impl(
        context: &B::Context,
        spec: AnyWeightMatrixSpec,
        input_dim: u32,
        output_dim: u32,
        matrix_count: Option<NonZeroU32>,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        weights_tree: &ParameterTree<B>,
        bias_tree: Option<&ParameterTree<B>>,
        output_hadamard_factors: Option<Allocation<B>>,
    ) -> Result<Self, LinearMatmulError<B>> {
        for data_type in [weights_data_type, input_data_type, output_data_type] {
            if !matches!(data_type, DataType::BF16 | DataType::F32) {
                return Err(LinearMatmulError::UnsupportedDataType(data_type));
            }
        }

        let matrix = match matrix_count {
            Some(matrix_count) => WeightMatrix::load_bank(
                weights_tree,
                spec,
                Layout::OutputInput,
                matrix_count,
                output_dim,
                input_dim,
                weights_data_type,
            )?,
            None => {
                WeightMatrix::load(weights_tree, spec, Layout::OutputInput, output_dim, input_dim, weights_data_type)?
            },
        };
        if output_hadamard_factors.is_some() && matrix.quantization().is_none() {
            return Err(LinearMatmulError::UnsupportedConfiguration(
                "fused output-hadamard factors require quantized weights".into(),
            ));
        }

        let biases = load_biases(weights_data_type, output_data_type, output_dim, matrix_count, bias_tree)?;

        let kernel =
            <B::Kernels as Kernels>::MatmulKernel::new(context, weights_data_type, input_data_type, output_data_type)
                .map_err(LinearMatmulError::BackendError)?;

        Ok(Self {
            kernel: Mutex::new(kernel),
            matrix,
            biases,
            output_hadamard_factors,
            input_dim,
            output_dim,
            output_data_type,
            matrix_count,
        })
    }

    pub(super) fn prepare_a8(
        &mut self,
        context: &B::Context,
    ) -> Option<A8ActivationPlan> {
        let mut candidate = self.matmul_shape(1, false, false, None);
        candidate.signed_codes = true;
        let plan = self.kernel.lock().a8_activation_plan(&candidate, context)?;
        self.matrix.make_codes_signed();
        Some(plan)
    }

    pub(super) fn encode_with_a(
        &self,
        a: MatmulA<'_, B>,
        batch_dim: u32,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut output =
            encoder.allocate_scratch(size_for_shape(&[batch_dim, self.output_dim], self.output_data_type))?;

        self.kernel.lock().encode(
            MatmulArguments {
                a,
                b: self.matmul_b(),
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut output,
                d_transform: self.d_ops(None),
                routing: MatmulRouting::Dense,
                m: batch_dim,
                n: self.output_dim,
                k: self.input_dim,
            },
            encoder,
        )?;

        Ok(output)
    }

    fn matmul_shape(
        &self,
        batch_dim: u32,
        a_full_precision: bool,
        expert_routed: bool,
        gate_act: Option<GateActMulDOps>,
    ) -> MatmulShape {
        let b = self.matmul_b();
        MatmulShape {
            m: batch_dim,
            n: self.output_dim,
            k: self.input_dim,
            b_transpose: true,
            b_leading_dimension: None,
            b_kind: b.kind(),
            b_prologue: b.b_prologue(),
            b_bits: b.bits_per_b(),
            b_group_size: b.group_size(),
            signed_codes: b.signed_codes(),
            a_full_precision,
            sparse_readout: false,
            expert_routed,
            expert_bias: expert_routed && self.biases.is_some(),
            d_transform: self.d_ops(gate_act).mask(),
        }
    }

    pub(super) fn select_activation_format(
        &self,
        batch_dim: u32,
        context: &B::Context,
    ) -> ActivationFormat {
        if !self.matmul_b().signed_codes() {
            return ActivationFormat::Bf16;
        }
        let bf16_shape = self.matmul_shape(batch_dim, true, false, None);
        self.kernel.lock().select_activation_format(&bf16_shape, context)
    }

    pub(crate) fn supports_routed_gate_act(
        &self,
        route_count: u32,
    ) -> bool {
        let shape = self.matmul_shape(
            route_count,
            true,
            true,
            Some(GateActMulDOps {
                activation_alpha: None,
                gate_clipping: None,
                value_clipping: None,
            }),
        );
        self.kernel.lock().supports_fused_gate_act(&shape)
    }

    pub(crate) fn encode_routed(
        &self,
        input: &Allocation<B>,
        route_count: u32,
        routes: ExpertRoutes<'_, B>,
        gate_act: Option<GateActMulDOps>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        assert!(self.matrix_count.is_some(), "routed linear requires a matrix bank");
        let output_dim = if gate_act.is_some() {
            self.output_dim / 2
        } else {
            self.output_dim
        };
        let mut output = encoder.allocate_scratch_for_shape(&[route_count, output_dim], self.output_data_type)?;
        self.kernel.lock().encode(
            MatmulArguments {
                a: MatmulA::FullPrecision {
                    values: input,
                    offset: 0,
                },
                b: self.matmul_b(),
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut output,
                d_transform: self.d_ops(gate_act),
                routing: MatmulRouting::Experts(routes),
                m: route_count,
                n: self.output_dim,
                k: self.input_dim,
            },
            encoder,
        )?;
        Ok(output)
    }

    fn matmul_b(&self) -> MatmulB<'_, B> {
        self.matrix.matmul_b()
    }

    fn d_ops(
        &self,
        gate_act: Option<GateActMulDOps>,
    ) -> MatmulDOps<'_, B> {
        let (bias, per_matrix_bias) = if self.matrix_count.is_some() {
            (None, self.biases.as_ref())
        } else {
            (self.biases.as_ref(), None)
        };
        MatmulDOps {
            bias,
            per_matrix_bias,
            rht_factors: self.output_hadamard_factors.as_ref(),
            gate_act,
            ..MatmulDOps::none()
        }
    }
}

impl<B: Backend> Linear<B> for LinearMatmul<B> {
    fn encode(
        &self,
        input: Allocation<B>,
        batch_dim: u32,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        encoder.push_debug_group("matmul");

        let output = self.encode_with_a(
            MatmulA::FullPrecision {
                values: &input,
                offset: 0,
            },
            batch_dim,
            encoder,
        )?;

        encoder.pop_debug_group();

        Ok(output)
    }

    fn encode_input(
        &self,
        input: LinearInput<B>,
        batch_dim: u32,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        match input {
            LinearInput::FullPrecision(input) => self.encode(input, batch_dim, encoder),
            LinearInput::Int8Symmetric {
                values,
                scales,
                group_sums,
                group_size,
            } => self.encode_with_a(
                MatmulA::Int8Symmetric {
                    values: &values,
                    scales: &scales,
                    group_sums: group_sums.as_ref(),
                    group_size,
                },
                batch_dim,
                encoder,
            ),
        }
    }

    fn select_activation_format(
        &self,
        batch_dim: u32,
        context: &B::Context,
    ) -> ActivationFormat {
        Self::select_activation_format(self, batch_dim, context)
    }
}
