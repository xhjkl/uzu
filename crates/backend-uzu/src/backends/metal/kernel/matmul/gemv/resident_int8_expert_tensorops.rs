use std::{
    collections::{HashMap, hash_map::Entry},
    mem::size_of,
    num::NonZeroU32,
};

use crate::{
    backends::{
        common::{
            Allocation, Encoder,
            kernel::matmul::{ExpertInput, MatmulError},
        },
        metal::{
            Int8Execution, Metal,
            context::MetalContext,
            error::MetalError,
            kernel::{ResidentInt8ExpertEmulatedMetalKernel, ResidentInt8ExpertTensorOpsMetalKernel},
        },
    },
    data_type::DataType,
};

/// Resident group-32 signed-INT8 expert projection with hardware and portable implementations.
pub(crate) struct ResidentInt8ExpertTensorOpsDispatch {
    weight_scale_data_type: DataType,
    bias_data_type: DataType,
    output_data_type: DataType,
    hardware_pipelines: HashMap<bool, ResidentInt8ExpertTensorOpsMetalKernel>,
    emulated_pipelines: HashMap<bool, ResidentInt8ExpertEmulatedMetalKernel>,
}

impl ResidentInt8ExpertTensorOpsDispatch {
    pub(crate) fn new(
        weight_scale_data_type: DataType,
        bias_data_type: DataType,
        output_data_type: DataType,
    ) -> Self {
        Self {
            weight_scale_data_type,
            bias_data_type,
            output_data_type,
            hardware_pipelines: HashMap::new(),
            emulated_pipelines: HashMap::new(),
        }
    }

    fn get_or_create_hardware(
        &mut self,
        context: &MetalContext,
        expert_bias: bool,
    ) -> Result<&ResidentInt8ExpertTensorOpsMetalKernel, MetalError> {
        match self.hardware_pipelines.entry(expert_bias) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let kernel = ResidentInt8ExpertTensorOpsMetalKernel::new(
                    context,
                    self.weight_scale_data_type,
                    self.bias_data_type,
                    self.output_data_type,
                    expert_bias,
                )?;
                Ok(entry.insert(kernel))
            },
        }
    }

    fn get_or_create_emulated(
        &mut self,
        context: &MetalContext,
        expert_bias: bool,
    ) -> Result<&ResidentInt8ExpertEmulatedMetalKernel, MetalError> {
        match self.emulated_pipelines.entry(expert_bias) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let kernel = ResidentInt8ExpertEmulatedMetalKernel::new(
                    context,
                    self.weight_scale_data_type,
                    self.bias_data_type,
                    self.output_data_type,
                    expert_bias,
                )?;
                Ok(entry.insert(kernel))
            },
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode(
        &mut self,
        weight_codes: &Allocation<Metal>,
        weight_scales: &Allocation<Metal>,
        activation_codes: &Allocation<Metal>,
        activation_scales: &Allocation<Metal>,
        output: &mut Allocation<Metal>,
        expert_biases: Option<&Allocation<Metal>>,
        expert_ids: &Allocation<Metal>,
        input_size: u32,
        output_size: u32,
        route_count: u32,
        routes_per_token: NonZeroU32,
        expert_count: NonZeroU32,
        input: ExpertInput,
        execution: Int8Execution,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        const PATH: &str = "ResidentInt8ExpertTensorOps";
        if input_size == 0 || output_size == 0 || !input_size.is_multiple_of(32) {
            return Err(MatmulError::<Metal>::InvalidStorage {
                path: PATH,
                operand: "dimensions",
                reason: "N and K must be nonzero and K must be divisible by 32",
            }
            .into());
        }
        let activation_rows = match input {
            ExpertInput::Routes => route_count,
            ExpertInput::Tokens => {
                if !route_count.is_multiple_of(routes_per_token.get()) {
                    return Err(MatmulError::<Metal>::UnsupportedRouting {
                        path: PATH,
                        reason: "route count must be divisible by routes_per_token for token-major inputs",
                    }
                    .into());
                }
                route_count / routes_per_token.get()
            },
        };
        let checked_size =
            |factors: &[usize]| factors.iter().try_fold(1usize, |size, &factor| size.checked_mul(factor));
        let require = |operand: &'static str, actual: usize, required: Option<usize>| -> Result<(), MetalError> {
            let Some(required) = required else {
                return Err(MatmulError::<Metal>::InvalidStorage {
                    path: PATH,
                    operand,
                    reason: "required storage size overflows usize",
                }
                .into());
            };
            if actual < required {
                return Err(MatmulError::<Metal>::InvalidStorage {
                    path: PATH,
                    operand,
                    reason: "allocation is smaller than required by the dispatch dimensions",
                }
                .into());
            }
            Ok(())
        };
        let experts = expert_count.get() as usize;
        let routes = route_count as usize;
        let activation_rows = activation_rows as usize;
        let output_size = output_size as usize;
        let input_size = input_size as usize;
        let group_count = input_size / 32;
        require("expert IDs", expert_ids.size(), checked_size(&[routes, size_of::<i32>()]))?;
        require("weight codes", weight_codes.size(), checked_size(&[experts, output_size, input_size]))?;
        require(
            "weight scales",
            weight_scales.size(),
            checked_size(&[experts, output_size, group_count, self.weight_scale_data_type.size_in_bytes()]),
        )?;
        require("activation codes", activation_codes.size(), checked_size(&[activation_rows, input_size]))?;
        require(
            "activation scales",
            activation_scales.size(),
            checked_size(&[activation_rows, group_count, size_of::<f32>()]),
        )?;
        require("output", output.size(), checked_size(&[routes, output_size, self.output_data_type.size_in_bytes()]))?;
        if let Some(expert_biases) = expert_biases {
            require(
                "expert biases",
                expert_biases.size(),
                checked_size(&[experts, output_size, self.bias_data_type.size_in_bytes()]),
            )?;
        }

        let output_tile_count = (output_size as u32).div_ceil(32);
        let input_is_route_major = matches!(input, ExpertInput::Routes);
        match execution {
            Int8Execution::Emulated => {
                let pipeline = self.get_or_create_emulated(encoder.context(), expert_biases.is_some())?;
                pipeline.encode(
                    weight_codes,
                    weight_scales,
                    activation_codes,
                    activation_scales,
                    output,
                    expert_biases,
                    expert_ids,
                    input_size as u32,
                    output_size as u32,
                    route_count,
                    routes_per_token.get(),
                    expert_count.get(),
                    input_is_route_major,
                    output_tile_count,
                    encoder,
                );
            },
            Int8Execution::HardwareTensorOps => {
                let pipeline = self.get_or_create_hardware(encoder.context(), expert_biases.is_some())?;
                pipeline.encode(
                    weight_codes,
                    weight_scales,
                    activation_codes,
                    activation_scales,
                    output,
                    expert_biases,
                    expert_ids,
                    input_size as u32,
                    output_size as u32,
                    route_count,
                    routes_per_token.get(),
                    expert_count.get(),
                    input_is_route_major,
                    output_tile_count,
                    encoder,
                );
            },
        }
        Ok(())
    }
}
