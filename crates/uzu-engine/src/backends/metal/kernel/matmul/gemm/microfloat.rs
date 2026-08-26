use std::collections::{HashMap, hash_map::Entry};

use crate::{
    backends::{
        common::{
            BufferArg, Encoder,
            gpu_types::gemm::GemmDTransform,
            kernel::matmul::{MatmulA, MatmulArguments, MatmulB, MatmulError},
        },
        metal::{Metal, context::MetalContext, error::MetalError, kernel::MicrofloatGemmMetalKernel},
    },
    data_type::DataType,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MicrofloatGemmSpecialization {
    group_size: u32,
    output_transform: GemmDTransform,
}

pub(super) struct MicrofloatGemmDispatch {
    pipelines: HashMap<MicrofloatGemmSpecialization, MicrofloatGemmMetalKernel>,
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
}

impl MicrofloatGemmDispatch {
    pub(super) fn new(
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Self {
        Self {
            pipelines: HashMap::new(),
            weights_data_type,
            input_data_type,
            output_data_type,
        }
    }

    fn get_or_create(
        &mut self,
        context: &MetalContext,
        specialization: MicrofloatGemmSpecialization,
    ) -> Result<&MicrofloatGemmMetalKernel, MetalError> {
        match self.pipelines.entry(specialization) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let pipeline = MicrofloatGemmMetalKernel::new(
                    context,
                    self.input_data_type,
                    self.weights_data_type,
                    self.output_data_type,
                    specialization.group_size,
                    specialization.output_transform,
                )?;
                Ok(entry.insert(pipeline))
            },
        }
    }

    pub(super) fn encode<'a, 'b, 'd, TB: BufferArg<'b, Metal>>(
        &mut self,
        arguments: MatmulArguments<'a, 'b, 'd, Metal, TB>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        let output_transform = arguments.d_transform.mask();
        if output_transform.contains(GemmDTransform::RHT) {
            return Err(MatmulError::<Metal>::UnsupportedDOp {
                bit: GemmDTransform::RHT,
                path: "MicrofloatGemm",
            }
            .into());
        }
        let MatmulA::FullPrecision {
            values: input,
            offset: input_offset,
        } = arguments.a
        else {
            return Err(MatmulError::IncompatibleA {
                path: "MicrofloatGemm",
                reason: "prepared int8 activations require integer weights",
            }
            .into());
        };
        let MatmulB::Microfloat {
            codes,
            scales,
            outer_scales,
            metadata,
        } = arguments.b
        else {
            unreachable!("microfloat GEMM is selected only for microfloat weights");
        };
        let specialization = MicrofloatGemmSpecialization {
            group_size: metadata.group_size(),
            output_transform,
        };
        let pipeline = self.get_or_create(encoder.context(), specialization)?;
        pipeline.encode(
            codes,
            scales,
            outer_scales,
            (input, input_offset),
            &mut *arguments.d,
            arguments.d_transform.bias,
            arguments.m,
            arguments.n,
            arguments.k,
            arguments.d_transform.ab_scale,
            arguments.d_transform.soft_cap,
            encoder,
        );
        Ok(())
    }
}
