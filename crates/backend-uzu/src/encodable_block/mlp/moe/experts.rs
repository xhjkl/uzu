use super::router::MoeRoutes;
use crate::{
    ClippingBounds,
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        gpu_types::ActivationType,
        kernel::{
            MoeFinalizeKernel,
            matmul::{ActivationFormat, ExpertInput, ExpertRoutes, GateActMulDOps},
        },
    },
    config::activation::AnyActivation,
    data_type::DataType,
    encodable_block::{
        linear::{LinearInput, LinearMatmul},
        mlp::gate_act_mul::MlpGateActMulEncodable,
    },
};

pub struct MoeExperts<B: Backend> {
    up_projection: LinearMatmul<B>,
    gate: MlpGateActMulEncodable<B>,
    down_projection: LinearMatmul<B>,
    finalize: <B::Kernels as Kernels>::MoeFinalizeKernel,
    model_dim: u32,
    expert_count: u32,
    data_type: DataType,
    silu_alpha: Option<f32>,
    gate_clipping: Option<(f32, f32)>,
    up_clipping: Option<(f32, f32)>,
}

impl<B: Backend> MoeExperts<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context: &B::Context,
        up_projection: LinearMatmul<B>,
        down_projection: LinearMatmul<B>,
        model_dim: u32,
        hidden_dim: u32,
        expert_count: u32,
        activation: AnyActivation,
        gate_clipping: ClippingBounds,
        up_clipping: ClippingBounds,
        data_type: DataType,
    ) -> Result<Self, B::Error> {
        let silu_alpha = (activation.act_type() == ActivationType::SILU).then(|| activation.alpha());
        Ok(Self {
            up_projection,
            gate: MlpGateActMulEncodable::new(
                context,
                DataType::F32,
                activation,
                gate_clipping,
                up_clipping,
                hidden_dim,
                None,
            )?,
            down_projection,
            finalize: <B::Kernels as Kernels>::MoeFinalizeKernel::new(context, data_type)?,
            model_dim,
            expert_count,
            data_type,
            silu_alpha,
            gate_clipping: clipping_bounds(gate_clipping),
            up_clipping: clipping_bounds(up_clipping),
        })
    }

    fn encode_hidden(
        &self,
        input: &Allocation<B>,
        routes: &MoeRoutes<B>,
        route_count: u32,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let fused_up = self.up_projection.encode_routed(
            input,
            route_count,
            ExpertRoutes {
                expert_ids: &routes.expert_ids,
                routes_per_token: routes.routes_per_token,
                expert_count: std::num::NonZeroU32::new(self.expert_count).expect("MoeBlock validates expert_count"),
                input: ExpertInput::Tokens,
            },
            None,
            encoder,
        )?;
        let gate_input = self.gate.encode_for_linear(encoder, &fused_up, route_count, ActivationFormat::Bf16)?;
        let hidden = match gate_input {
            LinearInput::FullPrecision(hidden) => hidden,
            _ => unreachable!("BF16 activation format always yields full-precision hidden states"),
        };
        Ok(hidden)
    }

    pub fn encode(
        &self,
        input: &Allocation<B>,
        routes: &MoeRoutes<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let route_count = routes.route_count();
        let route_metadata = |input| ExpertRoutes {
            expert_ids: &routes.expert_ids,
            routes_per_token: routes.routes_per_token,
            expert_count: std::num::NonZeroU32::new(self.expert_count).expect("MoeBlock validates expert_count"),
            input,
        };

        let hidden = match self.silu_alpha {
            Some(alpha) if self.up_projection.supports_routed_gate_act(route_count) => {
                self.up_projection.encode_routed(
                    input,
                    route_count,
                    route_metadata(ExpertInput::Tokens),
                    Some(GateActMulDOps {
                        activation_alpha: (alpha != 1.0).then_some(alpha),
                        gate_clipping: self.gate_clipping,
                        value_clipping: self.up_clipping,
                    }),
                    encoder,
                )?
            },
            _ => self.encode_hidden(input, routes, route_count, encoder)?,
        };
        let route_outputs = self.down_projection.encode_routed(
            &hidden,
            route_count,
            route_metadata(ExpertInput::Routes),
            None,
            encoder,
        )?;

        let mut output = encoder.allocate_scratch_for_shape(&[routes.token_count, self.model_dim], self.data_type)?;
        self.finalize.encode(
            &routes.route_weights,
            &route_outputs,
            &mut output,
            routes.token_count,
            self.model_dim,
            routes.routes_per_token.get(),
            encoder,
        );
        Ok(output)
    }
}

fn clipping_bounds(clipping: ClippingBounds) -> Option<(f32, f32)> {
    if clipping.min.is_none() && clipping.max.is_none() {
        return None;
    }
    Some((clipping.min.unwrap_or(f32::NEG_INFINITY), clipping.max.unwrap_or(f32::INFINITY)))
}
