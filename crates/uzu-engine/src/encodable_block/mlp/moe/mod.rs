//! Mixture-of-experts routing and expert execution.

mod experts;
mod router;

use std::num::NonZeroU32;

use experts::MoeExperts;
use router::MoeRouter;
use thiserror::Error;

use crate::{
    backends::common::{
        Allocation, Backend, Encoder,
        gpu_types::{
            ActivationType,
            router_topk::{ROUTER_TOPK_MAX_EXPERTS, ROUTER_TOPK_MAX_MODEL_DIM, ROUTER_TOPK_MAX_SELECTED_EXPERTS},
        },
    },
    config::{
        mlp::mixture_of_experts::MixtureOfExpertsConfig,
        weight_matrix::{AnyWeightMatrixSpec, Layout, full_precision_spec::FullPrecisionSpec},
    },
    data_type::DataType,
    encodable_block::{
        linear::{LinearMatmul, LinearMatmulError},
        mlp::Mlp,
    },
    parameters::{ParameterLoaderError, ParameterTree},
};

pub struct MoeBlock<B: Backend> {
    router: MoeRouter<B>,
    experts: MoeExperts<B>,
}

struct MoeShape {
    expert_count: NonZeroU32,
    routes_per_token: NonZeroU32,
    fused_hidden_dim: u32,
}

#[derive(Debug, Error)]
pub enum MoeBlockError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Parameter loader error: {0}")]
    ParameterLoaderError(#[from] ParameterLoaderError<B>),
    #[error("Expert linear loading error: {0}")]
    ExpertLinearError(#[from] LinearMatmulError<B>),
    #[error("MoE model_dim must be nonzero, divisible by 4, and within the fused router capacity")]
    InvalidModelDim,
    #[error("MoE num_routed_experts must be nonzero and within the fused router capacity")]
    InvalidRoutedExpertCount,
    #[error("MoE num_active_routed_experts must be nonzero, within TopK capacity, and <= num_routed_experts")]
    InvalidActiveExpertCount,
    #[error("MoE expert_hidden_dim must be > 0 and 2 * expert_hidden_dim must fit in u32")]
    InvalidExpertHiddenDim,
    #[error("MoE shared experts are not supported")]
    UnsupportedSharedExperts,
    #[error("MoE expert gate is not supported")]
    UnsupportedExpertGate,
    #[error("MoE without router, up, and down biases is not supported")]
    UnsupportedNoBiases,
    #[error("Unsupported MoE router configuration: {0}")]
    UnsupportedRouterConfiguration(String),
    #[error("Unsupported MoE expert activation: {0:?}")]
    UnsupportedExpertActivation(ActivationType),
}

impl<B: Backend> MoeBlock<B> {
    fn expert_spec(weights_tree: &ParameterTree<B>) -> Result<AnyWeightMatrixSpec, ParameterLoaderError<B>> {
        match weights_tree.metadata::<AnyWeightMatrixSpec>("spec") {
            Ok(spec) => Ok(spec),
            Err(ParameterLoaderError::KeyNotFound(_)) => {
                Ok(AnyWeightMatrixSpec::FullPrecisionSpec(FullPrecisionSpec::output_input()))
            },
            Err(error) => Err(error),
        }
    }

    fn validate_config(
        moe_config: &MixtureOfExpertsConfig,
        model_dim: u32,
    ) -> Result<MoeShape, MoeBlockError<B>> {
        if model_dim == 0 || model_dim > ROUTER_TOPK_MAX_MODEL_DIM || !model_dim.is_multiple_of(4) {
            return Err(MoeBlockError::InvalidModelDim);
        }
        let Some(expert_count) = NonZeroU32::new(moe_config.num_routed_experts) else {
            return Err(MoeBlockError::InvalidRoutedExpertCount);
        };
        if expert_count.get() > ROUTER_TOPK_MAX_EXPERTS {
            return Err(MoeBlockError::InvalidRoutedExpertCount);
        }
        let Some(routes_per_token) = NonZeroU32::new(moe_config.num_active_routed_experts) else {
            return Err(MoeBlockError::InvalidActiveExpertCount);
        };
        if routes_per_token.get() > ROUTER_TOPK_MAX_SELECTED_EXPERTS.min(expert_count.get()) {
            return Err(MoeBlockError::InvalidActiveExpertCount);
        }
        let fused_hidden_dim = moe_config
            .expert_hidden_dim
            .checked_mul(2)
            .filter(|_| moe_config.expert_hidden_dim > 0)
            .ok_or(MoeBlockError::InvalidExpertHiddenDim)?;
        if moe_config.num_shared_experts != 0 {
            return Err(MoeBlockError::UnsupportedSharedExperts);
        }
        if moe_config.gate_config.is_some() {
            return Err(MoeBlockError::UnsupportedExpertGate);
        }
        if !moe_config.router_has_biases
            || !moe_config.expert_config.has_up_biases
            || !moe_config.expert_config.has_down_biases
        {
            return Err(MoeBlockError::UnsupportedNoBiases);
        }
        match moe_config.expert_config.activation.act_type() {
            ActivationType::GELUApprox | ActivationType::SILU => {},
            activation => return Err(MoeBlockError::UnsupportedExpertActivation(activation)),
        }
        Ok(MoeShape {
            expert_count,
            routes_per_token,
            fused_hidden_dim,
        })
    }

    fn load_router(
        context: &B::Context,
        moe_config: &MixtureOfExpertsConfig,
        shape: &MoeShape,
        model_dim: u32,
        data_type: DataType,
        parameter_tree: &ParameterTree<B>,
    ) -> Result<MoeRouter<B>, MoeBlockError<B>> {
        let router_tree = parameter_tree.subtree("router");
        let weights_tree = router_tree.subtree("weights");
        let spec = weights_tree.metadata::<AnyWeightMatrixSpec>("spec")?;
        let AnyWeightMatrixSpec::FullPrecisionSpec(FullPrecisionSpec {
            layout: Layout::OutputInput,
            ..
        }) = &spec
        else {
            return Err(MoeBlockError::UnsupportedRouterConfiguration(format!("{spec:?}")));
        };
        let weights = weights_tree
            .leaf("weights")?
            .validate(&[shape.expert_count.get(), model_dim], data_type)?
            .read_allocation()?;
        let biases = router_tree.leaf("biases")?.validate(&[shape.expert_count.get()], data_type)?.read_allocation()?;
        MoeRouter::new(
            context,
            weights,
            biases,
            model_dim,
            shape.expert_count,
            shape.routes_per_token,
            matches!(
                moe_config.routing_function,
                crate::config::mlp::routing_function::AnyRoutingFunction::SoftmaxRouting(_)
            ),
            data_type,
        )
        .map_err(MoeBlockError::BackendError)
    }

    fn load_experts(
        context: &B::Context,
        moe_config: &MixtureOfExpertsConfig,
        shape: &MoeShape,
        model_dim: u32,
        data_type: DataType,
        parameter_tree: &ParameterTree<B>,
    ) -> Result<MoeExperts<B>, MoeBlockError<B>> {
        let experts_tree = parameter_tree.subtree("experts");
        let up_tree = experts_tree.subtree("up_projection");
        let up_weights_tree = up_tree.subtree("weights");
        let up_projection = LinearMatmul::load_bank(
            context,
            Self::expert_spec(&up_weights_tree)?,
            model_dim,
            shape.fused_hidden_dim,
            shape.expert_count,
            data_type,
            data_type,
            DataType::F32,
            &up_weights_tree,
            Some(&up_tree),
        )?;
        let down_tree = experts_tree.subtree("down_projection");
        let down_weights_tree = down_tree.subtree("weights");
        let down_projection = LinearMatmul::load_bank(
            context,
            Self::expert_spec(&down_weights_tree)?,
            moe_config.expert_hidden_dim,
            model_dim,
            shape.expert_count,
            data_type,
            DataType::F32,
            data_type,
            &down_weights_tree,
            Some(&down_tree),
        )?;
        MoeExperts::new(
            context,
            up_projection,
            down_projection,
            model_dim,
            moe_config.expert_hidden_dim,
            shape.expert_count,
            moe_config.expert_config.activation.clone(),
            moe_config.expert_config.gate_clipping,
            moe_config.expert_config.up_clipping,
            data_type,
        )
        .map_err(MoeBlockError::BackendError)
    }

    pub fn new(
        context: &B::Context,
        moe_config: &MixtureOfExpertsConfig,
        model_dim: u32,
        data_type: DataType,
        parameter_tree: &ParameterTree<B>,
    ) -> Result<Self, MoeBlockError<B>> {
        let shape = Self::validate_config(moe_config, model_dim)?;
        let router = Self::load_router(context, moe_config, &shape, model_dim, data_type, parameter_tree)?;
        let experts = Self::load_experts(context, moe_config, &shape, model_dim, data_type, parameter_tree)?;
        Ok(Self {
            router,
            experts,
        })
    }
}

impl<B: Backend> Mlp<B> for MoeBlock<B> {
    fn encode(
        &self,
        input: Allocation<B>,
        batch_dim: u32,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        encoder.push_debug_group("mlp (moe)");
        let routes = self.router.route(&input, batch_dim, encoder)?;
        let output = self.experts.encode(&input, &routes, encoder)?;
        encoder.pop_debug_group();
        Ok(output)
    }
}

#[cfg(test)]
#[path = "../../../../unit/encodable_block/moe/mod.rs"]
mod tests;
