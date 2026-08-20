use proc_macros::uzu_test;

use super::super::{MoeBlockError, validate_expert_counts};
use crate::backends::cpu::Cpu;

fn model_dim_error(model_dim: u32) -> MoeBlockError<Cpu> {
    let context = <Cpu as Backend>::Context::new().expect("create CPU context");
    let file = empty_parameter_file();
    let loader =
        ParameterLoader::<Cpu>::new_random(file.as_file(), context.as_ref(), 0).expect("load empty parameter file");

    match MoeBlock::<Cpu>::new(context.as_ref(), &config(1, 1), model_dim, DataType::BF16, &loader.tree()) {
        Ok(_) => panic!("invalid model dimension was accepted"),
        Err(error) => error,
    }
}

#[uzu_test]
fn rejects_invalid_routed_expert_counts() {
    for routed_experts in [0, 513] {
        let error = validate_expert_counts::<Cpu>(routed_experts, 1);
        assert!(matches!(error, Err(MoeBlockError::InvalidRoutedExpertCount)));
    }
}

#[uzu_test]
fn rejects_invalid_active_expert_counts() {
    for (routed_experts, active_experts) in [(1, 0), (1, 2), (512, 129)] {
        let error = validate_expert_counts::<Cpu>(routed_experts, active_experts);
        assert!(matches!(error, Err(MoeBlockError::InvalidActiveExpertCount)));
    }
}

#[uzu_test]
fn rejects_router_dimensions_beyond_its_threadgroup_cache() {
    for model_dim in [0, 2, 4100] {
        assert!(matches!(model_dim_error(model_dim), MoeBlockError::InvalidModelDim));
    }
}
