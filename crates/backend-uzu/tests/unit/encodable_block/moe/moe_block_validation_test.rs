use proc_macros::uzu_test;

use super::super::{MoeBlockError, validate_expert_counts};
use crate::backends::cpu::Cpu;

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
