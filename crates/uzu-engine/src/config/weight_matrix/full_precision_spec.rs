use uzu_engine_macros::uzu_config;

use crate::config::weight_matrix::Layout;

#[uzu_config(super::WeightMatrixSpec)]
pub struct FullPrecisionSpec {
    pub layout: Layout,
}

impl FullPrecisionSpec {
    pub(crate) fn output_input() -> Self {
        Self {
            ty: monostate::MustBeStr,
            layout: Layout::OutputInput,
        }
    }
}
