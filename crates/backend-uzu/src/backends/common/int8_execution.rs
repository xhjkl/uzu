use std::fmt;

use serde::{Deserialize, Serialize};

const INT8_TENSOROPS_ENV: &str = "UZU_INT8_TENSOROPS";

/// Implementation used for decode-shaped native-INT8 expert projections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Int8Execution {
    Emulated,
    HardwareTensorOps,
}

impl Int8Execution {
    pub(crate) fn emulation_requested() -> bool {
        std::env::var(INT8_TENSOROPS_ENV).is_ok_and(|value| value.eq_ignore_ascii_case("emulate"))
    }
}

impl fmt::Display for Int8Execution {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::Emulated => formatter.write_str("emulated"),
            Self::HardwareTensorOps => formatter.write_str("hardware TensorOps"),
        }
    }
}
