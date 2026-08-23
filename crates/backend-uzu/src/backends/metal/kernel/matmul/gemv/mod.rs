mod kernel;
pub(crate) mod mxfp4_expert_decode;
mod policy;

#[cfg(test)]
pub(crate) use kernel::GemvSpecialization;
pub(crate) use kernel::{GemvDispatch, GemvPlan};
pub(crate) use policy::DEFAULT_GEMV_MAX_BATCH;
