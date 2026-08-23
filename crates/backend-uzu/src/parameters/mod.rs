// TODO: This is overdue for a complete rewrite

mod loader;
mod safetensors_metadata;

pub use loader::{ParameterLoader, ParameterLoaderError, ParameterTree};
pub use safetensors_metadata::HeaderLoadingError;

pub(crate) fn has_native_int8_expert_weights(file: &std::fs::File) -> Result<bool, HeaderLoadingError> {
    let (_, metadata) = safetensors_metadata::read_metadata(file)?;
    Ok(metadata.tensors.iter().any(|(name, tensor)| {
        name.contains(".experts.")
            && name.ends_with(".weights.weights")
            && tensor.dtype == safetensors_metadata::Dtype::I8
    }))
}
