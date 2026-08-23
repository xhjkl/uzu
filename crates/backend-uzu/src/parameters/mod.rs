// TODO: This is overdue for a complete rewrite

mod loader;
mod safetensors_metadata;

pub use loader::{ParameterLoader, ParameterLoaderError, ParameterTree};
pub use safetensors_metadata::HeaderLoadingError;

pub(crate) fn has_mxfp4_expert_weights(file: &std::fs::File) -> Result<bool, HeaderLoadingError> {
    let (_, metadata) = safetensors_metadata::read_metadata(file)?;
    let Some(metadata) = metadata.metadata else {
        return Ok(false);
    };
    Ok(metadata.iter().any(|(name, spec)| {
        if !name.contains(".experts.") || !name.ends_with(".weights.spec") {
            return false;
        }
        let spec = serde_json::from_str::<serde_json::Value>(spec);
        let Ok(spec) = spec else {
            return false;
        };
        spec.get("type").and_then(serde_json::Value::as_str) == Some("MicrofloatSpec")
            && spec.get("scale_mode").and_then(serde_json::Value::as_str) == Some("mxfp4")
    }))
}
