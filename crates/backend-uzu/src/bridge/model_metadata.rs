use std::{fs::File, io::BufReader, path::Path};

use shoji::types::model::ModelSpecialization;

#[cfg(backend = "metal")]
use crate::backends::{common::Context, metal::MetalContext};
use crate::{
    backends::common::Int8Execution,
    config::model::AnyModelConfig,
    parameters::{HeaderLoadingError, has_native_int8_expert_weights},
};

#[derive(Debug, thiserror::Error)]
pub enum ModelMetadataError {
    #[error("Unable to open model configuration: {0}")]
    UnableToOpenConfig(#[from] std::io::Error),
    #[error("Unable to deserialize model configuration: {0}")]
    UnableToDeserializeConfig(#[from] serde_json::Error),
    #[error("Unable to open model weights: {0}")]
    UnableToOpenWeights(#[source] std::io::Error),
    #[error("Unable to inspect model weights: {0}")]
    UnableToInspectWeights(#[from] HeaderLoadingError),
    #[error("Unable to inspect INT8 execution support: {0}")]
    UnableToInspectInt8Execution(String),
}

pub fn resolve_model_specialization(model_path: &Path) -> Result<ModelSpecialization, ModelMetadataError> {
    let config_path = model_path.join("config.json");
    let file = File::open(&config_path)?;
    let config: AnyModelConfig = serde_json::from_reader(BufReader::new(file))?;
    Ok(match config {
        AnyModelConfig::LanguageModelConfig(_) => ModelSpecialization::Chat {},
        AnyModelConfig::ClassifierModelConfig(_) => ModelSpecialization::Classification {},
    })
}

/// Runtime selected for native-INT8 expert projections in this process.
pub fn resolve_int8_execution(model_path: &Path) -> Result<Option<Int8Execution>, ModelMetadataError> {
    let weights = File::open(model_path.join("model.safetensors")).map_err(ModelMetadataError::UnableToOpenWeights)?;
    if !has_native_int8_expert_weights(&weights)? {
        return Ok(None);
    }
    #[cfg(backend = "metal")]
    {
        let context = <MetalContext as Context>::new()
            .map_err(|error| ModelMetadataError::UnableToInspectInt8Execution(error.to_string()))?;
        return Ok(Some(context.int8_execution()));
    }
    #[cfg(not(backend = "metal"))]
    Ok(None)
}
