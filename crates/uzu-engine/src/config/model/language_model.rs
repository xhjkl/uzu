use uzu_engine_macros::uzu_config;

use crate::{
    config::{decoder::DecoderConfig, model::generation::GenerationConfig},
    data_type::DataType,
};

#[uzu_config(super::ModelConfig)]
pub struct LanguageModelConfig {
    pub decoder_config: DecoderConfig,
    pub generation_config: GenerationConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_type: Option<DataType>,
}
