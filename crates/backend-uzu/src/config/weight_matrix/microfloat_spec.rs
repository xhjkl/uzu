use proc_macros::uzu_config;

use crate::config::weight_matrix::Layout;

#[uzu_config]
#[serde(rename_all = "snake_case")]
pub enum MicrofloatScaleMode {
    Mxfp4,
    Nvfp4,
}

#[uzu_config(super::WeightMatrixSpec)]
pub struct MicrofloatSpec {
    pub bits: u32,
    pub group_size: usize,
    pub scale_mode: MicrofloatScaleMode,
    pub layout: Layout,
}

#[cfg(test)]
mod tests {
    use proc_macros::uzu_test;
    use serde_json::json;

    use super::*;
    use crate::{
        backends::cpu::Cpu, config::weight_matrix::AnyWeightMatrixSpec, encodable_block::weight_matrix::parse_spec,
    };

    #[uzu_test]
    fn parses_mxfp4_without_aliasing_nvfp4() {
        for (mode, expected) in [("mxfp4", MicrofloatScaleMode::Mxfp4), ("nvfp4", MicrofloatScaleMode::Nvfp4)] {
            let spec: AnyWeightMatrixSpec = serde_json::from_value(json!({
                "type": "MicrofloatSpec",
                "bits": 4,
                "group_size": 16,
                "scale_mode": mode,
                "layout": "output_input"
            }))
            .unwrap();
            let AnyWeightMatrixSpec::MicrofloatSpec(spec) = spec else {
                panic!("expected MicrofloatSpec");
            };
            assert_eq!(spec.scale_mode, expected);
        }
    }

    #[uzu_test]
    fn accepts_scale_formats_independently_of_group_size() {
        use crate::backends::common::microfloat::MicrofloatScaleFormat;

        for (scale_mode, scale_format) in
            [("mxfp4", MicrofloatScaleFormat::E8m0), ("nvfp4", MicrofloatScaleFormat::E4m3)]
        {
            for group_size in [16, 32] {
                let spec: AnyWeightMatrixSpec = serde_json::from_value(json!({
                    "type": "MicrofloatSpec",
                    "bits": 4,
                    "group_size": group_size,
                    "scale_mode": scale_mode,
                    "layout": "output_input"
                }))
                .unwrap();
                let parsed = parse_spec::<Cpu>(&spec).expect("supported scale-format/group-size combination");
                let info = parsed.microfloat.expect("microfloat info");
                assert_eq!(info.scale_format, scale_format);
                assert_eq!(info.group_size, group_size);
            }
        }
    }
}
