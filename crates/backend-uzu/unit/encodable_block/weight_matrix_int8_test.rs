use std::io::Write;

use backend_uzu_macros::uzu_test;
use half::bf16;
use serde_json::{Map, Value, json};
use tempfile::NamedTempFile;

use super::WeightMatrix;
use crate::{
    backends::{
        common::{Backend, Context, gpu_types::QuantizationMode, kernel::matmul::MatmulB},
        cpu::Cpu,
    },
    config::weight_matrix::{AnyWeightMatrixSpec, Layout},
    data_type::DataType,
    parameters::ParameterLoader,
};

fn add_tensor(
    header: &mut Map<String, Value>,
    payload: &mut Vec<u8>,
    name: &str,
    shape: &[u32],
    data_type: &str,
    data: &[u8],
) {
    let begin = payload.len();
    payload.extend_from_slice(data);
    header.insert(
        name.into(),
        json!({
            "dtype": data_type,
            "shape": shape,
            "data_offsets": [begin, payload.len()]
        }),
    );
}

/// Native signed expert-bank artifact emitted by Lalamo.
fn int8_parameter_file() -> NamedTempFile {
    const EXPERTS: u32 = 2;
    const ROWS: u32 = 2;
    const COLUMNS: u32 = 32;

    let mut header = Map::new();
    header.insert(
        "spec".into(),
        json!({
            "type": "IntSpec",
            "bits": 8,
            "group_size": 32,
            "is_symmetric": true,
            "layout": "output_input"
        })
        .to_string()
        .into(),
    );
    let mut payload = Vec::new();
    let codes: Vec<i8> = (0..EXPERTS * ROWS * COLUMNS).map(|index| index as i8 % 25 - 12).collect();
    let scales = vec![bf16::from_f32(0.125); (EXPERTS * ROWS) as usize];
    add_tensor(&mut header, &mut payload, "weights", &[EXPERTS, ROWS, COLUMNS], "I8", bytemuck::cast_slice(&codes));
    add_tensor(&mut header, &mut payload, "scales", &[EXPERTS, ROWS, 1], "BF16", bytemuck::cast_slice(&scales));

    let mut header = serde_json::to_vec(&Value::Object(header)).expect("serialize safetensors header");
    header.extend(std::iter::repeat_n(b' ', (8 - header.len() % 8) % 8));
    let mut file = NamedTempFile::new().expect("create native INT8 fixture");
    file.write_all(&(header.len() as u64).to_le_bytes()).expect("write safetensors header length");
    file.write_all(&header).expect("write safetensors header");
    file.write_all(&payload).expect("write safetensors payload");
    file
}

#[uzu_test]
fn loads_native_signed_int8_expert_banks_without_rewriting_codes() {
    let context = <Cpu as Backend>::Context::new().expect("create CPU context");
    let file = int8_parameter_file();
    let loader = ParameterLoader::<Cpu>::new(file.as_file(), context.as_ref()).expect("load native INT8 fixture");
    let tree = loader.tree();
    let spec = tree.metadata::<AnyWeightMatrixSpec>("spec").expect("load native INT8 spec");

    let matrix = WeightMatrix::load_bank(&tree, spec, Layout::OutputInput, 2, 2, 32, DataType::BF16)
        .expect("load native signed INT8 bank");
    let MatmulB::ScaleSymmetricDequant {
        b,
        mode,
        group_size,
        signed_codes,
        ..
    } = matrix.matmul_b()
    else {
        panic!("native INT8 bank did not preserve its integer operand format");
    };
    assert_eq!(mode, QuantizationMode::I8);
    assert_eq!(group_size, 32);
    assert!(signed_codes);
    assert_eq!(&b.as_slice::<i8>()[..4], &[-12, -11, -10, -9]);
    tree.assert_all_tensors_validated().expect("validate native INT8 tensors");
}
