use std::mem::size_of;

use half::bf16;
use proc_macros::uzu_test;

use super::{Microfloat, WeightMatrix, WeightMatrixError};
use crate::{
    backends::{
        common::{
            Backend, Context,
            microfloat::{MicrofloatFormat, MicrofloatLayout, MicrofloatMetadata, e2m1_to_exact_i8},
        },
        cpu::Cpu,
    },
    data_type::DataType,
    tests::helpers::{alloc_allocation_with_data, allocation_to_vec},
};

fn matrix(
    exponents: &[u8],
    group_size: u32,
) -> (std::sync::Arc<<Cpu as Backend>::Context>, WeightMatrix<Cpu>) {
    let context = <Cpu as Backend>::Context::new().expect("CPU context");
    let metadata =
        MicrofloatMetadata::new(MicrofloatFormat::Mxfp4, 4, group_size, MicrofloatLayout::OutputInput, 1, 2, 32)
            .unwrap();
    let packed_codes: Vec<u8> = (0..metadata.required_code_bytes())
        .map(|index| ((index * 2 + 1) as u8 & 0x0f) | (((index * 2 + 2) as u8 & 0x0f) << 4))
        .collect();
    let values = alloc_allocation_with_data::<Cpu, u8>(&context, &packed_codes);
    let scales = alloc_allocation_with_data::<Cpu, u8>(&context, exponents);
    let outer_scales = alloc_allocation_with_data::<Cpu, bf16>(&context, &[bf16::from_f32(-0.75)]);
    (
        context,
        WeightMatrix {
            values,
            quantized: None,
            microfloat: Some(Microfloat {
                scales,
                outer_scales,
                outer_scale_data_type: DataType::BF16,
                metadata,
            }),
        },
    )
}

#[uzu_test]
fn materializes_exact_int8_codes_and_fp32_scales_on_demand() {
    let (context, matrix) = matrix(&[0, 127], 32);
    let bank = matrix.materialize_mxfp4_int8_bank(&context).expect("derived exact bank");
    assert_eq!(bank.group_size(), 32);

    let expected_codes: Vec<i8> = (0..64).map(|index| e2m1_to_exact_i8((index as u8 + 1) & 0x0f)).collect();
    assert_eq!(allocation_to_vec::<Cpu, i8>(bank.codes()), expected_codes);
    let scales = allocation_to_vec::<Cpu, f32>(bank.scales());
    assert_eq!(scales[0].to_bits(), (crate::backends::common::microfloat::decode_e8m0(0) * -0.75 * 0.5).to_bits());
    assert_eq!(scales[1].to_bits(), (-0.75f32 * 0.5).to_bits());

    let statistics = bank.statistics();
    assert_eq!(statistics.source_code_bytes, 32);
    assert_eq!(statistics.derived_code_bytes, 64);
    assert_eq!(statistics.derived_scale_bytes, 2 * size_of::<f32>());
    assert_eq!(statistics.group_count, 2);
}

#[uzu_test]
fn rejects_invalid_e8m0_exponents() {
    let (context, matrix) = matrix(&[255, 127], 32);
    assert!(matches!(
        matrix.materialize_mxfp4_int8_bank(&context),
        Err(WeightMatrixError::UnsupportedConfiguration(message)) if message.contains("exponent 255")
    ));
}

#[uzu_test]
fn rejects_group16_weights_that_do_not_match_the_tensorops_k_tile() {
    let (context, matrix) = matrix(&[127; 4], 16);
    assert!(matches!(
        matrix.materialize_mxfp4_int8_bank(&context),
        Err(WeightMatrixError::UnsupportedConfiguration(message)) if message.contains("group-32")
    ));
}
