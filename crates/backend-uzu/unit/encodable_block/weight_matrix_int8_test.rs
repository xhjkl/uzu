use backend_uzu_macros::uzu_test;
use half::bf16;

use super::{Microfloat, WeightMatrix, WeightMatrixError, WeightStorage};
use crate::{
    backends::{
        common::{
            Backend, Context,
            gpu_types::QuantizationMode,
            kernel::matmul::MatmulB,
            microfloat::{MicrofloatFormat, MicrofloatLayout, MicrofloatMetadata, e2m1_to_exact_i8},
        },
        cpu::Cpu,
    },
    data_type::DataType,
    tests::helpers::{alloc_allocation_with_data, allocation_to_vec},
};

fn matrix(exponents: &[u8]) -> (std::sync::Arc<<Cpu as Backend>::Context>, WeightMatrix<Cpu>) {
    let context = <Cpu as Backend>::Context::new().expect("CPU context");
    let metadata =
        MicrofloatMetadata::new(MicrofloatFormat::Mxfp4, 4, 16, MicrofloatLayout::OutputInput, 1, 2, 32).unwrap();
    let packed_codes: Vec<u8> = (0..metadata.required_code_bytes())
        .map(|index| ((index * 2 + 1) as u8 & 0x0f) | (((index * 2 + 2) as u8 & 0x0f) << 4))
        .collect();
    let values = alloc_allocation_with_data::<Cpu, u8>(&context, &packed_codes);
    let scales = alloc_allocation_with_data::<Cpu, u8>(&context, exponents);
    let outer_scales = alloc_allocation_with_data::<Cpu, bf16>(&context, &[bf16::from_f32(-0.75)]);
    (
        context,
        WeightMatrix {
            storage: WeightStorage::Microfloat {
                values,
                microfloat: Microfloat {
                    scales,
                    outer_scales,
                    outer_scale_data_type: DataType::BF16,
                    metadata,
                },
            },
        },
    )
}

#[uzu_test]
fn prepares_all_group16_codes_and_merges_duplicate_scales() {
    let (context, mut matrix) = matrix(&[127, 127, 128, 128]);
    matrix.prepare_mxfp4_int8(&context).expect("prepare exact INT8 bank");

    let MatmulB::ScaleSymmetricDequant {
        b,
        scales,
        mode,
        group_size,
        signed_codes,
    } = matrix.matmul_b()
    else {
        panic!("prepared MXFP4 bank did not become symmetric INT8");
    };
    let expected_codes: Vec<i8> = (0..64).map(|index| e2m1_to_exact_i8((index as u8 + 1) & 0x0f)).collect();
    assert_eq!(allocation_to_vec::<Cpu, i8>(b), expected_codes);
    assert_eq!(allocation_to_vec::<Cpu, bf16>(scales), [bf16::from_f32(-0.375), bf16::from_f32(-0.75)]);
    assert_eq!(mode, QuantizationMode::I8);
    assert_eq!(group_size, 32);
    assert!(signed_codes);
}

#[uzu_test]
fn rejects_group16_scales_that_cannot_be_merged_exactly() {
    let (context, mut matrix) = matrix(&[127, 128, 128, 128]);
    assert!(matches!(
        matrix.prepare_mxfp4_int8(&context),
        Err(WeightMatrixError::UnsupportedConfiguration(message)) if message.contains("cannot be merged exactly")
    ));
}
