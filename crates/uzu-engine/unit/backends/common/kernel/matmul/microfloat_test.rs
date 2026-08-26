use std::fmt::Display;

use half::bf16;
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, Context, Encoder, Kernels,
            kernel::matmul::{MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, MatmulRouting},
            microfloat::{MicrofloatEncoding, MicrofloatFormat, MicrofloatMetadata, decode_mxfp4},
        },
        cpu::Cpu,
    },
    data_type::DataType,
    tests::{
        assert::assert_eq_float,
        helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec, for_each_non_cpu_backend},
    },
};

const K: usize = 32;
const N: usize = 4;
const E2M1_THREE_CODE: u8 = 5;

#[derive(Clone, Copy)]
enum OutputOps {
    None,
    Scale,
    All,
}

#[derive(Clone, Copy)]
struct Mxfp4Case {
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
    outer_scale: f32,
    uniform_code: Option<u8>,
    uniform_input: Option<f32>,
    output_ops: OutputOps,
    tolerance: f32,
}

fn packed_codes(
    rows: usize,
    columns: usize,
) -> Vec<u8> {
    (0..rows * columns / 2)
        .map(|index| {
            let low = ((index * 5 + 1) % 16) as u8;
            let high = ((index * 7 + 3) % 16) as u8;
            low | (high << 4)
        })
        .collect()
}

fn execute_mxfp4<B: Backend, T: ArrayElement + Float>(case: Mxfp4Case) -> Vec<T> {
    let Mxfp4Case {
        m,
        n,
        k,
        group_size,
        outer_scale,
        uniform_code,
        uniform_input,
        output_ops,
        tolerance: _,
    } = case;
    let encoding =
        MicrofloatEncoding::new(MicrofloatFormat::Mxfp4, 4, group_size as u32).expect("valid MXFP4 encoding");
    let metadata = MicrofloatMetadata::new(encoding, 1, n as u32, k as u32).expect("valid dense MXFP4 metadata");
    let context = B::Context::new().expect("create backend context");
    let input_values: Vec<T> =
        (0..m * k).map(|index| T::from(uniform_input.unwrap_or((index % 13) as f32 * 0.125 - 0.5)).unwrap()).collect();
    let codes = match uniform_code {
        Some(code) => vec![code | code << 4; n * k / 2],
        None => packed_codes(n, k),
    };
    let scales: Vec<u8> = (0..n * k / group_size).map(|index| 126 + (index % 3) as u8).collect();
    let outer_scale = [T::from(outer_scale).unwrap()];
    let output_values: Vec<T> =
        (0..m * n).map(|index| T::from((index % 7) as f32 * 0.03125 - 0.0625).unwrap()).collect();
    let bias_values: Vec<T> = (0..n).map(|index| T::from((index % 5) as f32 * 0.0625 - 0.125).unwrap()).collect();
    let input = alloc_allocation_with_data::<B, T>(context.as_ref(), &input_values);
    let codes = alloc_allocation_with_data::<B, u8>(context.as_ref(), &codes);
    let scales = alloc_allocation_with_data::<B, u8>(context.as_ref(), &scales);
    let outer_scales = alloc_allocation_with_data::<B, T>(context.as_ref(), &outer_scale);
    let bias = alloc_allocation_with_data::<B, T>(context.as_ref(), &bias_values);
    let mut output = alloc_allocation_with_data::<B, T>(context.as_ref(), &output_values);
    let mut kernel =
        <B::Kernels as Kernels>::MatmulKernel::new(context.as_ref(), T::data_type(), T::data_type(), T::data_type())
            .expect("create matmul kernel");
    let mut encoder = Encoder::<B>::new(context.as_ref()).expect("create encoder");
    let d_transform = match output_ops {
        OutputOps::None => MatmulDOps::none(),
        OutputOps::Scale => MatmulDOps {
            ab_scale: 0.75,
            ..MatmulDOps::none()
        },
        OutputOps::All => MatmulDOps {
            ab_scale: 0.75,
            accumulate: true,
            bias: Some(&bias),
            rht_factors: None,
            soft_cap: Some(2.0),
            ..MatmulDOps::none()
        },
    };
    kernel
        .encode(
            MatmulArguments {
                a: MatmulA::FullPrecision {
                    values: &input,
                    offset: 0,
                },
                b: MatmulB::<B>::Microfloat {
                    codes: &codes,
                    scales: &scales,
                    outer_scales: &outer_scales,
                    metadata,
                },
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut output,
                d_transform,
                routing: MatmulRouting::Dense,
                m: m as u32,
                n: n as u32,
                k: k as u32,
            },
            &mut encoder,
        )
        .expect("encode MXFP4 matmul");
    encoder.end_encoding().submit().wait_until_completed().expect("execute MXFP4 matmul");
    allocation_to_vec::<B, T>(&output)
}

#[uzu_test]
fn cpu_executes_dense_mxfp4_matmul() {
    for row_count in [1, 5] {
        for group_size in [16, 32] {
            let input_values: Vec<f32> = (0..row_count * K).map(|index| (index % 13) as f32 * 0.125 - 0.5).collect();
            let codes = packed_codes(N, K);
            let scales: Vec<u8> = (0..N * K / group_size).map(|index| 126 + (index % 3) as u8).collect();
            let outer_scales = [1.25f32];
            let encoding =
                MicrofloatEncoding::new(MicrofloatFormat::Mxfp4, 4, group_size as u32).expect("valid MXFP4 encoding");
            let metadata =
                MicrofloatMetadata::new(encoding, 1, N as u32, K as u32).expect("valid dense MXFP4 metadata");

            let context = <Cpu as Backend>::Context::new().expect("create CPU context");
            let input = alloc_allocation_with_data::<Cpu, f32>(context.as_ref(), &input_values);
            let codes = alloc_allocation_with_data::<Cpu, u8>(context.as_ref(), &codes);
            let scales = alloc_allocation_with_data::<Cpu, u8>(context.as_ref(), &scales);
            let outer_scales = alloc_allocation_with_data::<Cpu, f32>(context.as_ref(), &outer_scales);
            let mut output = alloc_allocation::<Cpu, f32>(context.as_ref(), row_count * N);
            let mut kernel = <<Cpu as Backend>::Kernels as Kernels>::MatmulKernel::new(
                context.as_ref(),
                DataType::F32,
                DataType::F32,
                DataType::F32,
            )
            .expect("create CPU matmul kernel");
            let mut encoder = Encoder::<Cpu>::new(context.as_ref()).expect("create CPU encoder");
            kernel
                .encode(
                    MatmulArguments {
                        a: MatmulA::FullPrecision {
                            values: &input,
                            offset: 0,
                        },
                        b: MatmulB::<Cpu>::Microfloat {
                            codes: &codes,
                            scales: &scales,
                            outer_scales: &outer_scales,
                            metadata,
                        },
                        b_leading_dimension: None,
                        b_transpose: true,
                        d: &mut output,
                        d_transform: MatmulDOps::none(),
                        routing: MatmulRouting::Dense,
                        m: row_count as u32,
                        n: N as u32,
                        k: K as u32,
                    },
                    &mut encoder,
                )
                .expect("encode dense MXFP4 matmul");
            encoder.end_encoding().submit().wait_until_completed().expect("execute dense MXFP4 matmul");
            let actual = allocation_to_vec::<Cpu, f32>(&output);

            let codes: &[u8] = codes.as_slice();
            let scales: &[u8] = scales.as_slice();
            let mut expected = vec![0.0f32; row_count * N];
            for row in 0..row_count {
                for output_row in 0..N {
                    for inner in 0..K {
                        let packed = codes[output_row * K / 2 + inner / 2];
                        let code = if inner.is_multiple_of(2) {
                            packed & 0x0f
                        } else {
                            packed >> 4
                        };
                        let scale = scales[output_row * K / group_size + inner / group_size];
                        expected[row * N + output_row] +=
                            input_values[row * K + inner] * decode_mxfp4(code, scale, 1.25);
                    }
                }
            }
            assert_eq_float(&expected, &actual, 1e-5, "CPU dense MXFP4");
        }
    }
}

#[uzu_test]
fn metal_matches_cpu_for_mxfp4_gemv_and_gemm() {
    fn compare<T: ArrayElement + Float + Display>(case: Mxfp4Case) {
        let expected = execute_mxfp4::<Cpu, T>(case);
        for_each_non_cpu_backend!(|B| {
            let actual = execute_mxfp4::<B, T>(case);
            assert_eq_float(&expected, &actual, case.tolerance, std::any::type_name::<B>());
        });
    }

    let case = |m, n, group_size| Mxfp4Case {
        m,
        n,
        k: 160,
        group_size,
        outer_scale: 1.3,
        uniform_code: None,
        uniform_input: None,
        output_ops: OutputOps::All,
        tolerance: 0.05,
    };

    compare::<f32>(Mxfp4Case {
        m: 1,
        n: 1,
        k: 32,
        group_size: 16,
        outer_scale: 1.3,
        uniform_code: None,
        uniform_input: None,
        output_ops: OutputOps::None,
        tolerance: 1e-4,
    });
    compare::<bf16>(case(8, 32, 32));
    compare::<bf16>(case(9, 17, 16));
    compare::<bf16>(case(9, 17, 32));
    compare::<f32>(Mxfp4Case {
        tolerance: 1e-3,
        ..case(9, 17, 32)
    });
    let outer_scale_rounding = Mxfp4Case {
        m: 9,
        n: 1,
        k: 32,
        group_size: 32,
        outer_scale: 1.0078125,
        uniform_code: Some(E2M1_THREE_CODE),
        uniform_input: Some(1.0),
        output_ops: OutputOps::Scale,
        tolerance: 0.0,
    };
    compare::<bf16>(outer_scale_rounding);
}
