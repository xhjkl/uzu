use std::num::NonZeroU32;

use proc_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, Context, Encoder, Kernels,
            kernel::matmul::{
                ExpertInput, ExpertRoutes, MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel, MatmulRouting,
            },
            microfloat::{MicrofloatLayout, MicrofloatMetadata, MicrofloatScaleFormat},
        },
        cpu::Cpu,
    },
    data_type::DataType,
    tests::{
        assert::assert_eq_float,
        helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec, for_each_non_cpu_backend},
    },
};

const EXPERTS: usize = 3;
const K: usize = 32;
const N: usize = 4;
const GROUP_SIZE: u32 = 16;

fn independent_e2m1(code: u8) -> f32 {
    const VALUES: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];
    VALUES[usize::from(code)]
}

fn independent_e4m3(bits: u8) -> f32 {
    let sign = if bits & 0x80 == 0 {
        1.0
    } else {
        -1.0
    };
    let exponent = (bits >> 3) & 0x0f;
    let mantissa = bits & 0x07;
    if exponent == 0 {
        return sign * f32::from(mantissa) / 512.0;
    }
    if exponent == 15 && mantissa == 7 {
        return f32::NAN;
    }
    sign * 2.0f32.powi(i32::from(exponent) - 7) * (1.0 + f32::from(mantissa) / 8.0)
}

fn independent_nvfp4(
    code: u8,
    scale: u8,
    outer_scale: f32,
) -> f32 {
    independent_e2m1(code) * independent_e4m3(scale) * outer_scale
}

fn packed_codes(
    experts: usize,
    n: usize,
    k: usize,
) -> Vec<u8> {
    let mut codes = vec![0u8; experts * n * k / 2];
    for matrix in 0..experts {
        for row in 0..n {
            for inner in (0..k).step_by(2) {
                let low = ((matrix + row + inner) % 7 + 1) as u8;
                let high = ((matrix * 3 + row + inner + 1) % 7 + 1) as u8;
                codes[(matrix * n + row) * k / 2 + inner / 2] = low | (high << 4);
            }
        }
    }
    codes
}

fn e4m3_scales(count: usize) -> Vec<u8> {
    const CODES: [u8; 4] = [0x30, 0x38, 0x3c, 0x40];
    (0..count).map(|index| CODES[index % CODES.len()]).collect()
}

fn run<B: Backend>(
    format: MicrofloatScaleFormat,
    group_size: u32,
    route_count: usize,
    routes_per_token: u32,
    input_layout: ExpertInput,
) -> (Vec<f32>, Vec<f32>) {
    let input_rows = if input_layout == ExpertInput::Tokens {
        assert!(route_count.is_multiple_of(routes_per_token as usize));
        route_count / routes_per_token as usize
    } else {
        route_count
    };
    let input: Vec<f32> = (0..input_rows * K).map(|index| (index % 11) as f32 * 0.1 - 0.4).collect();
    let codes = packed_codes(EXPERTS, N, K);
    let scales = e4m3_scales(EXPERTS * N * K / group_size as usize);
    let outer_scales = [1.0f32, 2.0, 0.5];
    let biases: Vec<f32> = (0..EXPERTS * N).map(|index| index as f32 * 0.01).collect();
    let expert_ids: Vec<i32> = (0..route_count)
        .map(|route| {
            if route % 7 == 3 {
                -1
            } else {
                ((route * 2 + 1) % EXPERTS) as i32
            }
        })
        .collect();
    let metadata = MicrofloatMetadata::new(
        format,
        4,
        group_size,
        MicrofloatLayout::OutputInput,
        EXPERTS as u32,
        N as u32,
        K as u32,
    )
    .unwrap();

    let context = B::Context::new().expect("create backend context");
    let input_alloc = alloc_allocation_with_data::<B, f32>(context.as_ref(), &input);
    let codes_alloc = alloc_allocation_with_data::<B, u8>(context.as_ref(), &codes);
    let scales_alloc = alloc_allocation_with_data::<B, u8>(context.as_ref(), &scales);
    let outer_scales_alloc = alloc_allocation_with_data::<B, f32>(context.as_ref(), &outer_scales);
    let biases_alloc = alloc_allocation_with_data::<B, f32>(context.as_ref(), &biases);
    let ids_alloc = alloc_allocation_with_data::<B, i32>(context.as_ref(), &expert_ids);
    let mut output = alloc_allocation::<B, f32>(context.as_ref(), route_count * N);
    let mut kernel =
        <B::Kernels as Kernels>::MatmulKernel::new(context.as_ref(), DataType::F32, DataType::F32, DataType::F32)
            .unwrap();
    let mut encoder = Encoder::<B>::new(context.as_ref()).unwrap();
    kernel
        .encode(
            MatmulArguments {
                a: MatmulA::FullPrecision {
                    values: &input_alloc,
                    offset: 0,
                },
                b: MatmulB::<B>::Microfloat {
                    codes: &codes_alloc,
                    scales: &scales_alloc,
                    outer_scales: &outer_scales_alloc,
                    metadata,
                },
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut output,
                d_transform: MatmulDOps {
                    per_matrix_bias: Some(&biases_alloc),
                    ..MatmulDOps::none()
                },
                routing: MatmulRouting::Experts(ExpertRoutes {
                    expert_ids: &ids_alloc,
                    routes_per_token: NonZeroU32::new(routes_per_token).unwrap(),
                    expert_count: NonZeroU32::new(EXPERTS as u32).unwrap(),
                    input: input_layout,
                }),
                m: route_count as u32,
                n: N as u32,
                k: K as u32,
            },
            &mut encoder,
        )
        .unwrap();
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    let actual = allocation_to_vec::<B, f32>(&output);

    let mut expected = vec![0.0f32; route_count * N];
    for route in 0..route_count {
        let expert = expert_ids[route];
        if expert < 0 {
            continue;
        }
        let expert = expert as usize;
        let input_row = if input_layout == ExpertInput::Tokens {
            route / routes_per_token as usize
        } else {
            route
        };
        for row in 0..N {
            let mut value = biases[expert * N + row];
            for inner in 0..K {
                let packed = codes[(expert * N + row) * K / 2 + inner / 2];
                let code = if inner.is_multiple_of(2) {
                    packed & 0x0f
                } else {
                    packed >> 4
                };
                let scale_index = (expert * N + row) * K / group_size as usize + inner / group_size as usize;
                value += input[input_row * K + inner]
                    * match format {
                        MicrofloatScaleFormat::E4m3 => {
                            independent_nvfp4(code, scales[scale_index], outer_scales[expert])
                        },
                        MicrofloatScaleFormat::E8m0 => {
                            independent_e2m1(code) * format.decode(scales[scale_index]) * outer_scales[expert]
                        },
                    };
            }
            expected[route * N + row] = value;
        }
    }
    (actual, expected)
}

fn run_tiny_output<B: Backend>(
    n: usize,
    k: usize,
) -> Vec<f32> {
    let input: Vec<f32> = (0..EXPERTS * k).map(|index| (index % 19) as f32 * 0.03 - 0.27).collect();
    let codes = packed_codes(EXPERTS, n, k);
    let scales = e4m3_scales(EXPERTS * n * k / GROUP_SIZE as usize);
    let outer_scales = [0.5f32, 1.0, 1.5];
    let biases: Vec<f32> = (0..EXPERTS * n).map(|index| index as f32 * 0.01 - 0.02).collect();
    let expert_ids = [2, 0, 1];
    let metadata = MicrofloatMetadata::new(
        MicrofloatScaleFormat::E4m3,
        4,
        GROUP_SIZE,
        MicrofloatLayout::OutputInput,
        EXPERTS as u32,
        n as u32,
        k as u32,
    )
    .unwrap();

    let context = B::Context::new().expect("create backend context");
    let input = alloc_allocation_with_data::<B, f32>(context.as_ref(), &input);
    let codes = alloc_allocation_with_data::<B, u8>(context.as_ref(), &codes);
    let scales = alloc_allocation_with_data::<B, u8>(context.as_ref(), &scales);
    let outer_scales = alloc_allocation_with_data::<B, f32>(context.as_ref(), &outer_scales);
    let biases = alloc_allocation_with_data::<B, f32>(context.as_ref(), &biases);
    let expert_ids = alloc_allocation_with_data::<B, i32>(context.as_ref(), &expert_ids);
    let mut output = alloc_allocation::<B, f32>(context.as_ref(), EXPERTS * n);
    let mut kernel =
        <B::Kernels as Kernels>::MatmulKernel::new(context.as_ref(), DataType::F32, DataType::F32, DataType::F32)
            .unwrap();
    let mut encoder = Encoder::<B>::new(context.as_ref()).unwrap();
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
                d_transform: MatmulDOps {
                    per_matrix_bias: Some(&biases),
                    ..MatmulDOps::none()
                },
                routing: MatmulRouting::Experts(ExpertRoutes {
                    expert_ids: &expert_ids,
                    routes_per_token: NonZeroU32::new(1).unwrap(),
                    expert_count: NonZeroU32::new(EXPERTS as u32).unwrap(),
                    input: ExpertInput::Tokens,
                }),
                m: EXPERTS as u32,
                n: n as u32,
                k: k as u32,
            },
            &mut encoder,
        )
        .unwrap();
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    allocation_to_vec::<B, f32>(&output)
}

fn rejection<B: Backend>(
    direct_routes: bool,
    matrix_count: u32,
    expert_count: u32,
) -> String {
    let metadata = MicrofloatMetadata::new(
        MicrofloatScaleFormat::E4m3,
        4,
        GROUP_SIZE,
        MicrofloatLayout::OutputInput,
        matrix_count,
        N as u32,
        K as u32,
    )
    .unwrap();
    let context = B::Context::new().expect("create backend context");
    let input = alloc_allocation_with_data::<B, f32>(context.as_ref(), &[1.0; K]);
    let codes = alloc_allocation_with_data::<B, u8>(context.as_ref(), &vec![0x22; metadata.required_code_bytes()]);
    let scales = alloc_allocation_with_data::<B, u8>(context.as_ref(), &e4m3_scales(metadata.required_scale_bytes()));
    let outer_scales = alloc_allocation_with_data::<B, f32>(context.as_ref(), &vec![1.0; matrix_count as usize]);
    let expert_ids = alloc_allocation_with_data::<B, i32>(context.as_ref(), &[0]);
    let mut output = alloc_allocation::<B, f32>(context.as_ref(), N);
    let mut kernel =
        <B::Kernels as Kernels>::MatmulKernel::new(context.as_ref(), DataType::F32, DataType::F32, DataType::F32)
            .unwrap();
    let mut encoder = Encoder::<B>::new(context.as_ref()).unwrap();
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
                d_transform: MatmulDOps::none(),
                routing: if direct_routes {
                    MatmulRouting::Experts(ExpertRoutes {
                        expert_ids: &expert_ids,
                        routes_per_token: NonZeroU32::new(1).unwrap(),
                        expert_count: NonZeroU32::new(expert_count).unwrap(),
                        input: ExpertInput::Tokens,
                    })
                } else {
                    MatmulRouting::Dense
                },
                m: 1,
                n: N as u32,
                k: K as u32,
            },
            &mut encoder,
        )
        .expect_err("invalid NVFP4 routing was accepted")
        .to_string()
}

#[uzu_test]
fn backends_decode_small_and_large_nvfp4_routes() {
    for group_size in [16, 32] {
        for (route_count, routes_per_token) in [(4, 2), (33, 3)] {
            for input_layout in [ExpertInput::Tokens, ExpertInput::Routes] {
                let (cpu, expected) =
                    run::<Cpu>(MicrofloatScaleFormat::E4m3, group_size, route_count, routes_per_token, input_layout);
                assert_eq_float(&expected, &cpu, 1e-5, "CPU E4M3-scaled expert routes");
                for_each_non_cpu_backend!(|B| {
                    let (actual, _) =
                        run::<B>(MicrofloatScaleFormat::E4m3, group_size, route_count, routes_per_token, input_layout);
                    assert_eq_float(&cpu, &actual, 1e-4, "Metal E4M3-scaled expert routes");
                });
            }
        }
    }
}

#[uzu_test]
fn nvfp4_is_not_an_mxfp4_group16_alias() {
    let (nvfp4, nvfp4_expected) = run::<Cpu>(MicrofloatScaleFormat::E4m3, 16, 4, 2, ExpertInput::Tokens);
    let (mxfp4, _) = run::<Cpu>(MicrofloatScaleFormat::E8m0, 16, 4, 2, ExpertInput::Tokens);
    assert_eq_float(&nvfp4_expected, &nvfp4, 1e-5, "NVFP4 independent E4M3 oracle");
    let max_delta = nvfp4.iter().zip(mxfp4.iter()).map(|(left, right)| (left - right).abs()).fold(0.0f32, f32::max);
    assert!(max_delta > 0.1, "NVFP4 matched MXFP4 on E4M3 scale bytes (delta {max_delta})");
}

#[uzu_test]
fn tiny_nvfp4_output_rows_cover_aligned_and_unaligned_k() {
    for n in 1..=3 {
        for k in [32, 48, 256] {
            let expected = run_tiny_output::<Cpu>(n, k);
            for_each_non_cpu_backend!(|B| {
                let actual = run_tiny_output::<B>(n, k);
                assert_eq_float(&expected, &actual, 1e-4, "tiny direct NVFP4 rows");
            });
        }
    }
}

#[uzu_test]
fn backends_reject_incomplete_nvfp4_route_contracts() {
    for (direct_routes, matrix_count, expert_count, expected) in [
        (false, 1, 1, "microfloat weights require direct expert routes"),
        (true, 1, 2, "microfloat storage does not match the requested expert bank"),
    ] {
        let error = rejection::<Cpu>(direct_routes, matrix_count, expert_count);
        assert!(error.contains(expected), "{error}");
        for_each_non_cpu_backend!(|B| {
            let error = rejection::<B>(direct_routes, matrix_count, expert_count);
            assert!(error.contains(expected), "{error}");
        });
    }
}
