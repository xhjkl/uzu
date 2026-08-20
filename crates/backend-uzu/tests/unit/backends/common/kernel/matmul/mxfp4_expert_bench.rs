#![cfg(backend = "metal")]

//! Production-shape oracle benches for the GPT-OSS MXFP4 expert paths.
//! Reported throughput estimates physical packed-weight traffic, not FLOP/s.

use std::{num::NonZeroU32, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput};
use half::bf16;
use proc_macros::{uzu_bench, uzu_test};

use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Allocation, Backend, Encoder, Kernels,
            gpu_types::ActivationType,
            kernel::{
                GatedActMul, MoeFinalizeKernel,
                matmul::{
                    ExpertInput, ExpertRouteIdentity, ExpertRoutes, MatmulA, MatmulArguments, MatmulB, MatmulDOps,
                    MatmulKernel, MatmulRouting,
                },
            },
            microfloat::{MicrofloatFormat, MicrofloatLayout, MicrofloatMetadata},
        },
        metal::{DEFAULT_GEMV_MAX_BATCH, Metal, MetalContext},
    },
    tests::{
        helpers::{alloc_allocation, alloc_allocation_with_data},
        matmul::iter_encode_loop,
        util::{shared_metal_context, type_short_name},
    },
};

type MetalKernels = <Metal as Backend>::Kernels;
type MetalMatmul = <MetalKernels as Kernels>::MatmulKernel;
type MetalMoeFinalize = <MetalKernels as Kernels>::MoeFinalizeKernel;

struct Mxfp4Shape {
    label: &'static str,
    routes: u32,
    routes_per_token: u32,
    experts: u32,
    n: u32,
    k: u32,
    group_size: u32,
    input_is_bf16: bool,
}

#[derive(Clone, Copy)]
enum Mxfp4RouteDistribution {
    Uniform,
    FourHot,
    OneHot,
    ManyEmpty,
}

impl Mxfp4RouteDistribution {
    fn label(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::FourHot => "four_hot",
            Self::OneHot => "one_hot",
            Self::ManyEmpty => "many_empty",
        }
    }

    fn expert_ids(
        self,
        route_count: u32,
        expert_count: u32,
    ) -> Vec<i32> {
        (0..route_count)
            .map(|route| match self {
                Self::Uniform => (route % expert_count) as i32,
                Self::FourHot => (route % 4) as i32,
                Self::OneHot => 0,
                Self::ManyEmpty => ((route * 7 + 3) % 8) as i32,
            })
            .collect()
    }
}

fn mxfp4_expert_bytes(
    shape: &Mxfp4Shape,
    expert_ids: &[i32],
) -> u64 {
    let rows_per_weight_fetch = if shape.routes <= DEFAULT_GEMV_MAX_BATCH {
        1
    } else {
        4
    };
    mxfp4_expert_bytes_for_rows(shape, expert_ids, rows_per_weight_fetch)
}

fn mxfp4_expert_bytes_for_rows(
    shape: &Mxfp4Shape,
    expert_ids: &[i32],
    rows_per_weight_fetch: u64,
) -> u64 {
    let per_expert = shape.n as u64 * (shape.k as u64 / 2 + shape.k as u64 / shape.group_size as u64);
    let mut routes_per_expert = vec![0u64; shape.experts as usize];
    for &expert in expert_ids {
        let Ok(expert) = usize::try_from(expert) else {
            continue;
        };
        if let Some(route_count) = routes_per_expert.get_mut(expert) {
            *route_count += 1;
        }
    }
    let row_tiles: u64 = routes_per_expert.into_iter().map(|routes| routes.div_ceil(rows_per_weight_fetch)).sum();
    row_tiles * per_expert
}

#[allow(clippy::too_many_arguments)]
fn encode_mxfp4_moe(
    w13_kernel: &mut MetalMatmul,
    w2_kernel: &mut MetalMatmul,
    gate: &GatedActMul<Metal>,
    finalize: &MetalMoeFinalize,
    input: &Allocation<Metal>,
    w13_codes: &Allocation<Metal>,
    w13_scales: &Allocation<Metal>,
    w13_outer_scales: &Allocation<Metal>,
    w13_biases: &Allocation<Metal>,
    w13_metadata: MicrofloatMetadata,
    w2_codes: &Allocation<Metal>,
    w2_scales: &Allocation<Metal>,
    w2_outer_scales: &Allocation<Metal>,
    w2_biases: &Allocation<Metal>,
    w2_metadata: MicrofloatMetadata,
    route_identity: &ExpertRouteIdentity,
    expert_ids: &Allocation<Metal>,
    route_weights: &Allocation<Metal>,
    fused_up: &mut Allocation<Metal>,
    hidden: &mut Allocation<Metal>,
    route_outputs: &mut Allocation<Metal>,
    output: &mut Allocation<Metal>,
    token_count: u32,
    routes_per_token: NonZeroU32,
    expert_count: NonZeroU32,
    model_dim: u32,
    hidden_dim: u32,
    encoder: &mut Encoder<Metal>,
) {
    let route_count = token_count * routes_per_token.get();
    w13_kernel
        .encode(
            MatmulArguments {
                a: MatmulA::FullPrecision {
                    values: input,
                    offset: 0,
                },
                b: MatmulB::Microfloat {
                    codes: w13_codes,
                    scales: w13_scales,
                    outer_scales: w13_outer_scales,
                    metadata: w13_metadata,
                },
                b_leading_dimension: None,
                b_transpose: true,
                d: fused_up,
                d_transform: MatmulDOps {
                    per_matrix_bias: Some(w13_biases),
                    ..MatmulDOps::none()
                },
                routing: MatmulRouting::Experts(ExpertRoutes {
                    identity: route_identity,
                    expert_ids,
                    routes_per_token,
                    expert_count,
                    input: ExpertInput::Tokens,
                }),
                m: route_count,
                n: 2 * hidden_dim,
                k: model_dim,
            },
            encoder,
        )
        .expect("whole-MoE W13");
    gate.encode_fp(
        fused_up,
        None,
        hidden,
        None,
        hidden_dim,
        route_count,
        0,
        0,
        ActivationType::SILU,
        Some(1.702),
        Some(-7.0),
        Some(7.0),
        Some(-7.0),
        Some(7.0),
        encoder,
    );
    w2_kernel
        .encode(
            MatmulArguments {
                a: MatmulA::FullPrecision {
                    values: hidden,
                    offset: 0,
                },
                b: MatmulB::Microfloat {
                    codes: w2_codes,
                    scales: w2_scales,
                    outer_scales: w2_outer_scales,
                    metadata: w2_metadata,
                },
                b_leading_dimension: None,
                b_transpose: true,
                d: route_outputs,
                d_transform: MatmulDOps {
                    per_matrix_bias: Some(w2_biases),
                    ..MatmulDOps::none()
                },
                routing: MatmulRouting::Experts(ExpertRoutes {
                    identity: route_identity,
                    expert_ids,
                    routes_per_token,
                    expert_count,
                    input: ExpertInput::Routes,
                }),
                m: route_count,
                n: model_dim,
                k: hidden_dim,
            },
            encoder,
        )
        .expect("whole-MoE W2");
    finalize.encode(route_weights, route_outputs, output, token_count, model_dim, routes_per_token.get(), encoder);
}

#[uzu_test]
fn grouped_byte_estimate_tracks_route_row_tiles() {
    let shape = Mxfp4Shape {
        label: "estimate",
        routes: 284,
        routes_per_token: 4,
        experts: 32,
        n: 5760,
        k: 2880,
        group_size: 16,
        input_is_bf16: true,
    };
    let bytes_per_matrix = shape.n as u64 * (shape.k as u64 / 2 + shape.k as u64 / shape.group_size as u64);
    let uniform = Mxfp4RouteDistribution::Uniform.expert_ids(shape.routes, shape.experts);
    let one_hot = Mxfp4RouteDistribution::OneHot.expert_ids(shape.routes, shape.experts);

    assert_eq!(mxfp4_expert_bytes_for_rows(&shape, &uniform, 4) / bytes_per_matrix, 92);
    assert_eq!(mxfp4_expert_bytes_for_rows(&shape, &one_hot, 4) / bytes_per_matrix, 71);
}

fn bench_mxfp4_expert_projection(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    context: &MetalContext,
    shape: &Mxfp4Shape,
    expert_ids_value: &[i32],
    ids_label: &str,
) {
    let routes = shape.routes as usize;
    let experts = shape.experts as usize;
    let (n, k) = (shape.n as usize, shape.k as usize);
    let codes = alloc_allocation_with_data::<Metal, u8>(context, &vec![0x1Eu8; experts * n * k / 2]);
    let scales =
        alloc_allocation_with_data::<Metal, u8>(context, &vec![127u8; experts * n * k / shape.group_size as usize]);
    let outer_scales = alloc_allocation_with_data::<Metal, bf16>(context, &vec![bf16::from_f32(1.0); experts]);
    let biases = alloc_allocation_with_data::<Metal, bf16>(context, &vec![bf16::from_f32(0.0); experts * n]);
    let metadata = MicrofloatMetadata::new(
        MicrofloatFormat::Mxfp4,
        4,
        shape.group_size,
        MicrofloatLayout::OutputInput,
        shape.experts,
        shape.n,
        shape.k,
    )
    .unwrap();
    let expert_ids = alloc_allocation_with_data::<Metal, i32>(context, expert_ids_value);
    let route_identity = ExpertRouteIdentity::new();
    let mut output = alloc_allocation::<Metal, u8>(context, routes * n * 4);
    let routes_per_token = NonZeroU32::new(shape.routes_per_token).unwrap();
    let expert_count = NonZeroU32::new(shape.experts).unwrap();

    let label = format!("{}_{}", shape.label, ids_label);
    group.throughput(Throughput::Bytes(mxfp4_expert_bytes(shape, expert_ids_value)));
    if shape.input_is_bf16 {
        let input = alloc_allocation::<Metal, bf16>(context, routes / shape.routes_per_token as usize * k);
        let mut matmul = <MetalKernels as Kernels>::MatmulKernel::new(
            context,
            bf16::data_type(),
            bf16::data_type(),
            crate::data_type::DataType::F32,
        )
        .expect("MatmulKernel");
        group.bench_function(BenchmarkId::new("mxfp4_routed", &label), |bencher| {
            iter_encode_loop::<Metal, _>(context, bencher, |encoder| {
                let b: MatmulB<'_, Metal> = MatmulB::Microfloat {
                    codes: &codes,
                    scales: &scales,
                    outer_scales: &outer_scales,
                    metadata,
                };
                matmul
                    .encode(
                        MatmulArguments {
                            a: MatmulA::FullPrecision {
                                values: &input,
                                offset: 0,
                            },
                            b,
                            b_leading_dimension: None,
                            b_transpose: true,
                            d: &mut output,
                            d_transform: MatmulDOps {
                                per_matrix_bias: Some(&biases),
                                ..MatmulDOps::none()
                            },
                            routing: MatmulRouting::Experts(ExpertRoutes {
                                identity: &route_identity,
                                expert_ids: &expert_ids,
                                routes_per_token,
                                expert_count,
                                input: ExpertInput::Tokens,
                            }),
                            m: shape.routes,
                            n: shape.n,
                            k: shape.k,
                        },
                        encoder,
                    )
                    .expect("mxfp4 routed matmul");
            });
        });
    } else {
        let input = alloc_allocation::<Metal, f32>(context, routes * k);
        let mut matmul = <MetalKernels as Kernels>::MatmulKernel::new(
            context,
            bf16::data_type(),
            crate::data_type::DataType::F32,
            bf16::data_type(),
        )
        .expect("MatmulKernel");
        group.bench_function(BenchmarkId::new("mxfp4_routed", &label), |bencher| {
            iter_encode_loop::<Metal, _>(context, bencher, |encoder| {
                let b: MatmulB<'_, Metal> = MatmulB::Microfloat {
                    codes: &codes,
                    scales: &scales,
                    outer_scales: &outer_scales,
                    metadata,
                };
                matmul
                    .encode(
                        MatmulArguments {
                            a: MatmulA::FullPrecision {
                                values: &input,
                                offset: 0,
                            },
                            b,
                            b_leading_dimension: None,
                            b_transpose: true,
                            d: &mut output,
                            d_transform: MatmulDOps {
                                per_matrix_bias: Some(&biases),
                                ..MatmulDOps::none()
                            },
                            routing: MatmulRouting::Experts(ExpertRoutes {
                                identity: &route_identity,
                                expert_ids: &expert_ids,
                                routes_per_token,
                                expert_count,
                                input: ExpertInput::Routes,
                            }),
                            m: shape.routes,
                            n: shape.n,
                            k: shape.k,
                        },
                        encoder,
                    )
                    .expect("mxfp4 routed matmul");
            });
        });
    }
}

#[uzu_bench]
fn bench_mxfp4_expert_decode_production(c: &mut Criterion) {
    let context = &*shared_metal_context();
    let mut group = c.benchmark_group(format!("{}/Kernel/Matmul/Mxfp4ExpertDecode", type_short_name::<Metal>()));

    const W13: Mxfp4Shape = Mxfp4Shape {
        label: "W13_N5760_K2880_G16",
        routes: 4,
        routes_per_token: 4,
        experts: 32,
        n: 5760,
        k: 2880,
        group_size: 16,
        input_is_bf16: true,
    };
    const W2: Mxfp4Shape = Mxfp4Shape {
        label: "W2_N2880_K2880_G32",
        routes: 4,
        routes_per_token: 4,
        experts: 32,
        n: 2880,
        k: 2880,
        group_size: 32,
        input_is_bf16: false,
    };

    let spread_ids: Vec<i32> = vec![0, 8, 16, 24];
    let fixed_ids: Vec<i32> = vec![3, 7, 11, 19];
    for shape in [&W13, &W2] {
        bench_mxfp4_expert_projection(&mut group, context, shape, &spread_ids, "spread");
        bench_mxfp4_expert_projection(&mut group, context, shape, &fixed_ids, "fixed");
    }

    // Fused W13: identical weight traffic, gate/up epilogue folded in, output
    // is the activated hidden half-width (routes x 2880 F32).
    {
        let routes = W13.routes as usize;
        let experts = W13.experts as usize;
        let (n, k) = (W13.n as usize, W13.k as usize);
        let hidden = n / 2;
        let codes = alloc_allocation_with_data::<Metal, u8>(context, &vec![0x1Eu8; experts * n * k / 2]);
        let scales =
            alloc_allocation_with_data::<Metal, u8>(context, &vec![127u8; experts * n * k / W13.group_size as usize]);
        let outer_scales = alloc_allocation_with_data::<Metal, bf16>(context, &vec![bf16::from_f32(1.0); experts]);
        let biases = alloc_allocation_with_data::<Metal, bf16>(context, &vec![bf16::from_f32(0.0); experts * n]);
        let metadata = MicrofloatMetadata::new(
            MicrofloatFormat::Mxfp4,
            4,
            W13.group_size,
            MicrofloatLayout::OutputInput,
            W13.experts,
            W13.n,
            W13.k,
        )
        .unwrap();
        let expert_ids = alloc_allocation_with_data::<Metal, i32>(context, &spread_ids);
        let route_identity = ExpertRouteIdentity::new();
        let input = alloc_allocation::<Metal, bf16>(context, k);
        let mut output = alloc_allocation::<Metal, u8>(context, routes * hidden * 4);
        let routes_per_token = NonZeroU32::new(W13.routes).unwrap();
        let expert_count = NonZeroU32::new(W13.experts).unwrap();
        let mut matmul = <MetalKernels as Kernels>::MatmulKernel::new(
            context,
            bf16::data_type(),
            bf16::data_type(),
            crate::data_type::DataType::F32,
        )
        .expect("MatmulKernel");
        group.throughput(Throughput::Bytes(mxfp4_expert_bytes(&W13, &spread_ids)));
        group.bench_function(BenchmarkId::new("mxfp4_routed_fused_gate_up", "W13_N5760_K2880_G16_spread"), |bencher| {
            iter_encode_loop::<Metal, _>(context, bencher, |encoder| {
                let b: MatmulB<'_, Metal> = MatmulB::Microfloat {
                    codes: &codes,
                    scales: &scales,
                    outer_scales: &outer_scales,
                    metadata,
                };
                matmul
                    .encode(
                        MatmulArguments {
                            a: MatmulA::FullPrecision {
                                values: &input,
                                offset: 0,
                            },
                            b,
                            b_leading_dimension: None,
                            b_transpose: true,
                            d: &mut output,
                            d_transform: MatmulDOps {
                                per_matrix_bias: Some(&biases),
                                gate_act: Some(crate::backends::common::kernel::matmul::GateActMulDOps {
                                    activation_alpha: Some(1.702),
                                    gate_clipping: Some((-7.0, 7.0)),
                                    value_clipping: Some((-7.0, 7.0)),
                                }),
                                ..MatmulDOps::none()
                            },
                            routing: MatmulRouting::Experts(ExpertRoutes {
                                identity: &route_identity,
                                expert_ids: &expert_ids,
                                routes_per_token,
                                expert_count,
                                input: ExpertInput::Tokens,
                            }),
                            m: W13.routes,
                            n: W13.n,
                            k: W13.k,
                        },
                        encoder,
                    )
                    .expect("fused mxfp4 routed matmul");
            });
        });
    }
}

#[uzu_bench]
fn bench_mxfp4_expert_prefill_production(c: &mut Criterion) {
    let context = &*shared_metal_context();
    let mut group = c.benchmark_group(format!("{}/Kernel/Matmul/Mxfp4ExpertPrefill", type_short_name::<Metal>()));
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_millis(800));

    const W13: Mxfp4Shape = Mxfp4Shape {
        label: "W13_T71_R284_N5760_K2880_G16",
        routes: 284,
        routes_per_token: 4,
        experts: 32,
        n: 5760,
        k: 2880,
        group_size: 16,
        input_is_bf16: true,
    };
    const W2: Mxfp4Shape = Mxfp4Shape {
        label: "W2_T71_R284_N2880_K2880_G32",
        routes: 284,
        routes_per_token: 4,
        experts: 32,
        n: 2880,
        k: 2880,
        group_size: 32,
        input_is_bf16: false,
    };

    for distribution in [
        Mxfp4RouteDistribution::Uniform,
        Mxfp4RouteDistribution::FourHot,
        Mxfp4RouteDistribution::OneHot,
        Mxfp4RouteDistribution::ManyEmpty,
    ] {
        let expert_ids = distribution.expert_ids(W13.routes, W13.experts);
        for shape in [&W13, &W2] {
            bench_mxfp4_expert_projection(&mut group, context, shape, &expert_ids, distribution.label());
        }
    }

    let experts = W13.experts as usize;
    let w13_codes =
        alloc_allocation_with_data::<Metal, u8>(context, &vec![0x1Eu8; experts * W13.n as usize * W13.k as usize / 2]);
    let w13_scales = alloc_allocation_with_data::<Metal, u8>(
        context,
        &vec![127u8; experts * W13.n as usize * W13.k as usize / W13.group_size as usize],
    );
    let w13_outer_scales = alloc_allocation_with_data::<Metal, bf16>(context, &vec![bf16::from_f32(1.0); experts]);
    let w13_biases =
        alloc_allocation_with_data::<Metal, bf16>(context, &vec![bf16::from_f32(0.0); experts * W13.n as usize]);
    let w13_metadata = MicrofloatMetadata::new(
        MicrofloatFormat::Mxfp4,
        4,
        W13.group_size,
        MicrofloatLayout::OutputInput,
        W13.experts,
        W13.n,
        W13.k,
    )
    .unwrap();
    let w2_codes =
        alloc_allocation_with_data::<Metal, u8>(context, &vec![0x1Eu8; experts * W2.n as usize * W2.k as usize / 2]);
    let w2_scales = alloc_allocation_with_data::<Metal, u8>(
        context,
        &vec![127u8; experts * W2.n as usize * W2.k as usize / W2.group_size as usize],
    );
    let w2_outer_scales = alloc_allocation_with_data::<Metal, bf16>(context, &vec![bf16::from_f32(1.0); experts]);
    let w2_biases =
        alloc_allocation_with_data::<Metal, bf16>(context, &vec![bf16::from_f32(0.0); experts * W2.n as usize]);
    let w2_metadata = MicrofloatMetadata::new(
        MicrofloatFormat::Mxfp4,
        4,
        W2.group_size,
        MicrofloatLayout::OutputInput,
        W2.experts,
        W2.n,
        W2.k,
    )
    .unwrap();

    let token_count = W13.routes / W13.routes_per_token;
    let routes_per_token = NonZeroU32::new(W13.routes_per_token).unwrap();
    let expert_count = NonZeroU32::new(W13.experts).unwrap();
    let input = alloc_allocation::<Metal, bf16>(context, token_count as usize * W13.k as usize);
    let route_weights = alloc_allocation_with_data::<Metal, bf16>(
        context,
        &vec![bf16::from_f32(1.0 / W13.routes_per_token as f32); W13.routes as usize],
    );
    let mut fused_up = alloc_allocation::<Metal, u8>(context, W13.routes as usize * W13.n as usize * 4);
    let mut hidden = alloc_allocation::<Metal, u8>(context, W13.routes as usize * W2.k as usize * 4);
    let mut route_outputs = alloc_allocation::<Metal, u8>(context, W13.routes as usize * W2.n as usize * 2);
    let mut output = alloc_allocation::<Metal, u8>(context, token_count as usize * W2.n as usize * 2);

    let mut grouped_w13 = <MetalMatmul as MatmulKernel>::new(
        context,
        crate::data_type::DataType::BF16,
        crate::data_type::DataType::BF16,
        crate::data_type::DataType::F32,
    )
    .expect("grouped W13 kernel");
    let mut grouped_w2 = <MetalMatmul as MatmulKernel>::new(
        context,
        crate::data_type::DataType::BF16,
        crate::data_type::DataType::F32,
        crate::data_type::DataType::BF16,
    )
    .expect("grouped W2 kernel");
    let mut forced_gemv_w13 = <MetalMatmul as MatmulKernel>::new(
        context,
        crate::data_type::DataType::BF16,
        crate::data_type::DataType::BF16,
        crate::data_type::DataType::F32,
    )
    .expect("forced-GEMV W13 kernel");
    let mut forced_gemv_w2 = <MetalMatmul as MatmulKernel>::new(
        context,
        crate::data_type::DataType::BF16,
        crate::data_type::DataType::F32,
        crate::data_type::DataType::BF16,
    )
    .expect("forced-GEMV W2 kernel");
    forced_gemv_w13.force_gemv_for_benchmark();
    forced_gemv_w2.force_gemv_for_benchmark();
    let gate =
        GatedActMul::<Metal>::full_precision(context, crate::data_type::DataType::F32, true, false, true, true, true)
            .expect("gate kernel");
    let finalize = MetalMoeFinalize::new(context, crate::data_type::DataType::BF16).expect("MoE finalize kernel");

    for distribution in [
        Mxfp4RouteDistribution::Uniform,
        Mxfp4RouteDistribution::FourHot,
        Mxfp4RouteDistribution::OneHot,
        Mxfp4RouteDistribution::ManyEmpty,
    ] {
        let expert_ids_value = distribution.expert_ids(W13.routes, W13.experts);
        let expert_ids = alloc_allocation_with_data::<Metal, i32>(context, &expert_ids_value);
        let route_identity = ExpertRouteIdentity::new();
        let grouped_bytes = mxfp4_expert_bytes_for_rows(&W13, &expert_ids_value, 4)
            + mxfp4_expert_bytes_for_rows(&W2, &expert_ids_value, 4);
        group.throughput(Throughput::Bytes(grouped_bytes));
        group.bench_function(BenchmarkId::new("full_moe_grouped", distribution.label()), |bencher| {
            iter_encode_loop::<Metal, _>(context, bencher, |encoder| {
                encode_mxfp4_moe(
                    &mut grouped_w13,
                    &mut grouped_w2,
                    &gate,
                    &finalize,
                    &input,
                    &w13_codes,
                    &w13_scales,
                    &w13_outer_scales,
                    &w13_biases,
                    w13_metadata,
                    &w2_codes,
                    &w2_scales,
                    &w2_outer_scales,
                    &w2_biases,
                    w2_metadata,
                    &route_identity,
                    &expert_ids,
                    &route_weights,
                    &mut fused_up,
                    &mut hidden,
                    &mut route_outputs,
                    &mut output,
                    token_count,
                    routes_per_token,
                    expert_count,
                    W13.k,
                    W2.k,
                    encoder,
                );
            });
        });

        let direct_bytes = mxfp4_expert_bytes_for_rows(&W13, &expert_ids_value, 1)
            + mxfp4_expert_bytes_for_rows(&W2, &expert_ids_value, 1);
        group.throughput(Throughput::Bytes(direct_bytes));
        group.bench_function(BenchmarkId::new("full_moe_forced_gemv", distribution.label()), |bencher| {
            iter_encode_loop::<Metal, _>(context, bencher, |encoder| {
                encode_mxfp4_moe(
                    &mut forced_gemv_w13,
                    &mut forced_gemv_w2,
                    &gate,
                    &finalize,
                    &input,
                    &w13_codes,
                    &w13_scales,
                    &w13_outer_scales,
                    &w13_biases,
                    w13_metadata,
                    &w2_codes,
                    &w2_scales,
                    &w2_outer_scales,
                    &w2_biases,
                    w2_metadata,
                    &route_identity,
                    &expert_ids,
                    &route_weights,
                    &mut fused_up,
                    &mut hidden,
                    &mut route_outputs,
                    &mut output,
                    token_count,
                    routes_per_token,
                    expert_count,
                    W13.k,
                    W2.k,
                    encoder,
                );
            });
        });
    }
}

#[uzu_bench]
fn bench_bf16_lm_head_preflight(c: &mut Criterion) {
    let context = &*shared_metal_context();
    let mut group = c.benchmark_group(format!("{}/Kernel/Matmul/Bf16LmHead", type_short_name::<Metal>()));

    // GPT-OSS LM head: dense BF16 GEMV, M=1, N=201088, K=2880.
    let (m, n, k) = (1usize, 201088usize, 2880usize);
    let input = alloc_allocation::<Metal, bf16>(context, m * k);
    let weights = alloc_allocation::<Metal, bf16>(context, n * k);
    let mut output = alloc_allocation::<Metal, bf16>(context, m * n);
    let mut matmul =
        <MetalKernels as Kernels>::MatmulKernel::new(context, bf16::data_type(), bf16::data_type(), bf16::data_type())
            .expect("MatmulKernel");
    let bytes = (n * k * 2) as u64;
    group.throughput(Throughput::Bytes(bytes));
    group.sample_size(20);
    group.bench_function(BenchmarkId::new("BF16", "M1_N201088_K2880"), |bencher| {
        iter_encode_loop::<Metal, _>(context, bencher, |encoder| {
            matmul
                .encode(
                    MatmulArguments {
                        a: MatmulA::FullPrecision {
                            values: &input,
                            offset: 0,
                        },
                        b: MatmulB::FullPrecision {
                            b: &weights,
                        },
                        b_leading_dimension: None,
                        b_transpose: true,
                        d: &mut output,
                        d_transform: MatmulDOps::none(),
                        routing: MatmulRouting::Dense,
                        m: m as u32,
                        n: n as u32,
                        k: k as u32,
                    },
                    encoder,
                )
                .expect("lm head matmul");
        });
    });
}
