#![cfg(backend = "metal")]

use std::num::NonZeroU32;

use proc_macros::uzu_test;

use crate::{
    backends::{
        common::{Backend, Context, Encoder, kernel::matmul::ExpertInput},
        metal::{Metal, ResidentInt8ExpertTensorOpsDispatch},
    },
    data_type::DataType,
    tests::helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
};

#[uzu_test]
fn resident_int8_expert_tensorops_rejects_incomplete_storage() {
    let context = <Metal as Backend>::Context::new().expect("Metal context");
    let allocation = alloc_allocation::<Metal, u8>(&context, 1);
    let expert_ids = alloc_allocation_with_data::<Metal, i32>(&context, &[0]);
    let mut output = alloc_allocation::<Metal, f32>(&context, 32);
    let mut dispatch = ResidentInt8ExpertTensorOpsDispatch::new(DataType::F32, DataType::F32);
    let mut encoder = Encoder::<Metal>::new(&context).expect("encoder");
    let error = dispatch
        .encode(
            &allocation,
            &allocation,
            &allocation,
            &allocation,
            &mut output,
            None,
            &expert_ids,
            32,
            32,
            1,
            NonZeroU32::new(1).unwrap(),
            NonZeroU32::new(1).unwrap(),
            ExpertInput::Routes,
            &mut encoder,
        )
        .expect_err("undersized resident storage must be rejected");
    assert!(error.to_string().contains("weight codes"));
}

#[uzu_test]
fn resident_int8_expert_tensorops_matches_grouped_cpu_dot() {
    let context = <Metal as Backend>::Context::new().expect("Metal context");
    if !context.supports_int8_tensorops() {
        return;
    }

    let (experts, routes, n, k) = (2usize, 2usize, 32usize, 32usize);
    let weight_codes: Vec<i8> = (0..experts * n * k).map(|index| (index % 25) as i8 - 12).collect();
    let weight_scales: Vec<f32> = (0..experts * n).map(|index| 0.00390625 * (1 + index % 4) as f32).collect();
    let activation_codes: Vec<i8> = (0..routes * k).map(|index| ((index * 7) % 255) as i16 as i8).collect();
    let activation_scales = vec![0.03125f32, 0.0625];
    let expert_ids = vec![1i32, 0];
    let biases: Vec<f32> = (0..experts * n).map(|index| (index % 7) as f32 * 0.125).collect();

    let weight_codes = alloc_allocation_with_data::<Metal, i8>(&context, &weight_codes);
    let weight_scales = alloc_allocation_with_data::<Metal, f32>(&context, &weight_scales);
    let activation_codes = alloc_allocation_with_data::<Metal, i8>(&context, &activation_codes);
    let activation_scales = alloc_allocation_with_data::<Metal, f32>(&context, &activation_scales);
    let expert_ids = alloc_allocation_with_data::<Metal, i32>(&context, &expert_ids);
    let biases = alloc_allocation_with_data::<Metal, f32>(&context, &biases);
    let mut output = alloc_allocation::<Metal, f32>(&context, routes * n);
    let mut dispatch = ResidentInt8ExpertTensorOpsDispatch::new(DataType::F32, DataType::F32);
    let mut encoder = Encoder::<Metal>::new(&context).expect("encoder");
    dispatch
        .encode(
            &weight_codes,
            &weight_scales,
            &activation_codes,
            &activation_scales,
            &mut output,
            Some(&biases),
            &expert_ids,
            k as u32,
            n as u32,
            routes as u32,
            NonZeroU32::new(1).unwrap(),
            NonZeroU32::new(experts as u32).unwrap(),
            ExpertInput::Routes,
            &mut encoder,
        )
        .expect("resident INT8 TensorOps encode");
    encoder.end_encoding().submit().wait_until_completed().expect("resident INT8 TensorOps completion");

    let weight_codes = weight_codes.as_slice::<i8>();
    let weight_scales = weight_scales.as_slice::<f32>();
    let activation_codes = activation_codes.as_slice::<i8>();
    let activation_scales = activation_scales.as_slice::<f32>();
    let biases = biases.as_slice::<f32>();
    let actual = allocation_to_vec::<Metal, f32>(&output);
    for route in 0..routes {
        let expert = [1usize, 0][route];
        for row in 0..n {
            let weights = &weight_codes[(expert * n + row) * k..(expert * n + row + 1) * k];
            let activation = &activation_codes[route * k..(route + 1) * k];
            let integer_product: i32 = weights
                .iter()
                .zip(activation)
                .map(|(&weight, &activation)| i32::from(weight) * i32::from(activation))
                .sum();
            let expected = integer_product as f32 * weight_scales[expert * n + row] * activation_scales[route]
                + biases[expert * n + row];
            let actual = actual[route * n + row];
            assert!((actual - expected).abs() <= 1e-5, "route={route}, row={row}: {actual} != {expected}");
        }
    }
}
