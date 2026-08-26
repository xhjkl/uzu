use std::mem::size_of;

use super::{
    MatmulError,
    d_ops::MatmulDOps,
    matmul_a::MatmulA,
    matmul_b::MatmulB,
    routing::{ExpertInput, MatmulRouting},
};
use crate::{
    backends::common::{Allocation, Backend, BufferArg, gpu_types::QuantizationMode},
    data_type::DataType,
};

enum IntegerCorrection<'a, B: Backend> {
    None,
    Bias(&'a Allocation<B>),
    ZeroPoint(&'a Allocation<B>),
}

pub struct MatmulArguments<'a, 'b, 'd, B: Backend, TB: BufferArg<'b, B> = &'b Allocation<B>> {
    pub a: MatmulA<'a, B>,
    pub b: MatmulB<'b, B, TB>,
    pub b_leading_dimension: Option<u32>,
    pub b_transpose: bool,
    pub d: &'d mut Allocation<B>,
    pub d_transform: MatmulDOps<'d, B>,
    pub routing: MatmulRouting<'a, B>,
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

fn validate_storage<'a, 'b, 'd, B: Backend, TB: BufferArg<'b, B>>(
    arguments: &MatmulArguments<'a, 'b, 'd, B, TB>,
    input_data_type: DataType,
    output_data_type: DataType,
    path: &'static str,
) -> Result<(), MatmulError<B>> {
    if let MatmulA::FullPrecision {
        values,
        offset,
    } = &arguments.a
    {
        let element_size = input_data_type.size_in_bytes();
        if !offset.is_multiple_of(element_size) {
            return Err(MatmulError::InvalidStorage {
                path,
                operand: "A",
                reason: "full-precision byte offset is not aligned to the input data type",
            });
        }
        let input_rows = match arguments.routing.expert_routes() {
            Some(routes) if routes.input == ExpertInput::Tokens => arguments.m / routes.routes_per_token.get(),
            _ => arguments.m,
        };
        let required_end = (input_rows as usize)
            .checked_mul(arguments.k as usize)
            .and_then(|size| size.checked_mul(element_size))
            .and_then(|size| size.checked_add(*offset));
        if required_end.is_none_or(|required| values.size() < required) {
            return Err(MatmulError::InvalidStorage {
                path,
                operand: "A",
                reason: "full-precision input allocation does not cover every referenced row",
            });
        }
    }

    let output_width = if arguments.d_transform.gate_act.is_some() {
        arguments.n / 2
    } else {
        arguments.n
    };
    let required_output = (arguments.m as usize)
        .checked_mul(output_width as usize)
        .and_then(|size| size.checked_mul(output_data_type.size_in_bytes()));
    if required_output.is_none_or(|required| arguments.d.size() < required) {
        return Err(MatmulError::InvalidStorage {
            path,
            operand: "D",
            reason: "output allocation does not cover M * N elements",
        });
    }
    Ok(())
}

pub(crate) fn validate_matmul_arguments<'a, 'b, 'd, B: Backend, TB: BufferArg<'b, B>>(
    arguments: &MatmulArguments<'a, 'b, 'd, B, TB>,
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
    path: &'static str,
) -> Result<(), MatmulError<B>> {
    validate_expert_routes(arguments, weights_data_type, path)?;
    validate_microfloat(arguments, weights_data_type, path)?;
    validate_storage(arguments, input_data_type, output_data_type, path)
}

fn validate_expert_routes<'a, 'b, 'd, B: Backend, TB: BufferArg<'b, B>>(
    arguments: &MatmulArguments<'a, 'b, 'd, B, TB>,
    weights_data_type: DataType,
    path: &'static str,
) -> Result<(), MatmulError<B>> {
    if arguments.d_transform.per_matrix_bias.is_some() && arguments.routing.expert_routes().is_none() {
        return Err(MatmulError::UnsupportedRouting {
            path,
            reason: "per-matrix bias requires direct expert routes",
        });
    }
    let Some(routes) = arguments.routing.expert_routes() else {
        return Ok(());
    };
    if arguments.d_transform.rht_factors.is_some() {
        return Err(MatmulError::UnsupportedRouting {
            path,
            reason: "direct expert routes do not support output RHT",
        });
    }
    let required_ids = (arguments.m as usize).checked_mul(size_of::<i32>());
    if required_ids.is_none_or(|required| routes.expert_ids.size() < required) {
        return Err(MatmulError::UnsupportedRouting {
            path,
            reason: "expert_ids must contain at least M entries",
        });
    }
    if routes.input == ExpertInput::Tokens && !arguments.m.is_multiple_of(routes.routes_per_token.get()) {
        return Err(MatmulError::UnsupportedRouting {
            path,
            reason: "M must be divisible by routes_per_token for token inputs",
        });
    }
    let required_biases = (routes.expert_count.get() as usize)
        .checked_mul(arguments.n as usize)
        .and_then(|size| size.checked_mul(weights_data_type.size_in_bytes()));
    if arguments
        .d_transform
        .per_matrix_bias
        .is_some_and(|biases| required_biases.is_none_or(|required| biases.size() < required))
    {
        return Err(MatmulError::UnsupportedRouting {
            path,
            reason: "expert bias bank must contain expert_count * N values",
        });
    }
    match &arguments.b {
        MatmulB::FullPrecision {
            b,
        } => validate_full_precision_bank(arguments, routes.expert_count, *b, weights_data_type, path),
        MatmulB::Microfloat {
            ..
        } => Ok(()),
        MatmulB::ScaleBiasDequant {
            b,
            scales,
            biases,
            mode,
            group_size,
            ..
        } => validate_integer_bank(
            arguments,
            routes.expert_count,
            b,
            scales,
            IntegerCorrection::Bias(biases),
            *mode,
            *group_size,
            weights_data_type,
            path,
        ),
        MatmulB::ScaleZeroPointDequant {
            b,
            scales,
            zero_points,
            mode,
            group_size,
            ..
        } => validate_integer_bank(
            arguments,
            routes.expert_count,
            b,
            scales,
            IntegerCorrection::ZeroPoint(zero_points),
            *mode,
            *group_size,
            weights_data_type,
            path,
        ),
        MatmulB::ScaleSymmetricDequant {
            b,
            scales,
            mode,
            group_size,
            ..
        } => validate_integer_bank(
            arguments,
            routes.expert_count,
            b,
            scales,
            IntegerCorrection::None,
            *mode,
            *group_size,
            weights_data_type,
            path,
        ),
    }
}

fn validate_full_precision_bank<'a, 'b, 'd, B: Backend, TB: BufferArg<'b, B>>(
    arguments: &MatmulArguments<'a, 'b, 'd, B, TB>,
    matrix_count: std::num::NonZeroU32,
    b: TB,
    weights_data_type: DataType,
    path: &'static str,
) -> Result<(), MatmulError<B>> {
    let leading_dimension = arguments.b_leading_dimension.unwrap_or(if arguments.b_transpose {
        arguments.k
    } else {
        arguments.n
    });
    let major_dimension = if arguments.b_transpose {
        arguments.n
    } else {
        arguments.k
    };
    let minimum_leading_dimension = if arguments.b_transpose {
        arguments.k
    } else {
        arguments.n
    };
    let required_bytes = (matrix_count.get() as usize)
        .checked_mul(major_dimension as usize)
        .and_then(|size| size.checked_mul(leading_dimension as usize))
        .and_then(|size| size.checked_mul(weights_data_type.size_in_bytes()));
    if leading_dimension < minimum_leading_dimension
        || required_bytes.is_none_or(|required| b.into_parts().2 < required)
    {
        return Err(MatmulError::UnsupportedRouting {
            path,
            reason: "full-precision weight storage does not cover every expert matrix",
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_integer_bank<'a, 'b, 'd, B: Backend, TB: BufferArg<'b, B>>(
    arguments: &MatmulArguments<'a, 'b, 'd, B, TB>,
    matrix_count: std::num::NonZeroU32,
    weights: &Allocation<B>,
    scales: &Allocation<B>,
    correction: IntegerCorrection<'_, B>,
    mode: QuantizationMode,
    group_size: u32,
    weights_data_type: DataType,
    path: &'static str,
) -> Result<(), MatmulError<B>> {
    let packing_divisor = mode.packing_divisor();
    if group_size == 0 || !arguments.k.is_multiple_of(packing_divisor) {
        return Err(MatmulError::UnsupportedRouting {
            path,
            reason: "integer expert storage has an invalid group size or packed K dimension",
        });
    }
    let matrix_rows = (matrix_count.get() as usize).checked_mul(arguments.n as usize);
    let groups = arguments.k.div_ceil(group_size) as usize;
    let required_weights = matrix_rows.and_then(|rows| {
        rows.checked_mul((arguments.k / packing_divisor) as usize)
            .and_then(|size| size.checked_mul(mode.storage_type().size_in_bytes()))
    });
    let required_scales = matrix_rows
        .and_then(|rows| rows.checked_mul(groups).and_then(|size| size.checked_mul(weights_data_type.size_in_bytes())));
    let correction_valid = match correction {
        IntegerCorrection::None => true,
        IntegerCorrection::Bias(biases) => required_scales.is_some_and(|required| biases.size() >= required),
        IntegerCorrection::ZeroPoint(zero_points) => matrix_rows
            .and_then(|rows| rows.checked_mul(groups.div_ceil(packing_divisor as usize)))
            .and_then(|size| size.checked_mul(mode.storage_type().size_in_bytes()))
            .is_some_and(|required| zero_points.size() >= required),
    };
    if required_weights.is_none_or(|required| weights.size() < required)
        || required_scales.is_none_or(|required| scales.size() < required)
        || !correction_valid
    {
        return Err(MatmulError::UnsupportedRouting {
            path,
            reason: "integer weight storage does not cover every expert matrix",
        });
    }
    Ok(())
}

fn validate_microfloat<'a, 'b, 'd, B: Backend, TB: BufferArg<'b, B>>(
    arguments: &MatmulArguments<'a, 'b, 'd, B, TB>,
    weights_data_type: DataType,
    path: &'static str,
) -> Result<(), MatmulError<B>> {
    let MatmulB::Microfloat {
        codes,
        scales,
        outer_scales,
        metadata,
    } = &arguments.b
    else {
        return Ok(());
    };
    let matrix_count = arguments.routing.expert_routes().map_or(1, |routes| routes.expert_count.get());
    let rows_match = arguments.routing.sparse_readout_rows().is_some() || metadata.rows() == arguments.n;
    let outer_scale_bytes = (metadata.matrix_count() as usize).checked_mul(weights_data_type.size_in_bytes());
    if !arguments.b_transpose
        || arguments.b_leading_dimension.is_some()
        || metadata.matrix_count() < matrix_count
        || !rows_match
        || metadata.columns() != arguments.k
        || codes.size() < metadata.required_code_bytes()
        || scales.size() < metadata.required_scale_bytes()
        || outer_scale_bytes.is_none_or(|required| outer_scales.size() < required)
    {
        return Err(MatmulError::UnsupportedRouting {
            path,
            reason: "microfloat storage does not match the requested matrix operand",
        });
    }
    Ok(())
}
