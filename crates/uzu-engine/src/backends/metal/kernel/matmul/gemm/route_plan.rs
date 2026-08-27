use crate::{
    backends::{
        common::{Allocation, Encoder, kernel::matmul::ExpertRoutes},
        metal::{
            Metal,
            context::MetalContext,
            error::MetalError,
            kernel::{
                ExpertRouteClearCountsMetalKernel, ExpertRouteCountMetalKernel, ExpertRoutePrefixMetalKernel,
                ExpertRouteScatterMetalKernel, ExpertRouteZeroInvalidMetalKernel,
            },
        },
    },
    data_type::DataType,
};

pub(super) struct ExpertRoutePlan {
    pub(super) offsets: Allocation<Metal>,
    pub(super) grouped_routes: Allocation<Metal>,
}

pub(super) struct ExpertRoutePlanner {
    clear_counts: ExpertRouteClearCountsMetalKernel,
    count: ExpertRouteCountMetalKernel,
    prefix: ExpertRoutePrefixMetalKernel,
    scatter: ExpertRouteScatterMetalKernel,
    zero_invalid: ExpertRouteZeroInvalidMetalKernel,
}

impl ExpertRoutePlanner {
    pub(super) fn new(
        context: &MetalContext,
        output_data_type: DataType,
    ) -> Result<Self, MetalError> {
        Ok(Self {
            clear_counts: ExpertRouteClearCountsMetalKernel::new(context)?,
            count: ExpertRouteCountMetalKernel::new(context)?,
            prefix: ExpertRoutePrefixMetalKernel::new(context)?,
            scatter: ExpertRouteScatterMetalKernel::new(context)?,
            zero_invalid: ExpertRouteZeroInvalidMetalKernel::new(context, output_data_type)?,
        })
    }

    pub(super) fn encode(
        &self,
        routes: ExpertRoutes<'_, Metal>,
        route_count: u32,
        output_width: u32,
        output: &mut Allocation<Metal>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<ExpertRoutePlan, MetalError> {
        let expert_count = routes.expert_count.get();
        let mut offsets = encoder.allocate_scratch_for_shape(&[expert_count + 1], DataType::U32)?;
        let mut cursors = encoder.allocate_scratch_for_shape(&[expert_count], DataType::U32)?;
        let mut grouped_routes = encoder.allocate_scratch_for_shape(&[route_count], DataType::U32)?;

        self.clear_counts.encode(&mut offsets, expert_count, encoder);
        self.count.encode(routes.expert_ids, &mut offsets, route_count, expert_count, encoder);
        self.prefix.encode(&mut offsets, &mut cursors, expert_count, encoder);
        self.scatter.encode(routes.expert_ids, &mut cursors, &mut grouped_routes, route_count, expert_count, encoder);
        self.zero_invalid.encode(routes.expert_ids, output, route_count, output_width, expert_count, encoder);

        Ok(ExpertRoutePlan {
            offsets,
            grouped_routes,
        })
    }
}
