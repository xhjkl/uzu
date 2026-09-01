use crate::{
    backends::{
        common::{Allocation, Encoder, kernel::matmul::ExpertRoutes},
        metal::{
            Metal,
            command_buffer::{CachedExpertRoutePlan, ExpertRoutePlanKey},
            context::MetalContext,
            error::MetalError,
            kernel::{
                ExpertRouteClearCountsMetalKernel, ExpertRouteCountMetalKernel,
                ExpertRouteDispatchArgumentsMetalKernel, ExpertRoutePrefixMetalKernel, ExpertRouteScatterMetalKernel,
                ExpertRouteZeroInvalidMetalKernel,
            },
        },
    },
    data_type::DataType,
};

pub(super) struct PreparedExpertRoutes {
    pub(super) key: ExpertRoutePlanKey,
    pub(super) plan: CachedExpertRoutePlan,
    pub(super) retain: bool,
}

pub(super) struct ExpertRoutePlanner {
    clear_counts: ExpertRouteClearCountsMetalKernel,
    count: ExpertRouteCountMetalKernel,
    prefix: ExpertRoutePrefixMetalKernel,
    scatter: ExpertRouteScatterMetalKernel,
    dispatch_arguments: ExpertRouteDispatchArgumentsMetalKernel,
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
            dispatch_arguments: ExpertRouteDispatchArgumentsMetalKernel::new(context)?,
            zero_invalid: ExpertRouteZeroInvalidMetalKernel::new(context, output_data_type)?,
        })
    }

    pub(super) fn prepare(
        &self,
        routes: ExpertRoutes<'_, Metal>,
        route_count: u32,
        block_m: u32,
        encoder: &mut Encoder<Metal>,
    ) -> Result<PreparedExpertRoutes, MetalError> {
        let expert_count = routes.expert_count.get();
        let key = ExpertRoutePlanKey::new(
            routes.expert_ids,
            route_count,
            expert_count,
            routes.routes_per_token.get(),
            block_m,
        );
        let plan = encoder.as_command_buffer_mut().take_expert_route_plan(&key);
        if let Some(plan) = plan {
            return Ok(PreparedExpertRoutes {
                key,
                plan,
                retain: false,
            });
        }

        let mut offsets = encoder.allocate_scratch_for_shape(&[expert_count + 1], DataType::U32)?;
        let mut tiles = encoder.allocate_scratch_for_shape(&[route_count, 3], DataType::U32)?;
        let mut cursors = encoder.allocate_scratch_for_shape(&[expert_count], DataType::U32)?;
        let mut grouped_routes = encoder.allocate_scratch_for_shape(&[route_count], DataType::U32)?;
        let mut tile_count = encoder.allocate_scratch_for_shape(&[1], DataType::U32)?;

        self.clear_counts.encode(&mut offsets, expert_count, encoder);
        self.count.encode(routes.expert_ids, &mut offsets, route_count, expert_count, encoder);
        self.prefix.encode(&mut offsets, &mut tiles, &mut cursors, &mut tile_count, expert_count, block_m, encoder);
        self.scatter.encode(routes.expert_ids, &mut cursors, &mut grouped_routes, route_count, expert_count, encoder);

        Ok(PreparedExpertRoutes {
            key,
            plan: CachedExpertRoutePlan {
                tiles,
                grouped_routes,
                tile_count,
            },
            retain: true,
        })
    }

    pub(super) fn encode_projection(
        &self,
        routes: ExpertRoutes<'_, Metal>,
        route_count: u32,
        output_width: u32,
        column_tiles: u32,
        tile_count: &Allocation<Metal>,
        output: &mut Allocation<Metal>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<Allocation<Metal>, MetalError> {
        let mut dispatch_arguments = encoder.allocate_scratch_for_shape(&[3], DataType::U32)?;
        self.dispatch_arguments.encode(tile_count, &mut dispatch_arguments, column_tiles, encoder);
        self.zero_invalid.encode(
            routes.expert_ids,
            output,
            route_count,
            output_width,
            routes.expert_count.get(),
            encoder,
        );
        Ok(dispatch_arguments)
    }
}
