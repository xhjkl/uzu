mod allocator;
mod range_allocator;

pub(crate) use allocator::AllocationIdentity;
pub use allocator::{Allocation, AllocationPool, AllocationType, Allocator};
use range_allocator::{AllocationType as RangeAllocationType, RangeAllocator};
