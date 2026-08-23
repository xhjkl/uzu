use bitflags::bitflags;

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct DeviceCapabilities: u32 {
        const SPARSE_BUFFERS = 1 << 0;
        const INT8_TENSOROPS = 1 << 1;
    }
}
