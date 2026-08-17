use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct Particle {
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    pub mass: f32,
    pub brightness: f32,
    pub alive: u32,
    pub padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct SimParams {
    pub count: u32,
    pub dt: f32,
    pub gravity: f32,
    pub softening_squared: f32,
}

// These plain-C layouts contain no pointers or Rust-managed data.
unsafe impl rustacuda::memory::DeviceCopy for SimParams {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn device_layout_is_stable() {
        assert_eq!(size_of::<Particle>(), 32);
        assert_eq!(align_of::<Particle>(), 4);
        assert_eq!(size_of::<SimParams>(), 16);
    }
}
