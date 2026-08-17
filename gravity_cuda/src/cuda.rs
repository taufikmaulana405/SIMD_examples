use crate::{
    collision::merge_colliding_particles,
    error::CudaError,
    types::{Particle, SimParams},
};
use bytemuck::Zeroable;
use rustacuda::{
    CudaFlags,
    context::{Context, ContextFlags},
    device::Device,
    function::{BlockSize, GridSize},
    memory::{CopyDestination, DeviceBuffer, DeviceCopy},
    module::Module,
    stream::{Stream, StreamFlags},
};
use std::ffi::CString;

pub struct CudaBackend {
    // CUDA resources must be dropped before the context that owns them.
    module: Module,
    stream: Stream,
    particles_a: DeviceBuffer<Particle>,
    particles_b: DeviceBuffer<Particle>,
    acceleration_x: DeviceBuffer<f32>,
    acceleration_y: DeviceBuffer<f32>,
    active_count: usize,
    use_a_as_source: bool,
    _context: Context,
}

unsafe impl DeviceCopy for Particle {}

impl CudaBackend {
    pub fn new(particles: &[Particle]) -> Result<Self, CudaError> {
        rustacuda::init(CudaFlags::empty())?;
        let device = Device::get_device(0)?;
        let device_name = device
            .name()
            .unwrap_or_else(|_| "unknown CUDA device".into());
        eprintln!("CUDA device: {device_name}");
        let context =
            Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
        let ptx = CString::new(include_bytes!(env!("GRAVITY_CUDA_PTX")).as_slice())?;
        let module = Module::load_from_string(&ptx)?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let particles_a = DeviceBuffer::from_slice(particles)?;
        let particles_b = DeviceBuffer::from_slice(particles)?;
        let acceleration_x = DeviceBuffer::from_slice(&vec![0.0; particles.len()])?;
        let acceleration_y = DeviceBuffer::from_slice(&vec![0.0; particles.len()])?;
        Ok(Self {
            module,
            stream,
            particles_a,
            particles_b,
            acceleration_x,
            acceleration_y,
            active_count: particles.len(),
            use_a_as_source: true,
            _context: context,
        })
    }

    pub fn active_count(&self) -> usize {
        self.active_count
    }

    pub fn reset(&mut self, particles: &[Particle]) -> Result<(), CudaError> {
        if particles.len() != self.particles_a.len() {
            return Err(CudaError::CountMismatch {
                active: self.particles_a.len(),
                requested: particles.len(),
            });
        }
        self.stream.synchronize()?;
        self.particles_a.copy_from(particles)?;
        self.particles_b.copy_from(particles)?;
        let zeros = vec![0.0_f32; particles.len()];
        self.acceleration_x.copy_from(&zeros)?;
        self.acceleration_y.copy_from(&zeros)?;
        self.stream.synchronize()?;
        self.active_count = particles.len();
        self.use_a_as_source = true;
        Ok(())
    }

    pub fn step(&mut self, mut params: SimParams) -> Result<(), CudaError> {
        let requested_count = params.count as usize;
        if requested_count != self.active_count {
            return Err(CudaError::CountMismatch {
                active: self.active_count,
                requested: requested_count,
            });
        }
        if self.active_count == 0 {
            return Ok(());
        }

        let grid = GridSize::x((self.active_count as u32).div_ceil(128));
        params.count = self.active_count as u32;
        let drift_name = CString::new("gravity_drift")?;
        let drift = self.module.get_function(&drift_name)?;
        {
            let stream = &self.stream;
            let (source, destination) = if self.use_a_as_source {
                (&mut self.particles_a, &mut self.particles_b)
            } else {
                (&mut self.particles_b, &mut self.particles_a)
            };
            unsafe {
                rustacuda::launch!(drift<<<grid, BlockSize::x(128), 0, stream>>>(
                    source.as_device_ptr(),
                    destination.as_device_ptr(),
                    self.acceleration_x.as_device_ptr(),
                    self.acceleration_y.as_device_ptr(),
                    params
                ))?;
            }
        }
        self.stream.synchronize()?;
        self.use_a_as_source = !self.use_a_as_source;

        // Collision is deliberately host-controlled for deterministic AVX2
        // ordering. The unsafe parallel CUDA merge kernel is not used.
        let mut host_particles = self.snapshot()?;
        merge_colliding_particles(&mut host_particles);
        self.active_count = host_particles.len();
        let mut compacted = vec![Particle::zeroed(); self.particles_a.len()];
        compacted[..host_particles.len()].copy_from_slice(&host_particles);
        let current = if self.use_a_as_source {
            &mut self.particles_a
        } else {
            &mut self.particles_b
        };
        current.copy_from(&compacted)?;
        self.stream.synchronize()?;

        if self.active_count == 0 {
            return Ok(());
        }
        params.count = self.active_count as u32;
        let grid = GridSize::x((self.active_count as u32).div_ceil(128));
        let finish_name = CString::new("gravity_finish")?;
        let finish = self.module.get_function(&finish_name)?;
        let current = if self.use_a_as_source {
            &mut self.particles_a
        } else {
            &mut self.particles_b
        };
        let stream = &self.stream;
        unsafe {
            rustacuda::launch!(finish<<<grid, BlockSize::x(128), 0, stream>>>(
                current.as_device_ptr(),
                self.acceleration_x.as_device_ptr(),
                self.acceleration_y.as_device_ptr(),
                params
            ))?;
        }
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn snapshot(&mut self) -> Result<Vec<Particle>, CudaError> {
        let mut output = vec![Particle::zeroed(); self.active_count];
        if self.use_a_as_source {
            self.particles_a[..self.active_count].copy_to(&mut output)?;
        } else {
            self.particles_b[..self.active_count].copy_to(&mut output)?;
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{reference, types::Particle};

    fn fixture() -> Particle {
        Particle {
            x: 1.0,
            y: -2.0,
            vx: 3.0,
            vy: 4.0,
            mass: 5.0,
            brightness: 0.8,
            alive: 1,
            padding: 0,
        }
    }

    #[test]
    #[ignore = "requires a CUDA device and driver"]
    fn cuda_one_particle_matches_reference() {
        let initial = vec![fixture()];
        let params = SimParams {
            count: 1,
            dt: 0.01,
            gravity: 15.0,
            softening_squared: 9.0,
        };
        let mut expected = initial.clone();
        reference::step(&mut expected, params);
        let mut backend = CudaBackend::new(&initial).expect("CUDA initialization failed");
        backend.step(params).expect("CUDA step failed");
        let actual = backend.snapshot().expect("CUDA snapshot failed");
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual.x - expected.x).abs() < 1e-4);
            assert!((actual.y - expected.y).abs() < 1e-4);
            assert!((actual.vx - expected.vx).abs() < 1e-4);
            assert!((actual.vy - expected.vy).abs() < 1e-4);
        }
    }
}
