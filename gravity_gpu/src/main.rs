use std::{
    borrow::Cow,
    collections::VecDeque,
    sync::{mpsc, Arc},
    time::Instant,
};

use bytemuck::{Pod, Zeroable};
use rand::Rng;
use wgpu::util::DeviceExt;
use winit::{
    dpi::PhysicalSize,
    event::{ElementState, Event, KeyEvent, MouseScrollDelta, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
};

const PARTICLE_COUNT: usize = 10_000;
const WORKGROUP_SIZE: u32 = 64;
const G: f32 = 15.0;
const DT: f32 = 1.0 / 120.0;
const SOFTENING_SQUARED: f32 = 9.0;
const RADIUS_SCALE: f32 = 0.45;
const SPAWN_RADIUS: f32 = 1200.0;
const TRAIL_SAMPLE_INTERVAL: f32 = 1.0 / 45.0;
const TRAIL_MAX_AGE: f32 = 40.0;
// Keep a fixed, predictable trail budget: 90 snapshots (2 seconds) at 45 Hz.
// 900,000 records × 32 bytes is about 28.8 MB on the GPU.
const TRAIL_MAX_SAMPLES: usize = 90;
const TRAIL_CAPACITY: usize = PARTICLE_COUNT * TRAIL_MAX_SAMPLES;
const MAX_PHYSICS_STEPS_PER_FRAME: u32 = 4;
// The serialized merge kernel is intentionally disabled for the 10,000-particle
// default; it remains available for smaller runs where its bounded workload is usable.
const COLLISION_MAX_PARTICLES: usize = 4_000;
const COLLISION_INTERVAL: u32 = 8;
const TRAIL_STRIDE: u64 = 32;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SimParams {
    count: u32,
    _padding: u32,
    dt: f32,
    gravity: f32,
    softening_squared: f32,
    radius_scale: f32,
    _padding2: [f32; 6],
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SimControl {
    active_count: u32,
    _padding: [u32; 7],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RenderParams {
    viewport: [f32; 4],
    counts: [u32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TrailGpu {
    position: [f32; 2],
    radius: f32,
    brightness: f32,
    created: f32,
    _padding: [f32; 3],
}
#[derive(Clone, Copy)]
struct TrailPoint {
    position: [f32; 2],
    radius: f32,
    brightness: f32,
    created: f32,
}

struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    sim_params: wgpu::Buffer,
    sim_control: wgpu::Buffer,
    render_params: wgpu::Buffer,
    collision_pipeline: wgpu::ComputePipeline,
    particles: [wgpu::Buffer; 2],
    sim_bind_groups: [wgpu::BindGroup; 2],
    render_bind_groups: [wgpu::BindGroup; 2],
    trail_bind_groups: [wgpu::BindGroup; 2],
    integrate_pipeline: wgpu::ComputePipeline,
    finish_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
    trail_pipeline: wgpu::RenderPipeline,
    particle_index: usize,
    zoom: f32,
    paused: bool,
    time_scale: f32,
    accumulator: f32,
    last_frame: Instant,
    total_mass: f32,
    trails_visible: bool,
    trail_clock: f32,
    trail_accumulator: f32,
    trails: VecDeque<TrailPoint>,
    trail_buffer: wgpu::Buffer,
    trail_upload: Vec<TrailGpu>,
    trail_sample_in_flight: bool,
    staging_buffer: wgpu::Buffer,
    physics_steps: u64,
    fps: f32,
    fps_elapsed: f32,
    fps_frames: u32,
}

fn create_particles() -> (Vec<f32>, f32) {
    let mut rng = rand::thread_rng();
    let mut data = vec![0.0; PARTICLE_COUNT * 8];
    let mut total_mass = 0.0;
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut mx = 0.0;
    let mut my = 0.0;
    for i in 0..PARTICLE_COUNT {
        let angle = rng.gen_range(0.0..std::f32::consts::TAU);
        let u: f32 = rng.gen();
        let distance = (u.sqrt() + (u.powf(1.4) - u.sqrt()) * 0.75) * SPAWN_RADIUS;
        let (rx, ry) = (angle.cos(), angle.sin());
        let speed = 6.0 + 14.0 * (distance / SPAWN_RADIUS).sqrt();
        let mass = rng.gen_range(5.0..20.0);
        let radial = rng.gen_range(-0.01..0.01);
        let base = i * 8;
        let vx = -ry * speed + rx * radial;
        let vy = rx * speed + ry * radial;
        data[base..base + 8].copy_from_slice(&[
            rx * distance,
            ry * distance,
            vx,
            vy,
            mass,
            rng.gen_range(0.55..1.0),
            1.0,
            0.0,
        ]);
        total_mass += mass;
        cx += data[base] * mass;
        cy += data[base + 1] * mass;
        mx += vx * mass;
        my += vy * mass;
    }
    cx /= total_mass;
    cy /= total_mass;
    mx /= total_mass;
    my /= total_mass;
    for p in data.chunks_exact_mut(8) {
        p[0] -= cx;
        p[1] -= cy;
        p[2] -= mx;
        p[3] -= my;
    }
    (data, total_mass)
}

impl State {
    async fn new(window: Arc<Window>) -> Result<Self, String> {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| e.to_string())?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or("No compatible GPU adapter was found")?;
        let info = adapter.get_info();
        eprintln!("GPU: {} ({:?})", info.name, info.backend);
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("gravity_gpu device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| e.to_string())?;
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: caps.present_modes[0],
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
        let (initial, total_mass) = create_particles();
        let particles = [
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("particles A"),
                contents: bytemuck::cast_slice(&initial),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            }),
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("particles B"),
                contents: bytemuck::cast_slice(&initial),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            }),
        ];
        let sim_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("simulation parameters"),
            size: 48,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sim_control = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("simulation control"),
            contents: bytemuck::bytes_of(&SimControl {
                active_count: PARTICLE_COUNT as u32,
                _padding: [0; 7],
            }),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let render_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render parameters"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle readback"),
            size: (PARTICLE_COUNT * 32) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let trail_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trail storage"),
            size: (TRAIL_CAPACITY as u64) * TRAIL_STRIDE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sim_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("simulation layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let sim_bind_groups = [
            make_sim_bind_group(
                &device,
                &sim_layout,
                &sim_params,
                &particles[0],
                &particles[1],
                &sim_control,
            ),
            make_sim_bind_group(
                &device,
                &sim_layout,
                &sim_params,
                &particles[1],
                &particles[0],
                &sim_control,
            ),
        ];
        let render_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("render layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let render_bind_groups = particles.each_ref().map(|b| {
            make_render_bind_group(&device, &render_layout, &render_params, b, &trail_buffer)
        });
        let trail_bind_groups = [
            make_render_bind_group(
                &device,
                &render_layout,
                &render_params,
                &particles[0],
                &trail_buffer,
            ),
            make_render_bind_group(
                &device,
                &render_layout,
                &render_params,
                &particles[1],
                &trail_buffer,
            ),
        ];

        let sim_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gravity compute shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/simulation.wgsl"
            ))),
        });
        let render_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("particle and trail shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/render.wgsl"))),
        });
        let sim_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("simulation pipeline layout"),
            bind_group_layouts: &[&sim_layout],
            push_constant_ranges: &[],
        });
        let integrate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("integrate pipeline"),
            layout: Some(&sim_pipeline_layout),
            module: &sim_module,
            entry_point: "integrate",
        });
        let finish_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("finish pipeline"),
            layout: Some(&sim_pipeline_layout),
            module: &sim_module,
            entry_point: "finish",
        });
        let collision_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("collision merge pipeline"),
            layout: Some(&sim_pipeline_layout),
            module: &sim_module,
            entry_point: "merge_serial",
        });
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("render pipeline layout"),
                bind_group_layouts: &[&render_layout],
                push_constant_ranges: &[],
            });
        let mk_pipeline = |label, entry| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &render_module,
                    entry_point: entry,
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &render_module,
                    entry_point: if entry == "trail_vertex" {
                        "trail_fragment"
                    } else {
                        "particle_fragment"
                    },
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            })
        };
        let render_pipeline = mk_pipeline("particle render pipeline", "particle_vertex");
        let trail_pipeline = mk_pipeline("trail render pipeline", "trail_vertex");
        let zoom = size.width.min(size.height) as f32 / (SPAWN_RADIUS * 2.25);
        Ok(Self {
            window: window.clone(),
            surface,
            device,
            queue,
            config,
            sim_params,
            sim_control,
            render_params,
            collision_pipeline,
            particles,
            sim_bind_groups,
            render_bind_groups,
            trail_bind_groups,
            integrate_pipeline,
            finish_pipeline,
            render_pipeline,
            trail_pipeline,
            particle_index: 0,
            zoom,
            paused: false,
            time_scale: 1.0,
            accumulator: 0.0,
            last_frame: Instant::now(),
            total_mass,
            trails_visible: true,
            trail_clock: 0.0,
            trail_accumulator: 0.0,
            trails: VecDeque::with_capacity(TRAIL_CAPACITY),
            trail_buffer,
            trail_upload: Vec::with_capacity(TRAIL_CAPACITY),
            trail_sample_in_flight: false,
            staging_buffer,
            physics_steps: 0,
            fps: 0.0,
            fps_elapsed: 0.0,
            fps_frames: 0,
        })
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width > 0 && size.height > 0 {
            self.config.width = size.width;
            self.config.height = size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }
    fn reset(&mut self) {
        let (data, mass) = create_particles();
        self.queue
            .write_buffer(&self.particles[0], 0, bytemuck::cast_slice(&data));
        self.queue
            .write_buffer(&self.particles[1], 0, bytemuck::cast_slice(&data));
        self.queue.write_buffer(
            &self.sim_control,
            0,
            bytemuck::bytes_of(&SimControl {
                active_count: PARTICLE_COUNT as u32,
                _padding: [0; 7],
            }),
        );
        self.total_mass = mass;
        self.particle_index = 0;
        self.accumulator = 0.0;
        self.trail_clock = 0.0;
        self.trail_accumulator = 0.0;
        self.physics_steps = 0;
        self.trails.clear();
        self.trail_upload.clear();
        self.trail_sample_in_flight = false;
    }
    fn step(&mut self) {
        let params = SimParams {
            count: PARTICLE_COUNT as u32,
            _padding: 0,
            dt: DT,
            gravity: G,
            softening_squared: SOFTENING_SQUARED,
            radius_scale: RADIUS_SCALE,
            _padding2: [0.0; 6],
        };
        self.queue
            .write_buffer(&self.sim_params, 0, bytemuck::bytes_of(&params));
        let current = self.particle_index;
        let next = 1 - current;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gravity compute"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("integrate"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.integrate_pipeline);
            pass.set_bind_group(0, &self.sim_bind_groups[current], &[]);
            pass.dispatch_workgroups((PARTICLE_COUNT as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        if PARTICLE_COUNT <= COLLISION_MAX_PARTICLES
            && self.physics_steps % COLLISION_INTERVAL as u64 == 0
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("collision merge"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.collision_pipeline);
            pass.set_bind_group(0, &self.sim_bind_groups[next], &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("finish"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.finish_pipeline);
            pass.set_bind_group(0, &self.sim_bind_groups[next], &[]);
            pass.dispatch_workgroups((PARTICLE_COUNT as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        self.particle_index = current;
        self.physics_steps += 1;
    }
    fn snapshot_trails(&mut self) {
        if self.trail_sample_in_flight {
            return;
        }
        self.trail_sample_in_flight = true;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("trail snapshot"),
            });
        encoder.copy_buffer_to_buffer(
            &self.particles[self.particle_index],
            0,
            &self.staging_buffer,
            0,
            (PARTICLE_COUNT * 32) as u64,
        );
        self.queue.submit(Some(encoder.finish()));
        let slice = self.staging_buffer.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        // One reusable staging buffer is used; wait for this bounded 320 KB
        // readback rather than allowing overlapping maps and queue growth.
        self.device.poll(wgpu::Maintain::Wait);
        if rx.recv().ok().and_then(Result::ok).is_none() {
            self.trail_sample_in_flight = false;
            return;
        }
        let view = slice.get_mapped_range();
        let records: &[f32] = bytemuck::cast_slice(&view);
        for p in records.chunks_exact(8) {
            if p[6] >= 0.5 {
                self.trails.push_back(TrailPoint {
                    position: [p[0], p[1]],
                    radius: RADIUS_SCALE * p[4].sqrt(),
                    brightness: p[5],
                    created: self.trail_clock,
                });
            }
        }
        drop(view);
        self.staging_buffer.unmap();
        while self
            .trails
            .front()
            .is_some_and(|p| self.trail_clock - p.created >= TRAIL_MAX_AGE)
        {
            self.trails.pop_front();
        }
        while self.trails.len() > TRAIL_CAPACITY {
            self.trails.pop_front();
        }
        self.trail_upload.clear();
        self.trail_upload
            .extend(self.trails.iter().map(|p| TrailGpu {
                position: p.position,
                radius: p.radius,
                brightness: p.brightness,
                created: p.created,
                _padding: [0.0; 3],
            }));
        if !self.trail_upload.is_empty() {
            self.queue.write_buffer(
                &self.trail_buffer,
                0,
                bytemuck::cast_slice(&self.trail_upload),
            );
        }
        self.trail_sample_in_flight = false;
    }
    fn update(&mut self, elapsed: f32) {
        self.fps_frames += 1;
        self.fps_elapsed += elapsed;
        if self.fps_elapsed >= 0.5 {
            self.fps = self.fps_frames as f32 / self.fps_elapsed;
            self.fps_frames = 0;
            self.fps_elapsed = 0.0;
        }
        if self.paused {
            return;
        }
        self.trail_clock += elapsed;
        self.trail_accumulator += elapsed;
        self.accumulator += elapsed.min(0.05) * self.time_scale;
        let mut steps = 0;
        while self.accumulator >= DT && steps < MAX_PHYSICS_STEPS_PER_FRAME {
            self.step();
            self.accumulator -= DT;
            steps += 1;
        }
        if steps == MAX_PHYSICS_STEPS_PER_FRAME {
            self.accumulator = 0.0;
        }
        if self.trails_visible && self.trail_accumulator >= TRAIL_SAMPLE_INTERVAL {
            self.trail_accumulator %= TRAIL_SAMPLE_INTERVAL;
            self.snapshot_trails();
        }
    }
    fn render(&mut self) {
        let data = RenderParams {
            viewport: [
                self.config.width as f32,
                self.config.height as f32,
                self.zoom,
                self.total_mass,
            ],
            counts: [
                PARTICLE_COUNT as u32,
                self.trails.len() as u32,
                self.trail_clock.to_bits(),
                0,
            ],
        };
        self.queue
            .write_buffer(&self.render_params, 0, bytemuck::bytes_of(&data));
        let output = match self.surface.get_current_texture() {
            Ok(v) => v,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            Err(wgpu::SurfaceError::OutOfMemory) => panic!("GPU is out of memory"),
            Err(wgpu::SurfaceError::Timeout) => return,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gravity render"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("particles and trails"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.005,
                            g: 0.008,
                            b: 0.025,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            if self.trails_visible && !self.trails.is_empty() {
                pass.set_pipeline(&self.trail_pipeline);
                pass.set_bind_group(0, &self.trail_bind_groups[0], &[]);
                pass.draw(0..6, 0..self.trails.len() as u32);
            }
            pass.set_pipeline(&self.render_pipeline);
            pass.set_bind_group(0, &self.render_bind_groups[self.particle_index], &[]);
            pass.draw(0..6, 0..PARTICLE_COUNT as u32);
        }
        self.queue.submit(Some(encoder.finish()));
        let state = if self.paused { "PAUSED" } else { "RUNNING" };
        let trail_state = if self.trails_visible {
            "VISIBLE"
        } else {
            "HIDDEN"
        };
        self.window.set_title(&format!(
            "Gravity GPU | {} particles | {} | {:.0} FPS | {:.2}x | Trails {} ({})",
            PARTICLE_COUNT,
            state,
            self.fps,
            self.time_scale,
            trail_state,
            self.trails.len()
        ));
        output.present();
    }
    fn redraw(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;
        self.update(elapsed);
        self.render();
    }
}
fn make_sim_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params: &wgpu::Buffer,
    source: &wgpu::Buffer,
    destination: &wgpu::Buffer,
    control: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("simulation bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: source.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: destination.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: control.as_entire_binding(),
            },
        ],
    })
}
fn make_render_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params: &wgpu::Buffer,
    particles: &wgpu::Buffer,
    trails: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("render bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: particles.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: trails.as_entire_binding(),
            },
        ],
    })
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Gravity GPU — AVX2 reference")
            .with_inner_size(PhysicalSize::new(1280, 800))
            .build(&event_loop)?,
    );
    let mut state = pollster::block_on(State::new(window.clone()))?;
    event_loop.run(move |event, target| {
        target.set_control_flow(ControlFlow::Poll);
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => target.exit(),
                WindowEvent::Resized(size) => state.resize(size),
                WindowEvent::MouseWheel {
                    delta: MouseScrollDelta::LineDelta(_, y),
                    ..
                } => {
                    state.zoom = (state.zoom * 1.03_f32.powf(y.clamp(-1.0, 1.0))).clamp(0.05, 5.0);
                }
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            physical_key: PhysicalKey::Code(code),
                            state: ElementState::Pressed,
                            ..
                        },
                    ..
                } => match code {
                    KeyCode::Space => state.paused = !state.paused,
                    KeyCode::KeyT => state.trails_visible = !state.trails_visible,
                    KeyCode::KeyR => state.reset(),
                    KeyCode::ArrowUp => state.time_scale = (state.time_scale * 2.0).min(4.0),
                    KeyCode::ArrowDown => state.time_scale = (state.time_scale * 0.5).max(0.125),
                    _ => {}
                },
                WindowEvent::RedrawRequested => state.redraw(),
                _ => {}
            },
            Event::AboutToWait => window.request_redraw(),
            _ => {}
        }
    })?;
    Ok(())
}
