use std::{
    borrow::Cow,
    collections::VecDeque,
    sync::{mpsc, Arc},
    time::Instant,
};

use bytemuck::{Pod, Zeroable};
use egui::Context;
use egui_wgpu::ScreenDescriptor;
use egui_winit::State as EguiWinitState;
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
// The all-pairs solver already performs 100 million pair evaluations per
// physics step at the default size. Keep frame catch-up bounded so a stalled
// frame cannot queue an unbounded amount of GPU work.
const MAX_PHYSICS_STEPS_PER_FRAME: u32 = 2;
// The deterministic merge kernel is serialized and O(N²). It is deliberately
// limited to smaller runs until a tiled GPU merge implementation is available.
const COLLISION_MAX_PARTICLES: usize = 2_000;
const COLLISION_INTERVAL: u32 = 8;
const TRAIL_STRIDE: u64 = 32; // vec2 position + radius + brightness + age + padding.

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
    output_count: u32,
    merge_count: u32,
    _padding: [u32; 5],
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
    collision_pipelines: [wgpu::ComputePipeline; 5],
    collision_bind_groups: [wgpu::BindGroup; 2],
    _collision_meta: wgpu::Buffer,
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
    egui_ctx: Context,
    egui_state: EguiWinitState,
    egui_renderer: egui_wgpu::Renderer,
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
                output_count: PARTICLE_COUNT as u32,
                merge_count: 0,
                _padding: [0; 5],
            }),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let render_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render parameters"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let collision_meta = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("collision grid metadata"),
            size: (32768 * 4 + 10000 * 4 + 10000 * 4 + 4 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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
        let collision_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("collision layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, false),
                storage_entry(4, false),
            ],
        });
        let collision_bind_groups = [
            make_collision_bind_group(
                &device,
                &collision_layout,
                &sim_params,
                &particles[0],
                &collision_meta,
                &particles[1],
                &sim_control,
            ),
            make_collision_bind_group(
                &device,
                &collision_layout,
                &sim_params,
                &particles[1],
                &collision_meta,
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
        let collision_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bounded collision shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/collision.wgsl"
            ))),
        });
        let collision_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("collision pipeline layout"),
                bind_group_layouts: &[&collision_layout],
                push_constant_ranges: &[],
            });
        let collision_pipelines = [
            ("collision clear", "clear"),
            ("collision grid", "build_grid"),
            ("collision proposals", "propose"),
            ("collision merge", "merge_compact"),
            ("collision finalize", "finalize"),
        ]
        .map(|(label, entry_point)| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&collision_pipeline_layout),
                module: &collision_module,
                entry_point,
            })
        });
        let collision_serial_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("serial collision pipeline layout"),
                bind_group_layouts: &[&sim_layout],
                push_constant_ranges: &[],
            });
        let collision_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("serial collision merge pipeline"),
            layout: Some(&collision_serial_layout),
            module: &sim_module,
            entry_point: "merge_serial",
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
        let egui_ctx = Context::default();
        let egui_state = EguiWinitState::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(&device, format, None, 1);
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
            collision_pipelines,
            collision_bind_groups,
            _collision_meta: collision_meta,
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
            egui_ctx,
            egui_state,
            egui_renderer,
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
                output_count: PARTICLE_COUNT as u32,
                merge_count: 0,
                _padding: [0; 5],
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
        let final_source;
        let final_destination;
        if PARTICLE_COUNT <= COLLISION_MAX_PARTICLES {
            if self.physics_steps % COLLISION_INTERVAL as u64 == 0 {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("serial collision merge"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.collision_pipeline);
                pass.set_bind_group(0, &self.sim_bind_groups[next], &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            final_source = next;
            final_destination = current;
        } else {
            for (round, _) in (0..2).enumerate() {
                let source_index = if round % 2 == 0 { next } else { current };
                let collision_bind_group = &self.collision_bind_groups[source_index];
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(if round == 0 {
                        "collision grid round 1"
                    } else {
                        "collision grid round 2"
                    }),
                    timestamp_writes: None,
                });
                pass.set_bind_group(0, collision_bind_group, &[]);
                pass.set_pipeline(&self.collision_pipelines[0]);
                pass.dispatch_workgroups(ceil_div(32768, WORKGROUP_SIZE), 1, 1);
                pass.set_pipeline(&self.collision_pipelines[1]);
                pass.dispatch_workgroups(ceil_div(PARTICLE_COUNT, WORKGROUP_SIZE), 1, 1);
                pass.set_pipeline(&self.collision_pipelines[2]);
                pass.dispatch_workgroups(ceil_div(PARTICLE_COUNT, WORKGROUP_SIZE), 1, 1);
                pass.set_pipeline(&self.collision_pipelines[3]);
                pass.dispatch_workgroups(ceil_div(PARTICLE_COUNT, WORKGROUP_SIZE), 1, 1);
                pass.set_pipeline(&self.collision_pipelines[4]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            final_source = next;
            final_destination = current;
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("finish"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.finish_pipeline);
            pass.set_bind_group(0, &self.sim_bind_groups[final_source], &[]);
            pass.dispatch_workgroups((PARTICLE_COUNT as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        self.particle_index = final_destination;
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
        self.accumulator += elapsed.min(0.02) * self.time_scale;
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
        let raw_input = self.egui_state.take_egui_input(self.window.as_ref());
        let ctx = self.egui_ctx.clone();
        let full_output = ctx.run(raw_input, |ctx| {
            egui::Window::new("Gravity GPU")
                .default_pos([18.0, 18.0])
                .resizable(false)
                .show(ctx, |ui| {
                    ui.heading("Simulation");
                    ui.label(format!("{} particles", PARTICLE_COUNT));
                    ui.label(format!("Total mass: {:.0}", self.total_mass));
                    ui.label(format!("Physics steps: {}", self.physics_steps));
                    ui.label(format!("FPS: {:.0}", self.fps));
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui
                            .button(if self.paused { "Resume" } else { "Pause" })
                            .clicked()
                        {
                            self.paused = !self.paused;
                        }
                        if ui.button("Reset").clicked() {
                            self.reset();
                        }
                    });
                    ui.checkbox(&mut self.trails_visible, "Show trails");
                    ui.add(egui::Slider::new(&mut self.time_scale, 0.125..=4.0).text("Time scale"));
                    ui.add(egui::Slider::new(&mut self.zoom, 0.05..=5.0).text("Zoom"));
                    ui.label("Space: pause • T: trails • R: reset • ↑/↓: speed");
                });
        });
        self.egui_state
            .handle_platform_output(self.window.as_ref(), full_output.platform_output);
        let paint_jobs = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: full_output.pixels_per_point,
        };
        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, image_delta);
        }
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
        self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &paint_jobs,
            &screen_descriptor,
        );
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui overlay"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            self.egui_renderer
                .render(&mut pass, &paint_jobs, &screen_descriptor);
        }
        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
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
fn ceil_div(value: usize, divisor: u32) -> u32 {
    (value as u32).div_ceil(divisor)
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn make_collision_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params: &wgpu::Buffer,
    source: &wgpu::Buffer,
    meta: &wgpu::Buffer,
    destination: &wgpu::Buffer,
    control: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("collision bind group"),
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
                resource: meta.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: destination.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: control.as_entire_binding(),
            },
        ],
    })
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
            Event::WindowEvent { event, .. } => {
                let response = state
                    .egui_state
                    .on_window_event(state.window.as_ref(), &event);
                if response.consumed {
                    if matches!(event, WindowEvent::RedrawRequested) {
                        state.redraw();
                    }
                    return;
                }
                match event {
                    WindowEvent::CloseRequested => target.exit(),
                    WindowEvent::Resized(size) => state.resize(size),
                    WindowEvent::MouseWheel {
                        delta: MouseScrollDelta::LineDelta(_, y),
                        ..
                    } => {
                        state.zoom =
                            (state.zoom * 1.03_f32.powf(y.clamp(-1.0, 1.0))).clamp(0.05, 5.0);
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
                        KeyCode::ArrowDown => {
                            state.time_scale = (state.time_scale * 0.5).max(0.125)
                        }
                        _ => {}
                    },
                    WindowEvent::RedrawRequested => state.redraw(),
                    _ => {}
                }
            }
            Event::AboutToWait => window.request_redraw(),
            _ => {}
        }
    })?;
    Ok(())
}
