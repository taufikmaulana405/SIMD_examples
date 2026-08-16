use std::{borrow::Cow, sync::Arc, time::Instant};

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
struct RenderParams {
    viewport: [f32; 4],
    counts: [u32; 4],
}

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    sim_params: wgpu::Buffer,
    render_params: wgpu::Buffer,
    particles: [wgpu::Buffer; 2],
    sim_bind_groups: [wgpu::BindGroup; 2],
    render_bind_groups: [wgpu::BindGroup; 2],
    integrate_pipeline: wgpu::ComputePipeline,
    finish_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
    particle_index: usize,
    zoom: f32,
    paused: bool,
    accumulator: f32,
    last_frame: Instant,
    total_mass: f32,
}

fn create_particles() -> (Vec<f32>, f32) {
    let mut rng = rand::thread_rng();
    let mut data = vec![0.0; PARTICLE_COUNT * 8];
    let mut total_mass = 0.0;
    let mut center_x = 0.0;
    let mut center_y = 0.0;
    let mut momentum_x = 0.0;
    let mut momentum_y = 0.0;

    for i in 0..PARTICLE_COUNT {
        let angle = rng.gen_range(0.0..std::f32::consts::TAU);
        let u: f32 = rng.gen();
        let distance = (u.sqrt() + (u.powf(1.4) - u.sqrt()) * 0.75) * SPAWN_RADIUS;
        let (rx, ry) = (angle.cos(), angle.sin());
        let speed = 6.0 + 14.0 * (distance / SPAWN_RADIUS).sqrt();
        let mass = rng.gen_range(5.0..20.0);
        let radial_speed = rng.gen_range(-0.01..0.01);
        let base = i * 8;
        let vx = -ry * speed + rx * radial_speed;
        let vy = rx * speed + ry * radial_speed;
        data[base..base + 8].copy_from_slice(&[rx * distance, ry * distance, vx, vy, mass, rng.gen_range(0.55..1.0), 1.0, 0.0]);
        total_mass += mass;
        center_x += data[base] * mass;
        center_y += data[base + 1] * mass;
        momentum_x += vx * mass;
        momentum_y += vy * mass;
    }

    let center_x = center_x / total_mass;
    let center_y = center_y / total_mass;
    let velocity_x = momentum_x / total_mass;
    let velocity_y = momentum_y / total_mass;
    for particle in data.chunks_exact_mut(8) {
        particle[0] -= center_x;
        particle[1] -= center_y;
        particle[2] -= velocity_x;
        particle[3] -= velocity_y;
    }
    (data, total_mass)
}

impl State {
    async fn new(window: Arc<Window>) -> Result<Self, String> {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window).map_err(|e| e.to_string())?;
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }).await.ok_or("No compatible GPU adapter was found")?;
        let info = adapter.get_info();
        eprintln!("GPU: {} ({:?})", info.name, info.backend);
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("gravity_gpu device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        }, None).await.map_err(|e| e.to_string())?;
        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
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
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("particles A"), contents: bytemuck::cast_slice(&initial), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST }),
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("particles B"), contents: bytemuck::cast_slice(&initial), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST }),
        ];
        let sim_params = device.create_buffer(&wgpu::BufferDescriptor { label: Some("simulation parameters"), size: 48, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let render_params = device.create_buffer(&wgpu::BufferDescriptor { label: Some("render parameters"), size: 32, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let sim_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("simulation layout"), entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ] });
        let sim_bind_groups = [
            make_sim_bind_group(&device, &sim_layout, &sim_params, &particles[0], &particles[1]),
            make_sim_bind_group(&device, &sim_layout, &sim_params, &particles[1], &particles[0]),
        ];
        let render_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("render layout"), entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ] });
        let render_bind_groups = particles.each_ref().map(|buffer| device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("particle render bind group"), layout: &render_layout, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: render_params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buffer.as_entire_binding() },
        ] }));
        let sim_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("gravity compute shader"), source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/simulation.wgsl"))) });
        let render_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("particle render shader"), source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/render.wgsl"))) });
        let sim_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("simulation pipeline layout"), bind_group_layouts: &[&sim_layout], push_constant_ranges: &[] });
        let integrate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("integrate pipeline"), layout: Some(&sim_pipeline_layout), module: &sim_module, entry_point: "integrate" });
        let finish_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("finish pipeline"), layout: Some(&sim_pipeline_layout), module: &sim_module, entry_point: "finish" });
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("render pipeline layout"), bind_group_layouts: &[&render_layout], push_constant_ranges: &[] });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor { label: Some("particle render pipeline"), layout: Some(&render_pipeline_layout), vertex: wgpu::VertexState { module: &render_module, entry_point: "vertex", buffers: &[] }, fragment: Some(wgpu::FragmentState { module: &render_module, entry_point: "fragment", targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })] }), primitive: wgpu::PrimitiveState::default(), depth_stencil: None, multisample: wgpu::MultisampleState::default(), multiview: None });
        Ok(Self { surface, device, queue, config, sim_params, render_params, particles, sim_bind_groups, render_bind_groups, integrate_pipeline, finish_pipeline, render_pipeline, particle_index: 0, zoom: 1.0, paused: false, accumulator: 0.0, last_frame: Instant::now(), total_mass })
    }

    fn resize(&mut self, size: PhysicalSize<u32>) { if size.width > 0 && size.height > 0 { self.config.width = size.width; self.config.height = size.height; self.surface.configure(&self.device, &self.config); } }

    fn reset(&mut self) { let (data, mass) = create_particles(); self.queue.write_buffer(&self.particles[0], 0, bytemuck::cast_slice(&data)); self.queue.write_buffer(&self.particles[1], 0, bytemuck::cast_slice(&data)); self.total_mass = mass; self.particle_index = 0; self.accumulator = 0.0; }

    fn step(&mut self) {
        let params = SimParams { count: PARTICLE_COUNT as u32, _padding: 0, dt: DT, gravity: G, softening_squared: SOFTENING_SQUARED, radius_scale: RADIUS_SCALE, _padding2: [0.0; 6] };
        self.queue.write_buffer(&self.sim_params, 0, bytemuck::bytes_of(&params));
        let current = self.particle_index;
        let temporary = 1 - current;
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gravity compute") });
        { let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("integrate"), timestamp_writes: None }); pass.set_pipeline(&self.integrate_pipeline); pass.set_bind_group(0, &self.sim_bind_groups[current], &[]); pass.dispatch_workgroups((PARTICLE_COUNT as u32).div_ceil(WORKGROUP_SIZE), 1, 1); }
        { let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("finish"), timestamp_writes: None }); pass.set_pipeline(&self.finish_pipeline); pass.set_bind_group(0, &self.sim_bind_groups[temporary], &[]); pass.dispatch_workgroups((PARTICLE_COUNT as u32).div_ceil(WORKGROUP_SIZE), 1, 1); }
        let render_data = RenderParams { viewport: [self.config.width as f32, self.config.height as f32, self.zoom, self.total_mass], counts: [PARTICLE_COUNT as u32, 0, 0, 0] };
        self.queue.write_buffer(&self.render_params, 0, bytemuck::bytes_of(&render_data));
        let output = match self.surface.get_current_texture() { Ok(output) => output, Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => { self.surface.configure(&self.device, &self.config); return; }, Err(wgpu::SurfaceError::OutOfMemory) => panic!("GPU is out of memory"), Err(wgpu::SurfaceError::Timeout) => return };
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        { let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor { label: Some("particles"), color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &view, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.005, g: 0.008, b: 0.025, a: 1.0 }), store: wgpu::StoreOp::Store } })], depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None }); pass.set_pipeline(&self.render_pipeline); pass.set_bind_group(0, &self.render_bind_groups[current], &[]); pass.draw(0..6, 0..PARTICLE_COUNT as u32); }
        self.queue.submit(Some(encoder.finish())); output.present();
        self.particle_index = current;
    }

    fn redraw(&mut self) { let now = Instant::now(); let elapsed = now.duration_since(self.last_frame).as_secs_f32().min(0.1); self.last_frame = now; if !self.paused { self.accumulator += elapsed; let mut steps = 0; while self.accumulator >= DT && steps < 8 { self.step(); self.accumulator -= DT; steps += 1; } if steps == 8 { self.accumulator = 0.0; } } else { self.step(); }
    }
}

fn make_sim_bind_group(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, params: &wgpu::Buffer, source: &wgpu::Buffer, destination: &wgpu::Buffer) -> wgpu::BindGroup { device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("simulation bind group"), layout, entries: &[wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: source.as_entire_binding() }, wgpu::BindGroupEntry { binding: 2, resource: destination.as_entire_binding() }] }) }

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    let window = Arc::new(WindowBuilder::new()
        .with_title("Gravity GPU — AVX2 reference")
        .with_inner_size(PhysicalSize::new(1280, 800))
        .build(&event_loop)?);
    let mut state = pollster::block_on(State::new(window.clone()))?;
    event_loop.run(move |event, target| {
        target.set_control_flow(ControlFlow::Poll);
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => target.exit(),
                WindowEvent::Resized(size) => state.resize(size),
                WindowEvent::MouseWheel { delta: MouseScrollDelta::LineDelta(_, y), .. } => {
                    state.zoom = (state.zoom * (1.0 + y * 0.05)).clamp(0.05, 5.0);
                }
                WindowEvent::KeyboardInput { event: KeyEvent { physical_key: PhysicalKey::Code(code), state: key_state, .. }, .. }
                    if key_state == ElementState::Pressed => match code {
                        KeyCode::Space => state.paused = !state.paused,
                        KeyCode::KeyR => state.reset(),
                        KeyCode::ArrowUp => state.zoom = (state.zoom * 1.1).min(5.0),
                        KeyCode::ArrowDown => state.zoom = (state.zoom / 1.1).max(0.05),
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
