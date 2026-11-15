//! Interactive windowed mode: winit event loop, wgpu surface, egui HUD.
//!
//! Preserves the legacy visible behavior: the three HUD lines (with the
//! backend label now reading `WGSL compute`), the controls (Space/R/T/
//! Up/Down/wheel), `dt = 1/120` with ≤ 8 substeps per frame, sim-time trail
//! sampling, and the mass-based colour transition. A fourth HUD line shows
//! step wall-time and the dropped-step counter (quirk g: legacy dropped the
//! backlog silently).

use gsim_core::cli::CliConfig;
use gsim_core::constants::*;
use gsim_core::gpu::{GpuContext, RenderParams, Sim, SimConfig};
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowAttributes};

pub fn run(cfg: &CliConfig) {
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(cfg.clone());
    event_loop.run_app(&mut app).expect("run app");
}

struct App {
    cfg: CliConfig,
    window: Option<Arc<Window>>,
    gfx: Option<Gfx>,
}

pub(crate) struct Gfx {
    pub(crate) window: Arc<Window>,
    pub(crate) sim: Sim,
    pub(crate) surface: wgpu::Surface<'static>,
    pub(crate) surface_format: wgpu::TextureFormat,
    pub(crate) egui_ctx: egui::Context,
    pub(crate) egui_state: egui_winit::State,
    pub(crate) egui_renderer: egui_wgpu::Renderer,
    // Legacy-equivalent UI state.
    pub(crate) paused: bool,
    pub(crate) trails_visible: bool,
    pub(crate) time_scale: f32,
    pub(crate) zoom: f32,
    pub(crate) accumulated: f32,
    pub(crate) dropped_steps: u64,
    pub(crate) last_frame: Instant,
    pub(crate) fps_ema: f32,
    pub(crate) step_wall_ms: f32,
    pub(crate) stats: gsim_core::stats::Stats,
}

impl App {
    fn new(cfg: CliConfig) -> Self {
        Self {
            cfg,
            window: None,
            gfx: None,
        }
    }

    fn init_gfx(&mut self, window: Arc<Window>) {
        let cfg = &self.cfg;
        let ctx = GpuContext::open(cfg.backend, cfg.adapter).expect("open adapter");
        let target = wgpu::SurfaceTarget::Window(Box::new(window.clone()));
        let surface = ctx
            .instance()
            .create_surface(target)
            .expect("create surface");
        let caps = surface.get_capabilities(ctx.adapter());
        let surface_format = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);
        let sim_cfg = SimConfig {
            n_max: cfg.n,
            seed: cfg.seed,
            mode: cfg.mode,
            merge_mode: cfg.merge_mode,
            no_accel_cache: cfg.no_accel_cache,
            spawn_radius: SPAWN_RADIUS,
            workgroup_size: 128,
        };
        let mut sim = Sim::from_context(ctx, sim_cfg, Some(surface_format)).expect("sim");
        sim.spawn();

        let size = window.inner_size();
        surface.configure(
            &sim.ctx.device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: surface_format,
                width: size.width.max(1),
                height: size.height.max(1),
                present_mode: wgpu::PresentMode::AutoVsync,
                desired_maximum_frame_latency: 2,
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
                color_space: wgpu::SurfaceColorSpace::Srgb,
            },
        );

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &*window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            &sim.ctx.device,
            surface_format,
            egui_wgpu::RendererOptions::default(),
        );

        let scale = window.scale_factor() as f32;
        let zoom = (size.width.min(size.height) as f32 / scale) / (SPAWN_RADIUS * 2.25);
        self.gfx = Some(Gfx {
            window,
            sim,
            surface,
            surface_format,
            egui_ctx,
            egui_state,
            egui_renderer,
            paused: false,
            trails_visible: false,
            time_scale: 1.0,
            zoom,
            accumulated: 0.0,
            dropped_steps: 0,
            last_frame: Instant::now(),
            fps_ema: 60.0,
            step_wall_ms: 0.0,
            stats: gsim_core::stats::Stats::default(),
        });
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window = Arc::new(
                event_loop
                    .create_window(
                        WindowAttributes::default()
                            .with_title("GSIM — WGSL compute gravity")
                            .with_inner_size(PhysicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT)),
                    )
                    .expect("create window"),
            );
            self.init_gfx(window.clone());
            self.window = Some(window);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let Some(gfx) = self.gfx.as_mut() else { return };
        // Feed egui first.
        let _ = gfx.egui_state.on_window_event(&gfx.window, &event);

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                gfx.surface.configure(
                    &gfx.sim.ctx.device,
                    &wgpu::SurfaceConfiguration {
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        format: gfx.surface_format,
                        width: size.width.max(1),
                        height: size.height.max(1),
                        present_mode: wgpu::PresentMode::AutoVsync,
                        desired_maximum_frame_latency: 2,
                        alpha_mode: wgpu::CompositeAlphaMode::Opaque,
                        view_formats: vec![],
                        color_space: wgpu::SurfaceColorSpace::Srgb,
                    },
                );
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed && !event.repeat {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::Space) => {
                            gfx.paused = !gfx.paused;
                        }
                        PhysicalKey::Code(KeyCode::KeyR) => {
                            gfx.sim.spawn();
                            gfx.accumulated = 0.0;
                            gfx.dropped_steps = 0;
                        }
                        PhysicalKey::Code(KeyCode::KeyT) => {
                            gfx.trails_visible = !gfx.trails_visible;
                        }
                        PhysicalKey::Code(KeyCode::ArrowUp) => {
                            gfx.time_scale = (gfx.time_scale * 2.0).min(4.0);
                        }
                        PhysicalKey::Code(KeyCode::ArrowDown) => {
                            gfx.time_scale = (gfx.time_scale * 0.5).max(0.125);
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let y = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 50.0,
                };
                gfx.zoom *= 1.03_f32.powf(y.clamp(-1.0, 1.0));
                gfx.zoom = gfx.zoom.clamp(0.05, 5.0);
            }
            WindowEvent::RedrawRequested => {
                gfx.frame();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

impl Gfx {
    pub(crate) fn frame(&mut self) {
        let now = Instant::now();
        let frame_dt = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;
        let frame_time = frame_dt.min(0.1);
        if frame_dt > 0.0 {
            let fps = 1.0 / frame_dt;
            self.fps_ema += (fps - self.fps_ema) * 0.05;
        }

        // Physics substeps: dt = 1/120, at most 8 per frame (quirk g: count
        // dropped debt instead of dropping it silently).
        let step_wall = Instant::now();
        if !self.paused {
            self.accumulated += frame_time.min(0.05) * self.time_scale;
            let mut completed = 0;
            while self.accumulated >= PHYSICS_DT && completed < MAX_PHYSICS_STEPS_PER_FRAME {
                self.sim.step();
                self.accumulated -= PHYSICS_DT;
                completed += 1;
            }
            if completed == MAX_PHYSICS_STEPS_PER_FRAME {
                self.dropped_steps += (self.accumulated / PHYSICS_DT) as u64;
                self.accumulated = 0.0;
            }
        }
        self.step_wall_ms = step_wall.elapsed().as_secs_f32() * 1000.0;

        self.sim.update_stats();
        self.stats = self.sim.read_stats();

        let size = self.window.inner_size();
        let (w, h) = (size.width.max(1), size.height.max(1));
        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(o)
            | wgpu::CurrentSurfaceTexture::Suboptimal(o) => o,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface
                    .configure(&self.sim.ctx.device, &self.surface_config(w, h));
                return;
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => {
                std::thread::sleep(Duration::from_millis(16));
                return;
            }
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.sim.render(RenderParams {
            view: &view,
            width: w,
            height: h,
            zoom: self.zoom,
            total_mass: self.stats.total_mass,
            trails_visible: self.trails_visible,
            trail_instances: self.stats.trail_points,
        });

        self.egui_frame(&view, w, h);
        self.sim.ctx.queue.present(output);
    }

    pub(crate) fn surface_config(&self, w: u32, h: u32) -> wgpu::SurfaceConfiguration {
        wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: self.surface_format,
            width: w,
            height: h,
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: vec![],
            color_space: wgpu::SurfaceColorSpace::Srgb,
        }
    }
}
