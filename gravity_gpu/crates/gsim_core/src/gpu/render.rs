//! [`Sim`] rendering: render pipelines, bind groups, per-frame draw.
//!
//! Trails are drawn first (instanced quads from the trail ring), then
//! particles (instanced quads with the legacy §8 colour mapping). The CPU
//! sizes the trail draw from the last stats readback; particles always draw
//! `n_max` instances with dead slots culled in the vertex shader.

use crate::constants::BACKGROUND_COLOR;
use crate::gpu::shaders;
use crate::gpu::sim::Sim;
use std::borrow::Cow;

/// Parameters for one rendered frame (bundled: clippy allows 7 args max).
pub struct RenderParams<'a> {
    /// Target view.
    pub view: &'a wgpu::TextureView,
    /// Viewport width in px.
    pub width: u32,
    /// Viewport height in px.
    pub height: u32,
    /// World-to-screen zoom.
    pub zoom: f32,
    /// Total system mass (drives the yellow transition).
    pub total_mass: f32,
    /// Whether the trail layer is drawn.
    pub trails_visible: bool,
    /// Live (unexpired) trail points from the last stats readback —
    /// the only points drawn.
    pub trail_live: u32,
    /// Ring slot of the oldest live point; the live arc is
    /// `[trail_start, trail_start + trail_live)`, wrapping at the ring
    /// capacity.
    pub trail_start: u32,
}

impl Sim {
    /// Build the render pipelines and bind groups for `format`.
    pub fn init_render(&mut self, format: wgpu::TextureFormat) {
        let device = &self.ctx.device;
        let layout_entry =
            |binding: u32, read_only: bool, uniform: bool| -> wgpu::BindGroupLayoutEntry {
                wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: if uniform {
                        wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        }
                    } else {
                        wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        }
                    },
                    count: None,
                }
            };
        let make = |name: &str,
                    entry_vs: &str,
                    entry_fs: &str,
                    buffers: &[&wgpu::Buffer],
                    cam: &wgpu::Buffer|
         -> (wgpu::RenderPipeline, wgpu::BindGroup) {
            let src = shaders::render_source(name);
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(src)),
            });
            let n_storage = buffers.len() as u32;
            let mut entries: Vec<wgpu::BindGroupLayoutEntry> = (0..n_storage)
                .map(|b| layout_entry(b, true, false))
                .collect();
            entries.push(layout_entry(n_storage, true, true));
            let bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(&format!("gsim render {name} layout")),
                entries: &entries,
            });
            let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("gsim render {name} pipeline layout")),
                bind_group_layouts: &[Some(&bg_layout)],
                immediate_size: 0,
            });
            let mut bg_entries: Vec<wgpu::BindGroupEntry> = buffers
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect();
            bg_entries.push(wgpu::BindGroupEntry {
                binding: n_storage,
                resource: cam.as_entire_binding(),
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("gsim render {name} bg")),
                layout: &bg_layout,
                entries: &bg_entries,
            });
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&format!("gsim render {name}")),
                layout: Some(&pl_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some(entry_vs),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(entry_fs),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
            (pipeline, bg)
        };
        let (pp, bgp) = make(
            "particles",
            "vs",
            "fs",
            &[&self.pos_a, &self.mass_a, &self.bright_a],
            &self.cam_particles,
        );
        let (pt, bgt) = make("trails", "vs", "fs", &[&self.trail_ring], &self.cam_trails);
        self.p_render_particles = Some(pp);
        self.render_bg_particles = Some(bgp);
        self.p_render_trails = Some(pt);
        self.render_bg_trails = Some(bgt);
        crate::log_debug!(
            "shader inventory (render):\n    particles::vs + particles::fs (format {format:?})\n    trails::vs + trails::fs (format {format:?})"
        );
    }

    /// Render trails + particles into the target view.
    pub fn render(&mut self, params: RenderParams<'_>) {
        let RenderParams {
            view,
            width,
            height,
            zoom,
            total_mass,
            trails_visible,
            trail_live,
            trail_start,
        } = params;
        let cam = |zoom: f32, a: f32, b: f32| {
            let mut raw = Vec::with_capacity(32);
            for v in [width as f32, height as f32, zoom, a, b] {
                raw.extend_from_slice(&v.to_le_bytes());
            }
            raw.extend_from_slice(&[0u8; 12]);
            raw
        };
        self.ctx
            .queue
            .write_buffer(&self.cam_particles, 0, &cam(zoom, total_mass, 0.0));
        self.ctx.queue.write_buffer(
            &self.cam_trails,
            0,
            &cam(zoom, self.sim_time, self.trail_max_age),
        );
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gsim render"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gsim render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: f64::from(BACKGROUND_COLOR[0]),
                            g: f64::from(BACKGROUND_COLOR[1]),
                            b: f64::from(BACKGROUND_COLOR[2]),
                            a: f64::from(BACKGROUND_COLOR[3]),
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            // Draw only the live arc of the ring: expired points are
            // never fetched. The arc wraps at most once.
            if trails_visible && trail_live > 0 {
                pass.set_pipeline(self.p_render_trails.as_ref().expect("render init"));
                pass.set_bind_group(0, self.render_bg_trails.as_ref().expect("render init"), &[]);
                let cap = self.trail_capacity;
                let end = trail_start + trail_live;
                pass.draw(0..6, trail_start..end.min(cap));
                if end > cap {
                    pass.draw(0..6, 0..(end - cap));
                }
            }
            pass.set_pipeline(self.p_render_particles.as_ref().expect("render init"));
            pass.set_bind_group(
                0,
                self.render_bg_particles.as_ref().expect("render init"),
                &[],
            );
            pass.draw(0..6, 0..self.cfg.n_max as u32);
        }
        self.ctx.queue.submit(Some(encoder.finish()));
    }
}
