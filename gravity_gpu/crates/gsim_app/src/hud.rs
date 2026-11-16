//! egui HUD: the three legacy HUD lines plus a GPU-port status line.
//!
//! Content and formats match the legacy `draw_interface` exactly; only the
//! backend label changed (`AVX-512: 16 x f32` -> `WGSL compute`). A fourth
//! line shows step wall-time, the dropped-step counter (quirk g: legacy
//! dropped the backlog silently) and sim time.

use crate::windowed::Gfx;
use gsim_core::constants::{CLOCKWISE_ROTATION_WEIGHT, COUNTERCLOCKWISE_ROTATION_WEIGHT};

impl Gfx {
    /// `w_px`/`h_px` are physical pixels (for the wgpu screen descriptor);
    /// `w_pts`/`h_pts` are egui points (for HUD layout). They differ on
    /// scaled displays (`pixels_per_point > 1`): using pixels for layout
    /// pushes the bottom-anchored lines off-screen, where egui's area
    /// constraining clamps them onto each other.
    pub(crate) fn egui_frame(
        &mut self,
        view: &wgpu::TextureView,
        w_px: u32,
        h_px: u32,
        _w_pts: f32,
        h_pts: f32,
    ) {
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let s = &self.stats;
        let cw_pct = CLOCKWISE_ROTATION_WEIGHT
            / (CLOCKWISE_ROTATION_WEIGHT + COUNTERCLOCKWISE_ROTATION_WEIGHT)
            * 100.0;
        let state = if self.paused { "PAUSED" } else { "RUNNING" };
        let trails = if self.trails_visible {
            format!("VISIBLE (TAIL {:.0}s)", self.sim.trail_max_age())
        } else {
            "HIDDEN".to_string()
        };
        let largest_pct = if s.total_mass > 0.0 {
            s.max_mass / s.total_mass * 100.0
        } else {
            0.0
        };
        // Legacy HUD formats, backend label now "WGSL compute".
        let l1 =
            format!(
            "{} particles | {} | Initial rotation: {:.0}% CW / {:.0}% CCW | WGSL compute | FPS: {}",
            s.count, state, cw_pct, 100.0 - cw_pct, self.fps_ema as u32
        );
        let l2 = format!(
            "Total mass: {:.0} | Largest: {:.1} ({:.1}%) | COM: ({:.2}, {:.2}) | Time: {:.2}x | Zoom: {:.2}x | Trails: {} ({})",
            s.total_mass, s.max_mass, largest_pct, s.com_x, s.com_y,
            self.time_scale, self.zoom, trails, s.trail_live
        );
        let l3 = "Space: pause | R: reset | T: trails | Up/Down: time speed | Mouse wheel: zoom";
        let l4 = format!(
            "step wall: {:.2} ms | dropped steps: {} | sim t: {:.1}s",
            self.step_wall_ms, self.dropped_steps, self.sim.sim_time
        );

        self.egui_ctx.begin_pass(raw_input);
        {
            let ctx = &self.egui_ctx;
            hud_line(ctx, "hud_l1", [20.0, 8.0], &l1, 16.0, [217, 230, 255]);
            hud_line(ctx, "hud_l2", [20.0, 32.0], &l2, 14.0, [179, 199, 230]);
            hud_line(
                ctx,
                "hud_l3",
                [20.0, h_pts - 28.0],
                l3,
                13.0,
                [153, 173, 204],
            );
            hud_line(
                ctx,
                "hud_l4",
                [20.0, h_pts - 50.0],
                &l4,
                13.0,
                [140, 150, 170],
            );
        }
        let full_output = self.egui_ctx.end_pass();
        self.egui_state
            .handle_platform_output(&self.window, full_output.platform_output);

        let tris = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        for (id, deltas) in &full_output.textures_delta.set {
            for delta in deltas {
                self.egui_renderer.update_texture(
                    &self.sim.ctx.device,
                    &self.sim.ctx.queue,
                    *id,
                    delta,
                );
            }
        }
        let screen_desc = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [w_px, h_px],
            pixels_per_point: full_output.pixels_per_point,
        };
        let mut encoder =
            self.sim
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("gsim egui"),
                });
        let command_buffers = self.egui_renderer.update_buffers(
            &self.sim.ctx.device,
            &self.sim.ctx.queue,
            &mut encoder,
            &tris,
            &screen_desc,
        );
        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gsim egui pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            let mut pass = pass.forget_lifetime();
            self.egui_renderer.render(&mut pass, &tris, &screen_desc);
        }
        self.sim.ctx.queue.submit(
            command_buffers
                .into_iter()
                .chain(std::iter::once(encoder.finish())),
        );
        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
    }
}

fn hud_line(ctx: &egui::Context, id: &str, pos: [f32; 2], text: &str, size: f32, rgb: [u8; 3]) {
    egui::Area::new(egui::Id::unique(id))
        .fixed_pos(egui::pos2(pos[0], pos[1]))
        .movable(false)
        .interactable(false)
        .show(ctx, |ui| {
            // Extend (never wrap): these are single-line status readouts, as
            // in the legacy app. The default Wrap mode can wrap mid-string
            // (the wrap width derives from the area's own measured width, so
            // once wrapped it stays wrapped) and the second line then
            // collides with the HUD line below.
            ui.add(
                egui::Label::new(
                    egui::RichText::new(text)
                        .size(size)
                        .color(egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2])),
                )
                .wrap_mode(egui::TextWrapMode::Extend),
            );
        });
}
