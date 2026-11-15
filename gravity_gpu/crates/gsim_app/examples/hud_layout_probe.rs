//! Wrap-threshold probe: render the EXACT hud.rs strings at several viewport
//! widths and report each area's rect, to find where lines start wrapping.

use std::cell::RefCell;

thread_local! {
    static AREA_RECTS: RefCell<Vec<(&'static str, egui::Rect)>> = const { RefCell::new(Vec::new()) };
}

fn hud_line(ctx: &egui::Context, id: &'static str, pos: [f32; 2], text: &str, size: f32) {
    egui::Area::new(egui::Id::unique(id))
        .fixed_pos(egui::pos2(pos[0], pos[1]))
        .movable(false)
        .interactable(false)
        .show(ctx, |ui| {
            // TEST: Extend (no wrapping) vs default Wrap
            ui.add(
                egui::Label::new(egui::RichText::new(text).size(size))
                    .wrap_mode(egui::TextWrapMode::Extend),
            );
            AREA_RECTS.with(|r| r.borrow_mut().push((id, ui.min_rect())));
        });
}

fn main() {
    // Exact strings from hud.rs (values as in the user's screenshot).
    let l1 =
        "40 particles | RUNNING | Initial rotation: 100% CW / 0% CCW | WGSL compute | FPS: 175";
    let l2 = "Total mass: 124499 | Largest: 84066.1 (67.5%) | COM: (-0.00, 0.00) | Time: 1.00x | Zoom: 0.44x | Trails: HIDDEN (3159762)";

    for (w_px, ppp) in [(1920u32, 1.5f32), (1600, 1.5), (1280, 1.5), (1920, 1.0)] {
        let h_px = 1000u32;
        let h_pts = h_px as f32 / ppp;
        println!(
            "=== window {w_px}x{h_px}px @ ppp={ppp} (viewport {:.0}x{:.0}pt) ===",
            w_px as f32 / ppp,
            h_pts
        );
        AREA_RECTS.with(|r| r.borrow_mut().clear());
        let ctx = egui::Context::default();
        let mut viewports = egui::ViewportIdMap::default();
        viewports.insert(
            egui::ViewportId::ROOT,
            egui::ViewportInfo {
                native_pixels_per_point: Some(ppp),
                ..Default::default()
            },
        );
        let raw_input = egui::RawInput {
            viewport_id: egui::ViewportId::ROOT,
            viewports,
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(w_px as f32 / ppp, h_pts),
            )),
            ..Default::default()
        };
        for frame in 0..8 {
            let mut input = raw_input.clone();
            input.time = Some(frame as f64 * 0.1);
            AREA_RECTS.with(|r| r.borrow_mut().clear());
            ctx.begin_pass(input);
            hud_line(&ctx, "hud_l1", [20.0, 8.0], l1, 16.0);
            hud_line(&ctx, "hud_l2", [20.0, 32.0], l2, 14.0);
            let mut out = ctx.end_pass();
            out.textures_delta.clear();
            if frame == 7 {
                AREA_RECTS.with(|r| {
                    for (id, rect) in r.borrow().iter() {
                        let h = rect.max.y - rect.min.y;
                        println!(
                            "{id}: rect y=[{:.1},{:.1}] h={:.1}pt x=[{:.1},{:.1}]",
                            rect.min.y, rect.max.y, h, rect.min.x, rect.max.x
                        );
                    }
                });
            }
        }
    }
}
