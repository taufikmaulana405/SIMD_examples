#![forbid(unsafe_code)]
//! `gsim_app`: thin frontend for the GPU gravity sim.
//!
//! Three entry modes, selected by CLI flags:
//! - `--bench`: headless benchmark ladder + N-sweep, then exit.
//! - `--headless`: run `--steps` physics steps with no window, print a JSON
//!   summary (works with no display at all).
//! - default: interactive winit window with egui HUD.

mod bench_cmd;
mod headless_cmd;
mod hud;
mod windowed;

use gsim_core::cli::CliConfig;

fn main() {
    let cfg = CliConfig::parse();
    if cfg.bench {
        bench_cmd::run(&cfg);
    } else if cfg.headless {
        headless_cmd::run(&cfg);
    } else {
        windowed::run(&cfg);
    }
}
