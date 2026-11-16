//! `--headless`: run `--steps` physics steps with no window at all and emit
//! a JSON summary (legacy-harness style). No display, surface or event loop
//! is touched.

use gsim_core::cli::CliConfig;
use gsim_core::constants::SPAWN_RADIUS;
use gsim_core::gpu::{Sim, SimConfig};
use gsim_core::{log_error, log_info};
use serde_json::json;
use std::time::Instant;

pub fn run(cfg: &CliConfig) {
    let sim_cfg = SimConfig {
        n_max: cfg.n,
        seed: cfg.seed,
        mode: cfg.mode,
        merge_mode: cfg.merge_mode,
        no_accel_cache: cfg.no_accel_cache,
        spawn_radius: SPAWN_RADIUS,
        workgroup_size: 128,
    };
    let mut sim = match Sim::new(cfg.backend, cfg.adapter, sim_cfg, None) {
        Ok(s) => s,
        Err(e) => {
            log_error!("cannot open adapter: {e}");
            std::process::exit(1);
        }
    };
    sim.spawn();

    log_info!(
        "headless: starting {} steps (n={}, mode={:?})",
        cfg.steps,
        cfg.n,
        cfg.mode
    );
    let wall = Instant::now();
    for _ in 0..cfg.steps {
        sim.step();
    }
    sim.ctx
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("drain queue");
    let wall_s = wall.elapsed().as_secs_f64();

    sim.update_stats();
    let stats = sim.read_stats();
    let hash = sim.readback_particles().positions_hash();
    let info = &sim.ctx.adapter_info;

    let summary = json!({
        "adapter": info,
        "mode": format!("{:?}", cfg.mode).to_lowercase(),
        "merge_mode": format!("{:?}", cfg.merge_mode).to_lowercase(),
        "no_accel_cache": cfg.no_accel_cache,
        "seed": cfg.seed,
        "steps": cfg.steps,
        "initial_count": cfg.n,
        "final_count": stats.count,
        "total_mass": stats.total_mass,
        "max_mass": stats.max_mass,
        "com": [stats.com_x, stats.com_y],
        "positions_hash": hash,
        "wall_seconds": wall_s,
        "steps_per_second": cfg.steps as f64 / wall_s.max(1e-9),
        "vram_bytes": sim.vram_bytes(),
        "gpu_timestamps": sim.timer.gpu_timestamps,
    });
    let text = serde_json::to_string_pretty(&summary).expect("json");
    log_info!(
        "headless done: {:.2} steps/s ({} steps, final count {})",
        cfg.steps as f64 / wall_s.max(1e-9),
        cfg.steps,
        stats.count
    );
    match &cfg.dump {
        Some(path) => std::fs::write(path, &text).expect("dump write"),
        None => println!("{text}"),
    }
}
