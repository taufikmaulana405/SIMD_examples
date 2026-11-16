//! `--bench`: headless benchmark ladder + N-scaling sweep.
//!
//! Rung sizes come from `--n`/`--steps`; the sweep list from `--sweep`
//! (default: a small smoke pair). On Mesa lavapipe (software Vulkan) run
//! with small values for a correctness/smoke pass: every timing is labeled
//! with the adapter and must be read as "lavapipe software, NOT GPU
//! performance". Real numbers come from the Windows run driven by
//! `dist/windows/bench.ps1`, which passes the full 1k→200k sweep.

use gsim_core::cli::{CliConfig, GpuMode, MergeMode};
use gsim_core::constants::SPAWN_RADIUS;
use gsim_core::gpu::bench::BenchmarkRunner;
use gsim_core::telemetry::default_path;
use gsim_core::{log_debug, log_error, log_info, log_warn};
use serde_json::{json, Value};
use std::path::PathBuf;

/// Parity-check size (correctness, fixed on all platforms).
const PARITY_N: usize = 256;
const PARITY_STEPS: u64 = 20;

/// Default smoke sweep (used when `--sweep` is absent, e.g. lavapipe).
const SMOKE_SWEEP: [usize; 2] = [256, 512];

fn parse_sweep(csv: &str) -> Vec<usize> {
    csv.split(',')
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("--sweep: cannot parse '{s}'"))
        })
        .collect()
}

pub fn run(cfg: &CliConfig) {
    let tele: PathBuf = cfg
        .telemetry
        .clone()
        .map(PathBuf::from)
        .unwrap_or_else(default_path);
    let runner = match BenchmarkRunner::new(
        cfg.backend,
        cfg.adapter,
        cfg.seed,
        tele,
        cfg.shader_debug,
        cfg.dxc_path.clone(),
        cfg.no_dxc,
        cfg.trails,
        cfg.trail_length,
    ) {
        Ok(r) => r,
        Err(e) => {
            log_error!("cannot open benchmark adapter: {e}");
            std::process::exit(1);
        }
    };
    let info = &runner.adapter_info;
    log_info!(
        "adapter: {} ({}, {})",
        info.name,
        info.device_type,
        info.backend
    );
    if info.device_type == "cpu" {
        log_warn!("software adapter in use: timings are smoke-only, NOT GPU performance");
    }
    log_debug!(
        "telemetry: {}",
        cfg.telemetry.as_deref().unwrap_or("default")
    );

    runner.log(
        "phase",
        json!({
            "phase": "bench_start",
            "cli": {
                "seed": cfg.seed, "n": cfg.n, "backend": format!("{:?}", cfg.backend),
                "adapter": cfg.adapter, "steps": cfg.steps,
                "mode": format!("{:?}", cfg.mode),
                "merge_mode": format!("{:?}", cfg.merge_mode),
                "sweep": cfg.sweep, "sweep_steps": cfg.sweep_steps,
            },
        }),
    );

    // Rung (1): cite the stage-1 legacy CPU baselines (not rerun).
    runner.log(
        "benchmark",
        json!({
            "context": "stage-1 legacy CPU baselines, CITED not rerun (Ubuntu AMD EPYC 9D25, headless, seed 12345)",
            "results": [
                {"crate": "gravity_scalar", "N": 1000, "median_steps_per_sec": 224.49},
                {"crate": "gravity_scalar", "N": 2000, "median_steps_per_sec": 50.68},
                {"crate": "gravity_simd_avx2", "N": 5000, "median_steps_per_sec": 54.75},
                {"crate": "gravity_simd_avx2", "N": 2000, "median_steps_per_sec": 331.66},
                {"crate": "gravity_simd_avx512", "N": 10000, "median_steps_per_sec": 26.18},
                {"crate": "gravity_simd_avx512", "N": 2000, "median_steps_per_sec": 589.50},
            ],
            "number_to_beat": "AVX-512 @ N=10000: 26.18 steps/s (0.22x realtime); GPU target >= 120 steps/s at N=10000",
        }),
    );

    // Autotune the tiled workgroup size on the running adapter (at rung N).
    let (best_wg, table) = runner.autotune(cfg.n);
    log_info!("autotune: best workgroup size = {best_wg}");
    log_debug!("autotune table: {table:?}");

    // Parity, redesigned 2026-10-18 for real-hardware honesty: GPU
    // naive-vs-tiled bitwise match, per-mode determinism across two runs,
    // and GPU-vs-CPU max position error under tolerance (see
    // BenchmarkRunner::parity_check). Bitwise GPU-vs-CPU over full steps is
    // unachievable across FP hardware (FMA contraction, sqrt precision).
    let parity = runner.parity_check(PARITY_N, PARITY_STEPS);
    let parity_ok = parity["ok"].as_bool().unwrap_or(false);
    log_info!(
        "parity: ok={parity_ok} cross_mode_bitwise={} det=[naive:{}, tiled:{}] cpu_max_err={} (tol {}) counts_match={}",
        parity["cross_mode_bitwise_match"],
        parity["modes"][0]["deterministic"],
        parity["modes"][1]["deterministic"],
        parity["cpu"]["max_position_error"],
        parity["cpu"]["tolerance"],
        parity["cpu"]["counts_match"],
    );
    if !parity_ok {
        log_warn!("parity check FAILED: {parity}");
    } else {
        log_debug!("parity detail: {parity}");
    }
    runner.log("parity", parity.clone());

    // Rungs (2) and (3): naive and tiled ladder rungs at --n/--steps.
    let mut rungs = Vec::new();
    for (mode, wg) in [(GpuMode::Naive, 64), (GpuMode::Tiled, best_wg)] {
        let r = runner.ladder_rung(
            mode,
            MergeMode::Parallel,
            cfg.n,
            cfg.steps,
            SPAWN_RADIUS,
            wg,
        );
        // One combined block: the rung headline plus its per-pass table,
        // separated by newlines — much easier to read than a run of
        // separately-prefixed log lines.
        let mut block = format!(
            "rung {:?}: {:.2} steps/s ({} steps, final count {}), timestamps={}",
            mode, r.steps_per_sec, r.steps, r.final_count, r.gpu_timestamps
        );
        for (name, ns) in &r.per_pass_ns {
            // Per-pass timings are the benchmark's product, not trivia:
            // they stay visible at the default INFO level in bench mode.
            block.push_str(&format!("\n    pass {name}: {ns:.1} ns"));
        }
        log_info!("{block}");
        rungs.push(json!({
            "mode": r.mode, "n": r.n, "steps": r.steps,
            "steps_per_sec": r.steps_per_sec,
            "per_pass_ns": r.per_pass_ns.iter().map(|(n, v)| json!({"pass": n, "ns": v})).collect::<Vec<_>>(),
            "final_count": r.final_count, "vram_bytes": r.vram_bytes,
            "gpu_timestamps": r.gpu_timestamps,
            "context": format!("measured on adapter '{}' ({}); software adapters (lavapipe) are smoke-only, not GPU performance", info.name, info.device_type),
        }));
        runner.log("benchmark", rungs.last().expect("rung").clone());
    }

    // Rung (4): N-scaling sweep. Spawn radius scales as
    // 1500*sqrt(N/10000) (constant surface density); the sweep finds the
    // largest N holding >= 60 FPS at 1x (2 steps/frame) and 4x (8
    // steps/frame) time scale.
    let sweep_ns: Vec<usize> = cfg
        .sweep
        .as_deref()
        .map(parse_sweep)
        .unwrap_or_else(|| SMOKE_SWEEP.to_vec());
    let sweep = runner.sweep(&sweep_ns, cfg.sweep_steps);
    let sweep_json: Vec<Value> = sweep
        .iter()
        .map(|s| {
            json!({
                "n": s.n, "spawn_radius": s.spawn_radius,
                "steps_per_sec": s.steps_per_sec,
                "fps_at_1x": s.fps_at_1x, "fps_at_4x": s.fps_at_4x,
                "meets_60fps_1x": s.fps_at_1x >= 60.0,
                "meets_60fps_4x": s.fps_at_4x >= 60.0,
                "vram_bytes": s.vram_bytes,
            })
        })
        .collect();
    let max_n_1x = sweep
        .iter()
        .filter(|s| s.fps_at_1x >= 60.0)
        .map(|s| s.n)
        .max();
    let max_n_4x = sweep
        .iter()
        .filter(|s| s.fps_at_4x >= 60.0)
        .map(|s| s.n)
        .max();
    log_info!(
        "sweep ok: {} points; max N @60fps: 1x={max_n_1x:?} 4x={max_n_4x:?}",
        sweep_json.len()
    );
    log_info!(
        "sweep points: {}",
        sweep_json
            .iter()
            .map(|s| format!("n={} {:.1}/s", s["n"], s["steps_per_sec"]))
            .collect::<Vec<_>>()
            .join(", ")
    );

    runner.log(
        "phase",
        json!({"phase": "bench_done", "parity_ok": parity_ok}),
    );

    let summary = json!({
        "adapter": info,
        "seed": cfg.seed,
        "parity_ok": parity_ok,
        "parity": parity,
        "autotune_best_workgroup": best_wg,
        "rungs": rungs,
        "sweep": sweep_json,
        "max_n_at_60fps_1x": max_n_1x,
        "max_n_at_60fps_4x": max_n_4x,
    });
    let text = serde_json::to_string_pretty(&summary).expect("json");
    match &cfg.dump {
        Some(path) => std::fs::write(path, &text).expect("dump write"),
        None => println!("{text}"),
    }
}
