//! Stage-3 parity test P4: long-run statistics vs the legacy's own noise
//! floor (task 4).
//!
//! Over >=5 seeds, the GPU's particle-count curve, largest-mass fraction,
//! energy drift, and COM drift must stay within the legacy's own
//! noise floor, defined per-seed per-metric as |scalar - avx2| (two
//! independent legacy kernel implementations at the same seed+N).
//!
//! The GPU port uses the AVX-512 spawn constants, so the GPU is compared
//! against the AVX-512 legacy variant (same initial conditions); the
//! scalar-vs-AVX2 spread measures how much two correct implementations
//! may legitimately disagree.

use gsim_core::cli::{Backend, GpuMode, MergeMode};
use gsim_core::constants::*;
use gsim_core::gpu::{Sim, SimConfig};
use gsim_core::spawn::Particles;
use gsim_core::telemetry::{append_telemetry_stage, AdapterInfo};
use serde_json::json;
use std::process::Command;

const SEEDS: [u64; 5] = [12345, 7, 99, 42, 777];
const N: usize = 1000;
const STEPS: u64 = 600;
const CHECKPOINTS: [u64; 6] = [100, 200, 300, 400, 500, 600];

#[derive(Clone, Default)]
struct Metrics {
    count: usize,
    largest_frac: f64,
    energy: f64,
    com_x: f64,
    com_y: f64,
}

/// Total energy E = KE - PE (softened), in f64, from [x,y,vx,vy,mass].
fn energy_of(bodies: &[[f64; 5]]) -> f64 {
    let n = bodies.len();
    let mut ke = 0.0;
    for b in bodies {
        ke += 0.5 * b[4] * (b[2] * b[2] + b[3] * b[3]);
    }
    let s2 = f64::from(GRAVITY_SOFTENING * GRAVITY_SOFTENING);
    let mut pe = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = bodies[j][0] - bodies[i][0];
            let dy = bodies[j][1] - bodies[i][1];
            let r = (dx * dx + dy * dy + s2).sqrt();
            pe += f64::from(GRAVITATIONAL_CONSTANT) * bodies[i][4] * bodies[j][4] / r;
        }
    }
    ke - pe
}

fn metrics_of(bodies: &[[f64; 5]]) -> Metrics {
    let n = bodies.len();
    let total: f64 = bodies.iter().map(|b| b[4]).sum();
    let largest: f64 = bodies.iter().map(|b| b[4]).fold(0.0, f64::max);
    let (mut cx, mut cy) = (0.0, 0.0);
    for b in bodies {
        cx += b[4] * b[0];
        cy += b[4] * b[1];
    }
    Metrics {
        count: n,
        largest_frac: largest / total.max(1e-30),
        energy: energy_of(bodies),
        com_x: cx / total.max(1e-30),
        com_y: cy / total.max(1e-30),
    }
}

fn particles_to_vec(p: &Particles) -> Vec<[f64; 5]> {
    (0..p.len())
        .map(|i| {
            [
                f64::from(p.x[i]),
                f64::from(p.y[i]),
                f64::from(p.vx[i]),
                f64::from(p.vy[i]),
                f64::from(p.mass[i]),
            ]
        })
        .collect()
}

fn legacy_run(crate_name: &str, seed: u64, steps: u64) -> Vec<[f64; 5]> {
    let bin = format!("../../legacy_work/{crate_name}/target/release/{crate_name}");
    let pos_path = std::env::temp_dir().join(format!("gsim_p4_{crate_name}_{seed}_{steps}.json"));
    let out = Command::new(&bin)
        .args([
            "--seed",
            &seed.to_string(),
            "--count",
            &N.to_string(),
            "--steps",
            &steps.to_string(),
            "--dump-positions",
            pos_path.to_str().unwrap(),
        ])
        .output()
        .unwrap_or_else(|e| panic!("failed to run {bin}: {e}"));
    assert!(
        out.status.success(),
        "{crate_name} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_str(&std::fs::read_to_string(&pos_path).unwrap()).unwrap()
}

#[test]
fn p4_long_run_stats_within_noise_floor() {
    let mut results = Vec::new();
    let mut all_pass = true;

    for &seed in &SEEDS {
        // GPU run (naive/parallel): collect metrics at checkpoints.
        let mut g = Sim::new(
            Backend::Vulkan,
            0,
            SimConfig {
                n_max: N,
                seed,
                mode: GpuMode::Naive,
                merge_mode: MergeMode::Parallel,
                no_accel_cache: false,
                spawn_radius: SPAWN_RADIUS,
                workgroup_size: 128,
                shader_debug: false,
                dxc_path: None,
                no_dxc: false,
                trails: false,
                trail_max_age: 40.0,
            },
            None,
        )
        .expect("lavapipe sim");
        g.spawn();
        let mut gpu_m: Vec<Metrics> = Vec::new();
        let mut done = 0u64;
        for &cp in &CHECKPOINTS {
            for _ in done..cp {
                g.step();
            }
            done = cp;
            gpu_m.push(metrics_of(&particles_to_vec(&g.readback_particles())));
        }

        // Legacy variants at the same checkpoints.
        let mut leg_m: Vec<Vec<Metrics>> = Vec::new(); // [variant][checkpoint]
        for variant in ["gravity_scalar", "gravity_simd_avx2", "gravity_simd_avx512"] {
            let mut vm = Vec::new();
            for &cp in &CHECKPOINTS {
                vm.push(metrics_of(&legacy_run(variant, seed, cp)));
            }
            leg_m.push(vm);
        }

        // Per-metric check: |gpu - avx512| <= |scalar - avx2| (noise floor),
        // with a small absolute floor so exactly-equal early checkpoints
        // don't produce vacuous 0-vs-0 comparisons.
        let e0 = leg_m[2][0].energy.abs().max(1.0);
        let mut seed_pass = true;
        let mut seed_detail = Vec::new();
        for (ci, &cp) in CHECKPOINTS.iter().enumerate() {
            let (gs, sa, a2, a5) = (&gpu_m[ci], &leg_m[0][ci], &leg_m[1][ci], &leg_m[2][ci]);
            let floor_count = ((sa.count as f64 - a2.count as f64).abs()).max(1.0);
            let floor_frac = (sa.largest_frac - a2.largest_frac).abs().max(1e-6);
            let floor_e = ((sa.energy - a2.energy).abs() / e0).max(1e-6);
            let floor_com = ((sa.com_x - a2.com_x).hypot(sa.com_y - a2.com_y)).max(1e-6);

            let d_count = (gs.count as f64 - a5.count as f64).abs();
            let d_frac = (gs.largest_frac - a5.largest_frac).abs();
            let d_e = ((gs.energy - a5.energy).abs() / e0).abs();
            let d_com = (gs.com_x - a5.com_x).hypot(gs.com_y - a5.com_y);

            let pass = d_count <= floor_count
                && d_frac <= floor_frac
                && d_e <= floor_e
                && d_com <= floor_com;
            seed_pass &= pass;
            seed_detail.push(json!({
                "step": cp,
                "gpu": {"count": gs.count, "largest_frac": gs.largest_frac, "energy": gs.energy, "com": [gs.com_x, gs.com_y]},
                "avx512": {"count": a5.count, "largest_frac": a5.largest_frac, "energy": a5.energy, "com": [a5.com_x, a5.com_y]},
                "noise_floor": {"count": floor_count, "frac": floor_frac, "energy": floor_e, "com": floor_com},
                "deviation": {"count": d_count, "frac": d_frac, "energy": d_e, "com": d_com},
                "pass": pass,
            }));
        }
        all_pass &= seed_pass;
        results.push(json!({"seed": seed, "pass": seed_pass, "checkpoints": seed_detail}));
        println!(
            "P4 seed {seed}: {}",
            if seed_pass { "PASS" } else { "FAIL" }
        );
    }

    let adapter = AdapterInfo {
        name: "llvmpipe (LLVM 20.1.2, 256 bits)".to_string(),
        device_type: "cpu".to_string(),
        backend: "vulkan".to_string(),
        driver: "llvmpipe Mesa 25.2.8 (LLVM 20.1.2)".to_string(),
    };
    let path =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../telemetry/stage3.jsonl");
    append_telemetry_stage(
        &path,
        3,
        "parity",
        &adapter,
        &json!({
            "test": "p4",
            "n": N, "steps": STEPS, "seeds": SEEDS,
            "gpu_mode": "naive/parallel",
            "noise_floor": "|scalar - avx2| per seed per checkpoint",
            "reference": "gravity_simd_avx512 (same spawn constants as the port)",
            "pass": all_pass,
            "per_seed": results,
        }),
    );
    assert!(
        all_pass,
        "P4: GPU stats exceeded legacy noise floor (see telemetry)"
    );
}
