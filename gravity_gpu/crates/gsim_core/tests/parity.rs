//! Stage-3 parity tests P1–P3 (task 4).
//!
//! Bit-exact GPU==legacy parity is impossible (f32, summation order,
//! chaos), so these verify the next-best guarantees:
//! - P1: GPU accelerations vs an f64 oracle, bounded by 2× the legacy
//!   f32 kernel's own error vs the same oracle.
//! - P2: 120 serial-merge steps vs the legacy AVX-512 headless binary
//!   (same spawn constants as the GPU port): max position error within a
//!   stated tolerance.
//! - P3: parallel-merge partitions vs legacy-order serial merge: identical
//!   for non-chain collisions; partition-difference rate reported.
//!
//! All GPU work uses the NAIVE kernel on Mesa lavapipe (software Vulkan):
//! it is always bitwise-exact vs the CPU reference, while the tiled kernel
//! has a lavapipe-only multi-workgroup nondeterminism (see stage3
//! telemetry blocker). Timings are smoke-only, never GPU performance.
//!
//! Run with the lavapipe ICD visible, e.g.
//! `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json cargo test`.

use gsim_core::cli::{Backend, GpuMode, MergeMode};
use gsim_core::constants::*;
use gsim_core::cpu::{calculate_accelerations, calculate_accelerations_f64, CpuSim};
use gsim_core::gpu::{Sim, SimConfig};
use gsim_core::spawn::{create_particles, Particles};
use gsim_core::telemetry::{append_telemetry_stage, AdapterInfo};
use serde_json::json;
use std::process::Command;

fn sim(n: usize, mode: GpuMode, merge: MergeMode, seed: u64) -> Sim {
    Sim::new(
        Backend::Vulkan,
        0,
        SimConfig {
            n_max: n,
            seed,
            mode,
            merge_mode: merge,
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
    .expect("lavapipe sim")
}

fn log(kind: &str, detail: serde_json::Value) {
    let adapter = AdapterInfo {
        name: "llvmpipe (LLVM 20.1.2, 256 bits)".to_string(),
        device_type: "cpu".to_string(),
        backend: "vulkan".to_string(),
        driver: "llvmpipe Mesa 25.2.8 (LLVM 20.1.2)".to_string(),
    };
    // Workspace-absolute: cargo test runs with CWD = the package dir.
    let path =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../telemetry/stage3.jsonl");
    append_telemetry_stage(&path, 3, kind, &adapter, &detail);
}

/// Legacy scalar force kernel replicated EXACTLY (paired Newton-3rd-law
/// update, `(displacement * common) * mass` association), from
/// legacy_work/gravity_scalar/src/main.rs `calculate_accelerations`.
fn legacy_scalar_accel(p: &Particles) -> (Vec<f32>, Vec<f32>) {
    let n = p.len();
    let mut ax = vec![0.0f32; n];
    let mut ay = vec![0.0f32; n];
    let s2 = GRAVITY_SOFTENING * GRAVITY_SOFTENING;
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = p.x[j] - p.x[i];
            let dy = p.y[j] - p.y[i];
            let r2 = dx * dx + dy * dy + s2;
            let inv = 1.0 / r2.sqrt();
            let c = GRAVITATIONAL_CONSTANT * inv * inv * inv;
            let tx = dx * c;
            let ty = dy * c;
            ax[i] += tx * p.mass[j];
            ay[i] += ty * p.mass[j];
            ax[j] -= tx * p.mass[i];
            ay[j] -= ty * p.mass[i];
        }
    }
    (ax, ay)
}

fn max_abs_diff(a: &[f32], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (f64::from(*x) - y).abs())
        .fold(0.0, f64::max)
}

/// P1: GPU accelerations vs the f64 oracle, bounded by 2× the legacy f32
/// kernel's error vs the same oracle.
#[test]
fn p1_accel_vs_f64_oracle() {
    let n = 256;
    let mut g = sim(n, GpuMode::Naive, MergeMode::Serial, 12345);
    g.spawn();
    g.step();
    let p = g.readback_particles();
    let (gax, gay) = g.readback_accel();
    assert_eq!(gax.len(), p.len());

    let (fax, fay) = calculate_accelerations_f64(&p);
    let (lax, lay) = legacy_scalar_accel(&p);
    let (mut cax, mut cay) = (vec![0.0f32; p.len()], vec![0.0f32; p.len()]);
    calculate_accelerations(&p, &mut cax, &mut cay);

    let err_gpu = max_abs_diff(&gax, &fax).max(max_abs_diff(&gay, &fay));
    let err_legacy = max_abs_diff(&lax, &fax).max(max_abs_diff(&lay, &fay));
    let err_cpu = max_abs_diff(&cax, &fax).max(max_abs_diff(&cay, &fay));

    // The naive GPU kernel must match the scalar CPU reference bitwise
    // (same summation order and op association, stage-2 design).
    let gpu_vs_cpu = gax
        .iter()
        .zip(cax.iter())
        .chain(gay.iter().zip(cay.iter()))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(gpu_vs_cpu, 0.0, "gpu must be bitwise vs cpu ref");
    // Sanity: the cpu ref and the legacy kernel should have comparable
    // f64 error (they differ only in summation structure).
    assert!(
        err_cpu <= 2.0 * err_legacy,
        "cpu ref unexpectedly worse than legacy"
    );

    let ratio = err_gpu / err_legacy.max(1e-30);
    log(
        "parity",
        json!({
            "test": "p1",
            "n": n,
            "err_gpu_vs_f64": err_gpu,
            "err_legacy_f32_vs_f64": err_legacy,
            "ratio_gpu_over_legacy": ratio,
            "bound": 2.0,
            "pass": ratio <= 2.0,
            "note": "legacy scalar uses paired Newton-3rd-law updates with (d*c)*m association; the port (like its cpu ref) accumulates per-row as dx*(G*inv^3*m). Both are valid f32; the 2x bound allows the port to be no worse than twice the legacy's own f32 error.",
        }),
    );
    assert!(
        ratio <= 2.0,
        "P1: gpu/f64 error {err_gpu:.3e} exceeds 2x legacy/f64 error {err_legacy:.3e}"
    );
}

/// Run a legacy headless binary and return its dump JSON + positions.
fn legacy_run(
    crate_name: &str,
    seed: u64,
    count: usize,
    steps: u64,
) -> (serde_json::Value, Vec<[f64; 5]>) {
    let bin = format!("../../legacy_work/{crate_name}/target/release/{crate_name}");
    let dump_path = std::env::temp_dir().join(format!("gsim_p2_{crate_name}_{seed}.json"));
    let pos_path = std::env::temp_dir().join(format!("gsim_p2_{crate_name}_{seed}.pos.json"));
    let out = Command::new(&bin)
        .args([
            "--seed",
            &seed.to_string(),
            "--count",
            &count.to_string(),
            "--steps",
            &steps.to_string(),
            "--dump",
            dump_path.to_str().unwrap(),
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
    let dump: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&dump_path).unwrap()).unwrap();
    let pos: Vec<[f64; 5]> =
        serde_json::from_str(&std::fs::read_to_string(&pos_path).unwrap()).unwrap();
    (dump, pos)
}

/// Max position error between two clouds, matched by sorted (x, y).
/// Returns None when counts differ.
fn max_position_error(a: &Particles, b: &[[f64; 5]]) -> Option<f64> {
    if a.len() != b.len() {
        return None;
    }
    let mut ga: Vec<(f32, f32)> = (0..a.len()).map(|i| (a.x[i], a.y[i])).collect();
    let mut gb: Vec<(f64, f64)> = b.iter().map(|p| (p[0], p[1])).collect();
    ga.sort_by(|x, y| x.partial_cmp(y).unwrap());
    gb.sort_by(|x, y| x.partial_cmp(y).unwrap());
    Some(
        ga.iter()
            .zip(gb.iter())
            .map(|(p, q)| {
                (f64::from(p.0) - q.0)
                    .abs()
                    .max((f64::from(p.1) - q.1).abs())
            })
            .fold(0.0, f64::max),
    )
}

/// P2: first 120 steps in serial merge mode vs the legacy AVX-512 binary
/// (same spawn constants as the GPU port). Max position error within the
/// stated tolerance.
#[test]
fn p2_120_steps_vs_legacy() {
    let (n, steps, seed) = (1000usize, 120u64, 12345u64);
    let (_dump, leg_pos) = legacy_run("gravity_simd_avx512", seed, n, steps);

    let mut g = sim(n, GpuMode::Naive, MergeMode::Serial, seed);
    g.spawn();
    for _ in 0..steps {
        g.step();
    }
    let gpu = g.readback_particles();

    // The GPU serial path must be bitwise-identical to the CPU reference.
    let mut cpu = CpuSim::new(create_particles(seed, n));
    for _ in 0..steps {
        cpu.step(PHYSICS_DT);
    }
    assert_eq!(
        gpu.positions_hash(),
        cpu.particles.positions_hash(),
        "gpu serial must be bitwise vs cpu ref"
    );

    let counts = (leg_pos.len(), gpu.len());
    let err = max_position_error(&gpu, &leg_pos);
    // Tolerance: 0.05 world units (absolute). Justification: the legacy
    // AVX-512 kernel uses FMA intrinsics and a different pair-summation
    // structure than the port's scalar-order kernel, so ulp-level force
    // differences are expected and grow through the chaotic dynamics;
    // 0.05 is ~3e-5 of the 1500-unit spawn radius (far below any
    // physically meaningful scale) yet ~1e4x the single-step f32
    // roundoff, so a systematic physics bug (wrong G/dt/merge) would
    // exceed it by orders of magnitude within 120 steps.
    let tol = 0.05;
    let pass = err.map(|e| e <= tol).unwrap_or(false);
    log(
        "parity",
        json!({
            "test": "p2",
            "n": n, "steps": steps, "seed": seed,
            "legacy": "gravity_simd_avx512",
            "legacy_count": counts.0, "gpu_count": counts.1,
            "max_position_error": err,
            "tolerance": tol,
            "pass": pass,
        }),
    );
    log(
        "tolerance",
        json!({
            "test": "p2",
            "tolerance": tol,
            "why": "0.05 world units absolute: legacy AVX-512 uses FMA + different summation structure, so ulp force diffs are expected and amplify chaotically; 0.05 is 3e-5 of the 1500-unit radius (physically negligible) but 1e4x single-step f32 roundoff, so any systematic bug (wrong G/dt/merge) would blow past it.",
        }),
    );
    match err {
        Some(e) => assert!(
            e <= tol,
            "P2: max position error {e:.6} exceeds tolerance {tol}"
        ),
        None => panic!(
            "P2: count divergence legacy={} gpu={} (merge decision flipped)",
            counts.0, counts.1
        ),
    }
}

/// Sorted mass multiset fingerprint of a particle set.
fn mass_multiset(p: &Particles) -> Vec<u32> {
    let mut v: Vec<u32> = (0..p.len()).map(|i| p.mass[i].to_bits()).collect();
    v.sort_unstable();
    v
}

/// P3: partition-difference rate of the parallel merge vs legacy-order
/// serial merge, measured step-by-step in lockstep from the same spawn.
/// A "merge event" is a step where either side merged; it counts as
/// different when the survivor mass multisets differ. Stops comparing a
/// seed at the first difference (states have diverged by design).
#[test]
fn p3_partition_difference_rate() {
    let (n, steps) = (500usize, 60u64);
    let mut total_events = 0u64;
    let mut diff_events = 0u64;
    let mut per_seed = Vec::new();

    for &seed in &[12345u64, 7, 99, 42, 777] {
        let mut g = sim(n, GpuMode::Naive, MergeMode::Parallel, seed);
        g.spawn();
        let mut cpu = CpuSim::new(create_particles(seed, n));
        let mut diverged_at: Option<u64> = None;
        let mut events = 0u64;
        for step in 0..steps {
            let gm0 = mass_multiset(&g.readback_particles());
            let cm0 = mass_multiset(&cpu.particles);
            g.step();
            cpu.step(PHYSICS_DT);
            let gm1 = mass_multiset(&g.readback_particles());
            let cm1 = mass_multiset(&cpu.particles);
            let gpu_merged = gm1.len() < gm0.len();
            let cpu_merged = cm1.len() < cm0.len();
            if gpu_merged || cpu_merged {
                events += 1;
                total_events += 1;
                if gm1 != cm1 {
                    diff_events += 1;
                    diverged_at = Some(step);
                    break;
                }
            }
        }
        per_seed.push(json!({
            "seed": seed, "merge_events": events,
            "diverged_at_step": diverged_at,
        }));
    }
    let rate = diff_events as f64 / total_events.max(1) as f64;
    log(
        "parity",
        json!({
            "test": "p3",
            "n": n, "steps": steps,
            "merge_events": total_events,
            "partition_differences": diff_events,
            "partition_difference_rate": rate,
            "per_seed": per_seed,
            "note": "Parallel (union-find components) vs serial (legacy order) merge; differences are expected by design (quirk b) whenever a chain collision occurs. Non-chain (isolated-pair) collisions must partition identically (covered by parallel_merge_matches_union_find_replica).",
        }),
    );
    // No assert on the rate itself: it is a reported measurement, not a
    // pass/fail gate. The test passes if the measurement completes.
    println!("P3 partition-difference rate: {rate:.3} ({diff_events}/{total_events})");
}
