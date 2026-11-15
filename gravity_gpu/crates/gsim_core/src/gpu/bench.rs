//! Benchmark ladder runner.
//!
//! - Rung (1): legacy CPU numbers are cited from stage-1 telemetry (no rerun).
//! - Rung (2): naive GPU port — one thread/particle, global memory,
//!   fixed summation order, no float atomics, legacy step structure.
//! - Rung (3): optimized GPU — tiled force, vec4 tiles, fused
//!   collision detection, cross-step acceleration cache, autotuned
//!   workgroup size.
//! - Rung (4): N-scaling sweep with `sqrt(N)` spawn-radius scaling, finding
//!   the largest N holding ≥60 FPS at 1× and 4× time scale.
//!
//! Every timing is labeled with the adapter it was measured on. On Mesa
//! lavapipe (software Vulkan) the numbers are smoke/correctness signals
//! only — never GPU performance.

use crate::cli::{Backend, GpuMode, MergeMode};
use crate::constants::*;
use crate::cpu::CpuSim;
use crate::gpu::context::{GpuContext, SimConfig};
use crate::gpu::sim::Sim;
use crate::spawn::create_particles;
use crate::telemetry::{append_telemetry, AdapterInfo};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::time::Instant;

/// GPU-vs-CPU max position error tolerance (world units; the spawn disc is
/// 3000 wide) for the parity physics-sanity check. Calibrated 2026-10-18:
/// simulating the worst plausible hardware FP difference — FMA contraction
/// of every mul+add in integrate/finish/merge — gives max 1.5e-5 after 20
/// steps at N=256/seed 12345 with no merges. 0.01 is ~700x headroom for
/// `sqrt` rounding and driver variance. This is the gross-sanity net (wrong
/// sign, wrong dt, scrambled buffers show O(1)+); data races are caught by
/// the determinism check and algorithmic force bugs by the cross-mode
/// bitwise check instead.
const PARITY_POS_TOLERANCE: f32 = 0.01;

/// Result of one ladder rung.
#[derive(Debug, Clone)]
pub struct RungResult {
    /// "naive" or "tiled".
    pub mode: String,
    /// Spawn count.
    pub n: usize,
    /// Timed steps.
    pub steps: u64,
    /// Steps per second (wall clock).
    pub steps_per_sec: f64,
    /// Per-pass GPU nanoseconds (one timed step; empty if unsupported).
    pub per_pass_ns: Vec<(String, f64)>,
    /// Final live body count.
    pub final_count: u32,
    /// VRAM budgeted by the sim's buffers.
    pub vram_bytes: u64,
    /// Whether GPU timestamp queries were available.
    pub gpu_timestamps: bool,
}

/// One N-sweep point.
#[derive(Debug, Clone)]
pub struct SweepPoint {
    /// Spawn count.
    pub n: usize,
    /// Spawn radius used (`1500·sqrt(N/10_000)`).
    pub spawn_radius: f32,
    /// Steps per second measured.
    pub steps_per_sec: f64,
    /// Estimated FPS at time_scale 1 (2 steps/frame at 60 FPS target).
    pub fps_at_1x: f64,
    /// Estimated FPS at time_scale 4 (8 steps/frame).
    pub fps_at_4x: f64,
    /// VRAM budgeted.
    pub vram_bytes: u64,
}

/// Drives the ladder; owns telemetry logging.
pub struct BenchmarkRunner {
    backend: Backend,
    adapter_index: usize,
    seed: u64,
    telemetry_path: PathBuf,
    /// Adapter identity, queried once.
    pub adapter_info: AdapterInfo,
}

impl BenchmarkRunner {
    /// Query the adapter (no device yet) and prepare telemetry.
    pub fn new(
        backend: Backend,
        adapter_index: usize,
        seed: u64,
        telemetry_path: PathBuf,
    ) -> Result<Self, String> {
        let adapters = GpuContext::list_adapters(backend);
        let adapter_info = adapters
            .into_iter()
            .nth(adapter_index)
            .ok_or_else(|| format!("adapter index {adapter_index} out of range"))?;
        Ok(Self {
            backend,
            adapter_index,
            seed,
            telemetry_path,
            adapter_info,
        })
    }

    /// Append a stage-2 telemetry record.
    pub fn log(&self, kind: &str, detail: Value) {
        append_telemetry(&self.telemetry_path, kind, &self.adapter_info, &detail);
    }

    fn sim_config(
        &self,
        mode: GpuMode,
        merge_mode: MergeMode,
        n: usize,
        spawn_radius: f32,
    ) -> SimConfig {
        SimConfig {
            n_max: n,
            seed: self.seed,
            mode,
            merge_mode,
            no_accel_cache: false,
            spawn_radius,
            workgroup_size: 0,
        }
    }

    /// Autotune the tiled workgroup size at `n`; logs the table.
    pub fn autotune(&self, n: usize) -> (u32, Vec<(u32, f64)>) {
        let mut sim = Sim::new(
            self.backend,
            self.adapter_index,
            self.sim_config(GpuMode::Tiled, MergeMode::Parallel, n, SPAWN_RADIUS),
            None,
        )
        .expect("sim for autotune");
        let (best, table) = sim.autotune();
        self.log(
            "benchmark",
            json!({
                "phase": "autotune",
                "context": "workgroup-size sweep over 64/128/256 on the running adapter; metric = mean fused-kernel ns over 4 timed steps (fresh spawn per candidate)",
                "n": n,
                "table": table.iter().map(|(wg, ns)| json!({"workgroup": wg, "fused_ns": ns})).collect::<Vec<_>>(),
                "best": best,
            }),
        );
        (best, table)
    }

    /// Run one ladder rung: warmup, then `steps` timed wall-clock steps,
    /// then a few timestamped steps for the per-pass breakdown.
    pub fn ladder_rung(
        &self,
        mode: GpuMode,
        merge_mode: MergeMode,
        n: usize,
        steps: u64,
        spawn_radius: f32,
        workgroup_size: u32,
    ) -> RungResult {
        let mut cfg = self.sim_config(mode, merge_mode, n, spawn_radius);
        cfg.workgroup_size = workgroup_size;
        let mut sim = Sim::new(self.backend, self.adapter_index, cfg, None).expect("sim for rung");
        sim.spawn();
        for _ in 0..2 {
            sim.step();
        }
        let wall = Instant::now();
        for _ in 0..steps {
            sim.step();
        }
        // Block until the queue drains so wall time is honest.
        sim.ctx
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("drain queue");
        let secs = wall.elapsed().as_secs_f64();
        let mut per_pass_ns = Vec::new();
        for _ in 0..3 {
            for (name, ns) in sim.step_timed() {
                if per_pass_ns.iter().all(|(n, _)| n != &name) {
                    per_pass_ns.push((name, ns));
                }
            }
        }
        sim.update_stats();
        let stats = sim.read_stats();
        RungResult {
            mode: format!("{mode:?}").to_lowercase(),
            n,
            steps,
            steps_per_sec: steps as f64 / secs.max(1e-9),
            per_pass_ns,
            final_count: stats.count,
            vram_bytes: sim.vram_bytes(),
            gpu_timestamps: sim.timer.gpu_timestamps,
        }
    }

    /// Parity check, redesigned 2026-10-18 for real-hardware honesty.
    ///
    /// Bitwise GPU-vs-CPU parity over full steps is unachievable across FP
    /// hardware: WGSL permits FMA contraction (`pos + v*DT` fusing to one
    /// rounding) and implementation-defined `sqrt` precision, so x86 and
    /// NVIDIA differ by 1 ulp somewhere within steps and chaos amplifies
    /// it (RTX 3060, 2026-10-18: both GPU modes agreed bitwise with each
    /// other yet differed from the CPU hash). The old check's premise —
    /// "same summation order, same op association ⇒ same bits" — is false
    /// across different FP implementations. Instead this asserts three
    /// independently meaningful properties:
    ///  1. cross-mode: naive/serial vs tiled/serial GPU hashes match
    ///     bit-for-bit (same hardware, two independent force algorithms —
    ///     a strong correctness signal for the tiled kernel);
    ///  2. determinism: each mode run twice hashes identically (catches
    ///     data races such as the 2026-10-18 pass1 `accel_valid` race);
    ///  3. physics sanity: GPU (tiled) vs scalar CPU max position error
    ///     after `steps` stays under [`PARITY_POS_TOLERANCE`] (P2-style;
    ///     catches gross divergence, not last-ulp noise).
    pub fn parity_check(&self, n: usize, steps: u64) -> Value {
        let mut modes = Vec::new();
        let mut hashes = Vec::new();
        let mut deterministic_ok = true;
        for mode in [GpuMode::Naive, GpuMode::Tiled] {
            let h1 = self.parity_run(mode, n, steps);
            let h2 = self.parity_run(mode, n, steps);
            let det = h1 == h2;
            deterministic_ok &= det;
            hashes.push(h1.clone());
            modes.push(json!({
                "mode": format!("{mode:?}").to_lowercase(),
                "merge": "serial",
                "n": n,
                "steps": steps,
                "hash_run1": h1,
                "hash_run2": h2,
                "deterministic": det,
            }));
        }
        let cross_mode_match = hashes[0] == hashes[1];

        let mut cpu = CpuSim::new(create_particles(self.seed, n));
        for _ in 0..steps {
            cpu.step(PHYSICS_DT);
        }
        let gpu_particles = self.parity_particles(GpuMode::Tiled, n, steps);
        let counts_match = cpu.particles.len() == gpu_particles.len();
        let max_err = cpu.particles.max_position_error(&gpu_particles);
        let tolerance_ok = counts_match && max_err.is_some_and(|e| e <= PARITY_POS_TOLERANCE);

        let ok = deterministic_ok && cross_mode_match && tolerance_ok;
        json!({
            "n": n,
            "steps": steps,
            "modes": modes,
            "cross_mode_bitwise_match": cross_mode_match,
            "cpu": {
                "counts_match": counts_match,
                "cpu_count": cpu.particles.len(),
                "gpu_count": gpu_particles.len(),
                "max_position_error": max_err,
                "tolerance": PARITY_POS_TOLERANCE,
                "tolerance_ok": tolerance_ok,
            },
            "ok": ok,
        })
    }

    /// One GPU serial-merge run; returns the positions hash.
    fn parity_run(&self, mode: GpuMode, n: usize, steps: u64) -> String {
        self.parity_particles(mode, n, steps).positions_hash()
    }

    /// One GPU serial-merge run; returns the read-back particle cloud.
    fn parity_particles(&self, mode: GpuMode, n: usize, steps: u64) -> crate::spawn::Particles {
        let mut cfg = self.sim_config(mode, MergeMode::Serial, n, SPAWN_RADIUS);
        cfg.workgroup_size = 128;
        let mut sim =
            Sim::new(self.backend, self.adapter_index, cfg, None).expect("sim for parity");
        sim.spawn();
        for _ in 0..steps {
            sim.step();
        }
        sim.readback_particles()
    }

    /// N-scaling sweep: for each N, spawn with `sqrt(N)`-scaled radius, run a
    /// few steps, and estimate FPS at 1× (2 steps/frame) and 4× (8
    /// steps/frame) time scale. Returns per-point results.
    pub fn sweep(&self, ns: &[usize], steps_per_point: u64) -> Vec<SweepPoint> {
        let mut out = Vec::new();
        for &n in ns {
            let radius = sweep_spawn_radius(n);
            let rung = self.ladder_rung(
                GpuMode::Tiled,
                MergeMode::Parallel,
                n,
                steps_per_point,
                radius,
                128,
            );
            let sps = rung.steps_per_sec;
            out.push(SweepPoint {
                n,
                spawn_radius: radius,
                steps_per_sec: sps,
                fps_at_1x: sps / 2.0,
                fps_at_4x: sps / 8.0,
                vram_bytes: rung.vram_bytes,
            });
            self.log(
                "benchmark",
                json!({
                    "phase": "sweep_point",
                    "n": n,
                    "spawn_radius": radius,
                    "steps_per_sec": sps,
                    "fps_at_1x": sps / 2.0,
                    "fps_at_4x": sps / 8.0,
                    "vram_bytes": rung.vram_bytes,
                }),
            );
        }
        out
    }
}
