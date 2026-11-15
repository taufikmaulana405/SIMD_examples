//! Headless WGSL compute tests on Mesa lavapipe (software Vulkan).
//!
//! Every test runs inside a wgpu validation-error scope, so a WGSL or API
//! misuse fails the test instead of silently producing garbage. Timings
//! here are lavapipe-software smoke signals, never GPU performance.
//!
//! Run with the lavapipe ICD visible, e.g.
//! `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json cargo test`.

use gsim_core::cli::{Backend, GpuMode, MergeMode};
use gsim_core::constants::*;
use gsim_core::cpu::CpuSim;
use gsim_core::gpu::{BenchmarkRunner, RenderParams, Sim, SimConfig};
use gsim_core::spawn::{create_particles, Particles};
use gsim_core::telemetry::AdapterInfo;

fn cfg(n: usize, mode: GpuMode, merge: MergeMode, seed: u64) -> SimConfig {
    SimConfig {
        n_max: n,
        seed,
        mode,
        merge_mode: merge,
        no_accel_cache: false,
        spawn_radius: SPAWN_RADIUS,
        workgroup_size: 128,
    }
}

fn sim(n: usize, mode: GpuMode, merge: MergeMode, seed: u64) -> Sim {
    Sim::new(Backend::Vulkan, 0, cfg(n, mode, merge, seed), None).expect("lavapipe sim")
}

/// Spawn, then step `steps` times inside a validation-error scope; panic
/// on any error.
fn run_guarded(s: &mut Sim, steps: u64) {
    s.spawn();
    let guard = s.ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    for _ in 0..steps {
        s.step();
    }
    s.ctx
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("drain");
    if let Some(err) = pollster::block_on(guard.pop()) {
        panic!("wgpu validation error: {err:?}");
    }
}

fn cpu_after(n: usize, seed: u64, steps: u64) -> CpuSim {
    let mut c = CpuSim::new(create_particles(seed, n));
    for _ in 0..steps {
        c.step(PHYSICS_DT);
    }
    c
}

#[test]
fn adapter_is_lavapipe_software() {
    let mut s = sim(16, GpuMode::Tiled, MergeMode::Serial, 12345);
    let info = &s.ctx.adapter_info;
    assert_eq!(info.backend, "vulkan", "test must run on Vulkan");
    assert_eq!(info.device_type, "cpu", "expected software adapter");
    assert!(
        info.name.contains("llvmpipe"),
        "expected llvmpipe, got {}",
        info.name
    );
    assert!(s.vram_bytes() > 0);
    assert_eq!(s.workgroup_size(), 128);
    s.set_workgroup_size(64);
    // Accessors used by windowed mode must not panic headless.
    let _ = s.ctx.instance();
    let _ = s.ctx.adapter();
}

#[test]
fn serial_mode_bitwise_parity_small() {
    // Bitwise GPU==CPU parity via the NAIVE kernel (always exact on
    // lavapipe). Stage-2 proved N=256/20 steps; this extends the matrix
    // down to N=1, two seeds. (Tiled mode has a lavapipe-only
    // nondeterminism, see stage3 telemetry blocker; it is smoke-tested
    // for validation errors separately.)
    for &n in &[1usize, 2, 3, 16, 64] {
        for &seed in &[12345u64, 7] {
            let mut s = sim(n, GpuMode::Naive, MergeMode::Serial, seed);
            run_guarded(&mut s, 5);
            let gpu = s.readback_particles();
            let cpu = cpu_after(n, seed, 5);
            assert_eq!(gpu.len(), cpu.particles.len(), "n={n} seed={seed}");
            assert_eq!(
                gpu.positions_hash(),
                cpu.particles.positions_hash(),
                "n={n} seed={seed}"
            );
        }
    }
    // Tiled mode: no validation errors, sane output (bitwise not asserted
    // on lavapipe — see blocker).
    for &n in &[16usize, 64] {
        let mut s = sim(n, GpuMode::Tiled, MergeMode::Serial, 12345);
        run_guarded(&mut s, 5);
        let gpu = s.readback_particles();
        let cpu = cpu_after(n, 12345, 5);
        assert_eq!(gpu.len(), cpu.particles.len(), "tiled n={n}");
        assert!(gpu.total_mass() > 0.0);
    }
}

#[test]
fn dispatch_edge_cases() {
    // Non-multiples of the 64/128/256 workgroup sizes must dispatch and
    // stay bitwise-equal (out-of-range threads early-out). Naive mode for
    // the bitwise assertion (lavapipe tiled quirk, see blocker).
    for &n in &[255usize, 256, 257, 1000] {
        let mut s = sim(n, GpuMode::Naive, MergeMode::Serial, 12345);
        run_guarded(&mut s, 3);
        let gpu = s.readback_particles();
        let cpu = cpu_after(n, 12345, 3);
        assert_eq!(
            gpu.positions_hash(),
            cpu.particles.positions_hash(),
            "edge N={n}"
        );
    }
    // Tiled dispatch smoke on the same edge sizes (validation only).
    for &n in &[255usize, 257, 1000] {
        let mut s = sim(n, GpuMode::Tiled, MergeMode::Serial, 12345);
        run_guarded(&mut s, 2);
        let gpu = s.readback_particles();
        assert_eq!(gpu.len(), cpu_after(n, 12345, 2).particles.len());
    }
}

#[test]
fn serial_parity_20_steps_matches_cpu() {
    // Second parity anchor next to stage-2's (N=256, 20 steps); naive mode
    // for lavapipe determinism.
    let mut s = sim(200, GpuMode::Naive, MergeMode::Serial, 999);
    run_guarded(&mut s, 20);
    let gpu = s.readback_particles();
    let cpu = cpu_after(200, 999, 20);
    assert_eq!(gpu.len(), cpu.particles.len());
    assert_eq!(gpu.positions_hash(), cpu.particles.positions_hash());
}

#[test]
fn no_accel_cache_still_bitwise() {
    // Quirk-c A/B: disabling the cache must not change results (the cache
    // is exact by construction). Naive mode for lavapipe determinism.
    let mut scfg = cfg(64, GpuMode::Naive, MergeMode::Serial, 12345);
    scfg.no_accel_cache = true;
    let mut s = Sim::new(Backend::Vulkan, 0, scfg, None).expect("lavapipe sim");
    run_guarded(&mut s, 5);
    let gpu = s.readback_particles();
    let cpu = cpu_after(64, 12345, 5);
    assert_eq!(gpu.positions_hash(), cpu.particles.positions_hash());
}

#[test]
fn stats_pipeline_matches_cpu() {
    // Naive/Parallel for lavapipe determinism (stats need exactness here).
    let mut s = sim(256, GpuMode::Naive, MergeMode::Parallel, 12345);
    run_guarded(&mut s, 8);
    s.update_stats();
    let st = s.read_stats();
    let p = s.readback_particles();
    assert_eq!(st.count as usize, p.len());
    let rel = |a: f32, b: f32| (a - b).abs() / b.abs().max(1e-9);
    assert!(rel(st.total_mass, p.total_mass()) < 1e-3, "total_mass");
    assert!(rel(st.max_mass, p.largest_mass()) < 1e-3, "max_mass");
    let (cx, cy) = p.center_of_mass();
    assert!((st.com_x - cx).abs() < 1e-2, "com_x");
    assert!((st.com_y - cy).abs() < 1e-2, "com_y");
    assert!(st.trail_points > 0, "trail ring should hold samples");
}

/// CPU replica of the GPU parallel merge (union-find components, ascending
/// member-order reduction, exactly like merge.wgsl's reduce kernel).
/// Used to verify the WGSL merge kernels produce the same partitions and
/// values. Returns the merged particle set.
fn union_find_merge_cpu(p: &Particles) -> Particles {
    let n = p.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = p.x[j] - p.x[i];
            let dy = p.y[j] - p.y[i];
            let rc = 0.45 * p.mass[i].sqrt() + 0.45 * p.mass[j].sqrt();
            if dx * dx + dy * dy <= rc * rc {
                let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                if ri != rj {
                    let (lo, hi) = (ri.min(rj), ri.max(rj));
                    parent[hi] = lo;
                }
            }
        }
    }
    // Roots in ascending order; members accumulated in ascending j order
    // (mirrors the WGSL reduce kernel's deterministic summation).
    let mut roots: Vec<usize> = (0..n).filter(|&i| find(&mut parent, i) == i).collect();
    roots.sort_unstable();
    let mut out = Particles::with_capacity(roots.len());
    for &r in &roots {
        let (mut m, mut x, mut y, mut vx, mut vy, mut b) =
            (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for j in 0..n {
            if find(&mut parent, j) == r {
                m += p.mass[j];
                x += p.mass[j] * p.x[j];
                y += p.mass[j] * p.y[j];
                vx += p.mass[j] * p.vx[j];
                vy += p.mass[j] * p.vy[j];
                b += p.mass[j] * p.brightness[j];
            }
        }
        out.push(gsim_core::spawn::ParticleData {
            x: x / m,
            y: y / m,
            vx: vx / m,
            vy: vy / m,
            mass: m,
            brightness: b / m,
        });
    }
    out
}

fn momentum_of(p: &Particles) -> (f64, f64) {
    let mut px = 0.0f64;
    let mut py = 0.0f64;
    for i in 0..p.len() {
        px += f64::from(p.vx[i]) * f64::from(p.mass[i]);
        py += f64::from(p.vy[i]) * f64::from(p.mass[i]);
    }
    (px, py)
}

#[test]
fn parallel_merge_matches_union_find_replica() {
    // Crafted state: two isolated overlapping pairs (non-chain), one
    // 3-body chain, and distant spectators. The GPU parallel merge must
    // produce the same partitions as the CPU union-find replica, with
    // mass/momentum/COM conserved to 1e-5 across the merge.
    //
    // Note: the GPU's survivor SLOT order is nondeterministic (atomic
    // slot allocation), so survivors are matched by mass.
    use gsim_core::cpu::{calculate_accelerations, finish_velocity, integrate};
    use gsim_core::spawn::ParticleData;

    let mut s = Particles::with_capacity(16);
    let mut add = |x: f32, y: f32, vx: f32, vy: f32, m: f32| {
        s.push(ParticleData {
            x,
            y,
            vx,
            vy,
            mass: m,
            brightness: 0.7,
        });
    };
    // Pair 1 (masses 6+10=16) at origin.
    add(0.0, 0.0, 1.0, 0.5, 6.0);
    add(1.0, 0.0, -1.0, -0.5, 10.0);
    // Pair 2 (masses 8+12=20) far away.
    add(500.0, 0.0, 0.0, 2.0, 8.0);
    add(501.5, 0.0, 0.0, -2.0, 12.0);
    // Chain of 3 (masses 5+7+9=21): A-B overlap, B-C overlap, A-C not.
    add(-500.0, 300.0, 1.0, 1.0, 5.0);
    add(-498.5, 300.0, -1.0, 0.0, 7.0);
    add(-497.0, 300.0, 0.0, -1.0, 9.0);
    // Spectators (no overlaps).
    add(200.0, 400.0, 3.0, -1.0, 11.0);
    add(-200.0, -400.0, -3.0, 1.0, 13.0);

    // CPU reference: integrate -> union-find merge -> force -> finish.
    let n0 = s.len();
    let mut cpu = s.clone();
    let (mut ax, mut ay) = (vec![0.0f32; n0], vec![0.0f32; n0]);
    calculate_accelerations(&cpu, &mut ax, &mut ay);
    integrate(&mut cpu, &ax, &ay, PHYSICS_DT);
    let (pmx, pmy) = momentum_of(&cpu);
    let merged = union_find_merge_cpu(&cpu);
    // Merge must conserve momentum/COM to 1e-5 (the task's P3 bar).
    let (qmx, qmy) = momentum_of(&merged);
    let mscale = pmx.abs().max(pmy.abs()).max(1.0);
    assert!(((qmx - pmx) / mscale).abs() < 1e-5, "replica momentum x");
    assert!(((qmy - pmy) / mscale).abs() < 1e-5, "replica momentum y");
    assert_eq!(merged.len(), 5, "2 pairs + 1 triple + 2 spectators");
    let n1 = merged.len();
    let (mut bx, mut by) = (vec![0.0f32; n1], vec![0.0f32; n1]);
    calculate_accelerations(&merged, &mut bx, &mut by);
    let mut expected = merged;
    finish_velocity(&mut expected, &bx, &by, PHYSICS_DT);

    // GPU: upload the same crafted state, one parallel step (naive force
    // for lavapipe determinism; the merge kernels are shared).
    let mut g = sim(16, GpuMode::Naive, MergeMode::Parallel, 12345);
    g.upload(&s);
    run_guarded_step(&mut g);
    let got = g.readback_particles();

    // Match survivors by mass (slot order nondeterministic).
    assert_eq!(got.len(), expected.len());
    let mut gm: Vec<f32> = (0..got.len()).map(|i| got.mass[i]).collect();
    let mut em: Vec<f32> = (0..expected.len()).map(|i| expected.mass[i]).collect();
    gm.sort_by(|a, b| a.partial_cmp(b).unwrap());
    em.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for (a, b) in gm.iter().zip(em.iter()) {
        assert!(
            (a - b).abs() / b.max(1.0) < 1e-5,
            "survivor mass {a} vs {b}"
        );
    }
    // Total mass conserved to 1e-5.
    let m0 = s.total_mass();
    assert!(((got.total_mass() - m0) / m0).abs() < 1e-5, "total mass");
    // Momentum conserved to 1e-5 relative across the GPU step's merge
    // (comparing against the replica's post-step momentum).
    let (gmx, gmy) = momentum_of(&got);
    let (emx, emy) = momentum_of(&expected);
    let escale = emx.abs().max(emy.abs()).max(1.0);
    assert!(((gmx - emx) / escale).abs() < 1e-5, "gpu momentum x");
    assert!(((gmy - emy) / escale).abs() < 1e-5, "gpu momentum y");
}

/// run_guarded without the spawn (state already uploaded).
fn run_guarded_step(s: &mut Sim) {
    let guard = s.ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    s.step();
    s.ctx
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("drain");
    if let Some(err) = pollster::block_on(guard.pop()) {
        panic!("wgpu validation error: {err:?}");
    }
}

#[test]
fn parallel_merge_conserves_mass() {
    // Total mass is exactly conserved by the parallel merge (masses only
    // move between bodies); unlike momentum/COM it is unaffected by
    // integration drift, so it can be checked across whole steps.
    // Momentum/COM conservation across the merge itself is covered by
    // parallel_merge_matches_union_find_replica (isolation via upload).
    for &seed in &[12345u64, 999] {
        let before = create_particles(seed, 512);
        let m0 = before.total_mass();
        let mut s = sim(512, GpuMode::Naive, MergeMode::Parallel, seed);
        run_guarded(&mut s, 15);
        let p = s.readback_particles();
        assert!(p.len() <= 512 && !p.is_empty());
        assert!(((p.total_mass() - m0) / m0).abs() < 1e-5, "mass");
    }
}

#[test]
fn step_timed_reports_sections() {
    let mut s = sim(64, GpuMode::Tiled, MergeMode::Parallel, 12345);
    s.spawn();
    let sections = s.step_timed();
    assert!(!sections.is_empty());
    for (name, ns) in &sections {
        assert!(!name.is_empty());
        assert!(ns.is_finite() && *ns >= 0.0, "{name}: {ns}");
    }
    let names: Vec<&str> = sections.iter().map(|(n, _)| n.as_str()).collect();
    for want in ["pass1", "integrate", "fused", "finish"] {
        assert!(names.contains(&want), "missing {want} in {names:?}");
    }
}

#[test]
fn render_pipelines_smoke() {
    // Build the render pipelines headless (format only, no surface) and
    // draw one frame into an offscreen texture: covers gpu/render.rs.
    let mut s = Sim::new(
        Backend::Vulkan,
        0,
        cfg(64, GpuMode::Tiled, MergeMode::Parallel, 12345),
        Some(wgpu::TextureFormat::Rgba8Unorm),
    )
    .expect("lavapipe sim with render");
    s.spawn();
    run_guarded(&mut s, 2);
    s.update_stats();
    let st = s.read_stats();
    let tex = s.ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test target"),
        size: wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    let guard = s.ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    for trails_visible in [true, false] {
        s.render(RenderParams {
            view: &view,
            width: 64,
            height: 64,
            zoom: 0.2,
            total_mass: st.total_mass,
            trails_visible,
            trail_instances: st.trail_points.min(1000),
        });
    }
    s.ctx
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("drain");
    if let Some(err) = pollster::block_on(guard.pop()) {
        panic!("render validation error: {err:?}");
    }
}

#[test]
fn shader_source_unknown_panics() {
    let r = std::panic::catch_unwind(|| gsim_core::gpu::shaders::source("nope"));
    assert!(r.is_err());
    let r = std::panic::catch_unwind(|| gsim_core::gpu::shaders::render_source("nope"));
    assert!(r.is_err());
}

#[test]
fn bench_runner_smoke() {
    // Exercises gpu/bench.rs end to end at tiny sizes.
    let tel = std::env::temp_dir().join("gsim_stage3_bench_smoke.jsonl");
    let runner = BenchmarkRunner::new(Backend::Vulkan, 0, 12345, tel).expect("runner");
    assert!(!runner.adapter_info.name.is_empty());
    let (best, table) = runner.autotune(64);
    assert!([64, 128, 256].contains(&best));
    assert_eq!(table.len(), 3);
    let rung = runner.ladder_rung(
        GpuMode::Tiled,
        MergeMode::Parallel,
        64,
        2,
        SPAWN_RADIUS,
        best,
    );
    assert!(rung.steps_per_sec > 0.0);
    assert!(rung.final_count > 0 && rung.final_count <= 64);
    assert_eq!(rung.mode, "tiled");
    let par = runner.parity_check(GpuMode::Naive, 32, 3);
    assert_eq!(par["match"], true);
    let pts = runner.sweep(&[32, 64], 2);
    assert_eq!(pts.len(), 2);
    assert!(pts[0].fps_at_1x > 0.0);
}

#[test]
fn list_adapters_finds_lavapipe() {
    let adapters = gsim_core::gpu::GpuContext::list_adapters(Backend::Vulkan);
    assert!(!adapters.is_empty());
    let _ = AdapterInfo::cpu_placeholder();
}
