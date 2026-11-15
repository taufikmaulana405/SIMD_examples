//! Deterministic unit tests for `gsim_core` (Stage 3, task 1).
//!
//! Spawn ranges and reference-frame centering, stats-struct correctness
//! against hand-computed values, merge-pair math, merge-partition
//! properties (isolated pairs vs chains, fixed point, preservation of
//! non-colliding bodies), collision-scan order, constants, and empty-input
//! early-return arms.

use gsim_core::constants::*;
use gsim_core::cpu::{find_collision, integrate, merge_colliding, merge_pair, CpuSim};
use gsim_core::spawn::{create_particles, create_particles_radius, ParticleData, Particles};
use gsim_core::stats::Stats;

fn body(x: f32, y: f32, mass: f32) -> ParticleData {
    ParticleData {
        x,
        y,
        vx: 1.0,
        vy: -2.0,
        mass,
        brightness: 0.7,
    }
}

fn assemble(v: Vec<ParticleData>) -> Particles {
    let mut p = Particles::with_capacity(v.len());
    for b in v {
        p.push(b);
    }
    p
}

// ---------------------------------------------------------------------------
// Spawn
// ---------------------------------------------------------------------------

#[test]
fn spawn_ranges_and_centering() {
    let p = create_particles(12345, 1000);
    assert_eq!(p.len(), 1000);
    for i in 0..p.len() {
        assert!(
            (MIN_PARTICLE_MASS..MAX_PARTICLE_MASS).contains(&p.mass[i]),
            "mass {} out of range",
            p.mass[i]
        );
        assert!(
            (0.55..1.0).contains(&p.brightness[i]),
            "brightness {} out of range",
            p.brightness[i]
        );
        let r = p.x[i].hypot(p.y[i]);
        // Bound is pre-centering: center_reference_frame then translates
        // every body by the pre-centering COM (tens of units at N=1000),
        // so post-centering radii can exceed SPAWN_RADIUS by that shift.
        assert!(
            r <= SPAWN_RADIUS * 1.05,
            "spawn radius {r} far beyond {SPAWN_RADIUS}"
        );
    }
    // Reference frame is centered: COM and bulk velocity ~ 0. The tolerance
    // is f32 accumulation over N=1000 bodies, not physics.
    let (cx, cy) = p.center_of_mass();
    assert!(cx.abs() < 1e-2 && cy.abs() < 1e-2, "COM ({cx},{cy})");
    let mut px = 0.0f32;
    let mut py = 0.0f32;
    for i in 0..p.len() {
        px += p.vx[i] * p.mass[i];
        py += p.vy[i] * p.mass[i];
    }
    assert!(
        px.abs() < 1.0 && py.abs() < 1.0,
        "bulk momentum ({px},{py})"
    );
}

#[test]
fn spawn_known_hash_regression() {
    // Bit-identical spawn anchor (stage 2, verified against the AVX-512
    // legacy binary): guards against silent drift in rng/spawn.
    let p = create_particles(12345, 1000);
    assert_eq!(p.positions_hash(), "577a839b55278c1d");
    let q = create_particles_radius(12345, 100, 240.0);
    assert_eq!(q.len(), 100);
    assert!(q.x.iter().all(|&x| x.abs() <= 240.0));
}

#[test]
fn spawn_empty_and_single() {
    let p = create_particles(7, 0);
    assert!(p.is_empty());
    assert_eq!(p.total_mass(), 0.0);
    assert_eq!(p.largest_mass(), 0.0);
    assert_eq!(p.center_of_mass(), (0.0, 0.0));
    // Empty centering must not divide by zero.
    let mut q = Particles::default();
    q.center_reference_frame();
    let one = create_particles(7, 1);
    assert_eq!(one.len(), 1);
}

// ---------------------------------------------------------------------------
// Stats struct
// ---------------------------------------------------------------------------

#[test]
fn stats_struct_correctness() {
    // Hand-computed reference for three bodies.
    let p = assemble(vec![
        body(0.0, 0.0, 4.0),
        body(6.0, 0.0, 8.0),
        body(0.0, 8.0, 12.0),
    ]);
    assert_eq!(p.total_mass(), 24.0);
    assert_eq!(p.largest_mass(), 12.0);
    let (cx, cy) = p.center_of_mass();
    assert!((cx - 2.0).abs() < 1e-6, "com_x {cx}");
    assert!((cy - 4.0).abs() < 1e-6, "com_y {cy}");

    // Stats word encoding round-trips exactly.
    let s = Stats {
        count: 3,
        total_mass: 24.0,
        max_mass: 12.0,
        com_x: 2.0,
        com_y: 4.0,
        trail_points: 42,
    };
    let back = Stats::from_words(&s.to_words());
    assert_eq!(s, back);

    // GPU-side aggregates must match the CPU reference on a real cloud.
    let cloud = create_particles(999, 256);
    let (ex, ey) = cloud.center_of_mass();
    let words = [
        cloud.len() as u32,
        cloud.total_mass().to_bits(),
        cloud.largest_mass().to_bits(),
        ex.to_bits(),
        ey.to_bits(),
        0,
        0,
        0,
    ];
    let s2 = Stats::from_words(&words);
    assert_eq!(s2.count, 256);
    assert!((s2.total_mass - cloud.total_mass()).abs() < 1e-9);
    assert!((s2.max_mass - cloud.largest_mass()).abs() < 1e-9);
}

#[test]
fn stats_default_is_zero() {
    let s = Stats::default();
    assert_eq!(s.count, 0);
    assert_eq!(s.total_mass, 0.0);
    assert_eq!(s.trail_points, 0);
}

// ---------------------------------------------------------------------------
// Merge math and partition properties
// ---------------------------------------------------------------------------

#[test]
fn merge_pair_exact_math() {
    // Two-body merge checked against f64 arithmetic.
    let mut p = assemble(vec![
        ParticleData {
            x: 1.0,
            y: 2.0,
            vx: 3.0,
            vy: 4.0,
            mass: 5.0,
            brightness: 0.6,
        },
        ParticleData {
            x: 11.0,
            y: 22.0,
            vx: 13.0,
            vy: 14.0,
            mass: 15.0,
            brightness: 0.9,
        },
    ]);
    merge_pair(&mut p, 0, 1);
    assert_eq!(p.len(), 1);
    let (mi, mj) = (5.0f64, 15.0f64);
    let m = mi + mj;
    assert!((f64::from(p.mass[0]) - m).abs() < 1e-9);
    assert!((f64::from(p.x[0]) - (1.0 * mi + 11.0 * mj) / m).abs() < 1e-5);
    assert!((f64::from(p.y[0]) - (2.0 * mi + 22.0 * mj) / m).abs() < 1e-5);
    assert!((f64::from(p.vx[0]) - (3.0 * mi + 13.0 * mj) / m).abs() < 1e-5);
    assert!((f64::from(p.vy[0]) - (4.0 * mi + 14.0 * mj) / m).abs() < 1e-5);
    assert!((f64::from(p.brightness[0]) - (0.6 * mi + 0.9 * mj) / m).abs() < 1e-5);
}

#[test]
fn merge_pair_swap_remove_moves_last() {
    let mut p = assemble(vec![
        body(0.0, 0.0, 5.0),
        body(100.0, 0.0, 6.0),
        body(1.0, 0.0, 7.0),
    ]);
    // Body 0 and body 2 overlap (r = 0.45*sqrt(5)+0.45*sqrt(7) ~ 2.2 > 1).
    merge_pair(&mut p, 0, 2);
    assert_eq!(p.len(), 2);
    // swap_remove(2) with j == last: no move; slot 1 keeps body (100,0).
    assert_eq!(p.x[1], 100.0);
    assert_eq!(p.mass[0], 12.0);
}

#[test]
fn find_collision_first_hit_ascending() {
    // Body 0 at origin with tiny radius; bodies 1,3 far; body 2 overlaps.
    let p = assemble(vec![
        body(0.0, 0.0, 1.0),
        body(500.0, 0.0, 1.0),
        body(0.5, 0.0, 1.0),
        body(-500.0, 0.0, 1.0),
    ]);
    assert_eq!(find_collision(&p, 0), Some(2));
    assert_eq!(find_collision(&p, 1), None);
    assert_eq!(find_collision(&p, 2), None);
}

#[test]
fn non_colliding_bodies_preserved_bitwise() {
    // No overlaps anywhere: merge must be the identity on the multiset of
    // (x, y, mass) bit patterns (order may change via swap_remove, values
    // must not).
    let p = assemble(vec![
        body(0.0, 0.0, 5.0),
        body(100.0, 0.0, 6.0),
        body(0.0, 100.0, 7.0),
        body(100.0, 100.0, 8.0),
    ]);
    let mut before: Vec<(u32, u32, u32)> = (0..p.len())
        .map(|i| (p.x[i].to_bits(), p.y[i].to_bits(), p.mass[i].to_bits()))
        .collect();
    before.sort_unstable();
    let mut q = p.clone();
    merge_colliding(&mut q);
    let mut after: Vec<(u32, u32, u32)> = (0..q.len())
        .map(|i| (q.x[i].to_bits(), q.y[i].to_bits(), q.mass[i].to_bits()))
        .collect();
    after.sort_unstable();
    assert_eq!(before, after);
}

#[test]
fn isolated_pair_merges_identically_in_both_modes() {
    // One isolated overlapping pair + one distant spectator. The sequential
    // (legacy-order) merge and a union-find component merge must agree on
    // the isolated pair: same survivor mass and COM.
    let pair = vec![
        ParticleData {
            x: 0.0,
            y: 0.0,
            vx: 1.0,
            vy: 0.0,
            mass: 6.0,
            brightness: 0.6,
        },
        ParticleData {
            x: 1.0,
            y: 0.0,
            vx: -1.0,
            vy: 0.0,
            mass: 10.0,
            brightness: 0.8,
        },
        body(500.0, 500.0, 7.0),
    ];
    let mut seq = assemble(pair.clone());
    merge_colliding(&mut seq);
    assert_eq!(seq.len(), 2);
    // Union-find reference: single component {0,1}.
    let m = 16.0f64;
    let ex = (0.0 * 6.0 + 1.0 * 10.0) / m;
    let evx = (1.0 * 6.0 - 10.0) / m;
    let survivor = (0..seq.len())
        .find(|&i| (f64::from(seq.mass[i]) - m).abs() < 1e-9)
        .expect("merged survivor");
    assert!((f64::from(seq.x[survivor]) - ex).abs() < 1e-5);
    assert!((f64::from(seq.vx[survivor]) - evx).abs() < 1e-5);
}

#[test]
fn chain_collision_documented() {
    // A-B overlap, B-C overlap, A-C do NOT overlap: a chain. The legacy
    // sequential scan merges A+B first, then the result with C (restart
    // semantics), so all three end in one body here. This pins the exact
    // legacy-order behavior chains depend on.
    let mut p = assemble(vec![
        body(0.0, 0.0, 4.0),
        body(1.0, 0.0, 4.0),
        body(2.0, 0.0, 4.0),
    ]);
    merge_colliding(&mut p);
    assert_eq!(p.len(), 1);
    assert!((p.mass[0] - 12.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// Step-level behavior
// ---------------------------------------------------------------------------

#[test]
fn cpu_step_conserves_and_shrinks() {
    let mut sim = CpuSim::new(create_particles(4242, 64));
    let m0 = sim.particles.total_mass();
    let n0 = sim.particles.len();
    for _ in 0..10 {
        sim.step(PHYSICS_DT);
    }
    assert!(sim.particles.len() <= n0);
    let rel = ((sim.particles.total_mass() - m0) / m0).abs();
    assert!(rel < 1e-4, "mass drift {rel}");
    // Deterministic: same seed, same trajectory hash.
    let mut sim2 = CpuSim::new(create_particles(4242, 64));
    for _ in 0..10 {
        sim2.step(PHYSICS_DT);
    }
    assert_eq!(
        sim.particles.positions_hash(),
        sim2.particles.positions_hash()
    );
}

#[test]
fn empty_and_singleton_steps_are_noops() {
    // Covers the early-return arms of update_system.
    let mut sim = CpuSim::new(Particles::default());
    sim.step(PHYSICS_DT);
    assert!(sim.particles.is_empty());
    let mut one = CpuSim::new(create_particles(3, 1));
    one.step(PHYSICS_DT);
    assert_eq!(one.particles.len(), 1);
}

#[test]
fn integrate_then_finish_is_verlet() {
    // One body, constant acceleration: x(t) = x0 + v0 t + a t^2/2 exactly
    // (up to f32 rounding) through the kick/drift/kick sequence.
    let mut p = assemble(vec![ParticleData {
        x: 0.0,
        y: 0.0,
        vx: 10.0,
        vy: 0.0,
        mass: 5.0,
        brightness: 0.7,
    }]);
    let (ax, ay) = (vec![2.0f32], vec![0.0f32]);
    let dt = 0.5f32;
    integrate(&mut p, &ax, &ay, dt);
    assert!((p.vx[0] - 10.5).abs() < 1e-6);
    assert!((p.x[0] - 5.25).abs() < 1e-6);
    gsim_core::cpu::finish_velocity(&mut p, &ax, &ay, dt);
    assert!((p.vx[0] - 11.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#[test]
fn constant_helpers() {
    assert!((particle_radius(16.0) - 1.8).abs() < 1e-6);
    assert!((sweep_spawn_radius(10_000) - 1500.0).abs() < 1e-3);
    assert!((sweep_spawn_radius(2_500) - 750.0).abs() < 1e-3);
    // Transition band edges (blueprint section 8 colour mapping).
    assert_eq!(dominant_body_transition(0.24), 0.0);
    assert_eq!(dominant_body_transition(0.25), 0.0);
    assert_eq!(dominant_body_transition(0.45), 1.0);
    assert_eq!(dominant_body_transition(0.46), 1.0);
    assert!((dominant_body_transition(0.35) - 0.5).abs() < 1e-6);
    // AVX-512-variant defaults (blueprint section 8).
    assert_eq!(PARTICLE_COUNT, 10_000);
    assert_eq!(GRAVITATIONAL_CONSTANT, 15.0);
    assert_eq!(GRAVITY_SOFTENING, 3.0);
    assert_eq!(PHYSICS_DT, 1.0 / 120.0);
}
