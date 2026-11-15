//! Scalar CPU reference implementation of one physics step.
//!
//! This is the parity oracle for Stage 3: it implements `update_system`
//! exactly as documented in blueprint §2/§6 (velocity Verlet, all-pairs
//! gravity with softening, perfectly-inelastic sequential merge), in plain
//! scalar `f32` arithmetic.
//!
//! Summation order matches the GPU **naive** kernel bit-for-bit: for each
//! body `i`, partners `j` are visited in ascending order (`j == i`
//! skipped) and only `a[i]` is accumulated (no Newton-3rd-law cross writes,
//! which would need float atomics on GPU). The legacy AVX kernels instead
//! use FMA intrinsics and split the pair sum across both bodies, so
//! legacy-vs-reference differs at the ulp level by construction; the
//! reference-vs-naive-GPU comparison is the bitwise one.

use crate::constants::*;
use crate::spawn::Particles;

/// All-pairs gravity: `a[i] = Σ G·m[j]·(x[j]−x[i]) / (|dx|²+soft²)^1.5`.
/// Partners visited in ascending `j` order, `j == i` skipped.
pub fn calculate_accelerations(p: &Particles, ax: &mut [f32], ay: &mut [f32]) {
    let n = p.len();
    assert!(ax.len() >= n && ay.len() >= n);
    let soft2 = GRAVITY_SOFTENING * GRAVITY_SOFTENING;
    for i in 0..n {
        let mut aix = 0.0_f32;
        let mut aiy = 0.0_f32;
        let xi = p.x[i];
        let yi = p.y[i];
        for j in 0..n {
            if j == i {
                continue;
            }
            let dx = p.x[j] - xi;
            let dy = p.y[j] - yi;
            let r2 = dx * dx + dy * dy + soft2;
            let inv = 1.0 / r2.sqrt();
            let f = GRAVITATIONAL_CONSTANT * inv * inv * inv * p.mass[j];
            aix += dx * f;
            aiy += dy * f;
        }
        ax[i] = aix;
        ay[i] = aiy;
    }
}

/// `f64` reference version of [`calculate_accelerations`]: identical
/// formula and ascending-`j` summation order, but every operation in
/// `f64`. Stage-3 P1 uses this as the oracle: the `f32` implementations
/// (legacy scalar, `gsim_core::cpu`, the WGSL kernels) are scored by their
/// max absolute deviation from it.
///
/// ```
/// use gsim_core::{cpu::{calculate_accelerations, calculate_accelerations_f64}, spawn::create_particles};
/// let p = create_particles(12345, 16);
/// let n = p.len();
/// let (mut ax, mut ay) = (vec![0.0f32; n], vec![0.0f32; n]);
/// calculate_accelerations(&p, &mut ax, &mut ay);
/// let (bx, by) = calculate_accelerations_f64(&p);
/// let err: f64 = ax.iter().zip(bx.iter()).map(|(a, b)| (*a as f64 - b).abs()).fold(0.0, f64::max);
/// assert!(err < 1e-2, "f32 vs f64 oracle must agree to ~1e-3, got {err}");
/// ```
pub fn calculate_accelerations_f64(p: &Particles) -> (Vec<f64>, Vec<f64>) {
    let n = p.len();
    let soft2 = f64::from(GRAVITY_SOFTENING) * f64::from(GRAVITY_SOFTENING);
    let g = f64::from(GRAVITATIONAL_CONSTANT);
    let mut ax = vec![0.0_f64; n];
    let mut ay = vec![0.0_f64; n];
    for i in 0..n {
        let mut aix = 0.0_f64;
        let mut aiy = 0.0_f64;
        let xi = f64::from(p.x[i]);
        let yi = f64::from(p.y[i]);
        for j in 0..n {
            if j == i {
                continue;
            }
            let dx = f64::from(p.x[j]) - xi;
            let dy = f64::from(p.y[j]) - yi;
            let r2 = dx * dx + dy * dy + soft2;
            let inv = 1.0 / r2.sqrt();
            let f = g * inv * inv * inv * f64::from(p.mass[j]);
            aix += dx * f;
            aiy += dy * f;
        }
        ax[i] = aix;
        ay[i] = aiy;
    }
    (ax, ay)
}

/// First Verlet half: `v += a·dt/2`, then `x += v·dt`.
pub fn integrate(p: &mut Particles, ax: &[f32], ay: &[f32], dt: f32) {
    for i in 0..p.len() {
        p.vx[i] += ax[i] * dt * 0.5;
        p.vy[i] += ay[i] * dt * 0.5;
        p.x[i] += p.vx[i] * dt;
        p.y[i] += p.vy[i] * dt;
    }
}

/// Second Verlet half: `v += a·dt/2`.
pub fn finish_velocity(p: &mut Particles, ax: &[f32], ay: &[f32], dt: f32) {
    for i in 0..p.len() {
        p.vx[i] += ax[i] * dt * 0.5;
        p.vy[i] += ay[i] * dt * 0.5;
    }
}

/// First index `> first` whose disk overlaps body `first` (ascending scan),
/// or `None`. Overlap test is the legacy one: `d² ≤ (rᵢ+rⱼ)²`.
pub fn find_collision(p: &Particles, first: usize) -> Option<usize> {
    let n = p.len();
    let ri = particle_radius(p.mass[first]);
    for j in (first + 1)..n {
        let dx = p.x[j] - p.x[first];
        let dy = p.y[j] - p.y[first];
        let rc = ri + particle_radius(p.mass[j]);
        if dx * dx + dy * dy <= rc * rc {
            return Some(j);
        }
    }
    None
}

/// Merge body `j` into body `i`: mass-weighted position/velocity/brightness
/// (perfectly inelastic), then `swap_remove(j)`.
pub fn merge_pair(p: &mut Particles, i: usize, j: usize) {
    let mi = p.mass[i];
    let mj = p.mass[j];
    let m = mi + mj;
    p.x[i] = (p.x[i] * mi + p.x[j] * mj) / m;
    p.y[i] = (p.y[i] * mi + p.y[j] * mj) / m;
    p.vx[i] = (p.vx[i] * mi + p.vx[j] * mj) / m;
    p.vy[i] = (p.vy[i] * mi + p.vy[j] * mj) / m;
    p.mass[i] = m;
    p.brightness[i] = (p.brightness[i] * mi + p.brightness[j] * mj) / m;
    for v in [
        &mut p.x,
        &mut p.y,
        &mut p.vx,
        &mut p.vy,
        &mut p.mass,
        &mut p.brightness,
    ] {
        v.swap_remove(j);
    }
}

/// Legacy-order sequential merge: ascending `first` scan, first overlap
/// wins, restart the scan for `first` after every merge (quirk b).
pub fn merge_colliding(p: &mut Particles) {
    let mut first = 0;
    while first < p.len() {
        while let Some(second) = find_collision(p, first) {
            merge_pair(p, first, second);
        }
        first += 1;
    }
}

/// One full velocity-Verlet step: force → kick+drift → merge → force → kick.
///
/// ```
/// use gsim_core::{cpu::CpuSim, spawn::create_particles};
/// let mut sim = CpuSim::new(create_particles(12345, 32));
/// sim.step(1.0 / 120.0);
/// assert!(sim.particles.len() <= 32);
/// assert!(sim.particles.total_mass() > 0.0);
/// ```
pub fn update_system(p: &mut Particles, ax: &mut Vec<f32>, ay: &mut Vec<f32>, dt: f32) {
    if p.is_empty() {
        return;
    }
    ax.resize(p.len(), 0.0);
    ay.resize(p.len(), 0.0);
    calculate_accelerations(p, ax, ay);
    integrate(p, ax, ay, dt);
    merge_colliding(p);
    if p.is_empty() {
        return;
    }
    ax.resize(p.len(), 0.0);
    ay.resize(p.len(), 0.0);
    calculate_accelerations(p, ax, ay);
    finish_velocity(p, ax, ay, dt);
}

/// Owns the particle store plus acceleration scratch buffers.
#[derive(Debug)]
pub struct CpuSim {
    /// Live bodies.
    pub particles: Particles,
    /// Scratch accelerations (x).
    pub ax: Vec<f32>,
    /// Scratch accelerations (y).
    pub ay: Vec<f32>,
}

impl CpuSim {
    /// New simulation from a spawned cloud.
    pub fn new(particles: Particles) -> Self {
        let n = particles.len();
        Self {
            particles,
            ax: vec![0.0; n],
            ay: vec![0.0; n],
        }
    }

    /// Advance one fixed-dt step.
    pub fn step(&mut self, dt: f32) {
        update_system(&mut self.particles, &mut self.ax, &mut self.ay, dt);
    }
}
