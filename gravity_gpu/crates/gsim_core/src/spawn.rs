//! Seeded initial particle cloud.
//!
//! Implements the legacy spawn algorithm exactly (blueprint §8): disk
//! sampling with center concentration, clockwise tangential velocities with
//! a sqrt radial profile, tiny radial jitter, uniform masses in
//! `[5, 20]`, brightness in `[0.55, 1.0]`, then
//! [`Particles::center_reference_frame`].
//!
//! Because [`crate::rng::QuadRand`] replicates `quad-rand 0.2.3` bit-for-bit,
//! `create_particles(seed, n)` is bit-identical to the legacy harness with
//! `--seed seed --count n` (verified by comparing `positions_hash` values).

use crate::constants::*;
use crate::rng::QuadRand;
use std::f32::consts::TAU;

/// One spawned body: position, velocity, mass and spawn brightness.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParticleData {
    /// World x.
    pub x: f32,
    /// World y.
    pub y: f32,
    /// World vx.
    pub vx: f32,
    /// World vy.
    pub vy: f32,
    /// Mass.
    pub mass: f32,
    /// Spawn-time brightness in `[0.55, 1.0]`.
    pub brightness: f32,
}

/// Structure-of-arrays particle store (CPU side: spawn + parity reference).
#[derive(Debug, Clone, Default)]
pub struct Particles {
    /// World x per body.
    pub x: Vec<f32>,
    /// World y per body.
    pub y: Vec<f32>,
    /// Velocity x per body.
    pub vx: Vec<f32>,
    /// Velocity y per body.
    pub vy: Vec<f32>,
    /// Mass per body.
    pub mass: Vec<f32>,
    /// Brightness per body.
    pub brightness: Vec<f32>,
}

impl Particles {
    /// Empty store with capacity for `n` bodies.
    pub fn with_capacity(n: usize) -> Self {
        Self {
            x: Vec::with_capacity(n),
            y: Vec::with_capacity(n),
            vx: Vec::with_capacity(n),
            vy: Vec::with_capacity(n),
            mass: Vec::with_capacity(n),
            brightness: Vec::with_capacity(n),
        }
    }

    /// Number of live bodies.
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// True when no bodies remain.
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// Push one body.
    pub fn push(&mut self, p: ParticleData) {
        self.x.push(p.x);
        self.y.push(p.y);
        self.vx.push(p.vx);
        self.vy.push(p.vy);
        self.mass.push(p.mass);
        self.brightness.push(p.brightness);
    }

    /// Total mass of the system.
    pub fn total_mass(&self) -> f32 {
        self.mass.iter().copied().sum()
    }

    /// Largest single body mass (0 when empty).
    pub fn largest_mass(&self) -> f32 {
        self.mass.iter().copied().fold(0.0_f32, f32::max)
    }

    /// Center of mass.
    pub fn center_of_mass(&self) -> (f32, f32) {
        let total = self.total_mass();
        if total == 0.0 {
            return (0.0, 0.0);
        }
        let mut wx = 0.0;
        let mut wy = 0.0;
        for i in 0..self.len() {
            wx += self.x[i] * self.mass[i];
            wy += self.y[i] * self.mass[i];
        }
        (wx / total, wy / total)
    }

    /// Shift positions so the COM is at the origin and subtract the bulk
    /// velocity, so the isolated cloud neither drifts nor spins away.
    pub fn center_reference_frame(&mut self) {
        let total = self.total_mass();
        if total == 0.0 {
            return;
        }
        let mut wx = 0.0;
        let mut wy = 0.0;
        let mut px = 0.0;
        let mut py = 0.0;
        for i in 0..self.len() {
            let m = self.mass[i];
            wx += self.x[i] * m;
            wy += self.y[i] * m;
            px += self.vx[i] * m;
            py += self.vy[i] * m;
        }
        let cx = wx / total;
        let cy = wy / total;
        let bx = px / total;
        let by = py / total;
        for i in 0..self.len() {
            self.x[i] -= cx;
            self.y[i] -= cy;
            self.vx[i] -= bx;
            self.vy[i] -= by;
        }
    }

    /// Permutation-invariant FNV-1a fingerprint over sorted
    /// `(x.to_bits, y.to_bits, mass.to_bits)` triples — the same hash the
    /// stage-1 harness emits as `positions_hash`.
    ///
    /// ```
    /// use gsim_core::spawn::create_particles;
    /// let a = create_particles(12345, 64);
    /// let b = create_particles(12345, 64);
    /// assert_eq!(a.positions_hash(), b.positions_hash());
    /// let c = create_particles(999, 64);
    /// assert_ne!(a.positions_hash(), c.positions_hash());
    /// ```
    pub fn positions_hash(&self) -> String {
        let mut keys: Vec<(u32, u32, u32)> = (0..self.len())
            .map(|i| {
                (
                    self.x[i].to_bits(),
                    self.y[i].to_bits(),
                    self.mass[i].to_bits(),
                )
            })
            .collect();
        keys.sort_unstable();
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for (x, y, m) in keys {
            for word in [x, y, m] {
                for byte in word.to_le_bytes() {
                    hash ^= byte as u64;
                    hash = hash.wrapping_mul(0x100000001b3);
                }
            }
        }
        format!("{hash:016x}")
    }
}

/// Fraction of particles spawned clockwise (legacy normalizes the weights).
fn clockwise_fraction() -> f32 {
    CLOCKWISE_ROTATION_WEIGHT / (CLOCKWISE_ROTATION_WEIGHT + COUNTERCLOCKWISE_ROTATION_WEIGHT)
}

/// Spawn radius sample: blends uniform-area and center-biased distributions.
fn sample_spawn_distance(rng: &mut QuadRand, spawn_radius: f32) -> f32 {
    let u = rng.gen_range_f32(0.0, 1.0);
    let uniform_area = u.sqrt();
    let biased = u.powf(CENTER_CONCENTRATION_EXPONENT);
    let strength = CENTER_CONCENTRATION_STRENGTH.clamp(0.0, 1.0);
    (uniform_area + (biased - uniform_area) * strength) * spawn_radius
}

/// Tangential spawn speed for a body at `distance` from the cloud center.
fn tangential_speed(distance: f32, spawn_radius: f32) -> f32 {
    let t = (distance / spawn_radius).clamp(0.0, 1.0);
    INITIAL_TANGENTIAL_SPEED_MIN
        + (INITIAL_TANGENTIAL_SPEED_MAX - INITIAL_TANGENTIAL_SPEED_MIN)
            * t.powf(INITIAL_TANGENTIAL_SPEED_RADIUS_EXPONENT)
}

/// Spawn one body; consumes exactly five RNG outputs in legacy order.
fn spawn_one(rng: &mut QuadRand, clockwise: bool, spawn_radius: f32) -> ParticleData {
    let angle = rng.gen_range_f32(0.0, TAU);
    let distance = sample_spawn_distance(rng, spawn_radius);
    let rx = angle.cos();
    let ry = angle.sin();
    // Screen-space Y grows downward; clockwise tangent is (-ry, rx).
    let (tx, ty) = if clockwise { (-ry, rx) } else { (ry, -rx) };
    let speed = tangential_speed(distance, spawn_radius);
    let radial = rng.gen_range_f32(-INITIAL_RADIAL_SPEED_MAX, INITIAL_RADIAL_SPEED_MAX);
    ParticleData {
        x: rx * distance,
        y: ry * distance,
        vx: tx * speed + rx * radial,
        vy: ty * speed + ry * radial,
        mass: rng.gen_range_f32(MIN_PARTICLE_MASS, MAX_PARTICLE_MASS),
        brightness: rng.gen_range_f32(0.55, 1.0),
    }
}

/// Build the initial cloud: `n` bodies from `seed`, reference frame
/// centered. With `spawn_radius = SPAWN_RADIUS` this is bit-identical to
/// legacy `--seed seed --count n`.
///
/// ```
/// use gsim_core::spawn::create_particles;
/// let p = create_particles(12345, 100);
/// assert_eq!(p.len(), 100);
/// assert!((p.total_mass() - 1250.0).abs() < 250.0);
/// ```
pub fn create_particles(seed: u64, n: usize) -> Particles {
    create_particles_radius(seed, n, SPAWN_RADIUS)
}

/// Variant with an explicit spawn radius (used by the N-scaling sweep,
/// which keeps surface density constant via [`sweep_spawn_radius`]).
pub fn create_particles_radius(seed: u64, n: usize, spawn_radius: f32) -> Particles {
    let mut rng = QuadRand::new();
    rng.srand(seed);
    let cw_count = (n as f32 * clockwise_fraction()).round() as usize;
    let mut particles = Particles::with_capacity(n);
    for i in 0..n {
        particles.push(spawn_one(&mut rng, i < cw_count, spawn_radius));
    }
    particles.center_reference_frame();
    particles
}
