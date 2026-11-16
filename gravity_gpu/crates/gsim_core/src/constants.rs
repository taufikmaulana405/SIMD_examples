//! Physics, spawn, trail and render constants.
//!
//! Every value is the **AVX-512 legacy variant's default**, copied verbatim
//! from `legacy_work/gravity_simd_avx512` (blueprint §8). The WGSL shaders in
//! `shaders/` mirror these values; if you change one here, change it there
//! too (see `gpu::shaders`).

/// Number of particles at spawn.
pub const PARTICLE_COUNT: usize = 10_000;
/// Gravitational constant in simulation units.
pub const GRAVITATIONAL_CONSTANT: f32 = 15.0;
/// Minimum spawn mass.
pub const MIN_PARTICLE_MASS: f32 = 5.0;
/// Maximum spawn mass.
pub const MAX_PARTICLE_MASS: f32 = 20.0;
/// World-space radius = `PARTICLE_RADIUS_SCALE * sqrt(mass)`.
pub const PARTICLE_RADIUS_SCALE: f32 = 0.45;
/// Initial cloud disk radius.
pub const SPAWN_RADIUS: f32 = 1500.0;
/// Blend between uniform-area (0) and center-biased (1) spawn distribution.
pub const CENTER_CONCENTRATION_STRENGTH: f32 = 0.75;
/// Exponent shaping the center-biased spawn distribution.
pub const CENTER_CONCENTRATION_EXPONENT: f32 = 1.40;
/// Relative weight of clockwise spawn rotation.
pub const CLOCKWISE_ROTATION_WEIGHT: f32 = 1.0;
/// Relative weight of counter-clockwise spawn rotation.
pub const COUNTERCLOCKWISE_ROTATION_WEIGHT: f32 = 0.0;
/// Tangential spawn speed at the cloud center.
pub const INITIAL_TANGENTIAL_SPEED_MIN: f32 = 6.0;
/// Tangential spawn speed at [`SPAWN_RADIUS`].
pub const INITIAL_TANGENTIAL_SPEED_MAX: f32 = 20.0;
/// Radial exponent of the tangential-speed profile (0.5 = sqrt profile).
pub const INITIAL_TANGENTIAL_SPEED_RADIUS_EXPONENT: f32 = 0.50;
/// Maximum absolute radial spawn speed.
pub const INITIAL_RADIAL_SPEED_MAX: f32 = 0.01;
/// Plummer-style softening length; avoids singular accelerations.
pub const GRAVITY_SOFTENING: f32 = 3.0;
/// Fixed physics timestep (seconds of sim time).
pub const PHYSICS_DT: f32 = 1.0 / 120.0;
/// Maximum physics substeps per rendered frame (quirk g: debt past this is
/// dropped, and the GPU port surfaces a counter instead of staying silent).
pub const MAX_PHYSICS_STEPS_PER_FRAME: usize = 8;

/// Trail sample interval in **sim** seconds (quirk e: legacy used wall time;
/// the GPU port samples on sim time so headless runs stay deterministic).
pub const TRAIL_SAMPLE_INTERVAL_SECONDS: f32 = 1.0 / 45.0;
/// Trail alpha half-life in sim seconds.
pub const TRAIL_HALF_LIFE_SECONDS: f32 = 10.0;
/// Trail points older than this are culled (sim seconds).
pub const TRAIL_MAX_AGE_SECONDS: f32 = 40.0;
/// Alpha of a freshly sampled trail point.
pub const TRAIL_INITIAL_ALPHA: f32 = 0.35;
/// Minimum trail width in screen pixels (full width).
pub const TRAIL_MIN_WIDTH_PIXELS: f32 = 1.0;
/// Maximum trail width in screen pixels (full width).
pub const TRAIL_MAX_WIDTH_PIXELS: f32 = 2.0;
/// Multiplier on world-space diameter before the pixel clamp.
pub const TRAIL_WIDTH_SCALE: f32 = 1.0;
/// Ring capacity when trails can never be recorded or shown (bench /
/// headless without `--trails`): the buffer must exist for the bind
/// group, but nothing is ever appended. When trails are in play the
/// ring holds `trail_max_age` seconds of points per particle at the
/// 45 Hz sample rate (the default 40 s — the legacy limit — is 1800
/// points per particle, the same footprint the previous program's
/// unbounded deque reached: 18 M points / 360 MB at N=10k, quirk f).
pub const TRAIL_RING_MIN_POINTS: usize = 1024;
/// Bytes per trail point on the GPU: `pos:vec2<f32>, radius:f32,
/// brightness:f32, birth:f32`.
pub const TRAIL_POINT_BYTES: usize = 20;

/// Mass fraction of total mass where the yellow transition starts.
pub const YELLOW_TRANSITION_START_MASS_FRACTION: f32 = 0.25;
/// Mass fraction where the central-body appearance is complete.
pub const CENTRAL_BODY_APPEARANCE_MASS_FRACTION: f32 = 0.45;

/// Background clear color (linear RGBA).
pub const BACKGROUND_COLOR: [f32; 4] = [0.005, 0.008, 0.025, 1.0];

/// Default window size (windowed mode).
pub const WINDOW_WIDTH: u32 = 1280;
/// Default window size (windowed mode).
pub const WINDOW_HEIGHT: u32 = 800;

/// Default RNG seed (arbitrary but fixed; matches the stage-1 harness).
pub const DEFAULT_SEED: u64 = 12345;

/// Spawn radius for the N-scaling sweep: `r(N) = 1500 * sqrt(N / 10_000)`.
/// Keeps the initial *surface density* constant as N grows so the dynamical
/// regime (collapse timescale, merger rate) stays comparable across the
/// sweep instead of turning into a super-dense bullet at large N.
pub fn sweep_spawn_radius(n: usize) -> f32 {
    SPAWN_RADIUS * (n as f32 / PARTICLE_COUNT as f32).sqrt()
}

/// Smoothstep of `x` over the [`YELLOW_TRANSITION_START_MASS_FRACTION`]..=
/// [`CENTRAL_BODY_APPEARANCE_MASS_FRACTION`] band, mirroring legacy's
/// `dominant_body_transition`.
///
/// ```
/// use gsim_core::constants::dominant_body_transition;
/// assert_eq!(dominant_body_transition(0.0), 0.0);
/// assert_eq!(dominant_body_transition(1.0), 1.0);
/// let mid = dominant_body_transition(0.35);
/// assert!((mid - 0.5).abs() < 1e-6);
/// ```
pub fn dominant_body_transition(mass_fraction: f32) -> f32 {
    let range = CENTRAL_BODY_APPEARANCE_MASS_FRACTION - YELLOW_TRANSITION_START_MASS_FRACTION;
    if range <= 0.0 {
        return if mass_fraction >= CENTRAL_BODY_APPEARANCE_MASS_FRACTION {
            1.0
        } else {
            0.0
        };
    }
    let t = ((mass_fraction - YELLOW_TRANSITION_START_MASS_FRACTION) / range).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// World-space display radius of a body of the given mass.
pub fn particle_radius(mass: f32) -> f32 {
    PARTICLE_RADIUS_SCALE * mass.sqrt()
}
