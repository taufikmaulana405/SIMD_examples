use macroquad::prelude::*;
use macroquad::rand::gen_range;
use std::collections::VecDeque;
use std::f32::consts::TAU;
use std::arch::x86_64::*;

const SIMD_LANES: usize = 8;
const PARTICLE_COUNT: usize = 10_000;

// Arbitrary simulation units.
const GRAVITATIONAL_CONSTANT: f32 = 15.0;

const MIN_PARTICLE_MASS: f32 = 5.0;
const MAX_PARTICLE_MASS: f32 = 20.0;
const PARTICLE_RADIUS_SCALE: f32 = 0.45;

// Particles begin inside this disk. There is no central body.
const SPAWN_RADIUS: f32 = 1200.0;

// Controls how strongly the initial cloud is concentrated toward its center.
//
// 0.0 = uniform particle density over the disk's area.
// 1.0 = use the full center-biased distribution selected below.
// Values between 0.0 and 1.0 blend smoothly between the two.
const CENTER_CONCENTRATION_STRENGTH: f32 = 0.75;

// Controls the shape of the center-biased distribution.
//
// 0.5 = uniform density over the disk's area.
// > 0.5 = increasingly concentrated toward the center.
// < 0.5 = increasingly concentrated toward the outer edge.
//
// This parameter has the strongest effect when
// CENTER_CONCENTRATION_STRENGTH is close to 1.0.
const CENTER_CONCENTRATION_EXPONENT: f32 = 1.40;

// Relative weights for the initial rotation directions. These do not have
// to add up to 1.0; the program normalizes them automatically.
// With 0.80 and 0.20, approximately 80% of the particles start clockwise
// and 20% start counterclockwise.
const CLOCKWISE_ROTATION_WEIGHT: f32 = 1.0;
const COUNTERCLOCKWISE_ROTATION_WEIGHT: f32 = 0.00;

// Initial tangential-speed range. The speed is determined primarily
// by distance from the cloud center: particles near the center begin slower,
// while particles near SPAWN_RADIUS approach the maximum speed.
const INITIAL_TANGENTIAL_SPEED_MIN: f32 = 6.0;
const INITIAL_TANGENTIAL_SPEED_MAX: f32 = 20.0;

// Controls the radial shape of the initial rotation-speed profile.
//
// 0.50 = square-root profile (recommended for this particle-only cloud)
// 1.00 = linear increase with radius
// >1.0 = keeps the inner region slower for longer
// <1.0 = speed rises more quickly near the center
const INITIAL_TANGENTIAL_SPEED_RADIUS_EXPONENT: f32 = 0.50;

// Adds a small inward or outward velocity component. Set this to 0.0 for
// initially pure circular/tangential motion.
const INITIAL_RADIAL_SPEED_MAX: f32 = 0.01;

// Avoids singular accelerations at extremely small separations.
const GRAVITY_SOFTENING: f32 = 3.0;

const PHYSICS_DT: f32 = 1.0 / 120.0;
const MAX_PHYSICS_STEPS_PER_FRAME: usize = 8;

const TRAIL_SAMPLE_INTERVAL_SECONDS: f32 = 1.0 / 45.0;
const TRAIL_HALF_LIFE_SECONDS: f32 = 10.0;
const TRAIL_MAX_AGE_SECONDS: f32 = 40.0;
const TRAIL_INITIAL_ALPHA: f32 = 0.35;

// Trail thickness is controlled independently from the object's visual size.
// These values describe the full trail width in screen pixels, not the radius.
const TRAIL_MIN_WIDTH_PIXELS: f32 = 1.0;
const TRAIL_MAX_WIDTH_PIXELS: f32 = 2.0;

// Multiplies the object's world-space diameter before applying the pixel limits.
// Lower values make every trail thinner while preserving relative differences.
const TRAIL_WIDTH_SCALE: f32 = 1.0;

const BACKGROUND_COLOR: Color =
    Color::new(0.005, 0.008, 0.025, 1.0);

// A body keeps its normal particle appearance below this fraction of the
// system's total mass. At this threshold it begins transitioning to yellow.
const YELLOW_TRANSITION_START_MASS_FRACTION: f32 = 0.25;

// At this fraction of the system's total mass, the transition is complete
// and the body uses the same glowing appearance as the original central mass.
const CENTRAL_BODY_APPEARANCE_MASS_FRACTION: f32 = 0.45;

struct ParticleData {
    x: f32,
    y: f32,
    velocity_x: f32,
    velocity_y: f32,
    mass: f32,
    brightness: f32,
}

// Structure-of-arrays layout: each property is contiguous in memory,
// making it convenient to load eight values into one SIMD vector.
struct Particles {
    x: Vec<f32>,
    y: Vec<f32>,
    velocity_x: Vec<f32>,
    velocity_y: Vec<f32>,
    mass: Vec<f32>,
    brightness: Vec<f32>,
}

impl Particles {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            x: Vec::with_capacity(capacity),
            y: Vec::with_capacity(capacity),
            velocity_x: Vec::with_capacity(capacity),
            velocity_y: Vec::with_capacity(capacity),
            mass: Vec::with_capacity(capacity),
            brightness: Vec::with_capacity(capacity),
        }
    }

    fn len(&self) -> usize {
        self.x.len()
    }

    fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    fn push(&mut self, particle: ParticleData) {
        self.x.push(particle.x);
        self.y.push(particle.y);
        self.velocity_x.push(particle.velocity_x);
        self.velocity_y.push(particle.velocity_y);
        self.mass.push(particle.mass);
        self.brightness.push(particle.brightness);
    }

    fn swap_remove(&mut self, index: usize) {
        self.x.swap_remove(index);
        self.y.swap_remove(index);
        self.velocity_x.swap_remove(index);
        self.velocity_y.swap_remove(index);
        self.mass.swap_remove(index);
        self.brightness.swap_remove(index);
    }

    fn total_mass(&self) -> f32 {
        self.mass.iter().copied().sum()
    }

    fn largest_mass(&self) -> f32 {
        self.mass
            .iter()
            .copied()
            .fold(0.0_f32, f32::max)
    }

    fn center_of_mass(&self) -> Vec2 {
        let total_mass = self.total_mass();

        if total_mass == 0.0 {
            return Vec2::ZERO;
        }

        let mut weighted_x = 0.0;
        let mut weighted_y = 0.0;

        for index in 0..self.len() {
            weighted_x += self.x[index] * self.mass[index];
            weighted_y += self.y[index] * self.mass[index];
        }

        vec2(weighted_x / total_mass, weighted_y / total_mass)
    }

    // Move the initial center of mass to the origin and remove the bulk
    // velocity so that the isolated system does not immediately drift away.
    fn center_reference_frame(&mut self) {
        let total_mass = self.total_mass();

        if total_mass == 0.0 {
            return;
        }

        let mut weighted_x = 0.0;
        let mut weighted_y = 0.0;
        let mut total_momentum_x = 0.0;
        let mut total_momentum_y = 0.0;

        for index in 0..self.len() {
            let mass = self.mass[index];

            weighted_x += self.x[index] * mass;
            weighted_y += self.y[index] * mass;

            total_momentum_x += self.velocity_x[index] * mass;
            total_momentum_y += self.velocity_y[index] * mass;
        }

        let center_x = weighted_x / total_mass;
        let center_y = weighted_y / total_mass;
        let bulk_velocity_x = total_momentum_x / total_mass;
        let bulk_velocity_y = total_momentum_y / total_mass;

        for index in 0..self.len() {
            self.x[index] -= center_x;
            self.y[index] -= center_y;
            self.velocity_x[index] -= bulk_velocity_x;
            self.velocity_y[index] -= bulk_velocity_y;
        }
    }
}

// Trail points are independent from particles. When particles merge, their
// previous trails remain and fade normally.
struct TrailPoint {
    x: f32,
    y: f32,
    radius: f32,
    brightness: f32,
    creation_time: f32,
}

fn window_configuration() -> Conf {
    Conf {
        window_title: "SIMD Rotating Particle Cloud".to_owned(),
        window_width: 1280,
        window_height: 800,
        high_dpi: true,
        window_resizable: true,
        ..Default::default()
    }
}

fn particle_radius(mass: f32) -> f32 {
    PARTICLE_RADIUS_SCALE * mass.sqrt()
}

#[inline]
fn mix_color(from: Color, to: Color, amount: f32) -> Color {
    let amount = amount.clamp(0.0, 1.0);

    Color::new(
        from.r + (to.r - from.r) * amount,
        from.g + (to.g - from.g) * amount,
        from.b + (to.b - from.b) * amount,
        from.a + (to.a - from.a) * amount,
    )
}

#[inline]
fn dominant_body_transition(mass_fraction: f32) -> f32 {
    let transition_range = CENTRAL_BODY_APPEARANCE_MASS_FRACTION
        - YELLOW_TRANSITION_START_MASS_FRACTION;

    if transition_range <= 0.0 {
        return if mass_fraction >= CENTRAL_BODY_APPEARANCE_MASS_FRACTION {
            1.0
        } else {
            0.0
        };
    }

    let linear_progress = (
        (mass_fraction - YELLOW_TRANSITION_START_MASS_FRACTION)
            / transition_range
    )
    .clamp(0.0, 1.0);

    // Smoothstep prevents a visually abrupt change at either threshold.
    linear_progress * linear_progress * (3.0 - 2.0 * linear_progress)
}

fn sample_spawn_distance() -> f32 {
    let random_fraction: f32 = gen_range(0.0_f32, 1.0_f32);

    // sqrt(U) produces uniform density over the area of a disk.
    let uniform_area_radius = random_fraction.sqrt();

    // Raising U to an exponent greater than 0.5 moves more samples
    // toward the center while preserving a smooth continuous cloud.
    let center_biased_radius =
        random_fraction.powf(CENTER_CONCENTRATION_EXPONENT);

    let strength = CENTER_CONCENTRATION_STRENGTH.clamp(0.0, 1.0);

    let normalized_radius = uniform_area_radius
        + (center_biased_radius - uniform_area_radius) * strength;

    normalized_radius * SPAWN_RADIUS
}

fn initial_tangential_speed(distance: f32) -> f32 {
    let normalized_radius =
        (distance / SPAWN_RADIUS).clamp(0.0, 1.0);

    let radial_speed_factor = normalized_radius.powf(
        INITIAL_TANGENTIAL_SPEED_RADIUS_EXPONENT,
    );

    INITIAL_TANGENTIAL_SPEED_MIN
        + (INITIAL_TANGENTIAL_SPEED_MAX
            - INITIAL_TANGENTIAL_SPEED_MIN)
            * radial_speed_factor
}

fn spawn_particle(clockwise: bool) -> ParticleData {
    let position_angle = gen_range(0.0, TAU);
    let distance = sample_spawn_distance();

    let radial_x = position_angle.cos();
    let radial_y = position_angle.sin();

    // Screen-space Y increases downward. At the right side of the cloud,
    // clockwise motion therefore points downward, giving the clockwise
    // tangent (-radial_y, radial_x).
    let (tangential_x, tangential_y) = if clockwise {
        (-radial_y, radial_x)
    } else {
        (radial_y, -radial_x)
    };

    // In the initial particle-only cloud, little mass is enclosed near
    // the center, so inner particles begin slower. Speed increases with
    // radius according to INITIAL_TANGENTIAL_SPEED_RADIUS_EXPONENT.
    let tangential_speed = initial_tangential_speed(distance);

    let radial_speed = gen_range(
        -INITIAL_RADIAL_SPEED_MAX,
        INITIAL_RADIAL_SPEED_MAX,
    );

    ParticleData {
        x: radial_x * distance,
        y: radial_y * distance,
        velocity_x:
            tangential_x * tangential_speed + radial_x * radial_speed,
        velocity_y:
            tangential_y * tangential_speed + radial_y * radial_speed,
        mass: gen_range(MIN_PARTICLE_MASS, MAX_PARTICLE_MASS),
        brightness: gen_range(0.55, 1.0),
    }
}

fn initial_clockwise_fraction() -> f32 {
    let total_weight =
        CLOCKWISE_ROTATION_WEIGHT + COUNTERCLOCKWISE_ROTATION_WEIGHT;

    assert!(
        CLOCKWISE_ROTATION_WEIGHT >= 0.0
            && COUNTERCLOCKWISE_ROTATION_WEIGHT >= 0.0
            && total_weight > 0.0,
        "Rotation weights must be non-negative and not both zero",
    );

    CLOCKWISE_ROTATION_WEIGHT / total_weight
}

fn create_particles() -> Particles {
    assert!(
        (0.0..=1.0).contains(&CENTER_CONCENTRATION_STRENGTH),
        "CENTER_CONCENTRATION_STRENGTH must be between 0.0 and 1.0",
    );

    assert!(
        CENTER_CONCENTRATION_EXPONENT > 0.0,
        "CENTER_CONCENTRATION_EXPONENT must be greater than zero",
    );

    assert!(
        INITIAL_TANGENTIAL_SPEED_MIN >= 0.0
            && INITIAL_TANGENTIAL_SPEED_MAX
                >= INITIAL_TANGENTIAL_SPEED_MIN,
        "Tangential speed limits must satisfy 0 <= MIN <= MAX",
    );

    assert!(
        INITIAL_TANGENTIAL_SPEED_RADIUS_EXPONENT > 0.0,
        "INITIAL_TANGENTIAL_SPEED_RADIUS_EXPONENT must be greater than zero",
    );

    let mut particles = Particles::with_capacity(PARTICLE_COUNT);

    // Using an exact count makes the configured split deterministic rather
    // than merely probable. For 1,000 particles and 0.80/0.20 weights, this
    // creates exactly 800 clockwise and 200 counterclockwise particles.
    let clockwise_count =
        ((PARTICLE_COUNT as f32) * initial_clockwise_fraction()).round()
            as usize;

    for index in 0..PARTICLE_COUNT {
        particles.push(spawn_particle(index < clockwise_count));
    }

    particles.center_reference_frame();
    particles
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load_f32x8(values: &[f32], start: usize) -> __m256 {
    debug_assert!(start + SIMD_LANES <= values.len());
    unsafe { _mm256_loadu_ps(values.as_ptr().add(start)) }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn store_f32x8(values: &mut [f32], start: usize, vector: __m256) {
    debug_assert!(start + SIMD_LANES <= values.len());
    unsafe { _mm256_storeu_ps(values.as_mut_ptr().add(start), vector) };
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_f32x8(vector: __m256) -> f32 {
    // Add the upper and lower 128-bit halves, then reduce four lanes to one.
    let lower_half = _mm256_castps256_ps128(vector);
    let upper_half = _mm256_extractf128_ps::<1>(vector);
    let sum128 = _mm_add_ps(lower_half, upper_half);

    let upper_two = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, upper_two);
    let second_lane = _mm_shuffle_ps::<0x55>(sum64, sum64);

    _mm_cvtss_f32(_mm_add_ss(sum64, second_lane))
}

// Exact all-pairs gravity. For each fixed first particle, interactions with
// eight second particles are evaluated together in one 256-bit register.
#[target_feature(enable = "avx2")]
unsafe fn calculate_accelerations_avx2(
    particles: &Particles,
    acceleration_x: &mut Vec<f32>,
    acceleration_y: &mut Vec<f32>,
) {
    let particle_count = particles.len();

    acceleration_x.resize(particle_count, 0.0);
    acceleration_y.resize(particle_count, 0.0);
    acceleration_x.fill(0.0);
    acceleration_y.fill(0.0);

    let gravitational_constant = _mm256_set1_ps(GRAVITATIONAL_CONSTANT);
    let softening_squared =
        _mm256_set1_ps(GRAVITY_SOFTENING * GRAVITY_SOFTENING);
    let one = _mm256_set1_ps(1.0);
    let zero = _mm256_setzero_ps();

    for first_index in 0..particle_count {
        let first_x_scalar = particles.x[first_index];
        let first_y_scalar = particles.y[first_index];
        let first_mass_scalar = particles.mass[first_index];

        let first_x = _mm256_set1_ps(first_x_scalar);
        let first_y = _mm256_set1_ps(first_y_scalar);
        let first_mass = _mm256_set1_ps(first_mass_scalar);

        let mut first_acceleration_x = zero;
        let mut first_acceleration_y = zero;

        let mut scalar_first_acceleration_x = 0.0;
        let mut scalar_first_acceleration_y = 0.0;

        let mut second_index = first_index + 1;

        while second_index + SIMD_LANES <= particle_count {
            let second_x =
                unsafe { load_f32x8(&particles.x, second_index) };
            let second_y =
                unsafe { load_f32x8(&particles.y, second_index) };
            let second_mass =
                unsafe { load_f32x8(&particles.mass, second_index) };

            let displacement_x = _mm256_sub_ps(second_x, first_x);
            let displacement_y = _mm256_sub_ps(second_y, first_y);

            let displacement_x_squared =
                _mm256_mul_ps(displacement_x, displacement_x);
            let displacement_y_squared =
                _mm256_mul_ps(displacement_y, displacement_y);

            let distance_squared = _mm256_add_ps(
                displacement_x_squared,
                _mm256_add_ps(displacement_y_squared, softening_squared),
            );

            let inverse_distance =
                _mm256_div_ps(one, _mm256_sqrt_ps(distance_squared));
            let inverse_distance_squared =
                _mm256_mul_ps(inverse_distance, inverse_distance);
            let inverse_distance_cubed =
                _mm256_mul_ps(inverse_distance_squared, inverse_distance);

            let common_factor = _mm256_mul_ps(
                gravitational_constant,
                inverse_distance_cubed,
            );

            // Acceleration of the first particle from eight partners.
            let second_weighted_factor =
                _mm256_mul_ps(common_factor, second_mass);

            first_acceleration_x = _mm256_add_ps(
                first_acceleration_x,
                _mm256_mul_ps(displacement_x, second_weighted_factor),
            );

            first_acceleration_y = _mm256_add_ps(
                first_acceleration_y,
                _mm256_mul_ps(displacement_y, second_weighted_factor),
            );

            // Equal and opposite force, converted into acceleration of the
            // eight second particles.
            let first_weighted_factor =
                _mm256_mul_ps(common_factor, first_mass);

            let updated_second_acceleration_x = _mm256_sub_ps(
                unsafe { load_f32x8(acceleration_x, second_index) },
                _mm256_mul_ps(displacement_x, first_weighted_factor),
            );

            let updated_second_acceleration_y = _mm256_sub_ps(
                unsafe { load_f32x8(acceleration_y, second_index) },
                _mm256_mul_ps(displacement_y, first_weighted_factor),
            );

            unsafe {
                store_f32x8(
                    acceleration_x,
                    second_index,
                    updated_second_acceleration_x,
                );
                store_f32x8(
                    acceleration_y,
                    second_index,
                    updated_second_acceleration_y,
                );
            }

            second_index += SIMD_LANES;
        }

        // Scalar remainder for the final zero to seven partners.
        while second_index < particle_count {
            let displacement_x =
                particles.x[second_index] - first_x_scalar;
            let displacement_y =
                particles.y[second_index] - first_y_scalar;

            let distance_squared =
                displacement_x * displacement_x
                    + displacement_y * displacement_y
                    + GRAVITY_SOFTENING * GRAVITY_SOFTENING;

            let inverse_distance = 1.0 / distance_squared.sqrt();
            let inverse_distance_cubed =
                inverse_distance * inverse_distance * inverse_distance;

            let common_factor =
                GRAVITATIONAL_CONSTANT * inverse_distance_cubed;

            scalar_first_acceleration_x +=
                displacement_x * common_factor * particles.mass[second_index];
            scalar_first_acceleration_y +=
                displacement_y * common_factor * particles.mass[second_index];

            acceleration_x[second_index] -=
                displacement_x * common_factor * first_mass_scalar;
            acceleration_y[second_index] -=
                displacement_y * common_factor * first_mass_scalar;

            second_index += 1;
        }

        acceleration_x[first_index] +=
            unsafe { horizontal_sum_f32x8(first_acceleration_x) }
                + scalar_first_acceleration_x;
        acceleration_y[first_index] +=
            unsafe { horizontal_sum_f32x8(first_acceleration_y) }
                + scalar_first_acceleration_y;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn integrate_positions_avx2(
    particles: &mut Particles,
    acceleration_x: &[f32],
    acceleration_y: &[f32],
    dt: f32,
) {
    let particle_count = particles.len();
    let half_dt = _mm256_set1_ps(dt * 0.5);
    let full_dt = _mm256_set1_ps(dt);

    let mut index = 0;

    while index + SIMD_LANES <= particle_count {
        let velocity_x = _mm256_add_ps(
            unsafe { load_f32x8(&particles.velocity_x, index) },
            _mm256_mul_ps(
                unsafe { load_f32x8(acceleration_x, index) },
                half_dt,
            ),
        );

        let velocity_y = _mm256_add_ps(
            unsafe { load_f32x8(&particles.velocity_y, index) },
            _mm256_mul_ps(
                unsafe { load_f32x8(acceleration_y, index) },
                half_dt,
            ),
        );

        let x = _mm256_add_ps(
            unsafe { load_f32x8(&particles.x, index) },
            _mm256_mul_ps(velocity_x, full_dt),
        );

        let y = _mm256_add_ps(
            unsafe { load_f32x8(&particles.y, index) },
            _mm256_mul_ps(velocity_y, full_dt),
        );

        unsafe {
            store_f32x8(&mut particles.velocity_x, index, velocity_x);
            store_f32x8(&mut particles.velocity_y, index, velocity_y);
            store_f32x8(&mut particles.x, index, x);
            store_f32x8(&mut particles.y, index, y);
        }

        index += SIMD_LANES;
    }

    while index < particle_count {
        particles.velocity_x[index] += acceleration_x[index] * dt * 0.5;
        particles.velocity_y[index] += acceleration_y[index] * dt * 0.5;

        particles.x[index] += particles.velocity_x[index] * dt;
        particles.y[index] += particles.velocity_y[index] * dt;

        index += 1;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn finish_velocity_update_avx2(
    particles: &mut Particles,
    acceleration_x: &[f32],
    acceleration_y: &[f32],
    dt: f32,
) {
    let particle_count = particles.len();
    let half_dt = _mm256_set1_ps(dt * 0.5);

    let mut index = 0;

    while index + SIMD_LANES <= particle_count {
        let velocity_x = _mm256_add_ps(
            unsafe { load_f32x8(&particles.velocity_x, index) },
            _mm256_mul_ps(
                unsafe { load_f32x8(acceleration_x, index) },
                half_dt,
            ),
        );

        let velocity_y = _mm256_add_ps(
            unsafe { load_f32x8(&particles.velocity_y, index) },
            _mm256_mul_ps(
                unsafe { load_f32x8(acceleration_y, index) },
                half_dt,
            ),
        );

        unsafe {
            store_f32x8(&mut particles.velocity_x, index, velocity_x);
            store_f32x8(&mut particles.velocity_y, index, velocity_y);
        }

        index += SIMD_LANES;
    }

    while index < particle_count {
        particles.velocity_x[index] += acceleration_x[index] * dt * 0.5;
        particles.velocity_y[index] += acceleration_y[index] * dt * 0.5;
        index += 1;
    }
}

// Tests eight possible collision partners simultaneously and returns the
// first one that overlaps the selected particle.
#[target_feature(enable = "avx2")]
unsafe fn find_collision_avx2(
    particles: &Particles,
    first_index: usize,
) -> Option<usize> {
    let particle_count = particles.len();

    let first_x = _mm256_set1_ps(particles.x[first_index]);
    let first_y = _mm256_set1_ps(particles.y[first_index]);
    let first_radius =
        _mm256_set1_ps(particle_radius(particles.mass[first_index]));
    let radius_scale = _mm256_set1_ps(PARTICLE_RADIUS_SCALE);

    let mut second_index = first_index + 1;

    while second_index + SIMD_LANES <= particle_count {
        let displacement_x = _mm256_sub_ps(
            unsafe { load_f32x8(&particles.x, second_index) },
            first_x,
        );
        let displacement_y = _mm256_sub_ps(
            unsafe { load_f32x8(&particles.y, second_index) },
            first_y,
        );

        let distance_squared = _mm256_add_ps(
            _mm256_mul_ps(displacement_x, displacement_x),
            _mm256_mul_ps(displacement_y, displacement_y),
        );

        let second_radius = _mm256_mul_ps(
            _mm256_sqrt_ps(unsafe {
                load_f32x8(&particles.mass, second_index)
            }),
            radius_scale,
        );

        let collision_distance = _mm256_add_ps(first_radius, second_radius);
        let collision_distance_squared =
            _mm256_mul_ps(collision_distance, collision_distance);

        let collision_comparison = _mm256_cmp_ps::<_CMP_LE_OQ>(
            distance_squared,
            collision_distance_squared,
        );
        let collision_mask = _mm256_movemask_ps(collision_comparison) as u32;

        if collision_mask != 0 {
            return Some(
                second_index + collision_mask.trailing_zeros() as usize,
            );
        }

        second_index += SIMD_LANES;
    }

    let first_x_scalar = particles.x[first_index];
    let first_y_scalar = particles.y[first_index];
    let first_radius_scalar =
        particle_radius(particles.mass[first_index]);

    while second_index < particle_count {
        let displacement_x =
            particles.x[second_index] - first_x_scalar;
        let displacement_y =
            particles.y[second_index] - first_y_scalar;

        let collision_distance = first_radius_scalar
            + particle_radius(particles.mass[second_index]);

        if displacement_x * displacement_x
            + displacement_y * displacement_y
            <= collision_distance * collision_distance
        {
            return Some(second_index);
        }

        second_index += 1;
    }

    None
}

fn merge_pair(
    particles: &mut Particles,
    first_index: usize,
    second_index: usize,
) {
    let first_mass = particles.mass[first_index];
    let second_mass = particles.mass[second_index];
    let combined_mass = first_mass + second_mass;

    // Center-of-mass position.
    let combined_x =
        (particles.x[first_index] * first_mass
            + particles.x[second_index] * second_mass)
            / combined_mass;

    let combined_y =
        (particles.y[first_index] * first_mass
            + particles.y[second_index] * second_mass)
            / combined_mass;

    // Conservation of linear momentum.
    let combined_velocity_x =
        (particles.velocity_x[first_index] * first_mass
            + particles.velocity_x[second_index] * second_mass)
            / combined_mass;

    let combined_velocity_y =
        (particles.velocity_y[first_index] * first_mass
            + particles.velocity_y[second_index] * second_mass)
            / combined_mass;

    let combined_brightness =
        (particles.brightness[first_index] * first_mass
            + particles.brightness[second_index] * second_mass)
            / combined_mass;

    particles.x[first_index] = combined_x;
    particles.y[first_index] = combined_y;
    particles.velocity_x[first_index] = combined_velocity_x;
    particles.velocity_y[first_index] = combined_velocity_y;
    particles.mass[first_index] = combined_mass;
    particles.brightness[first_index] = combined_brightness;

    particles.swap_remove(second_index);
}

#[target_feature(enable = "avx2")]
unsafe fn merge_colliding_particles_avx2(particles: &mut Particles) {
    let mut first_index = 0;

    while first_index < particles.len() {
        // Restart after each merge because the resulting particle has a new
        // radius, position, mass and velocity.
        while let Some(second_index) =
            unsafe { find_collision_avx2(particles, first_index) }
        {
            merge_pair(particles, first_index, second_index);
        }

        first_index += 1;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn update_system_avx2_impl(
    particles: &mut Particles,
    acceleration_x: &mut Vec<f32>,
    acceleration_y: &mut Vec<f32>,
    dt: f32,
) {
    if particles.is_empty() {
        return;
    }

    unsafe {
        calculate_accelerations_avx2(
            particles,
            acceleration_x,
            acceleration_y,
        );

        integrate_positions_avx2(
            particles,
            acceleration_x,
            acceleration_y,
            dt,
        );

        merge_colliding_particles_avx2(particles);
    }

    if particles.is_empty() {
        return;
    }

    unsafe {
        calculate_accelerations_avx2(
            particles,
            acceleration_x,
            acceleration_y,
        );

        finish_velocity_update_avx2(
            particles,
            acceleration_x,
            acceleration_y,
            dt,
        );
    }
}

fn update_system_simd(
    particles: &mut Particles,
    acceleration_x: &mut Vec<f32>,
    acceleration_y: &mut Vec<f32>,
    dt: f32,
) {
    assert!(
        std::arch::is_x86_feature_detected!("avx2"),
        "This build requires a CPU with AVX2 support",
    );

    unsafe {
        update_system_avx2_impl(
            particles,
            acceleration_x,
            acceleration_y,
            dt,
        );
    }
}

fn add_trail_points(
    trails: &mut VecDeque<TrailPoint>,
    particles: &Particles,
    trail_clock: f32,
) {
    trails.reserve(particles.len());

    for index in 0..particles.len() {
        trails.push_back(TrailPoint {
            x: particles.x[index],
            y: particles.y[index],
            radius: particle_radius(particles.mass[index]),
            brightness: particles.brightness[index],
            creation_time: trail_clock,
        });
    }
}

fn remove_expired_trails(
    trails: &mut VecDeque<TrailPoint>,
    trail_clock: f32,
) {
    while let Some(oldest) = trails.front() {
        if trail_clock - oldest.creation_time < TRAIL_MAX_AGE_SECONDS {
            break;
        }

        trails.pop_front();
    }
}

fn draw_trails(
    trails: &VecDeque<TrailPoint>,
    screen_center: Vec2,
    zoom: f32,
    trail_clock: f32,
) {
    let width = screen_width();
    let height = screen_height();

    for trail in trails {
        let screen_x = screen_center.x + trail.x * zoom;
        let screen_y = screen_center.y + trail.y * zoom;
        // Convert the stored world-space radius into a screen-space width,
        // then cap that width so very massive objects do not leave huge trails.
        let raw_trail_width =
            trail.radius * zoom * 2.0 * TRAIL_WIDTH_SCALE;

        let displayed_trail_width = raw_trail_width.clamp(
            TRAIL_MIN_WIDTH_PIXELS,
            TRAIL_MAX_WIDTH_PIXELS,
        );

        let displayed_radius = displayed_trail_width * 0.5;

        if screen_x < -displayed_radius
            || screen_x > width + displayed_radius
            || screen_y < -displayed_radius
            || screen_y > height + displayed_radius
        {
            continue;
        }

        let age = trail_clock - trail.creation_time;
        let fade = 0.5_f32.powf(age / TRAIL_HALF_LIFE_SECONDS);
        let alpha = TRAIL_INITIAL_ALPHA * fade;
        let brightness = trail.brightness.min(1.0);

        draw_circle(
            screen_x,
            screen_y,
            displayed_radius,
            Color::new(
                brightness * 0.60,
                brightness * 0.75,
                brightness,
                alpha,
            ),
        );
    }
}

fn draw_particles(
    particles: &Particles,
    screen_center: Vec2,
    zoom: f32,
) {
    let width = screen_width();
    let height = screen_height();
    let total_system_mass = particles.total_mass().max(f32::EPSILON);

    for index in 0..particles.len() {
        let screen_x = screen_center.x + particles.x[index] * zoom;
        let screen_y = screen_center.y + particles.y[index] * zoom;

        let displayed_radius =
            (particle_radius(particles.mass[index]) * zoom).clamp(0.7, 50.0);

        let mass_fraction = particles.mass[index] / total_system_mass;
        let star_transition = dominant_body_transition(mass_fraction);

        // The fully transitioned body has a glow extending to 2.7 times
        // its radius, so include that area in the off-screen test.
        let visibility_radius =
            displayed_radius * (1.0 + 1.7 * star_transition);

        if screen_x < -visibility_radius
            || screen_x > width + visibility_radius
            || screen_y < -visibility_radius
            || screen_y > height + visibility_radius
        {
            continue;
        }

        let mass_brightness =
            (particles.mass[index] / MAX_PARTICLE_MASS)
                .sqrt()
                .clamp(0.0, 1.0);

        let brightness = particles.brightness[index];

        let normal_color = Color::new(
            (brightness * 0.65 + mass_brightness * 0.25).min(1.0),
            (brightness * 0.75 + mass_brightness * 0.20).min(1.0),
            brightness.min(1.0),
            0.95,
        );

        let central_body_color = Color::new(1.0, 0.72, 0.15, 1.0);
        let body_color = mix_color(
            normal_color,
            central_body_color,
            star_transition,
        );

        // These two layers gradually appear from 30% to 50% mass.
        // At 50%, their radii and colors match the original central body.
        if star_transition > 0.0 {
            draw_circle(
                screen_x,
                screen_y,
                displayed_radius * 2.7,
                Color::new(1.0, 0.45, 0.05, 0.035 * star_transition),
            );

            draw_circle(
                screen_x,
                screen_y,
                displayed_radius * 1.8,
                Color::new(1.0, 0.55, 0.08, 0.08 * star_transition),
            );
        }

        draw_circle(
            screen_x,
            screen_y,
            displayed_radius,
            body_color,
        );

        // The highlight also fades in gradually. At 50% mass this is
        // identical to the highlight used by the original central mass.
        if star_transition > 0.0 {
            draw_circle(
                screen_x - displayed_radius * 0.25,
                screen_y - displayed_radius * 0.25,
                displayed_radius * 0.55,
                Color::new(
                    1.0,
                    0.92,
                    0.55,
                    0.8 * star_transition,
                ),
            );
        }
    }
}

fn draw_interface(
    particles: &Particles,
    trails: &VecDeque<TrailPoint>,
    paused: bool,
    trails_enabled: bool,
    time_scale: f32,
    zoom: f32,
) {
    let simulation_state = if paused { "PAUSED" } else { "RUNNING" };
    let trail_state = if trails_enabled { "VISIBLE" } else { "HIDDEN" };
    let center_of_mass = particles.center_of_mass();

    let clockwise_percent = initial_clockwise_fraction() * 100.0;
    let counterclockwise_percent = 100.0 - clockwise_percent;
    let total_mass = particles.total_mass();
    let largest_mass = particles.largest_mass();
    let largest_mass_percent = if total_mass > 0.0 {
        largest_mass / total_mass * 100.0
    } else {
        0.0
    };

    let first_line = format!(
        "{} particles | {} | Initial rotation: {:.0}% CW / {:.0}% CCW | AVX2: 8 x f32 | FPS: {}",
        particles.len(),
        simulation_state,
        clockwise_percent,
        counterclockwise_percent,
        get_fps(),
    );

    let second_line = format!(
        "Total mass: {:.0} | Largest: {:.1} ({:.1}%) | COM: ({:.2}, {:.2}) | Time: {:.2}x | Zoom: {:.2}x | Trails: {} ({})",
        total_mass,
        largest_mass,
        largest_mass_percent,
        center_of_mass.x,
        center_of_mass.y,
        time_scale,
        zoom,
        trail_state,
        trails.len(),
    );

    draw_text(
        &first_line,
        20.0,
        30.0,
        22.0,
        Color::new(0.85, 0.9, 1.0, 1.0),
    );

    draw_text(
        &second_line,
        20.0,
        56.0,
        20.0,
        Color::new(0.7, 0.78, 0.9, 1.0),
    );

    draw_text(
        "Space: pause | R: reset | T: trails | Up/Down: time speed | Mouse wheel: zoom",
        20.0,
        screen_height() - 20.0,
        19.0,
        Color::new(0.6, 0.68, 0.8, 1.0),
    );
}

#[macroquad::main(window_configuration)]
async fn main() {
    assert!(
        YELLOW_TRANSITION_START_MASS_FRACTION >= 0.0
            && YELLOW_TRANSITION_START_MASS_FRACTION
                < CENTRAL_BODY_APPEARANCE_MASS_FRACTION
            && CENTRAL_BODY_APPEARANCE_MASS_FRACTION <= 1.0,
        "Mass appearance thresholds must satisfy 0 <= start < complete <= 1",
    );

    let mut particles = create_particles();

    let mut acceleration_x = Vec::with_capacity(PARTICLE_COUNT);
    let mut acceleration_y = Vec::with_capacity(PARTICLE_COUNT);
    let mut trails = VecDeque::<TrailPoint>::new();

    let initial_view_size = screen_width().min(screen_height());
    let mut zoom = initial_view_size / (SPAWN_RADIUS * 2.25);

    let mut paused = false;
    let mut trails_enabled = false;
    let mut time_scale = 1.0_f32;

    let mut accumulated_time = 0.0_f32;
    let mut trail_clock = 0.0_f32;
    let mut trail_sample_accumulator = 0.0_f32;

    loop {
        let frame_time = get_frame_time().min(0.1);

        if is_key_pressed(KeyCode::Space) {
            paused = !paused;
        }

        if is_key_pressed(KeyCode::T) {
            trails_enabled = !trails_enabled;
        }

        if is_key_pressed(KeyCode::R) {
            particles = create_particles();
            acceleration_x.clear();
            acceleration_y.clear();
            trails.clear();
            accumulated_time = 0.0;
            trail_clock = 0.0;
            trail_sample_accumulator = 0.0;
        }

        if is_key_pressed(KeyCode::Up) {
            time_scale = (time_scale * 2.0).min(4.0);
        }

        if is_key_pressed(KeyCode::Down) {
            time_scale = (time_scale * 0.5).max(0.125);
        }

        let (_, wheel_y) = mouse_wheel();

        if wheel_y != 0.0 {
            zoom *= 1.03_f32.powf(wheel_y.clamp(-1.0, 1.0));
            zoom = zoom.clamp(0.05, 5.0);
        }

        if !paused {
            accumulated_time += frame_time.min(0.05) * time_scale;
            let mut completed_steps = 0;

            while accumulated_time >= PHYSICS_DT
                && completed_steps < MAX_PHYSICS_STEPS_PER_FRAME
            {
                update_system_simd(
                    &mut particles,
                    &mut acceleration_x,
                    &mut acceleration_y,
                    PHYSICS_DT,
                );

                accumulated_time -= PHYSICS_DT;
                completed_steps += 1;
            }

            if completed_steps == MAX_PHYSICS_STEPS_PER_FRAME {
                accumulated_time = 0.0;
            }

            trail_clock += frame_time;
            remove_expired_trails(&mut trails, trail_clock);

            trail_sample_accumulator += frame_time;

            while trail_sample_accumulator >= TRAIL_SAMPLE_INTERVAL_SECONDS {
                add_trail_points(&mut trails, &particles, trail_clock);
                trail_sample_accumulator -= TRAIL_SAMPLE_INTERVAL_SECONDS;
            }
        }

        clear_background(BACKGROUND_COLOR);

        let screen_center = vec2(
            screen_width() * 0.5,
            screen_height() * 0.5,
        );

        if trails_enabled {
            draw_trails(&trails, screen_center, zoom, trail_clock);
        }

        draw_particles(&particles, screen_center, zoom);

        draw_interface(
            &particles,
            &trails,
            paused,
            trails_enabled,
            time_scale,
            zoom,
        );

        next_frame().await;
    }
}