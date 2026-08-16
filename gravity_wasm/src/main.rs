use macroquad::prelude::*;
use macroquad::rand::gen_range;
use std::collections::VecDeque;
use std::f32::consts::TAU;

// Reduced particle count for browser/WASM performance.
// AVX2 version used 10_000; WASM scalar runs best with fewer particles.
const PARTICLE_COUNT: usize = 5_000;

// Arbitrary simulation units.
const GRAVITATIONAL_CONSTANT: f32 = 15.0;

const MIN_PARTICLE_MASS: f32 = 5.0;
const MAX_PARTICLE_MASS: f32 = 20.0;
const PARTICLE_RADIUS_SCALE: f32 = 0.45;

// Particles begin inside this disk. There is no initial central body.
const SPAWN_RADIUS: f32 = 1200.0;

// Initial cloud concentration.
const CENTER_CONCENTRATION_STRENGTH: f32 = 0.75;
const CENTER_CONCENTRATION_EXPONENT: f32 = 1.40;

// Relative initial rotation weights.
const CLOCKWISE_ROTATION_WEIGHT: f32 = 1.0;
const COUNTERCLOCKWISE_ROTATION_WEIGHT: f32 = 0.00;

// Initial tangential-speed range.
const INITIAL_TANGENTIAL_SPEED_MIN: f32 = 6.0;
const INITIAL_TANGENTIAL_SPEED_MAX: f32 = 20.0;
const INITIAL_TANGENTIAL_SPEED_RADIUS_EXPONENT: f32 = 0.50;
const INITIAL_RADIAL_SPEED_MAX: f32 = 0.01;

// Prevents singular acceleration at extremely short distances.
const GRAVITY_SOFTENING: f32 = 3.0;

const PHYSICS_DT: f32 = 1.0 / 120.0;
const MAX_PHYSICS_STEPS_PER_FRAME: usize = 4;

// Trail settings.
const TRAIL_SAMPLE_INTERVAL_SECONDS: f32 = 1.0 / 45.0;
const TRAIL_HALF_LIFE_SECONDS: f32 = 10.0;
const TRAIL_MAX_AGE_SECONDS: f32 = 40.0;
const TRAIL_INITIAL_ALPHA: f32 = 0.35;

const TRAIL_MIN_WIDTH_PIXELS: f32 = 1.0;
const TRAIL_MAX_WIDTH_PIXELS: f32 = 2.0;
const TRAIL_WIDTH_SCALE: f32 = 1.0;

const YELLOW_TRANSITION_START_MASS_FRACTION: f32 = 0.25;
const CENTRAL_BODY_APPEARANCE_MASS_FRACTION: f32 = 0.45;

const BACKGROUND_COLOR: Color =
    Color::new(0.005, 0.008, 0.025, 1.0);

#[derive(Clone, Copy)]
struct Particle {
    position: Vec2,
    velocity: Vec2,
    mass: f32,
    brightness: f32,
}

struct TrailPoint {
    position: Vec2,
    radius: f32,
    brightness: f32,
    creation_time: f32,
}

fn window_configuration() -> Conf {
    Conf {
        window_title: "Gravity Simulation (WASM)".to_owned(),
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

fn total_mass(particles: &[Particle]) -> f32 {
    particles.iter().map(|p| p.mass).sum()
}

fn largest_mass(particles: &[Particle]) -> f32 {
    particles.iter().map(|p| p.mass).fold(0.0_f32, f32::max)
}

fn center_of_mass(particles: &[Particle]) -> Vec2 {
    let system_mass = total_mass(particles);
    if system_mass <= f32::EPSILON {
        return Vec2::ZERO;
    }
    let weighted_position = particles
        .iter()
        .fold(Vec2::ZERO, |sum, p| sum + p.position * p.mass);
    weighted_position / system_mass
}

fn center_reference_frame(particles: &mut [Particle]) {
    let system_mass = total_mass(particles);
    if system_mass <= f32::EPSILON {
        return;
    }
    let mut weighted_position = Vec2::ZERO;
    let mut total_momentum = Vec2::ZERO;
    for particle in particles.iter() {
        weighted_position += particle.position * particle.mass;
        total_momentum += particle.velocity * particle.mass;
    }
    let center = weighted_position / system_mass;
    let bulk_velocity = total_momentum / system_mass;
    for particle in particles.iter_mut() {
        particle.position -= center;
        particle.velocity -= bulk_velocity;
    }
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
    let linear_progress = ((mass_fraction
        - YELLOW_TRANSITION_START_MASS_FRACTION)
        / transition_range)
        .clamp(0.0, 1.0);
    linear_progress * linear_progress * (3.0 - 2.0 * linear_progress)
}

fn sample_spawn_distance() -> f32 {
    let random_fraction: f32 = gen_range(0.0_f32, 1.0_f32);
    let uniform_area_radius = random_fraction.sqrt();
    let center_biased_radius =
        random_fraction.powf(CENTER_CONCENTRATION_EXPONENT);
    let strength = CENTER_CONCENTRATION_STRENGTH.clamp(0.0, 1.0);
    let normalized_radius = uniform_area_radius
        + (center_biased_radius - uniform_area_radius) * strength;
    normalized_radius * SPAWN_RADIUS
}

fn initial_tangential_speed(distance: f32) -> f32 {
    let normalized_radius = (distance / SPAWN_RADIUS).clamp(0.0, 1.0);
    let radial_speed_factor =
        normalized_radius.powf(INITIAL_TANGENTIAL_SPEED_RADIUS_EXPONENT);
    INITIAL_TANGENTIAL_SPEED_MIN
        + (INITIAL_TANGENTIAL_SPEED_MAX - INITIAL_TANGENTIAL_SPEED_MIN)
            * radial_speed_factor
}

fn spawn_particle(clockwise: bool) -> Particle {
    let position_angle: f32 = gen_range(0.0_f32, TAU);
    let distance = sample_spawn_distance();
    let radial_direction = vec2(position_angle.cos(), position_angle.sin());
    let tangential_direction = if clockwise {
        vec2(-radial_direction.y, radial_direction.x)
    } else {
        vec2(radial_direction.y, -radial_direction.x)
    };
    let tangential_speed = initial_tangential_speed(distance);
    let radial_speed: f32 =
        gen_range(-INITIAL_RADIAL_SPEED_MAX, INITIAL_RADIAL_SPEED_MAX);
    Particle {
        position: radial_direction * distance,
        velocity: tangential_direction * tangential_speed
            + radial_direction * radial_speed,
        mass: gen_range(MIN_PARTICLE_MASS, MAX_PARTICLE_MASS),
        brightness: gen_range(0.55_f32, 1.0_f32),
    }
}

fn initial_clockwise_fraction() -> f32 {
    let total_weight =
        CLOCKWISE_ROTATION_WEIGHT + COUNTERCLOCKWISE_ROTATION_WEIGHT;
    CLOCKWISE_ROTATION_WEIGHT / total_weight
}

fn create_particles() -> Vec<Particle> {
    let mut particles = Vec::with_capacity(PARTICLE_COUNT);
    let clockwise_count =
        ((PARTICLE_COUNT as f32) * initial_clockwise_fraction()).round()
            as usize;
    for index in 0..PARTICLE_COUNT {
        particles.push(spawn_particle(index < clockwise_count));
    }
    center_reference_frame(&mut particles);
    particles
}

fn calculate_accelerations(
    particles: &[Particle],
    accelerations: &mut Vec<Vec2>,
) {
    accelerations.resize(particles.len(), Vec2::ZERO);
    accelerations.fill(Vec2::ZERO);
    let softening_squared = GRAVITY_SOFTENING * GRAVITY_SOFTENING;
    for first_index in 0..particles.len() {
        for second_index in (first_index + 1)..particles.len() {
            let first = particles[first_index];
            let second = particles[second_index];
            let displacement = second.position - first.position;
            let softened_distance_squared =
                displacement.length_squared() + softening_squared;
            let inverse_distance =
                1.0 / softened_distance_squared.sqrt();
            let inverse_distance_cubed =
                inverse_distance * inverse_distance * inverse_distance;
            let common_factor =
                GRAVITATIONAL_CONSTANT * inverse_distance_cubed;
            accelerations[first_index] +=
                displacement * common_factor * second.mass;
            accelerations[second_index] -=
                displacement * common_factor * first.mass;
        }
    }
}

fn integrate_positions(
    particles: &mut [Particle],
    accelerations: &[Vec2],
    dt: f32,
) {
    for (particle, acceleration) in
        particles.iter_mut().zip(accelerations.iter())
    {
        particle.velocity += *acceleration * (dt * 0.5);
        particle.position += particle.velocity * dt;
    }
}

fn finish_velocity_update(
    particles: &mut [Particle],
    accelerations: &[Vec2],
    dt: f32,
) {
    for (particle, acceleration) in
        particles.iter_mut().zip(accelerations.iter())
    {
        particle.velocity += *acceleration * (dt * 0.5);
    }
}

fn find_collision(
    particles: &[Particle],
    first_index: usize,
) -> Option<usize> {
    let first = particles[first_index];
    let first_radius = particle_radius(first.mass);
    for second_index in (first_index + 1)..particles.len() {
        let second = particles[second_index];
        let collision_distance = first_radius + particle_radius(second.mass);
        if second.position.distance_squared(first.position)
            <= collision_distance * collision_distance
        {
            return Some(second_index);
        }
    }
    None
}

fn merge_pair(
    particles: &mut Vec<Particle>,
    first_index: usize,
    second_index: usize,
) {
    let first = particles[first_index];
    let second = particles[second_index];
    let combined_mass = first.mass + second.mass;
    let combined_position =
        (first.position * first.mass + second.position * second.mass)
            / combined_mass;
    let combined_velocity =
        (first.velocity * first.mass + second.velocity * second.mass)
            / combined_mass;
    let combined_brightness =
        (first.brightness * first.mass + second.brightness * second.mass)
            / combined_mass;
    particles[first_index] = Particle {
        position: combined_position,
        velocity: combined_velocity,
        mass: combined_mass,
        brightness: combined_brightness,
    };
    particles.swap_remove(second_index);
}

fn merge_colliding_particles(particles: &mut Vec<Particle>) {
    let mut first_index = 0;
    while first_index < particles.len() {
        while let Some(second_index) =
            find_collision(particles, first_index)
        {
            merge_pair(particles, first_index, second_index);
        }
        first_index += 1;
    }
}

fn update_system(
    particles: &mut Vec<Particle>,
    accelerations: &mut Vec<Vec2>,
    dt: f32,
) {
    if particles.is_empty() {
        return;
    }
    calculate_accelerations(particles, accelerations);
    integrate_positions(particles, accelerations, dt);
    merge_colliding_particles(particles);
    if particles.is_empty() {
        return;
    }
    calculate_accelerations(particles, accelerations);
    finish_velocity_update(particles, accelerations, dt);
}

fn add_trail_points(
    trails: &mut VecDeque<TrailPoint>,
    particles: &[Particle],
    trail_clock: f32,
) {
    trails.reserve(particles.len());
    for particle in particles {
        trails.push_back(TrailPoint {
            position: particle.position,
            radius: particle_radius(particle.mass),
            brightness: particle.brightness,
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
        let screen_position = screen_center + trail.position * zoom;
        let raw_trail_width =
            trail.radius * zoom * 2.0 * TRAIL_WIDTH_SCALE;
        let displayed_trail_width =
            raw_trail_width.clamp(TRAIL_MIN_WIDTH_PIXELS, TRAIL_MAX_WIDTH_PIXELS);
        let displayed_radius = displayed_trail_width * 0.5;
        if screen_position.x < -displayed_radius
            || screen_position.x > width + displayed_radius
            || screen_position.y < -displayed_radius
            || screen_position.y > height + displayed_radius
        {
            continue;
        }
        let age = trail_clock - trail.creation_time;
        let fade = 0.5_f32.powf(age / TRAIL_HALF_LIFE_SECONDS);
        let alpha = TRAIL_INITIAL_ALPHA * fade;
        let brightness = trail.brightness.min(1.0);
        draw_circle(
            screen_position.x,
            screen_position.y,
            displayed_radius,
            Color::new(brightness * 0.60, brightness * 0.75, brightness, alpha),
        );
    }
}

fn draw_particles(
    particles: &[Particle],
    screen_center: Vec2,
    zoom: f32,
) {
    let width = screen_width();
    let height = screen_height();
    let system_mass = total_mass(particles).max(f32::EPSILON);
    for particle in particles {
        let screen_position = screen_center + particle.position * zoom;
        let displayed_radius =
            (particle_radius(particle.mass) * zoom).clamp(0.7, 50.0);
        let mass_fraction = particle.mass / system_mass;
        let star_transition = dominant_body_transition(mass_fraction);
        let visibility_radius =
            displayed_radius * (1.0 + 1.7 * star_transition);
        if screen_position.x < -visibility_radius
            || screen_position.x > width + visibility_radius
            || screen_position.y < -visibility_radius
            || screen_position.y > height + visibility_radius
        {
            continue;
        }
        let mass_brightness =
            (particle.mass / MAX_PARTICLE_MASS).sqrt().clamp(0.0, 1.0);
        let normal_color = Color::new(
            (particle.brightness * 0.65 + mass_brightness * 0.25).min(1.0),
            (particle.brightness * 0.75 + mass_brightness * 0.20).min(1.0),
            particle.brightness.min(1.0),
            0.95,
        );
        let central_body_color = Color::new(1.0, 0.72, 0.15, 1.0);
        let body_color = mix_color(normal_color, central_body_color, star_transition);
        if star_transition > 0.0 {
            draw_circle(
                screen_position.x,
                screen_position.y,
                displayed_radius * 2.7,
                Color::new(1.0, 0.45, 0.05, 0.035 * star_transition),
            );
            draw_circle(
                screen_position.x,
                screen_position.y,
                displayed_radius * 1.8,
                Color::new(1.0, 0.55, 0.08, 0.08 * star_transition),
            );
        }
        draw_circle(
            screen_position.x,
            screen_position.y,
            displayed_radius,
            body_color,
        );
        if star_transition > 0.0 {
            draw_circle(
                screen_position.x - displayed_radius * 0.25,
                screen_position.y - displayed_radius * 0.25,
                displayed_radius * 0.55,
                Color::new(1.0, 0.92, 0.55, 0.8 * star_transition),
            );
        }
    }
}

fn draw_interface(
    particles: &[Particle],
    trails: &VecDeque<TrailPoint>,
    paused: bool,
    trails_enabled: bool,
    time_scale: f32,
    zoom: f32,
) {
    let simulation_state = if paused { "PAUSED" } else { "RUNNING" };
    let trail_state = if trails_enabled { "VISIBLE" } else { "HIDDEN" };
    let clockwise_percent = initial_clockwise_fraction() * 100.0;
    let counterclockwise_percent = 100.0 - clockwise_percent;
    let system_mass = total_mass(particles);
    let maximum_mass = largest_mass(particles);
    let largest_mass_percent = if system_mass > f32::EPSILON {
        maximum_mass / system_mass * 100.0
    } else {
        0.0
    };
    let com = center_of_mass(particles);
    let first_line = format!(
        "{} particles | {} | Rotation: {:.0}% CW / {:.0}% CCW | Scalar (WASM) | FPS: {}",
        particles.len(),
        simulation_state,
        clockwise_percent,
        counterclockwise_percent,
        get_fps(),
    );
    let second_line = format!(
        "Total mass: {:.0} | Largest: {:.1} ({:.1}%) | COM: ({:.2}, {:.2}) | Time: {:.2}x | Zoom: {:.2}x | Trails: {} ({})",
        system_mass,
        maximum_mass,
        largest_mass_percent,
        com.x,
        com.y,
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
    let mut particles = create_particles();
    let mut accelerations = Vec::<Vec2>::with_capacity(PARTICLE_COUNT);
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
            accelerations.clear();
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
                update_system(&mut particles, &mut accelerations, PHYSICS_DT);
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
        let screen_center = vec2(screen_width() * 0.5, screen_height() * 0.5);
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
