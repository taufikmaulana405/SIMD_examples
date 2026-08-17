mod collision;
mod cuda;
mod error;
#[cfg(test)]
mod reference;
mod types;

use cuda::CudaBackend;
use macroquad::prelude::*;
use std::collections::VecDeque;
use types::{Particle, SimParams};

const PARTICLE_COUNT: usize = 10_000;
const GRAVITY: f32 = 15.0;
const DT: f32 = 1.0 / 120.0;
const SOFTENING_SQUARED: f32 = 9.0;
const SPAWN_RADIUS: f32 = 1200.0;
const MIN_PARTICLE_MASS: f32 = 5.0;
const MAX_PARTICLE_MASS: f32 = 20.0;
const PARTICLE_RADIUS_SCALE: f32 = 0.45;
const MAX_PHYSICS_STEPS_PER_FRAME: u32 = 8;
const TIME_SCALE_MIN: f32 = 0.125;
const TIME_SCALE_MAX: f32 = 4.0;
const ZOOM_MIN: f32 = 0.05;
const ZOOM_MAX: f32 = 5.0;
const CENTER_CONCENTRATION_STRENGTH: f32 = 0.75;
const CENTER_CONCENTRATION_EXPONENT: f32 = 1.40;
const INITIAL_RADIAL_SPEED_MAX: f32 = 0.01;
const CLOCKWISE_ROTATION_WEIGHT: f32 = 1.0;
const COUNTERCLOCKWISE_ROTATION_WEIGHT: f32 = 0.0;
const INITIAL_TANGENTIAL_SPEED_MIN: f32 = 6.0;
const INITIAL_TANGENTIAL_SPEED_MAX: f32 = 20.0;
const INITIAL_TANGENTIAL_SPEED_RADIUS_EXPONENT: f32 = 0.50;

const TRAIL_SAMPLE_INTERVAL_SECONDS: f32 = 1.0 / 45.0;
const TRAIL_HALF_LIFE_SECONDS: f32 = 10.0;
const TRAIL_MAX_AGE_SECONDS: f32 = 40.0;
const TRAIL_INITIAL_ALPHA: f32 = 0.35;
const TRAIL_MIN_WIDTH_PIXELS: f32 = 1.0;
const TRAIL_MAX_WIDTH_PIXELS: f32 = 2.0;
const TRAIL_MAX_SAMPLES: usize = 90;
const TRAIL_CAPACITY: usize = PARTICLE_COUNT * TRAIL_MAX_SAMPLES;
const YELLOW_TRANSITION_START_MASS_FRACTION: f32 = 0.25;
const CENTRAL_BODY_APPEARANCE_MASS_FRACTION: f32 = 0.45;

#[derive(Clone, Copy)]
struct TrailPoint {
    x: f32,
    y: f32,
    radius: f32,
    brightness: f32,
    creation_time: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct SnapshotStats {
    total_mass: f32,
    largest_mass: f32,
    center_of_mass: Vec2,
}

fn sample_spawn_distance<R: ::rand::Rng + ?Sized>(rng: &mut R) -> f32 {
    let random_fraction = rng.r#gen::<f32>();
    let uniform_area_radius = random_fraction.sqrt();
    let center_biased_radius = random_fraction.powf(CENTER_CONCENTRATION_EXPONENT);
    let normalized_radius = uniform_area_radius
        + (center_biased_radius - uniform_area_radius) * CENTER_CONCENTRATION_STRENGTH;
    normalized_radius * SPAWN_RADIUS
}

fn initial_clockwise(index: usize, count: usize) -> bool {
    let total_weight = CLOCKWISE_ROTATION_WEIGHT + COUNTERCLOCKWISE_ROTATION_WEIGHT;
    let fraction = CLOCKWISE_ROTATION_WEIGHT / total_weight;
    index < (count as f32 * fraction).round() as usize
}

fn initial_tangential_speed(distance: f32) -> f32 {
    let normalized_radius = (distance / SPAWN_RADIUS).clamp(0.0, 1.0);
    let radial_factor = normalized_radius.powf(INITIAL_TANGENTIAL_SPEED_RADIUS_EXPONENT);
    INITIAL_TANGENTIAL_SPEED_MIN
        + (INITIAL_TANGENTIAL_SPEED_MAX - INITIAL_TANGENTIAL_SPEED_MIN) * radial_factor
}

fn make_particle<R: ::rand::Rng + ?Sized>(rng: &mut R, index: usize) -> Particle {
    let angle = rng.gen_range(0.0..std::f32::consts::TAU);
    let distance = sample_spawn_distance(rng);
    let (rx, ry) = (angle.cos(), angle.sin());
    let (tx, ty) = if initial_clockwise(index, PARTICLE_COUNT) {
        (-ry, rx)
    } else {
        (ry, -rx)
    };
    let tangential_speed = initial_tangential_speed(distance);
    let radial_speed = rng.gen_range(-INITIAL_RADIAL_SPEED_MAX..INITIAL_RADIAL_SPEED_MAX);
    let mass = rng.gen_range(MIN_PARTICLE_MASS..MAX_PARTICLE_MASS);
    Particle {
        x: rx * distance,
        y: ry * distance,
        vx: tx * tangential_speed + rx * radial_speed,
        vy: ty * tangential_speed + ry * radial_speed,
        mass,
        brightness: rng.gen_range(0.55..1.0),
        alive: 1,
        padding: 0,
    }
}

fn center_reference_frame(particles: &mut [Particle]) {
    let mut total_mass = 0.0;
    let mut center = Vec2::ZERO;
    let mut momentum = Vec2::ZERO;
    for particle in particles.iter().filter(|particle| particle.alive != 0) {
        total_mass += particle.mass;
        center += vec2(particle.x, particle.y) * particle.mass;
        momentum += vec2(particle.vx, particle.vy) * particle.mass;
    }
    if total_mass <= f32::EPSILON {
        return;
    }
    center /= total_mass;
    momentum /= total_mass;
    for particle in particles.iter_mut().filter(|particle| particle.alive != 0) {
        particle.x -= center.x;
        particle.y -= center.y;
        particle.vx -= momentum.x;
        particle.vy -= momentum.y;
    }
}

fn create_particles() -> Vec<Particle> {
    let mut rng = ::rand::thread_rng();
    let mut particles = (0..PARTICLE_COUNT)
        .map(|index| make_particle(&mut rng, index))
        .collect::<Vec<_>>();
    center_reference_frame(&mut particles);
    particles
}

fn window_conf() -> Conf {
    Conf {
        window_title: "CUDA Gravity Simulation".to_owned(),
        window_width: 1280,
        window_height: 800,
        window_resizable: true,
        high_dpi: true,
        ..Default::default()
    }
}

#[inline]
fn particle_radius(mass: f32) -> f32 {
    PARTICLE_RADIUS_SCALE * mass.max(0.0).sqrt()
}

#[inline]
fn dominant_body_transition(mass_fraction: f32) -> f32 {
    let range = CENTRAL_BODY_APPEARANCE_MASS_FRACTION - YELLOW_TRANSITION_START_MASS_FRACTION;
    if range <= 0.0 {
        return if mass_fraction >= CENTRAL_BODY_APPEARANCE_MASS_FRACTION {
            1.0
        } else {
            0.0
        };
    }
    let progress =
        ((mass_fraction - YELLOW_TRANSITION_START_MASS_FRACTION) / range).clamp(0.0, 1.0);
    progress * progress * (3.0 - 2.0 * progress)
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

fn snapshot_stats(particles: &[Particle]) -> SnapshotStats {
    let mut stats = SnapshotStats::default();
    for particle in particles.iter().filter(|particle| particle.alive != 0) {
        stats.total_mass += particle.mass;
        stats.largest_mass = stats.largest_mass.max(particle.mass);
        stats.center_of_mass += vec2(particle.x, particle.y) * particle.mass;
    }
    if stats.total_mass > f32::EPSILON {
        stats.center_of_mass /= stats.total_mass;
    } else {
        stats.center_of_mass = Vec2::ZERO;
    }
    stats
}

fn add_trail_points(trails: &mut VecDeque<TrailPoint>, particles: &[Particle], trail_clock: f32) {
    for particle in particles.iter().filter(|particle| particle.alive != 0) {
        trails.push_back(TrailPoint {
            x: particle.x,
            y: particle.y,
            radius: particle_radius(particle.mass),
            brightness: particle.brightness,
            creation_time: trail_clock,
        });
    }
    while trails.len() > TRAIL_CAPACITY {
        trails.pop_front();
    }
}

fn remove_expired_trails(trails: &mut VecDeque<TrailPoint>, trail_clock: f32) {
    while let Some(oldest) = trails.front() {
        if trail_clock - oldest.creation_time < TRAIL_MAX_AGE_SECONDS {
            break;
        }
        trails.pop_front();
    }
}

fn draw_trails(trails: &VecDeque<TrailPoint>, screen_center: Vec2, zoom: f32, trail_clock: f32) {
    for trail in trails {
        let screen_position = screen_center + vec2(trail.x, trail.y) * zoom;
        let width =
            (trail.radius * zoom * 2.0).clamp(TRAIL_MIN_WIDTH_PIXELS, TRAIL_MAX_WIDTH_PIXELS);
        let radius = width * 0.5;
        if screen_position.x < -radius
            || screen_position.x > screen_width() + radius
            || screen_position.y < -radius
            || screen_position.y > screen_height() + radius
        {
            continue;
        }
        let age = (trail_clock - trail.creation_time).max(0.0);
        let alpha = TRAIL_INITIAL_ALPHA * 0.5_f32.powf(age / TRAIL_HALF_LIFE_SECONDS);
        let brightness = trail.brightness.min(1.0);
        draw_circle(
            screen_position.x,
            screen_position.y,
            radius,
            Color::new(brightness * 0.60, brightness * 0.75, brightness, alpha),
        );
    }
}

fn draw_particles(particles: &[Particle], zoom: f32, stats: SnapshotStats) {
    let center = vec2(screen_width() * 0.5, screen_height() * 0.5);
    let total_mass = stats.total_mass.max(f32::EPSILON);
    for particle in particles.iter().filter(|particle| particle.alive != 0) {
        let position = center + vec2(particle.x, particle.y) * zoom;
        let radius = (particle_radius(particle.mass) * zoom).clamp(0.7, 50.0);
        let transition = dominant_body_transition(particle.mass / total_mass);
        let visibility_radius = radius * (1.0 + 1.7 * transition);
        if position.x < -visibility_radius
            || position.x > screen_width() + visibility_radius
            || position.y < -visibility_radius
            || position.y > screen_height() + visibility_radius
        {
            continue;
        }
        let mass_brightness = (particle.mass / MAX_PARTICLE_MASS).sqrt().clamp(0.0, 1.0);
        let brightness = particle.brightness;
        let normal_color = Color::new(
            (brightness * 0.65 + mass_brightness * 0.25).min(1.0),
            (brightness * 0.75 + mass_brightness * 0.20).min(1.0),
            brightness.min(1.0),
            0.95,
        );
        let body_color = mix_color(normal_color, Color::new(1.0, 0.72, 0.15, 1.0), transition);
        if transition > 0.0 {
            draw_circle(
                position.x,
                position.y,
                radius * 2.7,
                Color::new(1.0, 0.45, 0.05, 0.035 * transition),
            );
            draw_circle(
                position.x,
                position.y,
                radius * 1.8,
                Color::new(1.0, 0.55, 0.08, 0.08 * transition),
            );
        }
        draw_circle(position.x, position.y, radius, body_color);
        if transition > 0.0 {
            draw_circle(
                position.x - radius * 0.25,
                position.y - radius * 0.25,
                radius * 0.55,
                Color::new(1.0, 0.92, 0.55, 0.8 * transition),
            );
        }
    }
}

fn draw_interface(
    particles: &[Particle],
    stats: SnapshotStats,
    trails: &VecDeque<TrailPoint>,
    paused: bool,
    trails_visible: bool,
    time_scale: f32,
    zoom: f32,
    fps: f32,
    runtime_error: Option<&str>,
) {
    let state = if paused { "PAUSED" } else { "RUNNING" };
    let trail_state = if trails_visible { "VISIBLE" } else { "HIDDEN" };
    let total_mass = stats.total_mass;
    let largest_fraction = if total_mass > f32::EPSILON {
        stats.largest_mass / total_mass * 100.0
    } else {
        0.0
    };
    let clockwise_percent = CLOCKWISE_ROTATION_WEIGHT
        / (CLOCKWISE_ROTATION_WEIGHT + COUNTERCLOCKWISE_ROTATION_WEIGHT)
        * 100.0;
    let first_line = format!(
        "{} particles | {} | Initial rotation: {:.0}% CW / {:.0}% CCW | FPS: {:.0}",
        particles
            .iter()
            .filter(|particle| particle.alive != 0)
            .count(),
        state,
        clockwise_percent,
        100.0 - clockwise_percent,
        fps
    );
    let second_line = format!(
        "Total mass: {:.0} | Largest: {:.1} ({:.1}%) | COM: ({:.2}, {:.2}) | Time: {:.2}x | Zoom: {:.2}x | Trails: {} ({})",
        total_mass,
        stats.largest_mass,
        largest_fraction,
        stats.center_of_mass.x,
        stats.center_of_mass.y,
        time_scale,
        zoom,
        trail_state,
        trails.len()
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
    if let Some(error) = runtime_error {
        draw_rectangle(
            20.0,
            75.0,
            screen_width() - 40.0,
            70.0,
            Color::new(0.3, 0.02, 0.02, 0.92),
        );
        draw_text(
            "CUDA runtime error (simulation paused)",
            35.0,
            103.0,
            22.0,
            RED,
        );
        draw_text(error, 35.0, 130.0, 16.0, WHITE);
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let initial_particles = create_particles();
    let mut backend = match CudaBackend::new(&initial_particles) {
        Ok(backend) => backend,
        Err(error) => {
            let message = format!("CUDA initialization failed: {error}");
            loop {
                clear_background(BLACK);
                draw_text("CUDA device unavailable", 40.0, 70.0, 32.0, RED);
                draw_text(&message, 40.0, 115.0, 18.0, WHITE);
                draw_text(
                    "Install an NVIDIA driver/GPU, then restart. Press Escape to quit.",
                    40.0,
                    150.0,
                    22.0,
                    WHITE,
                );
                if is_key_pressed(KeyCode::Escape) {
                    return;
                }
                next_frame().await;
            }
        }
    };
    let mut paused = false;
    let mut trails_visible = false;
    let mut zoom = screen_width().min(screen_height()) / (SPAWN_RADIUS * 2.25);
    let mut time_scale: f32 = 1.0;
    let mut accumulator = 0.0;
    let mut last_time = get_time();
    let mut trail_clock = 0.0;
    let mut trail_sample_accumulator = 0.0;
    let mut trails = VecDeque::new();
    let mut last_snapshot = initial_particles;
    let mut runtime_error: Option<String> = None;
    let mut fps_elapsed = 0.0;
    let mut fps_frames = 0_u32;
    let mut fps = 0.0;

    loop {
        if is_key_pressed(KeyCode::Space) {
            paused = !paused;
        }
        if is_key_pressed(KeyCode::T) {
            trails_visible = !trails_visible;
        }
        if is_key_pressed(KeyCode::R) {
            let reset_particles = create_particles();
            match backend.reset(&reset_particles) {
                Ok(()) => {
                    last_snapshot = reset_particles;
                    accumulator = 0.0;
                    last_time = get_time();
                    trail_clock = 0.0;
                    trail_sample_accumulator = 0.0;
                    trails.clear();
                    runtime_error = None;
                }
                Err(error) => {
                    paused = true;
                    runtime_error = Some(format!("Reset failed: {error}"));
                }
            }
        }
        if is_key_pressed(KeyCode::Up) {
            time_scale = (time_scale * 2.0).min(TIME_SCALE_MAX);
        }
        if is_key_pressed(KeyCode::Down) {
            time_scale = (time_scale * 0.5).max(TIME_SCALE_MIN);
        }
        let (_, wheel_y) = mouse_wheel();
        if wheel_y != 0.0 {
            zoom *= 1.03_f32.powf(wheel_y.clamp(-1.0, 1.0));
        }
        zoom = zoom.clamp(ZOOM_MIN, ZOOM_MAX);

        let now = get_time();
        let frame_time = (now - last_time).min(0.1) as f32;
        last_time = now;
        fps_elapsed += frame_time;
        fps_frames += 1;
        if fps_elapsed >= 0.5 {
            fps = fps_frames as f32 / fps_elapsed.max(f32::EPSILON);
            fps_elapsed = 0.0;
            fps_frames = 0;
        }

        if !paused {
            accumulator += frame_time.min(0.05) * time_scale;
            let mut completed_steps = 0;
            while accumulator >= DT && completed_steps < MAX_PHYSICS_STEPS_PER_FRAME {
                let params = SimParams {
                    count: backend.active_count() as u32,
                    dt: DT,
                    gravity: GRAVITY,
                    softening_squared: SOFTENING_SQUARED,
                };
                if let Err(error) = backend.step(params) {
                    paused = true;
                    runtime_error = Some(format!("Physics step failed: {error}"));
                    break;
                }
                accumulator -= DT;
                completed_steps += 1;
            }
            if completed_steps == MAX_PHYSICS_STEPS_PER_FRAME {
                accumulator = 0.0;
            }
            trail_clock += frame_time;
            trail_sample_accumulator += frame_time;
            remove_expired_trails(&mut trails, trail_clock);
        }

        match backend.snapshot() {
            Ok(snapshot) => {
                last_snapshot = snapshot;
                if runtime_error.is_some() && !paused {
                    runtime_error = None;
                }
            }
            Err(error) => {
                paused = true;
                runtime_error = Some(format!("Particle readback failed: {error}"));
            }
        }
        if !paused {
            while trail_sample_accumulator >= TRAIL_SAMPLE_INTERVAL_SECONDS {
                add_trail_points(&mut trails, &last_snapshot, trail_clock);
                trail_sample_accumulator -= TRAIL_SAMPLE_INTERVAL_SECONDS;
            }
        }

        let stats = snapshot_stats(&last_snapshot);
        clear_background(Color::new(0.005, 0.008, 0.025, 1.0));
        let screen_center = vec2(screen_width() * 0.5, screen_height() * 0.5);
        if trails_visible {
            draw_trails(&trails, screen_center, zoom, trail_clock);
        }
        draw_particles(&last_snapshot, zoom, stats);
        draw_interface(
            &last_snapshot,
            stats,
            &trails,
            paused,
            trails_visible,
            time_scale,
            zoom,
            fps,
            runtime_error.as_deref(),
        );
        next_frame().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dominant_transition_has_expected_bounds() {
        assert_eq!(dominant_body_transition(0.1), 0.0);
        assert_eq!(dominant_body_transition(0.45), 1.0);
        assert!(dominant_body_transition(0.35) > 0.0);
        assert!(dominant_body_transition(0.35) < 1.0);
    }

    #[test]
    fn snapshot_stats_ignore_inactive_particles() {
        let particles = vec![
            Particle {
                x: 2.0,
                y: 0.0,
                vx: 0.0,
                vy: 0.0,
                mass: 2.0,
                brightness: 1.0,
                alive: 1,
                padding: 0,
            },
            Particle {
                x: 100.0,
                y: 0.0,
                vx: 0.0,
                vy: 0.0,
                mass: 100.0,
                brightness: 1.0,
                alive: 0,
                padding: 0,
            },
        ];
        let stats = snapshot_stats(&particles);
        assert_eq!(stats.total_mass, 2.0);
        assert_eq!(stats.largest_mass, 2.0);
        assert_eq!(stats.center_of_mass, vec2(2.0, 0.0));
    }
}
