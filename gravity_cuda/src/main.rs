mod collision;
mod cuda;
mod error;
mod types;

use cuda::CudaBackend;
use macroquad::prelude::*;
use types::{Particle, SimParams};

const PARTICLE_COUNT: usize = 10_000;
const GRAVITY: f32 = 15.0;
const DT: f32 = 1.0 / 120.0;
const SOFTENING_SQUARED: f32 = 9.0;
const SPAWN_RADIUS: f32 = 1200.0;
const MAX_PHYSICS_STEPS_PER_FRAME: u32 = 8;
const TIME_SCALE_MIN: f32 = 0.125;
const TIME_SCALE_MAX: f32 = 4.0;
const ZOOM_MIN: f32 = 0.05;
const ZOOM_MAX: f32 = 5.0;
const CENTER_CONCENTRATION_STRENGTH: f32 = 0.75;
const CENTER_CONCENTRATION_EXPONENT: f32 = 1.40;
const INITIAL_RADIAL_SPEED_MAX: f32 = 0.01;
const CLOCKWISE_ROTATION_WEIGHT: f32 = 0.8;
const COUNTERCLOCKWISE_ROTATION_WEIGHT: f32 = 0.2;

fn sample_spawn_distance<R: ::rand::Rng + ?Sized>(rng: &mut R) -> f32 {
    let random_fraction = rng.r#gen::<f32>();
    let uniform_area_radius = random_fraction.sqrt();
    let center_biased_radius = random_fraction.powf(CENTER_CONCENTRATION_EXPONENT);
    let normalized_radius = uniform_area_radius
        + (center_biased_radius - uniform_area_radius) * CENTER_CONCENTRATION_STRENGTH;
    normalized_radius * SPAWN_RADIUS
}

fn initial_clockwise(index: usize, count: usize) -> bool {
    let fraction =
        CLOCKWISE_ROTATION_WEIGHT / (CLOCKWISE_ROTATION_WEIGHT + COUNTERCLOCKWISE_ROTATION_WEIGHT);
    index < (count as f32 * fraction).round() as usize
}

fn initial_tangential_speed(distance: f32) -> f32 {
    6.0 + 14.0 * (distance / SPAWN_RADIUS).clamp(0.0, 1.0).powf(1.0)
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
    let mass = rng.gen_range(5.0..20.0);
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
    let mut center_x = 0.0;
    let mut center_y = 0.0;
    let mut momentum_x = 0.0;
    let mut momentum_y = 0.0;
    for particle in particles.iter() {
        total_mass += particle.mass;
        center_x += particle.x * particle.mass;
        center_y += particle.y * particle.mass;
        momentum_x += particle.vx * particle.mass;
        momentum_y += particle.vy * particle.mass;
    }
    if total_mass == 0.0 {
        return;
    }
    center_x /= total_mass;
    center_y /= total_mass;
    momentum_x /= total_mass;
    momentum_y /= total_mass;
    for particle in particles {
        particle.x -= center_x;
        particle.y -= center_y;
        particle.vx -= momentum_x;
        particle.vy -= momentum_y;
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

fn draw_snapshot(particles: &[Particle], zoom: f32) {
    let center = vec2(screen_width() * 0.5, screen_height() * 0.5);
    for particle in particles {
        if particle.alive == 0 {
            continue;
        }
        let position = center + vec2(particle.x, particle.y) * zoom;
        if position.x < -20.0
            || position.x > screen_width() + 20.0
            || position.y < -20.0
            || position.y > screen_height() + 20.0
        {
            continue;
        }
        let radius = (particle.mass.sqrt() * 0.45 * zoom).max(1.0);
        draw_circle(
            position.x,
            position.y,
            radius,
            Color::new(particle.brightness, particle.brightness * 0.8, 1.0, 1.0),
        );
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
    let mut zoom: f32 = 0.35;
    let mut time_scale: f32 = 1.0;
    let mut accumulator = 0.0;
    let mut last_time = get_time();

    loop {
        if is_key_pressed(KeyCode::Space) {
            paused = !paused;
        }
        if is_key_pressed(KeyCode::R) {
            let reset_particles = create_particles();
            match backend.reset(&reset_particles) {
                Ok(()) => {
                    accumulator = 0.0;
                    last_time = get_time();
                }
                Err(error) => {
                    eprintln!("CUDA reset failed: {error}");
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
        if !paused {
            accumulator += frame_time.min(0.05) * time_scale;
            let mut completed_steps = 0;
            while accumulator >= DT && completed_steps < MAX_PHYSICS_STEPS_PER_FRAME {
                if let Err(error) = backend.step(SimParams {
                    count: backend.active_count() as u32,
                    dt: DT,
                    gravity: GRAVITY,
                    softening_squared: SOFTENING_SQUARED,
                }) {
                    eprintln!("CUDA step failed: {error}");
                    paused = true;
                    break;
                }
                accumulator -= DT;
                completed_steps += 1;
            }
            if completed_steps == MAX_PHYSICS_STEPS_PER_FRAME {
                accumulator = 0.0;
            }
        }
        let snapshot = match backend.snapshot() {
            Ok(snapshot) => snapshot,
            Err(error) => {
                eprintln!("CUDA readback failed: {error}");
                paused = true;
                Vec::new()
            }
        };
        clear_background(Color::new(0.005, 0.008, 0.025, 1.0));
        draw_snapshot(&snapshot, zoom);
        draw_text(
            "CUDA gravity | Space: pause | R: reset | Up/Down: time | Wheel: zoom",
            20.0,
            30.0,
            20.0,
            WHITE,
        );
        draw_text(
            &format!(
                "state: {}  particles: {}  time: {:.3}x  zoom: {:.2}x",
                if paused { "paused" } else { "running" },
                snapshot.len(),
                time_scale,
                zoom
            ),
            20.0,
            58.0,
            18.0,
            GRAY,
        );
        next_frame().await;
    }
}
