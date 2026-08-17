use crate::{
    collision::merge_colliding_particles,
    types::{Particle, SimParams},
};

/// Scalar oracle that follows the CUDA kernels' per-target accumulation order.
pub fn step(particles: &mut Vec<Particle>, params: SimParams) {
    if particles.is_empty() {
        return;
    }

    let accelerations = calculate_accelerations(particles, params);
    for (particle, (ax, ay)) in particles.iter_mut().zip(accelerations) {
        if particle.alive == 0 {
            continue;
        }
        let half_dt = 0.5 * params.dt;
        particle.vx += ax * half_dt;
        particle.vy += ay * half_dt;
        particle.x += particle.vx * params.dt;
        particle.y += particle.vy * params.dt;
    }

    merge_colliding_particles(particles);
    if particles.is_empty() {
        return;
    }

    let accelerations = calculate_accelerations(
        particles,
        SimParams {
            count: particles.len() as u32,
            ..params
        },
    );
    for (particle, (ax, ay)) in particles.iter_mut().zip(accelerations) {
        if particle.alive == 0 {
            continue;
        }
        let half_dt = 0.5 * params.dt;
        particle.vx += ax * half_dt;
        particle.vy += ay * half_dt;
    }
}

fn calculate_accelerations(particles: &[Particle], params: SimParams) -> Vec<(f32, f32)> {
    let count = particles.len().min(params.count as usize);
    particles[..count]
        .iter()
        .enumerate()
        .map(|(i, current)| {
            if current.alive == 0 {
                return (0.0, 0.0);
            }
            let mut ax = 0.0;
            let mut ay = 0.0;
            for (j, other) in particles[..count].iter().enumerate() {
                if i == j || other.alive == 0 {
                    continue;
                }
                let dx = other.x - current.x;
                let dy = other.y - current.y;
                let distance_squared = dx * dx + dy * dy;
                let inverse_distance = distance_squared
                    .mul_add(1.0, params.softening_squared)
                    .sqrt()
                    .recip();
                let factor = params.gravity
                    * other.mass
                    * inverse_distance
                    * inverse_distance
                    * inverse_distance;
                ax += dx * factor;
                ay += dy * factor;
            }
            (ax, ay)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const PARAMS: SimParams = SimParams {
        count: 0,
        dt: 0.01,
        gravity: 15.0,
        softening_squared: 9.0,
    };

    fn particle(x: f32, y: f32, vx: f32, vy: f32, mass: f32) -> Particle {
        Particle {
            x,
            y,
            vx,
            vy,
            mass,
            brightness: 0.8,
            alive: 1,
            padding: 0,
        }
    }

    fn close(left: f32, right: f32, tolerance: f32) {
        assert!((left - right).abs() <= tolerance, "{left} != {right}");
    }

    #[test]
    fn one_particle_moves_without_acceleration() {
        let mut particles = vec![particle(1.0, -2.0, 3.0, 4.0, 5.0)];
        step(&mut particles, SimParams { count: 1, ..PARAMS });
        close(particles[0].x, 1.03, 1e-6);
        close(particles[0].y, -1.96, 1e-6);
        close(particles[0].vx, 3.0, 1e-6);
        close(particles[0].vy, 4.0, 1e-6);
    }

    #[test]
    fn empty_input_is_safe() {
        let mut particles = Vec::new();
        step(&mut particles, PARAMS);
        assert!(particles.is_empty());
    }

    #[test]
    fn two_body_step_has_finite_values() {
        let mut particles = vec![
            particle(-10.0, 0.0, 0.0, 1.0, 5.0),
            particle(10.0, 0.0, 0.0, -1.0, 7.0),
        ];
        step(&mut particles, SimParams { count: 2, ..PARAMS });
        assert_eq!(particles.len(), 2);
        for particle in particles {
            assert!(particle.x.is_finite());
            assert!(particle.y.is_finite());
            assert!(particle.vx.is_finite());
            assert!(particle.vy.is_finite());
        }
    }

    #[test]
    fn collision_is_resolved_before_final_kick() {
        let mut particles = vec![
            particle(0.0, 0.0, 0.0, 0.0, 4.0),
            particle(0.1, 0.0, 0.0, 0.0, 4.0),
        ];
        step(&mut particles, SimParams { count: 2, ..PARAMS });
        assert_eq!(particles.len(), 1);
        close(particles[0].mass, 8.0, 1e-6);
        assert!(particles[0].vx.is_finite());
    }
}
