use crate::types::Particle;

pub const PARTICLE_RADIUS_SCALE: f32 = 0.45;

#[inline]
fn particle_radius(mass: f32) -> f32 {
    PARTICLE_RADIUS_SCALE * mass.sqrt()
}

fn find_collision(particles: &[Particle], first_index: usize) -> Option<usize> {
    let first = particles[first_index];
    let first_radius = particle_radius(first.mass);

    ((first_index + 1)..particles.len()).find(|&second_index| {
        let second = particles[second_index];
        if second.alive == 0 {
            return false;
        }

        let dx = second.x - first.x;
        let dy = second.y - first.y;
        let collision_distance = first_radius + particle_radius(second.mass);
        dx * dx + dy * dy <= collision_distance * collision_distance
    })
}

fn merge_pair(particles: &mut Vec<Particle>, first_index: usize, second_index: usize) {
    let first = particles[first_index];
    let second = particles[second_index];
    let combined_mass = first.mass + second.mass;
    let inverse_mass = 1.0 / combined_mass;

    particles[first_index] = Particle {
        x: (first.x * first.mass + second.x * second.mass) * inverse_mass,
        y: (first.y * first.mass + second.y * second.mass) * inverse_mass,
        vx: (first.vx * first.mass + second.vx * second.mass) * inverse_mass,
        vy: (first.vy * first.mass + second.vy * second.mass) * inverse_mass,
        mass: combined_mass,
        brightness: (first.brightness * first.mass + second.brightness * second.mass)
            * inverse_mass,
        alive: 1,
        padding: 0,
    };
    particles.swap_remove(second_index);
}

/// Merge particles in the same deterministic order as the AVX2 reference.
///
/// The first surviving particle is scanned in ascending order. After every
/// merge its collision search restarts because its radius, position, and mass
/// have changed. The returned vector is always a compact active prefix.
pub fn merge_colliding_particles(particles: &mut Vec<Particle>) {
    let mut first_index = 0;
    while first_index < particles.len() {
        while let Some(second_index) = find_collision(particles, first_index) {
            merge_pair(particles, first_index, second_index);
        }
        first_index += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn particle(x: f32, y: f32, mass: f32, vx: f32, brightness: f32) -> Particle {
        Particle {
            x,
            y,
            vx,
            vy: 0.0,
            mass,
            brightness,
            alive: 1,
            padding: 0,
        }
    }

    fn close(left: f32, right: f32) {
        assert!((left - right).abs() < 1e-5, "{left} != {right}");
    }

    #[test]
    fn non_overlapping_particles_are_unchanged() {
        let mut particles = vec![
            particle(0.0, 0.0, 4.0, 2.0, 0.5),
            particle(10.0, 0.0, 4.0, -1.0, 0.8),
        ];
        let original = particles.clone();
        merge_colliding_particles(&mut particles);
        assert_eq!(particles, original);
    }

    #[test]
    fn merge_conserves_weighted_properties() {
        let radius_sum = particle_radius(4.0) + particle_radius(9.0);
        let mut particles = vec![
            particle(0.0, 0.0, 4.0, 2.0, 0.5),
            particle(radius_sum, 0.0, 9.0, -1.0, 1.0),
        ];
        merge_colliding_particles(&mut particles);
        assert_eq!(particles.len(), 1);
        let merged = particles[0];
        close(merged.mass, 13.0);
        close(merged.x, 9.0 * radius_sum / 13.0);
        close(merged.vx, -1.0 / 13.0);
        close(merged.brightness, 11.0 / 13.0);
        assert_eq!(merged.alive, 1);
    }

    #[test]
    fn merged_particle_is_rechecked_for_collision_chain() {
        let first_radius = particle_radius(4.0);
        let second_radius = particle_radius(4.0);
        let merged_radius = particle_radius(8.0);
        let second_x = first_radius + second_radius - 1e-3;
        let third_x = 2.0 * merged_radius - 0.1;
        let mut particles = vec![
            particle(0.0, 0.0, 4.0, 0.0, 1.0),
            particle(second_x, 0.0, 4.0, 0.0, 1.0),
            particle(third_x, 0.0, 4.0, 0.0, 1.0),
        ];
        merge_colliding_particles(&mut particles);
        assert_eq!(particles.len(), 1);
        close(particles[0].mass, 12.0);
    }

    #[test]
    fn touching_radii_merge() {
        let distance = particle_radius(4.0) + particle_radius(4.0);
        let mut particles = vec![
            particle(0.0, 0.0, 4.0, 0.0, 1.0),
            particle(distance, 0.0, 4.0, 0.0, 1.0),
        ];
        merge_colliding_particles(&mut particles);
        assert_eq!(particles.len(), 1);
    }
}
