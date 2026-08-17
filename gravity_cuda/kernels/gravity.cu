#include <cstddef>

struct Particle {
    float x, y;
    float vx, vy;
    float mass;
    float brightness;
    unsigned int alive;
    unsigned int padding;
};

struct SimParams {
    unsigned int count;
    float dt;
    float gravity;
    float softening_squared;
};

static_assert(sizeof(Particle) == 32, "Particle ABI size changed");
static_assert(offsetof(Particle, x) == 0, "Particle.x ABI offset changed");
static_assert(offsetof(Particle, vx) == 8, "Particle.vx ABI offset changed");
static_assert(offsetof(Particle, mass) == 16, "Particle.mass ABI offset changed");
static_assert(offsetof(Particle, alive) == 24, "Particle.alive ABI offset changed");
static_assert(sizeof(SimParams) == 16, "SimParams ABI size changed");
static_assert(offsetof(SimParams, count) == 0, "SimParams.count ABI offset changed");
static_assert(offsetof(SimParams, dt) == 4, "SimParams.dt ABI offset changed");

__device__ __forceinline__ bool is_active(const Particle& particle) {
    return particle.alive != 0;
}

__device__ __forceinline__ float acceleration_factor(
    float other_mass,
    float distance_squared,
    SimParams params
) {
    float inverse_distance = rsqrtf(distance_squared + params.softening_squared);
    return params.gravity * other_mass * inverse_distance
        * inverse_distance * inverse_distance;
}

__device__ __forceinline__ void calculate_acceleration(
    const Particle* particles,
    unsigned int i,
    SimParams params,
    float& ax,
    float& ay
) {
    Particle current = particles[i];
    for (unsigned int j = 0; j < params.count; ++j) {
        if (i == j || !is_active(particles[j])) continue;
        Particle other = particles[j];
        float dx = other.x - current.x;
        float dy = other.y - current.y;
        float distance_squared = dx * dx + dy * dy;
        float factor = acceleration_factor(other.mass, distance_squared, params);
        ax += dx * factor;
        ay += dy * factor;
    }
}

__device__ __forceinline__ void clear_inactive_particle(Particle& particle) {
    particle.x = 0.0f;
    particle.y = 0.0f;
    particle.vx = 0.0f;
    particle.vy = 0.0f;
    particle.mass = 0.0f;
    particle.brightness = 0.0f;
    particle.alive = 0;
    particle.padding = 0;
}

extern "C" __global__ void clear_particle_suffix(
    Particle* particles,
    unsigned int start,
    unsigned int capacity
) {
    unsigned int i = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= capacity) return;
    clear_inactive_particle(particles[i]);
}

extern "C" __global__ void clear_particle_buffer(
    Particle* particles,
    unsigned int capacity
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= capacity) return;
    clear_inactive_particle(particles[i]);
}

__device__ __forceinline__ void invalidate_suffix(
    Particle* particles,
    unsigned int count,
    unsigned int capacity
) {
    for (unsigned int i = count; i < capacity; ++i) {
        clear_inactive_particle(particles[i]);
    }
}

/* Host uploads a compacted full-capacity buffer in phase one, so these
 * helpers are available for future asynchronous suffix invalidation. */

/* Collision is intentionally resolved by the synchronized Rust host phase.
 * The former parallel merge kernel was removed because it could mutate the
 * same pair from multiple threads and did not implement active compaction. */

/* Collision is intentionally resolved by the synchronized Rust host phase.
 * The former parallel merge kernel was removed because it could mutate the
 * same pair from multiple threads and did not implement active compaction. */

extern "C" __global__ void gravity_drift(
    const Particle* source,
    Particle* destination,
    float* acceleration_x,
    float* acceleration_y,
    SimParams params
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= params.count) return;

    Particle current = source[i];
    if (!is_active(current)) {
        destination[i] = current;
        acceleration_x[i] = 0.0f;
        acceleration_y[i] = 0.0f;
        return;
    }
    float ax = 0.0f;
    float ay = 0.0f;
    calculate_acceleration(source, i, params, ax, ay);

    float half_dt = 0.5f * params.dt;
    current.vx += ax * half_dt;
    current.vy += ay * half_dt;
    current.x += current.vx * params.dt;
    current.y += current.vy * params.dt;
    destination[i] = current;
    acceleration_x[i] = ax;
    acceleration_y[i] = ay;
}

extern "C" __global__ void gravity_finish(
    Particle* particles,
    float* acceleration_x,
    float* acceleration_y,
    SimParams params
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= params.count) return;

    Particle current = particles[i];
    if (!is_active(current)) {
        acceleration_x[i] = 0.0f;
        acceleration_y[i] = 0.0f;
        return;
    }
    float ax = 0.0f;
    float ay = 0.0f;
    calculate_acceleration(particles, i, params, ax, ay);

    float half_dt = 0.5f * params.dt;
    current.vx += ax * half_dt;
    current.vy += ay * half_dt;
    particles[i] = current;
    acceleration_x[i] = ax;
    acceleration_y[i] = ay;
}

/* merge_overlaps is intentionally absent: host collision is the only phase-one path. */
