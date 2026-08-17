const GRID_BUCKETS: u32 = 32768u;
const PARTICLE_CAPACITY: u32 = 10000u;
const GRID_CELL_SIZE: f32 = 16.0;
const MAX_BUCKET_VISITS: u32 = 128u;
const NONE: u32 = 0xffffffffu;

struct SimParams {
  count: u32,
  _padding: u32,
  dt: f32,
  gravity: f32,
  softening_squared: f32,
  radius_scale: f32,
  _padding2: vec2<f32>,
  _padding3: vec2<f32>,
  _padding4: vec2<f32>,
};
struct Particle {
  position: vec2<f32>,
  velocity: vec2<f32>,
  mass: f32,
  brightness: f32,
  alive: f32,
  _padding: f32,
};
struct CollisionMeta {
  heads: array<atomic<u32>, 32768>,
  next: array<atomic<u32>, 10000>,
  proposal: array<atomic<u32>, 10000>,
  output_count: atomic<u32>,
  overflow: atomic<u32>,
};
struct SimControl { active_count: atomic<u32>, output_count: atomic<u32>, merge_count: atomic<u32>, _padding: array<u32, 5>, };

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> source: array<Particle>;
@group(0) @binding(2) var<storage, read_write> collision_state: CollisionMeta;
@group(0) @binding(3) var<storage, read_write> destination: array<Particle>;
@group(0) @binding(4) var<storage, read_write> control: SimControl;

fn hash_cell(cell: vec2<i32>) -> u32 {
  let mixed = (cell.x * 73856093) ^ (cell.y * 19349663);
  return u32(mixed & 2147483647) % GRID_BUCKETS;
}
fn bucket_for(position: vec2<f32>) -> u32 {
  return hash_cell(vec2<i32>(i32(floor(position.x / GRID_CELL_SIZE)), i32(floor(position.y / GRID_CELL_SIZE))));
}
fn particle_radius(p: Particle) -> f32 {
  return params.radius_scale * sqrt(max(p.mass, 0.0));
}
fn merge_into(first: Particle, second: Particle) -> Particle {
  let mass = first.mass + second.mass;
  var result = first;
  result.position = (first.position * first.mass + second.position * second.mass) / max(mass, 0.000001);
  result.velocity = (first.velocity * first.mass + second.velocity * second.mass) / max(mass, 0.000001);
  result.mass = mass;
  result.brightness = (first.brightness * first.mass + second.brightness * second.mass) / max(mass, 0.000001);
  result.alive = 1.0;
  return result;
}

@compute @workgroup_size(64)
fn clear(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index < GRID_BUCKETS) { atomicStore(&collision_state.heads[index], NONE); }
  if (index < PARTICLE_CAPACITY) {
    atomicStore(&collision_state.next[index], NONE);
    atomicStore(&collision_state.proposal[index], NONE);
  }
  if (index == 0u) {
    atomicStore(&collision_state.output_count, 0u);
    atomicStore(&collision_state.overflow, 0u);
  }
}

@compute @workgroup_size(64)
fn build_grid(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.count) { return; }
  let p = source[index];
  if (p.alive < 0.5) { return; }
  let bucket = bucket_for(p.position);
  let old = atomicExchange(&collision_state.heads[bucket], index);
  atomicStore(&collision_state.next[index], old);
}

@compute @workgroup_size(64)
fn propose(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.count) { return; }
  let particle = source[index];
  if (particle.alive < 0.5) { atomicStore(&collision_state.proposal[index], NONE); return; }
  let center = vec2<i32>(i32(floor(particle.position.x / GRID_CELL_SIZE)), i32(floor(particle.position.y / GRID_CELL_SIZE)));
  var best = NONE;
  var visits = 0u;
  for (var oy = -1; oy <= 1; oy++) {
    for (var ox = -1; ox <= 1; ox++) {
      let bucket = hash_cell(center + vec2<i32>(ox, oy));
      var candidate = atomicLoad(&collision_state.heads[bucket]);
      loop {
        if (candidate == NONE || visits >= MAX_BUCKET_VISITS) { break; }
        visits += 1u;
        if (candidate != index) {
          let other = source[candidate];
          let displacement = other.position - particle.position;
          let limit = particle_radius(particle) + particle_radius(other);
          if (other.alive >= 0.5 && dot(displacement, displacement) <= limit * limit && candidate < best) { best = candidate; }
        }
        candidate = atomicLoad(&collision_state.next[candidate]);
      }
    }
  }
  if (visits >= MAX_BUCKET_VISITS) { atomicStore(&collision_state.overflow, 1u); }
  atomicStore(&collision_state.proposal[index], best);
}

@compute @workgroup_size(64)
fn reset_output(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x == 0u) { atomicStore(&collision_state.output_count, 0u); }
}

@compute @workgroup_size(64)
fn merge_compact(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.count) { return; }
  let particle = source[index];
  if (particle.alive < 0.5) { return; }
  let partner = atomicLoad(&collision_state.proposal[index]);
  if (partner != NONE && atomicLoad(&collision_state.proposal[partner]) == index) {
    if (index > partner) { return; }
    let slot = atomicAdd(&collision_state.output_count, 1u);
    destination[slot] = merge_into(particle, source[partner]);
    return;
  }
  let slot = atomicAdd(&collision_state.output_count, 1u);
  destination[slot] = particle;
}

@compute @workgroup_size(1)
fn finalize(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x != 0u) { return; }
  let count = min(atomicLoad(&collision_state.output_count), params.count);
  atomicStore(&control.active_count, count);
  atomicStore(&control.output_count, count);
  atomicStore(&control.merge_count, atomicLoad(&collision_state.overflow));
}
