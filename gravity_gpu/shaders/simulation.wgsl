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
struct SimControl { active_count: atomic<u32>, _padding: array<u32, 7>, };

@group(0) @binding(3) var<storage, read_write> control: SimControl;

fn active_count() -> u32 { return min(atomicLoad(&control.active_count), params.count); }

fn collision_partner(index: u32, particle: Particle, count: u32) -> u32 {
  let radius = params.radius_scale * sqrt(max(particle.mass, 0.0));
  for (var other_index = index + 1u; other_index < count; other_index++) {
    let other = source[other_index];
    let displacement = other.position - particle.position;
    let collision_distance = radius + params.radius_scale * sqrt(max(other.mass, 0.0));
    if (other.alive >= 0.5 && dot(displacement, displacement) <= collision_distance * collision_distance) {
      return other_index;
    }
  }
  return count;
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

@compute @workgroup_size(1)
fn merge_serial(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x != 0u) { return; }
  let count = active_count();
  var output_count = count;
  var first = 0u;
  while (first < output_count) {
    var current = source[first];
    loop {
      let partner = collision_partner(first, current, output_count);
      if (partner >= output_count) { break; }
      current = merge_into(current, source[partner]);
      let last = output_count - 1u;
      if (partner != last) { source[partner] = source[last]; }
      output_count -= 1u;
    }
    source[first] = current;
    first += 1u;
  }
  for (var index = output_count; index < params.count; index++) {
    var dead = source[index]; dead.alive = 0.0; source[index] = dead;
  }
  atomicStore(&control.active_count, output_count);
}

@compute @workgroup_size(64)
fn copy_compacted(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.count) { return; }
  destination[index] = source[index];
}

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> source: array<Particle>;
@group(0) @binding(2) var<storage, read_write> destination: array<Particle>;

fn acceleration(index: u32, position: vec2<f32>) -> vec2<f32> {
  var result = vec2<f32>(0.0);
  let live_count = active_count();
  for (var other_index = 0u; other_index < live_count; other_index++) {
    let other = source[other_index];
    if (other.alive < 0.5) { continue; }
    let displacement = other.position - position;
    let inverse_distance = inverseSqrt(dot(displacement, displacement) + params.softening_squared);
    result += displacement * (params.gravity * other.mass * inverse_distance * inverse_distance * inverse_distance);
  }
  return result;
}

@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= active_count()) { return; }
  let particle = source[index];
  if (particle.alive < 0.5) { destination[index] = particle; return; }
  var result = particle;
  result.velocity += acceleration(index, particle.position) * (params.dt * 0.5);
  result.position += result.velocity * params.dt;
  destination[index] = result;
}

@compute @workgroup_size(64)
fn finish(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= active_count()) { return; }
  var result = source[index];
  if (result.alive >= 0.5) {
    result.velocity += acceleration(index, result.position) * (params.dt * 0.5);
  }
  destination[index] = result;
}
