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

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> source: array<Particle>;
@group(0) @binding(2) var<storage, read_write> destination: array<Particle>;

fn acceleration(index: u32, position: vec2<f32>) -> vec2<f32> {
  var result = vec2<f32>(0.0);
  for (var other_index = 0u; other_index < params.count; other_index++) {
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
  if (index >= params.count) { return; }
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
  if (index >= params.count) { return; }
  var result = source[index];
  if (result.alive >= 0.5) {
    result.velocity += acceleration(index, result.position) * (params.dt * 0.5);
  }
  destination[index] = result;
}
