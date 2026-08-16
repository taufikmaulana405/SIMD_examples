struct Params {
  count: u32,
  _pad0: u32,
  dt: f32,
  gravity: f32,
  softening_squared: f32,
  radius_scale: f32,
  width: f32,
  height: f32,
  zoom: f32,
  total_mass: f32,
  trail_sample: u32,
  trail_capacity: u32,
};

struct Particle {
  position: vec2<f32>,
  velocity: vec2<f32>,
  mass: f32,
  brightness: f32,
  // JavaScript uploads the particle array as Float32Array.
  alive: f32,
  _pad: f32,
};


@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<Particle>;
@group(0) @binding(2) var<storage, read_write> dst: array<Particle>;

@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.count) { return; }

  let current_particle = src[index];
  if (current_particle.alive < 0.5) {
    dst[index] = current_particle;
    return;
  }

  var acceleration = vec2<f32>(0.0);
  for (var j = 0u; j < params.count; j++) {
    let other = src[j];
    if (other.alive < 0.5) { continue; }
    let displacement = other.position - current_particle.position;
    let inverse_distance = inverseSqrt(dot(displacement, displacement) + params.softening_squared);
    acceleration += displacement * (params.gravity * other.mass * inverse_distance * inverse_distance * inverse_distance);
  }

  var result = current_particle;
  result.velocity += acceleration * (params.dt * 0.5);
  result.position += result.velocity * params.dt;
  dst[index] = result;
}

// Kept separate from integrate so the JavaScript side can run a complete
// velocity-Verlet step without reading particle data back from the GPU.
@compute @workgroup_size(64)
fn finish(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= params.count) { return; }
  var result = src[index];
  if (result.alive >= 0.5) {
    var acceleration = vec2<f32>(0.0);
    for (var j = 0u; j < params.count; j++) {
      let other = src[j];
      if (other.alive < 0.5) { continue; }
      let displacement = other.position - result.position;
      let inverse_distance = inverseSqrt(dot(displacement, displacement) + params.softening_squared);
      acceleration += displacement * (params.gravity * other.mass * inverse_distance * inverse_distance * inverse_distance);
    }
    result.velocity += acceleration * (params.dt * 0.5);
  }
  dst[index] = result;
}

/* finish is defined above; this marker makes accidental duplicate entry-point edits obvious. */


/* end of simulation shader */
