struct RenderParams {
  // vec4 members keep the uniform fields aligned to WGSL's 16-byte
  // uniform-buffer requirement while remaining exactly 32 bytes in Rust.
  viewport: vec4<f32>,
  counts: vec4<u32>,
};

struct Particle {
  position: vec2<f32>,
  velocity: vec2<f32>,
  mass: f32,
  brightness: f32,
  alive: f32,
  _padding: f32,
};

@group(0) @binding(0) var<uniform> params: RenderParams;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;

struct VertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) local: vec2<f32>,
  @location(1) color: vec4<f32>,
};

fn corner(vertex: u32) -> vec2<f32> {
  // Dynamic indexing of function-local constant arrays is not accepted by
  // all naga backends, so select the six quad corners explicitly.
  if (vertex == 0u) { return vec2<f32>(-1.0, -1.0); }
  if (vertex == 1u) { return vec2<f32>(1.0, -1.0); }
  if (vertex == 2u) { return vec2<f32>(-1.0, 1.0); }
  if (vertex == 3u) { return vec2<f32>(-1.0, 1.0); }
  if (vertex == 4u) { return vec2<f32>(1.0, -1.0); }
  return vec2<f32>(1.0, 1.0);
}

@vertex
fn vertex(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance: u32) -> VertexOut {
  let particle = particles[instance];
  let local = corner(vertex_index);
  let radius = clamp(params.viewport.z * 0.45 * sqrt(particle.mass), 0.7, 50.0);
  let pixel = particle.position * params.viewport.z + vec2<f32>(params.viewport.x, params.viewport.y) * 0.5;
  let center = vec2<f32>(pixel.x / params.viewport.x * 2.0 - 1.0, 1.0 - pixel.y / params.viewport.y * 2.0);
  let extent = local * vec2<f32>(radius / params.viewport.x * 2.0, -radius / params.viewport.y * 2.0);
  var output: VertexOut;
  output.position = vec4<f32>(center + extent, 0.0, 1.0);
  output.local = local;
  output.color = vec4<f32>(particle.brightness * 0.65, particle.brightness * 0.75, particle.brightness, 0.95);
  return output;
}

@fragment
fn fragment(input: VertexOut) -> @location(0) vec4<f32> {
  let distance = length(input.local);
  if (distance > 1.0) { discard; }
  let edge = 1.0 - smoothstep(0.72, 1.0, distance);
  return vec4<f32>(input.color.rgb, input.color.a * edge);
}
