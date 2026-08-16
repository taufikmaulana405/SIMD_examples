struct RenderParams {
  count: u32,
  trail_count: u32,
  width: f32,
  height: f32,
  zoom: f32,
  total_mass: f32,
  trail_clock: f32,
  _pad: f32,
};

struct Particle {
  position: vec2<f32>,
  velocity: vec2<f32>,
  mass: f32,
  brightness: f32,
  alive: f32,
  _pad: f32,
};

struct Trail {
  position: vec2<f32>,
  radius: f32,
  brightness: f32,
  created: f32,
  _pad0: f32,
  _pad1: f32,
};

@group(0) @binding(0) var<uniform> params: RenderParams;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var<storage, read> trails: array<Trail>;

struct VertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) local: vec2<f32>,
  @location(1) color: vec4<f32>,
  @location(2) radius: f32,
};

fn clip_position(world: vec2<f32>, size: vec2<f32>) -> vec4<f32> {
  let pixel = world + vec2<f32>(params.width, params.height) * 0.5;
  // Match Macroquad's screen coordinates: +Y points downward.
  let center = vec2<f32>(
    pixel.x / params.width * 2.0 - 1.0,
    1.0 - pixel.y / params.height * 2.0,
  );
  let extent = vec2<f32>(
    size.x / params.width * 2.0,
    -size.y / params.height * 2.0,
  );
  return vec4<f32>(center + extent, 0.0, 1.0);
}

fn transition(fraction: f32) -> f32 {
  let p = clamp((fraction - 0.25) / 0.20, 0.0, 1.0);
  return p * p * (3.0 - 2.0 * p);
}

fn quad_corner(vertex: u32) -> vec2<f32> {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
  return corners[vertex];
}

@vertex
fn particle_vertex(@builtin(vertex_index) vertex: u32,
                   @builtin(instance_index) instance: u32) -> VertexOut {
  let corner = quad_corner(vertex);
  let particle = particles[instance];
  var out: VertexOut;
  let radius = clamp(params.zoom * 0.45 * sqrt(particle.mass), 0.7, 50.0);
  let star = transition(particle.mass / max(params.total_mass, 0.0001));
  let mass_brightness = clamp(sqrt(particle.mass / 20.0), 0.0, 1.0);
  let normal = vec3<f32>(min(particle.brightness * 0.65 + mass_brightness * 0.25, 1.0),
                         min(particle.brightness * 0.75 + mass_brightness * 0.20, 1.0),
                         min(particle.brightness, 1.0));
  let body = mix(normal, vec3<f32>(1.0, 0.72, 0.15), star);
  let scale = radius * (1.0 + 1.7 * star);
  out.position = clip_position(particle.position * params.zoom, corner * scale);
  out.local = corner;
  out.color = vec4<f32>(body, 0.95);
  out.radius = radius;
  return out;
}

@fragment
fn particle_fragment(input: VertexOut) -> @location(0) vec4<f32> {
  let distance = length(input.local);
  if (distance > 1.0) { discard; }
  let edge = 1.0 - smoothstep(0.72, 1.0, distance);
  return vec4<f32>(input.color.rgb, input.color.a * edge);
}

struct TrailOut {
  @builtin(position) position: vec4<f32>,
  @location(0) local: vec2<f32>,
  @location(1) color: vec4<f32>,
};

@vertex
fn trail_vertex(@builtin(vertex_index) vertex: u32,
                @builtin(instance_index) instance: u32) -> TrailOut {
  let corner = quad_corner(vertex);
  let trail = trails[instance];
  let age = max(params.trail_clock - trail.created, 0.0);
  let fade = pow(0.5, age / 10.0);
  let width = clamp(trail.radius * params.zoom * 2.0, 1.0, 2.0);
  var out: TrailOut;
  out.position = clip_position(trail.position * params.zoom, corner * width * 0.5);
  out.local = corner;
  out.color = vec4<f32>(trail.brightness * 0.60, trail.brightness * 0.75, trail.brightness, 0.35 * fade);
  return out;
}

@fragment
fn trail_fragment(input: TrailOut) -> @location(0) vec4<f32> {
  let distance = length(input.local);
  if (distance > 1.0) { discard; }
  let edge = 1.0 - smoothstep(0.75, 1.0, distance);
  return vec4<f32>(input.color.rgb, input.color.a * edge);
}
