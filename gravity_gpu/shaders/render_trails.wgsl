// Instanced trail renderer. One instance per stored ring point (the CPU
// sizes the draw from the last stats readback). Points older than 40 sim-s
// or off-screen are culled in the vertex shader; alpha fades with a 10 s
// half-life per the legacy §8 trail spec.

const TRAIL_HALF_LIFE: f32 = 10.0;
const TRAIL_MAX_AGE: f32 = 40.0;
const TRAIL_INITIAL_ALPHA: f32 = 0.35;
const TRAIL_MIN_W: f32 = 1.0;
const TRAIL_MAX_W: f32 = 2.0;

struct TrailPoint {
    pos: vec2<f32>,
    radius: f32,
    brightness: f32,
    birth: f32,
};

struct Camera {
    screen: vec2<f32>,
    zoom: f32,
    sim_time: f32,
    _pad: f32,
};

@group(0) @binding(0) var<storage, read> ring: array<TrailPoint>;
@group(0) @binding(1) var<uniform> cam: Camera;

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) offset: vec2<f32>,
    @location(1) half_alpha: vec2<f32>, // (half width px, alpha)
    @location(2) col: vec3<f32>,
};

const QUAD: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, 1.0),
);

fn cull() -> VOut {
    var o: VOut;
    o.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    o.offset = vec2<f32>(0.0, 0.0);
    o.half_alpha = vec2<f32>(0.0, 0.0);
    o.col = vec3<f32>(0.0, 0.0, 0.0);
    return o;
}

@vertex
fn vs(
    @builtin(instance_index) inst: u32,
    @builtin(vertex_index) vi: u32,
) -> VOut {
    let p = ring[inst];
    let age = cam.sim_time - p.birth;
    if (age < 0.0 || age >= TRAIL_MAX_AGE) {
        return cull();
    }
    let w = clamp(p.radius * cam.zoom * 2.0, TRAIL_MIN_W, TRAIL_MAX_W);
    let half = w * 0.5;
    let c = cam.screen * 0.5 + p.pos * cam.zoom;
    if (c.x < -half || c.x > cam.screen.x + half || c.y < -half
        || c.y > cam.screen.y + half) {
        return cull();
    }
    let alpha = TRAIL_INITIAL_ALPHA * pow(0.5, age / TRAIL_HALF_LIFE);
    if (alpha < 0.004) {
        return cull();
    }
    let b = min(p.brightness, 1.0);
    let off = QUAD[vi] * half;
    let px = c + off;
    var o: VOut;
    o.clip = vec4<f32>(
        px.x / cam.screen.x * 2.0 - 1.0,
        1.0 - px.y / cam.screen.y * 2.0,
        0.0,
        1.0,
    );
    o.offset = off;
    o.half_alpha = vec2<f32>(half, alpha);
    o.col = vec3<f32>(b * 0.60, b * 0.75, b);
    return o;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    if (length(in.offset) > in.half_alpha.x) {
        discard;
    }
    return vec4<f32>(in.col, in.half_alpha.y);
}
