// Instanced particle renderer. One instance per body slot (dead slots are
// culled to degenerate triangles in the vertex shader). Colour mapping is
// the legacy §8 spec verbatim: mass-brightened base colour, smoothstep
// yellow transition over mass fraction [0.25, 0.45], outer/inner glow and a
// highlight for transitioned bodies.

const RADIUS_SCALE: f32 = 0.45;

struct Camera {
    screen: vec2<f32>, // viewport size in px
    zoom: f32,
    total_mass: f32,
    _pad: f32,
};

@group(0) @binding(0) var<storage, read> pos: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> mass: array<f32>;
@group(0) @binding(2) var<storage, read> bright: array<f32>;
@group(0) @binding(3) var<uniform> cam: Camera;

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) center: vec2<f32>,
    @location(1) offset: vec2<f32>, // px offset from center (interpolated)
    @location(2) r_t: vec2<f32>,   // (display radius px, transition t)
    @location(3) col: vec4<f32>,   // body colour
};

const QUAD: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, 1.0),
);

fn transition(mass_fraction: f32) -> f32 {
    let t = clamp((mass_fraction - 0.25) / 0.20, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

fn cull() -> VOut {
    var o: VOut;
    o.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    o.center = vec2<f32>(0.0, 0.0);
    o.offset = vec2<f32>(0.0, 0.0);
    o.r_t = vec2<f32>(0.0, 0.0);
    o.col = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    return o;
}

@vertex
fn vs(
    @builtin(instance_index) inst: u32,
    @builtin(vertex_index) vi: u32,
) -> VOut {
    let m = mass[inst];
    if (m <= 0.0) {
        return cull();
    }
    let r_disp = clamp(RADIUS_SCALE * sqrt(m) * cam.zoom, 0.7, 50.0);
    let t = transition(m / max(cam.total_mass, 0.000001));
    let vr = r_disp * (1.0 + 1.7 * t); // glow-aware visibility radius
    let c = cam.screen * 0.5 + pos[inst] * cam.zoom;
    if (c.x < -vr || c.x > cam.screen.x + vr || c.y < -vr
        || c.y > cam.screen.y + vr) {
        return cull();
    }
    let b = bright[inst];
    let mb = clamp(sqrt(m / 20.0), 0.0, 1.0);
    let normal = vec4<f32>(
        min(b * 0.65 + mb * 0.25, 1.0),
        min(b * 0.75 + mb * 0.20, 1.0),
        min(b, 1.0),
        0.95,
    );
    let body = mix(normal, vec4<f32>(1.0, 0.72, 0.15, 1.0), t);
    let half = r_disp * 2.7; // quad covers the outer glow
    let off = QUAD[vi] * half;
    let px = c + off;
    var o: VOut;
    o.clip = vec4<f32>(
        px.x / cam.screen.x * 2.0 - 1.0,
        1.0 - px.y / cam.screen.y * 2.0,
        0.0,
        1.0,
    );
    o.center = c;
    o.offset = off;
    o.r_t = vec2<f32>(r_disp, t);
    o.col = body;
    return o;
}

// fg over bg compositing; hardware alpha blending stacks instances.
fn over(fg: vec4<f32>, bg: vec4<f32>) -> vec4<f32> {
    let a = fg.a + bg.a * (1.0 - fg.a);
    let rgb = fg.rgb * fg.a + bg.rgb * bg.a * (1.0 - fg.a);
    return vec4<f32>(rgb / max(a, 0.00001), a);
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    let r = in.r_t.x;
    let t = in.r_t.y;
    let d = length(in.offset);
    var col = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    if (t > 0.0) {
        if (d <= r * 2.7) {
            col = over(vec4<f32>(1.0, 0.45, 0.05, 0.035 * t), col);
        }
        if (d <= r * 1.8) {
            col = over(vec4<f32>(1.0, 0.55, 0.08, 0.08 * t), col);
        }
    }
    if (d <= r) {
        col = over(in.col, col);
    }
    if (t > 0.0) {
        let hd = length(in.offset + vec2<f32>(0.25, 0.25) * r);
        if (hd <= r * 0.55) {
            col = over(vec4<f32>(1.0, 0.92, 0.55, 0.8 * t), col);
        }
    }
    return col;
}
