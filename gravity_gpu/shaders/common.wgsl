// Shared constants, prepended to every shader by `gpu::shaders`.
// Mirror of `gsim_core::constants` — keep in sync.

const G: f32 = 15.0;
const SOFTENING_SQ: f32 = 9.0; // 3.0^2
const DT: f32 = 1.0 / 120.0;
const RADIUS_SCALE: f32 = 0.45;

const TRAIL_MAX_POINTS: u32 = 4000000u;
const TRAIL_HALF_LIFE: f32 = 10.0;
const TRAIL_MAX_AGE: f32 = 40.0;
const TRAIL_INITIAL_ALPHA: f32 = 0.35;
const TRAIL_MIN_W: f32 = 1.0;
const TRAIL_MAX_W: f32 = 2.0;

const YELLOW_START: f32 = 0.25;
const CENTRAL_FULL: f32 = 0.45;

// Shared compute bind-group layout (group 0). Every compute shader declares
// the subset it needs; all particle buffers are read_write in the layout so
// shaders may declare them either way.

struct TrailPoint {
    pos: vec2<f32>,
    radius: f32,
    brightness: f32,
    birth: f32,
};

struct TrailUni {
    sim_time: f32,
    _pad: vec3<u32>,
};

@group(0) @binding(0) var<storage, read_write> pos_a: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> vel_a: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> mass_a: array<f32>;
@group(0) @binding(3) var<storage, read_write> bright_a: array<f32>;
@group(0) @binding(4) var<storage, read_write> accel_a: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> pos_b: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read_write> vel_b: array<vec2<f32>>;
@group(0) @binding(7) var<storage, read_write> mass_b: array<f32>;
@group(0) @binding(8) var<storage, read_write> bright_b: array<f32>;
@group(0) @binding(9) var<storage, read_write> parent: array<atomic<u32>>;
@group(0) @binding(10) var<storage, read_write> merge_flag: array<atomic<u32>, 1>;
@group(0) @binding(11) var<storage, read_write> accel_valid: array<atomic<u32>, 1>;
@group(0) @binding(12) var<storage, read_write> alive_count: array<atomic<u32>, 1>;
@group(0) @binding(13) var<storage, read_write> n_buf: array<u32, 1>;
@group(0) @binding(14) var<storage, read_write> trail_ring: array<TrailPoint>;
@group(0) @binding(15) var<storage, read_write> trail_head: array<atomic<u32>, 1>;
@group(0) @binding(16) var<storage, read_write> stats_buf: array<u32, 8>;
@group(0) @binding(17) var<storage, read_write> partials_f: array<vec4<f32>, 1024>;
@group(0) @binding(18) var<storage, read_write> partials_c: array<u32, 1024>;
@group(0) @binding(19) var<uniform> trail_uni: TrailUni;
