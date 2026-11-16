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
// Per-body flag (binding 20): 1 if the body's merge component is nontrivial
// (set by `compress` for every root that absorbed another body). Lets
// `reduce` skip the O(n) member scan for singleton components — the scan
// over all-n roots was O(n^2) per merge step (7.7 ms at N=10000 on DX12).
@group(0) @binding(20) var<storage, read_write> merged_root: array<atomic<u32>>;

// Shader-debug logging (opt-in via `--shader-debug`).
//
// WGSL has no printf; the mechanism is an atomically-appended debug buffer.
// `SHADER_DEBUG` is a pipeline override (default 0 = off). Verified with
// naga: at 0 the gate folds to `if (false)` and every downstream shader
// compiler strips it — zero cost when disabled. When 1, each instrumented
// force thread appends one 4-word record: (kernel_tag, thread_id,
// bitcast(ax), bitcast(ay)). The buffer holds 1024 records; overflow is
// dropped silently (capped sample). Reading the buffer back forces a
// GPU/CPU sync, so capture only happens on explicit CPU request.
override SHADER_DEBUG: u32 = 0u;
const SHADER_DBG_WORDS: u32 = 4096u;

@group(0) @binding(21) var<storage, read_write> dbg_buf: array<u32, SHADER_DBG_WORDS>;
@group(0) @binding(22) var<storage, read_write> dbg_count: atomic<u32>;

// Kernel tags for shader-debug records.
const DBG_NAIVE: u32 = 1u;
const DBG_PASS1: u32 = 2u;
const DBG_FORCE: u32 = 3u;
const DBG_FORCE_ALWAYS: u32 = 4u;
const DBG_FUSED: u32 = 5u;

fn dbg_log(tag: u32, thread_id: u32, ax: f32, ay: f32) {
    if (SHADER_DEBUG != 0u) {
        let i = atomicAdd(&dbg_count, 1u);
        if (i + 4u <= SHADER_DBG_WORDS) {
            dbg_buf[i] = tag;
            dbg_buf[i + 1u] = thread_id;
            dbg_buf[i + 2u] = bitcast<u32>(ax);
            dbg_buf[i + 3u] = bitcast<u32>(ay);
        }
    }
}
