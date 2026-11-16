//! WGSL shader sources, embedded at compile time.
//!
//! `common.wgsl` (shared constants + the compute bind-group declarations) is
//! prepended to every compute shader. Render shaders are self-contained.

macro_rules! shader {
    ($name:literal) => {
        include_str!(concat!("../../../../shaders/", $name))
    };
}

const COMMON: &str = shader!("common.wgsl");

/// Compute shader source with [`COMMON`] prepended.
pub fn with_common(src: &str) -> String {
    let mut out = String::with_capacity(COMMON.len() + src.len() + 1);
    out.push_str(COMMON);
    out.push('\n');
    out.push_str(src);
    out
}

/// Raw source of one compute shader (without the common prelude).
pub fn source(name: &str) -> &'static str {
    match name {
        "force_naive" => shader!("force_naive.wgsl"),
        "force_tiled" => shader!("force_tiled.wgsl"),
        "integrate" => shader!("integrate.wgsl"),
        "detect_naive" => shader!("detect_naive.wgsl"),
        "merge" => shader!("merge.wgsl"),
        "trail" => shader!("trail.wgsl"),
        "stats" => shader!("stats.wgsl"),
        _ => panic!("unknown compute shader: {name}"),
    }
}

/// Generate the tiled-force WGSL with the tile loop fully unrolled.
///
/// FXC (SM 5.1, DirectX 12) rejects `workgroupBarrier()` inside a loop that
/// contains a nested loop (error X4026), even when all bounds are uniform
/// and the body is branchless. Unrolling the tile loop eliminates the
/// nesting: barriers sit in straight-line code, and the per-tile consume
/// loops become sequential (not nested). The result is bit-identical to
/// the looped version (same operations, same order).
///
/// `ntiles_max` is ceil(n_max / wg); `wg` is the workgroup size (64/128/256).
pub fn force_tiled_unrolled(ntiles_max: u32, wg: u32) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        r#"// Tiled force kernel, tile loop unrolled (see shaders::force_tiled_unrolled).
// Bit-identical to force_tiled.wgsl; barriers are in straight-line code
// for FXC (SM 5.1) X4026 compliance.
override WG: u32 = {wg}u;

var<workgroup> tile: array<vec4<f32>, 256>; // (x, y, mass, _) per partner

// One tiled force row (pure force). Tile loop unrolled; each tile is
// loaded, synchronized, consumed, synchronized in straight-line code.
fn force_row(i: u32, n: u32, lid: u32) -> vec2<f32> {{
    let pi = pos_a[i];
    var ax = vec2<f32>(0.0, 0.0);
"#
    ));
    for t in 0..ntiles_max {
        s.push_str(&format!(
            r#"    // Tile {t}:
    {{
        let j = {t}u * WG + lid;
        let jc = min(j, max(n, 1u) - 1u);
        let m = f32(j < n);
        tile[lid] = vec4<f32>(pos_a[jc].x, pos_a[jc].y, mass_a[jc], 0.0) * m;
    }}
    workgroupBarrier();
    for (var k = 0u; k < WG; k++) {{
        let other = tile[k];
        let d = other.xy - pi;
        let r2 = dot(d, d) + SOFTENING_SQ;
        let inv = 1.0 / sqrt(r2);
        let f = G * inv * inv * inv * other.z;
        ax += d * f;
    }}
    workgroupBarrier();
"#
        ));
    }
    s.push_str(
        r#"    return ax;
}

@compute @workgroup_size(WG)
fn pass1(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    // NOTE (FXC X4026): the cache-valid early return
    // `if (atomicLoad(&accel_valid[0]) == 1u) { return; }` is NOT allowed
    // here — FXC treats atomicLoad as varying, so the barriers in
    // force_row would sit in varying control flow. Every thread therefore
    // runs force_row unconditionally (barriers converge).
    //
    // NOTE (race fix, 2026-10-18): the write-back must NOT be gated on the
    // flag either (`if (atomicLoad(&accel_valid[0]) == 0u)`). Thread 0 used
    // to store 1 to accel_valid in this same dispatch, so every other
    // thread's gated load raced with that store: a thread whose load
    // landed after the store skipped its write and left a stale
    // acceleration behind. Nondeterministic across runs; broke tiled-vs-CPU
    // bitwise parity on the RTX 3060 (parity_ok=false, 2026-10-18). Since
    // the compute is unconditional anyway, the write is unconditional too
    // — always correct. Cache validity is tracked on the CPU
    // (Sim::accel_valid_cpu), which simply skips this dispatch when the
    // cache is valid; no shader reads the accel_valid flag anymore.
    let i = gid.x;
    let n = n_buf[0];
    let ax = force_row(i, n, lid.x);
    if (i < n) {
        accel_a[i] = ax;
        dbg_log(DBG_PASS1, i, ax.x, ax.y);
    }
}

@compute @workgroup_size(WG)
fn force(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    // NOTE (FXC X4026): same as pass1 — no early return on
    // atomicLoad(&merge_flag[0]); run force_row unconditionally and gate
    // only the write-back.
    let i = gid.x;
    let n = n_buf[0];
    let ax = force_row(i, n, lid.x);
    if (atomicLoad(&merge_flag[0]) != 0u && i < n) {
        accel_a[i] = ax;
        dbg_log(DBG_FORCE, i, ax.x, ax.y);
    }
}

@compute @workgroup_size(WG)
fn force_always(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = gid.x;
    let n = n_buf[0];
    let ax = force_row(i, n, lid.x);
    if (i < n) {
        accel_a[i] = ax;
        dbg_log(DBG_FORCE_ALWAYS, i, ax.x, ax.y);
    }
}

@compute @workgroup_size(WG)
fn fused(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = gid.x;
    let n = n_buf[0];
    let ax = force_row(i, n, lid.x);
    var live = false;
    if (i < n) {
        accel_a[i] = ax;
        dbg_log(DBG_FUSED, i, ax.x, ax.y);
        live = true;
    }
    // Barrier-free naive detection (upper triangle).
    if (live) {
        let pi = pos_a[i];
        let ri = RADIUS_SCALE * sqrt(mass_a[i]);
        for (var j = i + 1u; j < n; j++) {
            let d = pos_a[j] - pi;
            let rs = ri + RADIUS_SCALE * sqrt(mass_a[j]);
            if (dot(d, d) <= rs * rs) {
                atomicMin(&parent[j], i);
                atomicOr(&merge_flag[0], 1u);
            }
        }
    }
    // NOTE: the old thread-0 `atomicStore(&accel_valid[0], 1u)` is gone —
    // cache validity is tracked on the CPU now (Sim::accel_valid_cpu);
    // no shader reads the flag.
}
"#,
    );
    s
}

/// Render shader source (self-contained, no prelude).
pub fn render_source(name: &str) -> &'static str {
    match name {
        "particles" => shader!("render_particles.wgsl"),
        "trails" => shader!("render_trails.wgsl"),
        _ => panic!("unknown render shader: {name}"),
    }
}
