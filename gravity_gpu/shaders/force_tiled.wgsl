// Tiled force kernel (benchmark ladder rung 3): shared-memory tiling with
// vec4-packed partner tiles, one thread per row i. Partners are streamed
// tile-by-tile in ascending global j order, so the accumulation order is
// bit-identical to `force_naive` (and to `gsim_core::cpu`).
//
// Entry points:
//   `pass1` — computes only when the cross-step cache is invalid
//             (quirk c: pass2(k) == pass1(k+1), proven bit-identical).
//   `force` — unconditional force evaluation (post-merge recompute).
//   `force_always` — unconditional pure force pass (serial-parity mode).
//   `fused` — tiled force, then a barrier-free naive collision-detection
//             scan (upper triangle). Detection emits union-find links via
//             atomicMin and raises merge_flag; the result is identical to
//             fusing detection into the pair loop (min is order-independent,
//             each pair is unioned once).
//
// `WG` is a pipeline-creation override (autotuned over 64/128/256).
// `NTILES_MAX` is a pipeline-creation override set to ceil(n_max / WG):
// the tiled loop MUST have a workgroup-uniform trip count because it
// contains workgroupBarrier() calls.
//
// FXC (SM 5.1, DirectX 12) X4026 note: the barrier loop must contain NO
// varying control flow whatsoever — not just a uniform trip count, but no
// varying `if`/`continue`/`break` anywhere in the loop, including the
// nested consume loop. Hence `force_row` is fully branchless:
//   * the tile load is predicated arithmetically (`* f32(j < n)`);
//   * the inner consume loop has no branches at all.
// Padding lanes (j >= n) contribute exactly +0.0 (mass 0 after masking)
// and self-interaction contributes exactly +0.0 (d == 0), both
// bit-identical to skipping. Collision detection lives in `fused`'s
// barrier-free scan, where varying control flow is fine.

override WG: u32 = 128u;
override NTILES_MAX: u32 = 1024u;

var<workgroup> tile: array<vec4<f32>, 256>; // (x, y, mass, _) per partner

// One tiled force row (pure force). Returns the accumulated acceleration.
fn force_row(i: u32, n: u32, lid: u32) -> vec2<f32> {
    let pi = pos_a[i];
    var ax = vec2<f32>(0.0, 0.0);
    // Uniform trip count (see NTILES_MAX above): every thread in the
    // workgroup executes both barriers exactly NTILES_MAX times, and the
    // loop body contains no varying control flow (see header note).
    for (var t = 0u; t < NTILES_MAX; t++) {
        let j = t * WG + lid;
        // Clamp the load index into range (n >= 1 always; max() guards the
        // hypothetical n == 0), then zero padding lanes arithmetically.
        // No `if (j < n)` branch — FXC X4026.
        let jc = min(j, max(n, 1u) - 1u);
        let m = f32(j < n);
        tile[lid] = vec4<f32>(pos_a[jc].x, pos_a[jc].y, mass_a[jc], 0.0) * m;
        workgroupBarrier();
        // Fully branchless consume loop: no `if`, no `continue`.
        for (var k = 0u; k < WG; k++) {
            let other = tile[k];
            let d = other.xy - pi;
            let r2 = dot(d, d) + SOFTENING_SQ;
            let inv = 1.0 / sqrt(r2);
            let f = G * inv * inv * inv * other.z;
            ax += d * f;
        }
        workgroupBarrier();
    }
    return ax;
}

@compute @workgroup_size(WG)
fn pass1(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    // Uniform branch before any barrier: the whole dispatch is a no-op when
    // the cache is valid.
    if (atomicLoad(&accel_valid[0]) == 1u) {
        return;
    }
    let i = gid.x;
    let n = n_buf[0];
    // Every thread runs the tiled loop (barriers must converge); only the
    // write-back is guarded. Out-of-range threads accumulate into ax and
    // discard it. pos_a[i] is always in-bounds: the dispatch covers exactly
    // the n_max-sized buffer.
    let ax = force_row(i, n, lid.x);
    if (i < n) {
        accel_a[i] = ax;
    }
    // No thread reads the flag inside this dispatch, so thread 0 may set it.
    if (gid.x == 0u) {
        atomicStore(&accel_valid[0], 1u);
    }
}

@compute @workgroup_size(WG)
fn force(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = gid.x;
    let n = n_buf[0];
    // Flag-gated: skipped (O(N) early-out) when no merge happened. Uniform
    // branch before any barrier, so the whole dispatch is a cheap no-op.
    if (atomicLoad(&merge_flag[0]) == 0u) {
        return;
    }
    let ax = force_row(i, n, lid.x);
    if (i < n) {
        accel_a[i] = ax;
    }
}

// Unconditional pure force pass (serial-parity mode's post-merge pass).
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
    }
}

@compute @workgroup_size(WG)
fn fused(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = gid.x;
    let n = n_buf[0];
    // Tiled force (branchless; see force_row). Barriers converge: every
    // thread runs the full tile loop; only the write-back is guarded.
    let ax = force_row(i, n, lid.x);
    var live = false;
    if (i < n) {
        accel_a[i] = ax;
        live = true;
    }
    // Collision detection: naive upper-triangle scan. This loop contains NO
    // barriers, so varying control flow is fine (no X4026). Each overlapping
    // pair (i, j), i < j, emits one union-find link; atomicMin is
    // order-independent, so the resulting parent[] and merge_flag match the
    // old fused pair loop exactly.
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
    if (gid.x == 0u) {
        atomicStore(&accel_valid[0], 1u);
    }
}
