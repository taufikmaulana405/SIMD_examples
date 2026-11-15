// Tiled force kernel (benchmark ladder rung 3): shared-memory tiling with
// vec4-packed partner tiles, one thread per row i. Partners are streamed
// tile-by-tile in ascending global j order, so the accumulation order is
// bit-identical to `force_naive` (and to `gsim_core::cpu`).
//
// Entry points:
//   `pass1` — computes only when the cross-step cache is invalid
//             (quirk c: pass2(k) == pass1(k+1), proven bit-identical).
//   `force` — unconditional force evaluation (post-merge recompute).
//   `fused` — force + collision detection in the same pair loop; emits
//             union-find links via atomicMin and raises merge_flag.
//
// `WG` is a pipeline-creation override (autotuned over 64/128/256).

override WG: u32 = 128u;

var<workgroup> tile: array<vec4<f32>, 256>; // (x, y, mass, _) per partner

// One tiled force row. When `detect` is true, overlapping pairs also emit
// union(i, j) links. Returns the accumulated acceleration.
fn force_row(
    i: u32,
    n: u32,
    lid: u32,
    detect: bool,
    ri: f32,
) -> vec2<f32> {
    let pi = pos_a[i];
    var ax = vec2<f32>(0.0, 0.0);
    let ntiles = (n + WG - 1u) / WG;
    for (var t = 0u; t < ntiles; t++) {
        let j = t * WG + lid;
        var packed = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        if (j < n) {
            packed = vec4<f32>(pos_a[j].x, pos_a[j].y, mass_a[j], 0.0);
        }
        tile[lid] = packed;
        workgroupBarrier();
        for (var k = 0u; k < WG; k++) {
            let gj = t * WG + k;
            if (gj >= n || gj == i) {
                continue;
            }
            let other = tile[k];
            let d = other.xy - pi;
            let r2 = dot(d, d) + SOFTENING_SQ;
            let inv = 1.0 / sqrt(r2);
            let f = G * inv * inv * inv * other.z;
            ax += d * f;
            if (detect) {
                let rr = RADIUS_SCALE * sqrt(other.z);
                let rs = ri + rr;
                if (dot(d, d) <= rs * rs) {
                    let lo = min(i, gj);
                    let hi = max(i, gj);
                    atomicMin(&parent[hi], lo);
                    atomicOr(&merge_flag[0], 1u);
                }
            }
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
    let ax = force_row(i, n, lid.x, false, 0.0);
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
    let ax = force_row(i, n, lid.x, false, 0.0);
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
    let ax = force_row(i, n, lid.x, false, 0.0);
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
    var ri = 0.0;
    var do_detect = false;
    if (i < n) {
        ri = RADIUS_SCALE * sqrt(mass_a[i]);
        do_detect = true;
    }
    let ax = force_row(i, n, lid.x, do_detect, ri);
    if (i < n) {
        accel_a[i] = ax;
    }
    if (gid.x == 0u) {
        atomicStore(&accel_valid[0], 1u);
    }
}
