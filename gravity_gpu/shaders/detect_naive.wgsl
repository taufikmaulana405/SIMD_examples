// Naive collision detection (ladder rung 2): one thread per particle,
// global memory, ascending j scan from i+1 (halves the pair tests; union is
// symmetric). Overlaps emit union(i, j) via atomicMin on parent[hi] and
// raise merge_flag. Resulting parent[] is deterministic (min is
// order-independent).

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = n_buf[0];
    if (i >= n) {
        return;
    }
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
