// Naive force kernel (benchmark ladder rung 2): one thread per particle,
// global memory, exact all-pairs, FIXED summation order (partners visited in
// ascending j, j == i skipped), no cross-thread writes (no float atomics).
// Mirrors `gsim_core::cpu::calculate_accelerations` bit-for-bit.

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = n_buf[0];
    if (i >= n) {
        return;
    }
    let pi = pos_a[i];
    var ax = vec2<f32>(0.0, 0.0);
    for (var j = 0u; j < n; j++) {
        if (j == i) {
            continue;
        }
        let d = pos_a[j] - pi;
        let r2 = dot(d, d) + SOFTENING_SQ;
        let inv = 1.0 / sqrt(r2);
        let f = G * inv * inv * inv * mass_a[j];
        ax += d * f;
    }
    accel_a[i] = ax;
    dbg_log(DBG_NAIVE, i, ax.x, ax.y);
}
