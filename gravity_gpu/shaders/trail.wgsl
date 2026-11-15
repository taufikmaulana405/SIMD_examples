// Trail ring append: every live body writes one point per sample tick into
// the fixed-capacity ring (quirk f). The slot is atomicAdd(head) % CAP, so
// when full the oldest points are overwritten — visually equivalent to
// expiry in steady state. Sampling is driven by the CPU on SIM time at
// 1/45 s (quirk-e fix), passing the current sim time via trail_uni.

@compute @workgroup_size(256)
fn append(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = n_buf[0];
    if (i >= n) {
        return;
    }
    let m = mass_a[i];
    if (m <= 0.0) {
        return;
    }
    let slot = atomicAdd(&trail_head[0], 1u) % TRAIL_MAX_POINTS;
    trail_ring[slot] = TrailPoint(
        pos_a[i],
        RADIUS_SCALE * sqrt(m),
        bright_a[i],
        trail_uni.sim_time,
    );
}
