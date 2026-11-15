// Integration kernels (Verlet halves). `integrate` also owns the per-step
// transient-state reset: merge_flag, alive_count (thread 0) and the
// union-find identity parent[i] = i. Dispatch ordering guarantees the
// previous step's merge work is complete before this runs.

@compute @workgroup_size(256)
fn integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i == 0u) {
        atomicStore(&merge_flag[0], 0u);
        atomicStore(&alive_count[0], 0u);
    }
    let n = n_buf[0];
    if (i >= n) {
        return;
    }
    atomicStore(&parent[i], i);
    // v += a*dt/2 ; x += v*dt  (legacy kick+drift; exact op association
    // `(a*DT)*0.5` matches the scalar CPU reference bit-for-bit)
    let v = vel_a[i] + accel_a[i] * DT * 0.5;
    vel_a[i] = v;
    pos_a[i] = pos_a[i] + v * DT;
}

@compute @workgroup_size(256)
fn finish(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = n_buf[0];
    if (i >= n) {
        return;
    }
    // v += a*dt/2  (legacy second kick; exact op association `(a*DT)*0.5`)
    vel_a[i] = vel_a[i] + accel_a[i] * DT * 0.5;
}
