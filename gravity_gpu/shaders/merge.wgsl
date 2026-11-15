// Merge kernels.
//
// Parallel mode (default): collision graph -> connected components ->
// mass/momentum-conserving merge. `compress` chases parent links to the
// canonical root (links strictly decrease, so chains terminate).
// `reduce` accumulates each component in ascending member order
// (deterministic sums) and compacts survivors into the B scratch buffers via
// an atomic slot counter; `copy_back` moves them to A, shrinks n and zeroes
// the tail so the renderer culls ghosts. All three early-out when
// merge_flag == 0, so merge-free steps pay only O(N) dispatch overhead.
//
// `serial_parity`: single-threaded kernel replicating the legacy merge scan
// EXACTLY (ascending first-index scan, first overlap wins, swap_remove,
// restart after every merge) for Stage-3 parity tests. Operates in place.

@compute @workgroup_size(256)
fn compress(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (atomicLoad(&merge_flag[0]) == 0u) {
        return;
    }
    let i = gid.x;
    let n = n_buf[0];
    if (i >= n) {
        return;
    }
    var r = i;
    var pr = atomicLoad(&parent[r]);
    while (pr != r) {
        r = pr;
        pr = atomicLoad(&parent[r]);
    }
    atomicStore(&parent[i], r);
    if (r != i) {
        // Non-root member: its component is nontrivial. (Benign race:
        // concurrent threads only ever write 1.)
        atomicOr(&merged_root[r], 1u);
    }
}

@compute @workgroup_size(256)
fn reduce(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (atomicLoad(&merge_flag[0]) == 0u) {
        return;
    }
    let i = gid.x;
    let n = n_buf[0];
    if (i >= n) {
        return;
    }
    if (atomicLoad(&parent[i]) != i) {
        return; // not a component root
    }
    if (atomicLoad(&merged_root[i]) == 0u) {
        // Singleton component: the center of mass is the body itself, so
        // skip the O(n) member scan (this turns merge steps from O(n^2)
        // into O(n)). Up to 1 ulp from the scan's `(m*x)/m` rounding —
        // physically identical.
        let slot = atomicAdd(&alive_count[0], 1u);
        pos_b[slot] = pos_a[i];
        vel_b[slot] = vel_a[i];
        mass_b[slot] = mass_a[i];
        bright_b[slot] = bright_a[i];
        return;
    }
    // Ascending member scan: deterministic accumulation order.
    var m_sum = 0.0;
    var x_sum = 0.0;
    var y_sum = 0.0;
    var vx_sum = 0.0;
    var vy_sum = 0.0;
    var b_sum = 0.0;
    for (var j = 0u; j < n; j++) {
        if (atomicLoad(&parent[j]) == i) {
            let m = mass_a[j];
            m_sum += m;
            x_sum += m * pos_a[j].x;
            y_sum += m * pos_a[j].y;
            vx_sum += m * vel_a[j].x;
            vy_sum += m * vel_a[j].y;
            b_sum += m * bright_a[j];
        }
    }
    // m_sum > 0 always: j == i is a member.
    let slot = atomicAdd(&alive_count[0], 1u);
    pos_b[slot] = vec2<f32>(x_sum / m_sum, y_sum / m_sum);
    vel_b[slot] = vec2<f32>(vx_sum / m_sum, vy_sum / m_sum);
    mass_b[slot] = m_sum;
    bright_b[slot] = b_sum / m_sum;
}

@compute @workgroup_size(256)
fn copy_back(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (atomicLoad(&merge_flag[0]) == 0u) {
        return;
    }
    let i = gid.x;
    let alive = atomicLoad(&alive_count[0]);
    if (i == 0u) {
        n_buf[0] = alive;
    }
    let cap = arrayLength(&mass_a);
    if (i >= cap) {
        return;
    }
    if (i < alive) {
        pos_a[i] = pos_b[i];
        vel_a[i] = vel_b[i];
        mass_a[i] = mass_b[i];
        bright_a[i] = bright_b[i];
    } else {
        // Zero the tail: stale bodies must not render as ghosts, and every
        // kernel guards on n_buf anyway.
        mass_a[i] = 0.0;
    }
}

fn parity_find_collision(first: u32, n: u32) -> u32 {
    let ri = RADIUS_SCALE * sqrt(mass_a[first]);
    for (var j = first + 1u; j < n; j++) {
        let d = pos_a[j] - pos_a[first];
        let rs = ri + RADIUS_SCALE * sqrt(mass_a[j]);
        if (dot(d, d) <= rs * rs) {
            return j;
        }
    }
    return 0xFFFFFFFFu;
}

// Exact legacy-order merge in a single thread. Deterministic by
// construction; used only by parity tests (O(N^2) serial).
@compute @workgroup_size(1)
fn serial_parity() {
    var n = n_buf[0];
    var first = 0u;
    while (first < n) {
        var second = parity_find_collision(first, n);
        while (second != 0xFFFFFFFFu) {
            // merge_pair(first, second): mass-weighted COM/momentum/
            // brightness, result stays at slot `first`.
            let mi = mass_a[first];
            let mj = mass_a[second];
            let m = mi + mj;
            pos_a[first] = (pos_a[first] * mi + pos_a[second] * mj) / m;
            vel_a[first] = (vel_a[first] * mi + vel_a[second] * mj) / m;
            mass_a[first] = m;
            bright_a[first] =
                (bright_a[first] * mi + bright_a[second] * mj) / m;
            // swap_remove(second): last body moves into the hole.
            n -= 1u;
            if (second != n) {
                pos_a[second] = pos_a[n];
                vel_a[second] = vel_a[n];
                mass_a[second] = mass_a[n];
                bright_a[second] = bright_a[n];
            }
            second = parity_find_collision(first, n);
        }
        first += 1u;
    }
    n_buf[0] = n;
    let cap = arrayLength(&mass_a);
    for (var i = n; i < cap; i++) {
        mass_a[i] = 0.0;
    }
}
