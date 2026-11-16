// Stats reduction: count, total mass, max mass, center of mass, trail
// points. Two dispatches: `partial` (fixed 1024 workgroups, grid-stride) and
// `finalize` (single workgroup). The CPU maps the 32-byte stats buffer once
// per frame for the HUD — the only per-frame readback.

var<workgroup> sh_f: array<vec4<f32>, 256>; // (mass_sum, max_mass, com_x, com_y)
var<workgroup> sh_c: array<u32, 256>;

@compute @workgroup_size(256)
fn partial(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let n = n_buf[0];
    var mass = 0.0;
    var maxm = 0.0;
    var cx = 0.0;
    var cy = 0.0;
    var c = 0u;
    var i = gid.x;
    let stride = 256u * nwg.x;
    while (i < n) {
        let m = mass_a[i];
        if (m > 0.0) {
            c += 1u;
            mass += m;
            maxm = max(maxm, m);
            cx += m * pos_a[i].x;
            cy += m * pos_a[i].y;
        }
        i += stride;
    }
    sh_f[lid.x] = vec4<f32>(mass, maxm, cx, cy);
    sh_c[lid.x] = c;
    workgroupBarrier();
    // Tree reduction, fully unrolled (was: for (var s = 128u; s > 0u; s >>= 1u)).
    // The loop form places workgroupBarrier() inside naga's while(true) loop
    // emulation; FXC (DX12) takes ~9s to compile that on EVERY launch
    // (stats::partial measured 8.56s cold / 9.31s warm). Unrolled, no barrier
    // sits in a loop. Same operations in the same order: bit-identical.
    if (lid.x < 128u) {
        let a = sh_f[lid.x];
        let b = sh_f[lid.x + 128u];
        sh_f[lid.x] = vec4<f32>(a.x + b.x, max(a.y, b.y), a.z + b.z, a.w + b.w);
        sh_c[lid.x] = sh_c[lid.x] + sh_c[lid.x + 128u];
    }
    workgroupBarrier();
    if (lid.x < 64u) {
        let a = sh_f[lid.x];
        let b = sh_f[lid.x + 64u];
        sh_f[lid.x] = vec4<f32>(a.x + b.x, max(a.y, b.y), a.z + b.z, a.w + b.w);
        sh_c[lid.x] = sh_c[lid.x] + sh_c[lid.x + 64u];
    }
    workgroupBarrier();
    if (lid.x < 32u) {
        let a = sh_f[lid.x];
        let b = sh_f[lid.x + 32u];
        sh_f[lid.x] = vec4<f32>(a.x + b.x, max(a.y, b.y), a.z + b.z, a.w + b.w);
        sh_c[lid.x] = sh_c[lid.x] + sh_c[lid.x + 32u];
    }
    workgroupBarrier();
    if (lid.x < 16u) {
        let a = sh_f[lid.x];
        let b = sh_f[lid.x + 16u];
        sh_f[lid.x] = vec4<f32>(a.x + b.x, max(a.y, b.y), a.z + b.z, a.w + b.w);
        sh_c[lid.x] = sh_c[lid.x] + sh_c[lid.x + 16u];
    }
    workgroupBarrier();
    if (lid.x < 8u) {
        let a = sh_f[lid.x];
        let b = sh_f[lid.x + 8u];
        sh_f[lid.x] = vec4<f32>(a.x + b.x, max(a.y, b.y), a.z + b.z, a.w + b.w);
        sh_c[lid.x] = sh_c[lid.x] + sh_c[lid.x + 8u];
    }
    workgroupBarrier();
    if (lid.x < 4u) {
        let a = sh_f[lid.x];
        let b = sh_f[lid.x + 4u];
        sh_f[lid.x] = vec4<f32>(a.x + b.x, max(a.y, b.y), a.z + b.z, a.w + b.w);
        sh_c[lid.x] = sh_c[lid.x] + sh_c[lid.x + 4u];
    }
    workgroupBarrier();
    if (lid.x < 2u) {
        let a = sh_f[lid.x];
        let b = sh_f[lid.x + 2u];
        sh_f[lid.x] = vec4<f32>(a.x + b.x, max(a.y, b.y), a.z + b.z, a.w + b.w);
        sh_c[lid.x] = sh_c[lid.x] + sh_c[lid.x + 2u];
    }
    workgroupBarrier();
    if (lid.x < 1u) {
        let a = sh_f[lid.x];
        let b = sh_f[lid.x + 1u];
        sh_f[lid.x] = vec4<f32>(a.x + b.x, max(a.y, b.y), a.z + b.z, a.w + b.w);
        sh_c[lid.x] = sh_c[lid.x] + sh_c[lid.x + 1u];
    }
    workgroupBarrier();
    if (lid.x == 0u) {
        partials_f[wid.x] = sh_f[0];
        partials_c[wid.x] = sh_c[0];
    }
}

@compute @workgroup_size(1)
fn finalize() {
    var mass = 0.0;
    var maxm = 0.0;
    var cx = 0.0;
    var cy = 0.0;
    var c = 0u;
    for (var w = 0u; w < 1024u; w++) {
        let p = partials_f[w];
        mass += p.x;
        maxm = max(maxm, p.y);
        cx += p.z;
        cy += p.w;
        c += partials_c[w];
    }
    var com_x = 0.0;
    var com_y = 0.0;
    if (mass > 0.0) {
        com_x = cx / mass;
        com_y = cy / mass;
    }
    let head = atomicLoad(&trail_head[0]);
    let tp = min(head, TRAIL_MAX_POINTS);
    stats_buf[0] = c;
    stats_buf[1] = bitcast<u32>(mass);
    stats_buf[2] = bitcast<u32>(maxm);
    stats_buf[3] = bitcast<u32>(com_x);
    stats_buf[4] = bitcast<u32>(com_y);
    stats_buf[5] = tp;
    stats_buf[6] = 0u;
    stats_buf[7] = 0u;
}
