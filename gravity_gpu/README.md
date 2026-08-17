# gravity_gpu

Native desktop GPU version of the gravity simulation. This is a separate project derived primarily from [`gravity_simd_avx2`](../gravity_simd_avx2): it keeps the 10,000-particle baseline, initialization profile, constants, controls, trails, and velocity-Verlet ordering while replacing AVX2 arithmetic with `wgpu` compute shaders.

## Run

```bash
cargo run --release
```

`wgpu` selects a supported Vulkan, Direct3D 12, Metal, or other backend. A working graphics driver and display server are required.

## Controls

| Input | Action |
|---|---|
| `Space` | Pause/resume |
| `R` | Reset the particle cloud |
| `↑` / `↓` | Double / halve time scale (0.125x–4x) |
| `T` | Toggle particle trails |
| Mouse wheel | Zoom (0.05x–5x) |
| Close window | Exit |

The renderer continues presenting frames while paused. Trails use a fixed bounded GPU allocation and the AVX2 reference's fading behavior. GPU floating-point evaluation order can produce small numerical differences from the CPU reference.

## Collision behavior

The default 10,000-particle configuration uses a bounded, GPU-resident collision path:

1. Live bodies are inserted into a fixed-size hashed uniform grid.
2. Each body searches a bounded 3×3 neighborhood and proposes one overlapping partner.
3. Only mutual proposals are merged, so a body cannot be written by two merges in one round.
4. Survivors and weighted merged bodies are compacted into the other particle buffer.
5. Two bounded rounds are run after each drift step so newly enlarged bodies can participate in another merge.

The grid uses fixed O(N) auxiliary storage and caps bucket visits. Dense buckets set an overflow diagnostic instead of entering an unbounded shader loop. Scalable GPU collision preserves overlap geometry, mass-weighted position/velocity/brightness, conservation, and compacted active-prefix semantics. Because matching is parallel, its grouping and merge order are not bit-for-bit identical to AVX2.

For small configurations up to 2,000 particles, the exact-style serialized swap-remove fallback remains available every eighth physics step. It is never used for the default 10,000-particle run because its single-invocation O(N²) scan can monopolize a GPU and make a desktop unresponsive.

## GPU design

- Gravity is an all-pairs O(N²) WGSL compute pass, matching the AVX2 reference model.
- Two particle storage buffers provide explicit ping-pong integration and collision compaction.
- The step order is drift, bounded collision rounds, final acceleration/half-step, then rendering.
- Active count and collision output state remain in GPU storage; dead records are cleared from the active tail.
- Particle and trail rendering are instanced and remain GPU-resident.
- CPU readback is limited to the bounded trail/statistics snapshot and is not used to perform collision physics.

The all-pairs gravity solver is intentionally demanding. Lower `PARTICLE_COUNT` in `src/main.rs` when testing a weaker GPU. The broad-phase collision path avoids materializing candidate pairs, whose worst-case count would be approximately 50 million at 10,000 particles.

## HUD statistics

The in-window HUD reports the authoritative GPU statistics for the completed simulation buffer:

- current live particle count;
- largest live particle mass and its percentage of the initial total mass;
- current position of the largest particle;
- signed particle-count change per second; and
- signed largest-mass change per second.

The rates are calculated from successive synchronized GPU samples. They start at zero after reset and remain signed, so a merge normally produces a negative particle-count rate and a positive largest-mass rate. Statistics are sampled independently of trail visibility.

## Current limitations

- GPU arithmetic is not expected to be bitwise identical to AVX2.
- The statistics snapshot scans the fixed particle capacity and uses a synchronized readback; this is bounded but can be optimized later with a GPU reduction.
- Grid overflow is reported by the collision shader and causes that round to remain bounded; extremely dense configurations may therefore merge more slowly.
- `wgpu_glyph` is not used because its older releases target an incompatible `wgpu` version.
