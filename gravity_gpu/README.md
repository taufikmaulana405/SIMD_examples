# gravity_gpu

Native desktop GPU version of the gravity simulation. This project is a new project derived primarily from [`gravity_simd_avx2`](../gravity_simd_avx2), including its 10,000-particle baseline, initialization profile, gravity constants, and velocity-Verlet step. The AVX2 CPU acceleration is replaced with GPU compute through `wgpu`.

## Run

From this directory:

```bash
cargo run --release
```

`wgpu` selects the available Vulkan, Direct3D 12, Metal, or other supported backend. A working graphics driver is required. The program prints the selected adapter and backend at startup.

## Controls

| Input | Action |
|---|---|
| `Space` | Pause/resume |
| `R` | Reset the particle cloud |
| `↑` / `↓` | Double / halve time scale (0.125x–4x) |
| `T` | Toggle particle trails |
| Mouse wheel | Zoom (0.05x–5x) |
| Close window | Exit |

The native renderer continuously presents frames, including while paused. Trails use a fixed 90-snapshot (2-second) GPU budget and a reusable bounded readback buffer; they fade with the AVX2 reference's 10-second half-life. Collision merging is throttled and disabled for the default 10,000-particle run because the current serialized fallback is not safe to execute every frame. GPU results may differ slightly from AVX2 because of floating-point evaluation order.

The window title provides a compact live HUD with simulation state, FPS, time scale, zoom, and trail count. A full in-window statistics overlay is planned for the next pass; current total mass remains available to the renderer and active particle state is maintained in a GPU control buffer.


## GPU design

- Gravity is calculated in a WGSL compute shader using the same O(N²) all-pairs model as `gravity_simd_avx2`.
- Two storage buffers are used for explicit ping-pong integration.
- Collision/merge runs after position drift and before the second acceleration evaluation; it uses mass-weighted position, velocity, and brightness and repeatedly merges the current body with overlapping bodies.
- The active particle count is stored in a GPU control buffer, and dead/merged records are excluded by the renderer.
- Particle rendering is instanced and stays on the GPU.
- The initial particle cloud and lightweight uniforms are uploaded by the CPU; collision physics does not require particle readback.

The all-pairs solver performs O(N²) work, so performance depends on the GPU and driver. To test a weaker GPU, lower `PARTICLE_COUNT` in `src/main.rs`.

## Current limitations

- The merge kernel uses a single serialized invocation for deterministic swap-remove-like semantics. It is enabled only for runs at or below 4,000 particles and executes every eighth physics step; the default 10,000-particle run skips it to avoid GPU watchdogs.
- Full AVX2-style in-window HUD statistics (largest mass and center of mass) still require a small asynchronous GPU reduction/readback pass.
- Collision convergence currently follows the available compacted pass; a future pass can add multiple GPU convergence iterations for chains of newly enlarged bodies.
- The font file is retained for a future native text overlay; no `wgpu_glyph` dependency is used because its older releases target incompatible `wgpu` versions.
- `egui-wgpu` compatibility has been checked but is not yet wired into the render pass.

The all-pairs solver performs O(N²) work, so performance depends on the GPU and driver. To test a weaker GPU, lower `PARTICLE_COUNT` in `src/main.rs`.

The all-pairs solver performs O(N²) work, so performance depends on the GPU and driver. To test a weaker GPU, lower `PARTICLE_COUNT` in `src/main.rs`.
