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
| `↑` / `↓` | Zoom in/out |
| Mouse wheel | Zoom |
| Close window | Exit |

## GPU design

- Gravity is calculated in a WGSL compute shader using the same O(N²) all-pairs model as `gravity_simd_avx2`.
- Two storage buffers are used for explicit ping-pong integration.
- Particle rendering is instanced and stays on the GPU.
- The initial particle cloud and lightweight uniforms are uploaded by the CPU; there is no per-frame particle readback.
- The first version intentionally omits CPU collision compaction so that the main simulation remains GPU-resident. Collision handling can be added later as a GPU compute pass.

The all-pairs solver performs O(N²) work, so performance depends on the GPU and driver. To test a weaker GPU, lower `PARTICLE_COUNT` in `src/main.rs`.
