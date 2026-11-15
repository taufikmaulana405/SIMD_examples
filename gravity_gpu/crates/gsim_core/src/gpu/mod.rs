//! wgpu compute driver: buffers, WGSL pipelines, stepping, timestamps, bench.
//!
//! # GPU-only rule
//!
//! Force evaluation, integration, collision detection, merge, trail
//! storage/aging and rendering all run on the GPU. The CPU submits command
//! buffers and reads back only the 32-byte stats buffer (once per frame for
//! the HUD; test/bench modes may additionally map particle buffers for
//! parity checks).
//!
//! # Step structure (tiled / parallel mode)
//!
//! ```text
//! pass1 (cached) -> integrate -> fused force+detect -> compress -> reduce
//!   -> copy_back -> force (iff merged) -> finish_velocity
//! ```
//! Quirk c halves the force cost: the `accel` buffer persists across steps
//! and `pass1` is a no-op unless the cache is invalid (first step, respawn,
//! or `--no-accel-cache`). Collision detection is fused into the single
//! force pass; when a merge happened the survivors' accelerations are
//! recomputed once (`force` entry, flag-gated to an O(N) early-out when no
//! merge occurred).

pub mod bench;
pub mod context;
pub(crate) mod pipelines;
pub mod render;
pub mod run;
pub mod shaders;
pub mod sim;
pub mod timing;

pub use bench::BenchmarkRunner;
pub use context::{GpuContext, SimConfig};
pub use render::RenderParams;
pub use sim::Sim;
pub use timing::GpuTimer;
