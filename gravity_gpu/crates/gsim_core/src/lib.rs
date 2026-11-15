#![forbid(unsafe_code)]
//! `gsim_core`: headless simulation core for the GPU gravity port.
//!
//! This crate owns everything that does not need a window:
//! - [`constants`]: physics/render constants (AVX-512 legacy defaults, verbatim)
//! - [`rng`]: bit-exact replica of `quad-rand 0.2.3`'s PCG generator
//! - [`spawn`]: seeded initial particle cloud (bit-identical to legacy)
//! - [`cpu`]: scalar CPU reference step (parity oracle for Stage 3)
//! - [`cli`]: `--bench/--headless/--seed/...` argument parsing
//! - [`telemetry`]: JSONL telemetry writer (stage 2 schema)
//! - [`stats`]: the tiny per-frame stats struct
//! - [`gpu`]: wgpu compute driver — buffers, WGSL pipelines, stepping,
//!   timestamp queries, and the benchmark ladder.
//!
//! # GPU-only rule
//!
//! Force calculation, integration, collision/merge, trail storage and aging,
//! and rendering all happen on the GPU. The CPU only submits commands and
//! asynchronously reads back the 32-byte [`stats::Stats`] struct once per
//! frame for the HUD (test/bench modes may read back more).

pub mod cli;
pub mod constants;
pub mod cpu;
pub mod gpu;
pub mod rng;
pub mod spawn;
pub mod stats;
pub mod telemetry;
