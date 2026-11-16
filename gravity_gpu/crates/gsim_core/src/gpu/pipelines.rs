//! Compute pipeline inventory for [`Sim`](crate::gpu::sim::Sim).
//!
//! The tiled-force pipelines (`pass1`/`force`/`force_always`/`fused`) are
//! built from one WGSL module with the `WG` override constant, so the
//! autotuner can rebuild them for 64/128/256 without touching the rest.

use crate::gpu::shaders;
use std::borrow::Cow;

/// All compute pipelines owned by a [`Sim`](crate::gpu::sim::Sim).
pub(crate) struct ComputePipelines {
    pub(crate) naive_force: wgpu::ComputePipeline,
    pub(crate) detect_naive: wgpu::ComputePipeline,
    pub(crate) integrate: wgpu::ComputePipeline,
    pub(crate) finish: wgpu::ComputePipeline,
    pub(crate) compress: wgpu::ComputePipeline,
    pub(crate) reduce: wgpu::ComputePipeline,
    pub(crate) copy_back: wgpu::ComputePipeline,
    pub(crate) serial: wgpu::ComputePipeline,
    pub(crate) trail: wgpu::ComputePipeline,
    pub(crate) stats_partial: wgpu::ComputePipeline,
    pub(crate) stats_final: wgpu::ComputePipeline,
    pub(crate) pass1: wgpu::ComputePipeline,
    pub(crate) force: wgpu::ComputePipeline,
    pub(crate) force_always: wgpu::ComputePipeline,
    pub(crate) fused: wgpu::ComputePipeline,
    tiled_module: wgpu::ShaderModule,
}

impl ComputePipelines {
    /// Build every compute pipeline; `wg` sizes the tiled force kernels and
    /// `ntiles_max` (= ceil(n_max / wg)) fixes their uniform tile-loop bound
    /// (see the `NTILES_MAX` override in `force_tiled.wgsl`).
    pub(crate) fn new(
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        wg: u32,
        ntiles_max: u32,
    ) -> Self {
        let one = |name: &str, entry: &str| {
            let src = shaders::with_common(shaders::source(name));
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(src)),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("gsim {name}::{entry}")),
                layout: Some(layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let tiled_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("force_tiled"),
            // The tile loop is unrolled at runtime (see
            // shaders::force_tiled_unrolled): FXC (SM 5.1, DX12) rejects
            // barriers in loops containing nested loops (X4026), even when
            // uniform and branchless. Unrolling eliminates the nesting.
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shaders::with_common(
                &shaders::force_tiled_unrolled(ntiles_max, wg),
            ))),
        });
        let tiled = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("gsim tiled::{entry}")),
                layout: Some(layout),
                module: &tiled_module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("WG", f64::from(wg))],
                    zero_initialize_workgroup_memory: false,
                },
                cache: None,
            })
        };
        let pipes = Self {
            naive_force: one("force_naive", "main"),
            detect_naive: one("detect_naive", "main"),
            integrate: one("integrate", "integrate"),
            finish: one("integrate", "finish"),
            compress: one("merge", "compress"),
            reduce: one("merge", "reduce"),
            copy_back: one("merge", "copy_back"),
            serial: one("merge", "serial_parity"),
            trail: one("trail", "append"),
            stats_partial: one("stats", "partial"),
            stats_final: one("stats", "finalize"),
            pass1: tiled("pass1"),
            force: tiled("force"),
            force_always: tiled("force_always"),
            fused: tiled("fused"),
            tiled_module,
        };
        // Pipeline inventory: which WGSL modules and entry points were
        // compiled, and with which overrides. The tiled force module is
        // generated at runtime with the tile loop fully unrolled (FXC
        // X4026 workaround); WG/NTILES_MAX are pipeline overrides.
        crate::log_debug!(
            "shader inventory (compute):\n    force_naive::main\n    detect_naive::main\n    integrate::integrate\n    integrate::finish\n    merge::compress\n    merge::reduce\n    merge::copy_back\n    merge::serial_parity\n    trail::append\n    stats::partial\n    stats::finalize\n    force_tiled::pass1 (overrides WG={wg}, NTILES_MAX={ntiles_max}; tile loop unrolled)\n    force_tiled::force (overrides WG={wg}, NTILES_MAX={ntiles_max}; tile loop unrolled)\n    force_tiled::force_always (overrides WG={wg}, NTILES_MAX={ntiles_max}; tile loop unrolled)\n    force_tiled::fused (overrides WG={wg}, NTILES_MAX={ntiles_max}; tile loop unrolled)"
        );
        pipes
    }

    /// Rebuild the four tiled-force pipelines for a new workgroup size
    /// (autotune); everything else is untouched.
    pub(crate) fn set_workgroup_size(
        &mut self,
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        wg: u32,
        ntiles_max: u32,
    ) {
        // The tile loop is unrolled at runtime, so the module itself must be
        // rebuilt for the new (wg, ntiles_max), not just the pipelines.
        self.tiled_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("force_tiled"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shaders::with_common(
                &shaders::force_tiled_unrolled(ntiles_max, wg),
            ))),
        });
        let tiled = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("gsim tiled::{entry}")),
                layout: Some(layout),
                module: &self.tiled_module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("WG", f64::from(wg))],
                    zero_initialize_workgroup_memory: false,
                },
                cache: None,
            })
        };
        self.pass1 = tiled("pass1");
        self.force = tiled("force");
        self.force_always = tiled("force_always");
        self.fused = tiled("fused");
        crate::log_debug!(
            "shader inventory: tiled pipelines rebuilt (overrides WG={wg}, NTILES_MAX={ntiles_max})"
        );
    }
}
