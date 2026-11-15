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
    /// Build every compute pipeline; `wg` sizes the tiled force kernels.
    pub(crate) fn new(device: &wgpu::Device, layout: &wgpu::PipelineLayout, wg: u32) -> Self {
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
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shaders::with_common(shaders::source(
                "force_tiled",
            )))),
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
        Self {
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
        }
    }

    /// Rebuild the four tiled-force pipelines for a new workgroup size
    /// (autotune); everything else is untouched.
    pub(crate) fn set_workgroup_size(
        &mut self,
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        wg: u32,
    ) {
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
    }
}
