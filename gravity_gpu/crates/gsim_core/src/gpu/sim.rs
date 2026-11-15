//! [`Sim`]: the GPU-resident N-body simulation.
//!
//! Buffer plan (SoA, blueprint §3): live set `A` (`pos/vel/mass/brightness/
//! accel`), scratch set `B` for merge compaction, union-find `parent`,
//! single-word flags (`merge_flag`, `accel_valid`, `alive_count`, `n_buf`),
//! the 4 M-point trail ring, and the 32-byte stats buffer. All compute
//! shaders share one bind-group layout; see `shaders/common.wgsl`.

use crate::cli::Backend;
use crate::constants::*;
use crate::gpu::context::{GpuContext, SimConfig};
use crate::gpu::pipelines::ComputePipelines;
use crate::gpu::timing::GpuTimer;

/// Bytes for `n` `vec2<f32>` elements.
fn bytes_vec2(n: usize) -> u64 {
    (n * 8) as u64
}

/// Bytes for `n` `f32` elements.
fn bytes_f32(n: usize) -> u64 {
    (n * 4) as u64
}

/// The GPU simulation: owns every buffer and pipeline.
pub struct Sim {
    /// Wgpu context.
    pub ctx: GpuContext,
    pub(crate) cfg: SimConfig,
    /// Total VRAM budgeted by all created buffers.
    pub(crate) vram_bytes: u64,
    /// Tiled-force workgroup size in use.
    pub(crate) wg_size: u32,

    // Live particle set A.
    pub(crate) pos_a: wgpu::Buffer,
    pub(crate) vel_a: wgpu::Buffer,
    pub(crate) mass_a: wgpu::Buffer,
    pub(crate) bright_a: wgpu::Buffer,
    pub(crate) accel_a: wgpu::Buffer,
    // Compaction scratch set B. (Fields are not read after construction:
    // the compute bind group holds its own references; they are kept here
    // to document the buffer inventory.)
    #[allow(dead_code)]
    pub(crate) pos_b: wgpu::Buffer,
    #[allow(dead_code)]
    pub(crate) vel_b: wgpu::Buffer,
    #[allow(dead_code)]
    pub(crate) mass_b: wgpu::Buffer,
    #[allow(dead_code)]
    pub(crate) bright_b: wgpu::Buffer,
    // Merge / control words. (Kept for inventory; live references live in
    // the compute bind group.)
    #[allow(dead_code)]
    pub(crate) parent: wgpu::Buffer,
    pub(crate) merge_flag: wgpu::Buffer,
    pub(crate) accel_valid: wgpu::Buffer,
    pub(crate) alive_count: wgpu::Buffer,
    pub(crate) n_buf: wgpu::Buffer,
    // Trails.
    pub(crate) trail_ring: wgpu::Buffer,
    pub(crate) trail_head: wgpu::Buffer,
    // Stats + readback staging.
    pub(crate) stats_buf: wgpu::Buffer,
    pub(crate) stats_staging: wgpu::Buffer,
    // Kept for inventory; live references live in the bind group.
    #[allow(dead_code)]
    pub(crate) partials_f: wgpu::Buffer,
    #[allow(dead_code)]
    pub(crate) partials_c: wgpu::Buffer,
    pub(crate) trail_uni: wgpu::Buffer,

    pub(crate) compute_layout: wgpu::PipelineLayout,
    pub(crate) compute_bg: wgpu::BindGroup,

    // Compute pipelines (tiled force rebuilt by autotune).
    pub(crate) pipes: ComputePipelines,

    // Render resources.
    pub(crate) p_render_particles: Option<wgpu::RenderPipeline>,
    pub(crate) p_render_trails: Option<wgpu::RenderPipeline>,
    pub(crate) render_bg_particles: Option<wgpu::BindGroup>,
    pub(crate) render_bg_trails: Option<wgpu::BindGroup>,
    pub(crate) cam_particles: wgpu::Buffer,
    pub(crate) cam_trails: wgpu::Buffer,

    /// Per-pass GPU timer.
    pub timer: GpuTimer,
    /// Sim seconds elapsed (drives trail sampling).
    pub sim_time: f32,
    pub(crate) trail_acc: f32,
    /// Physics steps executed.
    pub step_count: u64,
}

/// Bind-group entry borrowing a whole buffer.
fn bg_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

/// Create a STORAGE|COPY_DST buffer, accounting VRAM.
fn storage_buf(
    device: &wgpu::Device,
    vram: &mut u64,
    label: &str,
    size: u64,
    extra: wgpu::BufferUsages,
) -> wgpu::Buffer {
    *vram += size;
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra,
        mapped_at_creation: false,
    })
}

/// 4-byte control word buffer (also CPU-readable for readback).
fn word_buf(device: &wgpu::Device, vram: &mut u64, label: &str) -> wgpu::Buffer {
    storage_buf(device, vram, label, 4, wgpu::BufferUsages::COPY_SRC)
}

impl Sim {
    /// Build context, buffers and pipelines. `surface_format` enables the
    /// render pipelines (windowed mode); pass `None` for headless.
    pub fn new(
        backend: Backend,
        adapter_index: usize,
        cfg: SimConfig,
        surface_format: Option<wgpu::TextureFormat>,
    ) -> Result<Self, String> {
        let ctx = GpuContext::open(backend, adapter_index)?;
        Self::from_context(ctx, cfg, surface_format)
    }

    /// Build buffers and pipelines from an existing context (windowed mode:
    /// the app creates the surface from `ctx.instance()` first, then learns
    /// the surface format from `ctx.adapter()`).
    pub fn from_context(
        ctx: GpuContext,
        cfg: SimConfig,
        surface_format: Option<wgpu::TextureFormat>,
    ) -> Result<Self, String> {
        let device = &ctx.device;
        let n = cfg.n_max;
        let mut vram = 0u64;

        // (helper fns `storage_buf`/`word_buf` below take `&mut vram`;
        // closures cannot be reborrowed across calls, hence free fns)

        // Particle sets (A live, B scratch); A is COPY_SRC for test readback.
        let copy_src = wgpu::BufferUsages::COPY_SRC;
        let pos_a = storage_buf(device, &mut vram, "pos_a", bytes_vec2(n), copy_src);
        let vel_a = storage_buf(device, &mut vram, "vel_a", bytes_vec2(n), copy_src);
        let mass_a = storage_buf(device, &mut vram, "mass_a", bytes_f32(n), copy_src);
        let bright_a = storage_buf(device, &mut vram, "bright_a", bytes_f32(n), copy_src);
        let accel_a = storage_buf(device, &mut vram, "accel_a", bytes_vec2(n), copy_src);
        let pos_b = storage_buf(device, &mut vram, "pos_b", bytes_vec2(n), copy_src);
        let vel_b = storage_buf(device, &mut vram, "vel_b", bytes_vec2(n), copy_src);
        let mass_b = storage_buf(device, &mut vram, "mass_b", bytes_f32(n), copy_src);
        let bright_b = storage_buf(device, &mut vram, "bright_b", bytes_f32(n), copy_src);

        let parent = storage_buf(
            device,
            &mut vram,
            "parent",
            bytes_f32(n),
            wgpu::BufferUsages::empty(),
        );
        let merge_flag = word_buf(device, &mut vram, "merge_flag");
        let accel_valid = word_buf(device, &mut vram, "accel_valid");
        let alive_count = word_buf(device, &mut vram, "alive_count");
        let n_buf = word_buf(device, &mut vram, "n_buf");

        let trail_ring = storage_buf(
            device,
            &mut vram,
            "trail_ring",
            (TRAIL_MAX_POINTS * TRAIL_POINT_BYTES) as u64,
            wgpu::BufferUsages::empty(),
        );
        let trail_head = word_buf(device, &mut vram, "trail_head");

        let stats_buf = storage_buf(device, &mut vram, "stats", 32, wgpu::BufferUsages::COPY_SRC);
        let stats_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stats_staging"),
            size: 32,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        vram += 32;
        let partials_f = storage_buf(
            device,
            &mut vram,
            "partials_f",
            1024 * 16,
            wgpu::BufferUsages::empty(),
        );
        let partials_c = storage_buf(
            device,
            &mut vram,
            "partials_c",
            1024 * 4,
            wgpu::BufferUsages::empty(),
        );
        let trail_uni = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trail_uni"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        vram += 32;

        // One shared compute bind-group layout: 20 entries (see
        // shaders/common.wgsl). Everything is read_write storage so shaders
        // may declare either access.
        let storage_rw = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let mut entries: Vec<wgpu::BindGroupLayoutEntry> = (0..19).map(storage_rw).collect();
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 19,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        let bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gsim compute layout"),
            entries: &entries,
        });
        let compute_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gsim compute pipeline layout"),
            bind_group_layouts: &[Some(&bg_layout)],
            immediate_size: 0,
        });

        let compute_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gsim compute bg"),
            layout: &bg_layout,
            entries: &[
                bg_entry(0, &pos_a),
                bg_entry(1, &vel_a),
                bg_entry(2, &mass_a),
                bg_entry(3, &bright_a),
                bg_entry(4, &accel_a),
                bg_entry(5, &pos_b),
                bg_entry(6, &vel_b),
                bg_entry(7, &mass_b),
                bg_entry(8, &bright_b),
                bg_entry(9, &parent),
                bg_entry(10, &merge_flag),
                bg_entry(11, &accel_valid),
                bg_entry(12, &alive_count),
                bg_entry(13, &n_buf),
                bg_entry(14, &trail_ring),
                bg_entry(15, &trail_head),
                bg_entry(16, &stats_buf),
                bg_entry(17, &partials_f),
                bg_entry(18, &partials_c),
                bg_entry(19, &trail_uni),
            ],
        });

        let wg_size = if cfg.workgroup_size == 0 {
            128
        } else {
            cfg.workgroup_size
        };
        let pipes = ComputePipelines::new(device, &compute_layout, wg_size);

        // Camera uniforms for the render pipelines.
        let mut cam_buf = |label: &str| {
            vram += 32;
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: 32,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let cam_particles = cam_buf("cam_particles");
        let cam_trails = cam_buf("cam_trails");

        let timer = GpuTimer::new(&ctx.device, &ctx.queue);
        let mut sim = Self {
            ctx,
            cfg,
            vram_bytes: vram,
            wg_size,
            pos_a,
            vel_a,
            mass_a,
            bright_a,
            accel_a,
            pos_b,
            vel_b,
            mass_b,
            bright_b,
            parent,
            merge_flag,
            accel_valid,
            alive_count,
            n_buf,
            trail_ring,
            trail_head,
            stats_buf,
            stats_staging,
            partials_f,
            partials_c,
            trail_uni,
            compute_layout,
            compute_bg,
            pipes,
            p_render_particles: None,
            p_render_trails: None,
            render_bg_particles: None,
            render_bg_trails: None,
            cam_particles,
            cam_trails,
            timer,
            sim_time: 0.0,
            trail_acc: 0.0,
            step_count: 0,
        };
        if let Some(format) = surface_format {
            sim.init_render(format);
        }
        Ok(sim)
    }

    /// Total VRAM budgeted by every buffer created for this sim.
    pub fn vram_bytes(&self) -> u64 {
        self.vram_bytes
    }

    /// Tiled-force workgroup size currently in use.
    pub fn workgroup_size(&self) -> u32 {
        self.wg_size
    }

    /// Rebuild the tiled-force pipelines for a new workgroup size
    /// (autotune). Other pipelines are unaffected.
    pub fn set_workgroup_size(&mut self, wg: u32) {
        self.pipes
            .set_workgroup_size(&self.ctx.device, &self.compute_layout, wg);
        self.wg_size = wg;
    }

    /// Workgroup count covering the full capacity buffer.
    pub(crate) fn dispatch(&self, wg: u32) -> u32 {
        (self.cfg.n_max as u32).div_ceil(wg)
    }
}
