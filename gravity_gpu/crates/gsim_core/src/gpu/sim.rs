//! [`Sim`]: the GPU-resident N-body simulation.
//!
//! Buffer plan (SoA, blueprint §3): live set `A` (`pos/vel/mass/brightness/
//! accel`), scratch set `B` for merge compaction, union-find `parent`,
//! single-word flags (`merge_flag`, `accel_valid`, `alive_count`, `n_buf`),
//! the trail ring (sized to the configured trail length), and the
//! 32-byte stats buffer. All compute
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
    /// Per-body merge-component flag (binding 20): 1 if the body's root
    /// absorbed another body this step; lets `reduce` skip the O(n) member
    /// scan for singleton components.
    pub(crate) merged_root: wgpu::Buffer,
    /// Shader-debug log buffer (4096 u32 words = 1024 records).
    pub(crate) dbg_buf: wgpu::Buffer,
    /// Shader-debug atomic append counter (single u32).
    pub(crate) dbg_count: wgpu::Buffer,
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
    /// Whether trail points are being recorded. When false, `step`
    /// skips the trail append dispatch entirely.
    pub(crate) trails_enabled: bool,
    /// Trail ring capacity in points (sized from `trail_max_age`).
    pub(crate) trail_capacity: u32,
    /// Trail length limit in sim-seconds: points older than this are
    /// culled at render and excluded from the live count. Configured
    /// via `--trail-length` (default 40, the legacy limit).
    pub(crate) trail_max_age: f32,
    /// Physics steps executed.
    pub step_count: u64,
    /// CPU-tracked acceleration cache validity (replaces the retired GPU
    /// `accel_valid` flag protocol): true when `accel_a` holds accelerations
    /// for the current positions, so the tiled `pass1` dispatch can be
    /// skipped. Set false by spawn/upload (fresh state) and by
    /// `--no-accel-cache` each step; set true after pass1/fused/force.
    pub(crate) accel_valid_cpu: bool,
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
    crate::log_debug!("gpu buffer {label}: {size} bytes");
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
        let ctx = GpuContext::open(backend, adapter_index, cfg.dxc_path.as_deref(), cfg.no_dxc)?;
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
        let merged_root = storage_buf(
            device,
            &mut vram,
            "merged_root",
            bytes_f32(n),
            wgpu::BufferUsages::empty(),
        );
        // Shader-debug log: 4096-word ring of 4-word records
        // (kernel_tag, thread_id, ax_bits, ay_bits) = 1024 records max,
        // plus an atomic append counter. Always allocated (16 KiB); only
        // written when SHADER_DEBUG=1.
        let dbg_buf = storage_buf(
            device,
            &mut vram,
            "dbg_buf",
            4096 * 4,
            wgpu::BufferUsages::COPY_SRC,
        );
        let dbg_count = storage_buf(
            device,
            &mut vram,
            "dbg_count",
            4,
            wgpu::BufferUsages::COPY_SRC,
        );

        // Trail ring sized to the configured length limit
        // (trail_max_age seconds of points per particle at the 45 Hz
        // sample rate) whenever trails can be recorded or shown; a
        // token ring when they cannot (bench/headless without
        // --trails), since nothing is ever appended then.
        let points_per_particle =
            ((cfg.trail_max_age / TRAIL_SAMPLE_INTERVAL_SECONDS).round() as usize).max(1);
        let trail_capacity = if cfg.trails || surface_format.is_some() {
            cfg.n_max * points_per_particle
        } else {
            TRAIL_RING_MIN_POINTS
        };
        let trail_ring = storage_buf(
            device,
            &mut vram,
            "trail_ring",
            (trail_capacity * TRAIL_POINT_BYTES) as u64,
            wgpu::BufferUsages::empty(),
        );
        let trail_head = word_buf(device, &mut vram, "trail_head");

        let stats_buf = storage_buf(device, &mut vram, "stats", 32, wgpu::BufferUsages::COPY_SRC);
        crate::log_debug!("gpu buffer stats_staging: 32 bytes");
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
        crate::log_debug!("gpu buffer trail_uni: 32 bytes");
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
        // Binding 20: merged_root flag array (storage, read_write).
        entries.push(storage_rw(20));
        // Bindings 21/22: shader-debug log buffer + atomic append counter
        // (storage, read_write). Always bound; only written when the
        // SHADER_DEBUG pipeline override is 1 (`--shader-debug`).
        entries.push(storage_rw(21));
        entries.push(storage_rw(22));
        let bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gsim compute layout"),
            entries: &entries,
        });
        crate::log_debug!("compute bind-group layout created (23 bindings)");
        let compute_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gsim compute pipeline layout"),
            bind_group_layouts: &[Some(&bg_layout)],
            immediate_size: 0,
        });
        crate::log_debug!("compute pipeline layout created");

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
                bg_entry(20, &merged_root),
                bg_entry(21, &dbg_buf),
                bg_entry(22, &dbg_count),
                bg_entry(14, &trail_ring),
                bg_entry(15, &trail_head),
                bg_entry(16, &stats_buf),
                bg_entry(17, &partials_f),
                bg_entry(18, &partials_c),
                bg_entry(19, &trail_uni),
            ],
        });
        crate::log_debug!("compute bind group created (23 entries)");

        let wg_size = if cfg.workgroup_size == 0 {
            128
        } else {
            cfg.workgroup_size
        };
        // Uniform tile-loop bound for the tiled force kernels (see the
        // NTILES_MAX override in force_tiled.wgsl): n only shrinks, so the
        // startup capacity is a valid bound for the whole run.
        let ntiles_max = (cfg.n_max as u32).div_ceil(wg_size);
        let pipes = ComputePipelines::new(
            device,
            &compute_layout,
            wg_size,
            ntiles_max,
            cfg.shader_debug,
        );

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
        let trails_enabled = cfg.trails;
        let trail_max_age = cfg.trail_max_age;
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
            merged_root,
            dbg_buf,
            dbg_count,
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
            trails_enabled,
            trail_capacity: trail_capacity as u32,
            trail_max_age,
            step_count: 0,
            accel_valid_cpu: false,
        };
        crate::log_debug!(
            "sim created: n_max={} vram={:.1} MiB workgroup={}",
            sim.cfg.n_max,
            sim.vram_bytes as f64 / 1048576.0,
            sim.wg_size
        );
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
        let ntiles_max = (self.cfg.n_max as u32).div_ceil(wg);
        self.pipes
            .set_workgroup_size(&self.ctx.device, &self.compute_layout, wg, ntiles_max);
        self.wg_size = wg;
    }

    /// Workgroup count covering the full capacity buffer.
    pub(crate) fn dispatch(&self, wg: u32) -> u32 {
        (self.cfg.n_max as u32).div_ceil(wg)
    }
}
