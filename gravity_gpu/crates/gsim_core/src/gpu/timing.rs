//! Per-pass GPU timing via wgpu timestamp queries, with a CPU-`Instant`
//! fallback when the adapter lacks `Features::TIMESTAMP_QUERY`.
//!
//! Sections are recorded as encoder-level timestamp pairs around each
//! compute dispatch. [`GpuTimer::resolve_and_read`] copies the query set to
//! a staging buffer and maps it; callers must label the resulting numbers
//! with the adapter they came from (lavapipe software timings are smoke
//! numbers, never GPU performance).

use std::time::Instant;

/// Features needed for encoder-level timestamp queries.
pub const TIMESTAMP_FEATURES: wgpu::Features =
    wgpu::Features::TIMESTAMP_QUERY.union(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);

/// Number of timestamp slots in the query set (2 per section, 16 sections).
pub const QUERY_COUNT: u32 = 32;

/// Per-pass GPU timer.
pub struct GpuTimer {
    query_set: Option<wgpu::QuerySet>,
    resolve_buf: Option<wgpu::Buffer>,
    staging_buf: Option<wgpu::Buffer>,
    period: f32,
    /// False when the adapter lacks timestamp queries (CPU fallback).
    pub gpu_timestamps: bool,
    sections: Vec<String>,
    cpu_fallback: Vec<(String, Instant)>,
}

impl GpuTimer {
    /// Create the query set if `device` supports timestamps.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let supported = device.features().contains(TIMESTAMP_FEATURES);
        if !supported {
            return Self {
                query_set: None,
                resolve_buf: None,
                staging_buf: None,
                period: 0.0,
                gpu_timestamps: false,
                sections: Vec::new(),
                cpu_fallback: Vec::new(),
            };
        }
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("gsim timestamps"),
            count: QUERY_COUNT,
            ty: wgpu::QueryType::Timestamp,
        });
        let size = u64::from(QUERY_COUNT) * 8;
        let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gsim timestamp resolve"),
            size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gsim timestamp staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            query_set: Some(query_set),
            resolve_buf: Some(resolve_buf),
            staging_buf: Some(staging_buf),
            period: queue.get_timestamp_period(),
            gpu_timestamps: true,
            sections: Vec::new(),
            cpu_fallback: Vec::new(),
        }
    }

    /// Reset section bookkeeping for a new timed batch.
    pub fn begin_batch(&mut self) {
        self.sections.clear();
        self.cpu_fallback.clear();
    }

    /// Write the "start" timestamp for section `name` (slot `2*idx`).
    /// Panics if more than 16 sections are recorded per batch.
    pub fn section_start(&mut self, encoder: &mut wgpu::CommandEncoder, name: &str) {
        let idx = self.sections.len();
        assert!(
            2 * idx + 1 < QUERY_COUNT as usize,
            "too many timed sections"
        );
        self.sections.push(name.to_string());
        if let Some(qs) = &self.query_set {
            encoder.write_timestamp(qs, (2 * idx) as u32);
        } else {
            self.cpu_fallback
                .push((format!("{name}/start"), Instant::now()));
        }
    }

    /// Write the "end" timestamp for the current section (slot `2*idx+1`).
    pub fn section_end(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let idx = self.sections.len() - 1;
        if let Some(qs) = &self.query_set {
            encoder.write_timestamp(qs, (2 * idx + 1) as u32);
        } else if let Some(name) = self.sections.last().cloned() {
            self.cpu_fallback
                .push((format!("{name}/end"), Instant::now()));
        }
    }

    /// Resolve the query set into the staging buffer (call before submit).
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        if let (Some(qs), Some(resolve)) = (&self.query_set, &self.resolve_buf) {
            encoder.resolve_query_set(qs, 0..QUERY_COUNT, resolve, 0);
        }
    }

    /// Copy resolved timestamps to staging, map, and return per-section
    /// nanoseconds in recording order. Blocks on the GPU.
    pub fn resolve_and_read(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Vec<(String, f64)> {
        if !self.gpu_timestamps {
            // CPU fallback: pair up start/end instants.
            let mut out = Vec::new();
            let mut i = 0;
            while i + 1 < self.cpu_fallback.len() {
                let (name, t0) = &self.cpu_fallback[i];
                let (_, t1) = &self.cpu_fallback[i + 1];
                let base = name.strip_suffix("/start").unwrap_or(name);
                out.push((base.to_string(), t1.duration_since(*t0).as_secs_f64() * 1e9));
                i += 2;
            }
            return out;
        }
        let (resolve, staging) = (
            self.resolve_buf.as_ref().expect("resolve"),
            self.staging_buf.as_ref().expect("staging"),
        );
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gsim timestamp copy"),
        });
        encoder.copy_buffer_to_buffer(resolve, 0, staging, 0, u64::from(QUERY_COUNT) * 8);
        queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).expect("timestamp map callback");
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("poll timestamps");
        rx.recv()
            .expect("timestamp map result")
            .expect("timestamp map");
        let data = slice.get_mapped_range().expect("mapped");
        let mut out = Vec::new();
        for (i, name) in self.sections.iter().enumerate() {
            let word = |k: usize| {
                let b: [u8; 8] = data[8 * k..8 * k + 8].try_into().expect("8 bytes");
                u64::from_le_bytes(b) as f64
            };
            let ns = (word(2 * i + 1) - word(2 * i)) * f64::from(self.period);
            out.push((name.clone(), ns));
        }
        drop(data);
        staging.unmap();
        out
    }
}
