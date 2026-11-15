//! [`Sim`] orchestration: stepping, stats, rendering, autotune.
//!
//! Step layouts per configuration:
//!
//! - tiled/parallel: `pass1(cached) → integrate → fused → compress → reduce
//!   → copy_back → force(iff merged) → finish → trail?`
//! - tiled/serial: `pass1 → integrate → serial_parity → force_always → finish`
//! - naive/parallel: `force → integrate → detect → compress → reduce →
//!   copy_back → force → finish` (legacy structure, no cache)
//! - naive/serial: `force → integrate → serial_parity → force → finish`

use crate::constants::*;
use crate::gpu::sim::Sim;
use crate::spawn::{create_particles_radius, Particles};
use crate::stats::Stats;
use std::sync::mpsc;

impl Sim {
    /// Spawn the cloud on the CPU and upload it; reset all control words.
    /// Bit-identical to legacy `--seed seed --count n` for the same seed.
    pub fn spawn(&mut self) {
        let cloud = create_particles_radius(self.cfg.seed, self.cfg.n_max, self.cfg.spawn_radius);
        self.upload(&cloud);
    }

    /// Upload an explicit particle state (test/bench mode only): writes the
    /// live set A, resets all control words, and clears the accel cache.
    /// `p.len()` must not exceed the configured capacity.
    pub fn upload(&mut self, p: &Particles) {
        assert!(p.len() <= self.cfg.n_max, "upload exceeds capacity");
        let n = p.len();
        let q = &self.ctx.queue;
        let put2 = |buf: &wgpu::Buffer, xs: &[f32], ys: &[f32]| {
            let mut raw = Vec::with_capacity(xs.len() * 8);
            for i in 0..xs.len() {
                raw.extend_from_slice(&xs[i].to_le_bytes());
                raw.extend_from_slice(&ys[i].to_le_bytes());
            }
            q.write_buffer(buf, 0, &raw);
        };
        put2(&self.pos_a, &p.x, &p.y);
        put2(&self.vel_a, &p.vx, &p.vy);
        let put1 = |buf: &wgpu::Buffer, xs: &[f32]| {
            let mut raw = Vec::with_capacity(xs.len() * 4);
            for x in xs {
                raw.extend_from_slice(&x.to_le_bytes());
            }
            q.write_buffer(buf, 0, &raw);
        };
        put1(&self.mass_a, &p.mass);
        put1(&self.bright_a, &p.brightness);
        // Fresh buffers are zero-init, but respawn must clear explicitly.
        q.write_buffer(&self.accel_a, 0, &vec![0u8; n * 8]);
        let word = |buf: &wgpu::Buffer, v: u32| {
            q.write_buffer(buf, 0, &v.to_le_bytes());
        };
        word(&self.n_buf, n as u32);
        word(&self.merge_flag, 0);
        word(&self.accel_valid, 0);
        word(&self.alive_count, 0);
        word(&self.trail_head, 0);
        self.sim_time = 0.0;
        self.trail_acc = 0.0;
        self.step_count = 0;
    }

    /// One compute dispatch into `encoder`.
    fn dispatch_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        label: &'static str,
        pipeline: &wgpu::ComputePipeline,
        x: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.compute_bg, &[]);
        pass.dispatch_workgroups(x, 1, 1);
    }

    /// Advance one fixed-dt physics step (untimed fast path).
    pub fn step(&mut self) {
        self.step_inner(false);
    }

    /// Advance one step with per-pass GPU timestamps; returns
    /// `(section, nanoseconds)` in dispatch order.
    pub fn step_timed(&mut self) -> Vec<(String, f64)> {
        self.step_inner(true)
    }

    fn step_inner(&mut self, timed: bool) -> Vec<(String, f64)> {
        if self.cfg.no_accel_cache {
            // A/B parity: force pass1 to recompute every step.
            self.ctx
                .queue
                .write_buffer(&self.accel_valid, 0, &0u32.to_le_bytes());
        }
        if timed {
            self.timer.begin_batch();
        }
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gsim step"),
            });
        let d256 = self.dispatch(256);
        let d64 = self.dispatch(64);
        let dwg = self.dispatch(self.wg_size);

        macro_rules! run {
            ($name:expr, $pipe:expr, $x:expr) => {{
                let pipe: &wgpu::ComputePipeline = $pipe;
                if timed {
                    self.timer.section_start(&mut encoder, $name);
                }
                self.dispatch_into(&mut encoder, $name, pipe, $x);
                if timed {
                    self.timer.section_end(&mut encoder);
                }
            }};
        }

        let tiled = self.cfg.mode == crate::cli::GpuMode::Tiled;
        let serial = self.cfg.merge_mode == crate::cli::MergeMode::Serial;

        if tiled {
            run!("pass1", &self.pipes.pass1, dwg);
            run!("integrate", &self.pipes.integrate, d256);
            if serial {
                run!("serial_parity", &self.pipes.serial, 1);
                run!("force_always", &self.pipes.force_always, dwg);
            } else {
                run!("fused", &self.pipes.fused, dwg);
                run!("compress", &self.pipes.compress, d256);
                run!("reduce", &self.pipes.reduce, d256);
                run!("copy_back", &self.pipes.copy_back, d256);
                run!("force", &self.pipes.force, dwg);
            }
            run!("finish", &self.pipes.finish, d256);
        } else {
            run!("force1", &self.pipes.naive_force, d64);
            run!("integrate", &self.pipes.integrate, d256);
            if serial {
                run!("serial_parity", &self.pipes.serial, 1);
            } else {
                run!("detect", &self.pipes.detect_naive, d64);
                run!("compress", &self.pipes.compress, d256);
                run!("reduce", &self.pipes.reduce, d256);
                run!("copy_back", &self.pipes.copy_back, d256);
            }
            run!("force2", &self.pipes.naive_force, d64);
            run!("finish", &self.pipes.finish, d256);
        }

        // Trail sampling on SIM time (quirk-e fix): 1/45 s of sim time.
        self.trail_acc += PHYSICS_DT;
        if self.trail_acc >= TRAIL_SAMPLE_INTERVAL_SECONDS {
            self.trail_acc -= TRAIL_SAMPLE_INTERVAL_SECONDS;
            let t = self.sim_time + PHYSICS_DT;
            let mut raw = t.to_le_bytes().to_vec();
            raw.extend_from_slice(&[0u8; 12]);
            self.ctx.queue.write_buffer(&self.trail_uni, 0, &raw);
            if timed {
                self.timer.section_start(&mut encoder, "trail");
            }
            self.dispatch_into(&mut encoder, "trail", &self.pipes.trail, d256);
            if timed {
                self.timer.section_end(&mut encoder);
            }
        }

        if timed {
            self.timer.resolve(&mut encoder);
        }
        self.ctx.queue.submit(Some(encoder.finish()));
        self.sim_time += PHYSICS_DT;
        self.step_count += 1;

        if timed {
            self.timer
                .resolve_and_read(&self.ctx.device, &self.ctx.queue)
        } else {
            Vec::new()
        }
    }

    /// Dispatch the two-stage stats reduction (call before [`Sim::read_stats`]).
    pub fn update_stats(&mut self) {
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gsim stats"),
            });
        self.dispatch_into(
            &mut encoder,
            "stats_partial",
            &self.pipes.stats_partial,
            1024,
        );
        self.dispatch_into(&mut encoder, "stats_final", &self.pipes.stats_final, 1);
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    /// Copy the 32-byte stats buffer to staging and map it. This is the only
    /// per-frame CPU readback (test/bench modes may read more).
    pub fn read_stats(&self) -> Stats {
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gsim stats copy"),
            });
        encoder.copy_buffer_to_buffer(&self.stats_buf, 0, &self.stats_staging, 0, 32);
        self.ctx.queue.submit(Some(encoder.finish()));
        let slice = self.stats_staging.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).expect("stats map callback");
        });
        self.ctx
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("poll stats");
        rx.recv().expect("stats map result").expect("stats map");
        let data = slice.get_mapped_range().expect("mapped");
        let mut words = [0u32; 8];
        for (i, w) in words.iter_mut().enumerate() {
            let b: [u8; 4] = data[4 * i..4 * i + 4].try_into().expect("4 bytes");
            *w = u32::from_le_bytes(b);
        }
        drop(data);
        self.stats_staging.unmap();
        Stats::from_words(&words)
    }

    /// Map a copy of `src` (test/bench mode only).
    fn map_copy(&self, src: &wgpu::Buffer, size: u64) -> Vec<u8> {
        let staging = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gsim readback staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gsim readback"),
            });
        encoder.copy_buffer_to_buffer(src, 0, &staging, 0, size);
        self.ctx.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).expect("readback map callback");
        });
        self.ctx
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("poll readback");
        rx.recv()
            .expect("readback map result")
            .expect("readback map");
        let data = slice.get_mapped_range().expect("mapped").to_vec();
        staging.unmap();
        data
    }

    /// Read back the full live particle set (test/bench mode only — never
    /// per-frame in interactive mode).
    /// Read back the full live particle set (test/bench mode only — never
    /// per-frame in interactive mode).
    pub fn readback_particles(&self) -> Particles {
        let n = u32::from_le_bytes(
            self.map_copy(&self.n_buf, 4)[..4]
                .try_into()
                .expect("4 bytes"),
        ) as usize;
        let get2 = |buf: &wgpu::Buffer| -> (Vec<f32>, Vec<f32>) {
            let raw = self.map_copy(buf, (n * 8) as u64);
            let mut xs = Vec::with_capacity(n);
            let mut ys = Vec::with_capacity(n);
            for i in 0..n {
                xs.push(f32::from_le_bytes(
                    raw[8 * i..8 * i + 4].try_into().expect("f32"),
                ));
                ys.push(f32::from_le_bytes(
                    raw[8 * i + 4..8 * i + 8].try_into().expect("f32"),
                ));
            }
            (xs, ys)
        };
        let get1 = |buf: &wgpu::Buffer| -> Vec<f32> {
            let raw = self.map_copy(buf, (n * 4) as u64);
            (0..n)
                .map(|i| f32::from_le_bytes(raw[4 * i..4 * i + 4].try_into().expect("f32")))
                .collect()
        };
        let (x, y) = get2(&self.pos_a);
        let (vx, vy) = get2(&self.vel_a);
        let mass = get1(&self.mass_a);
        let brightness = get1(&self.bright_a);
        let mut p = Particles::with_capacity(n);
        for i in 0..n {
            p.push(crate::spawn::ParticleData {
                x: x[i],
                y: y[i],
                vx: vx[i],
                vy: vy[i],
                mass: mass[i],
                brightness: brightness[i],
            });
        }
        p
    }

    /// Read back the `accel_a` buffer for the live bodies (test/bench mode
    /// only — Stage-3 P1 compares GPU accelerations against the f64 CPU
    /// oracle on the identical post-step state).
    pub fn readback_accel(&self) -> (Vec<f32>, Vec<f32>) {
        let n = u32::from_le_bytes(
            self.map_copy(&self.n_buf, 4)[..4]
                .try_into()
                .expect("4 bytes"),
        ) as usize;
        let raw = self.map_copy(&self.accel_a, (n * 8) as u64);
        let mut ax = Vec::with_capacity(n);
        let mut ay = Vec::with_capacity(n);
        for i in 0..n {
            ax.push(f32::from_le_bytes(
                raw[8 * i..8 * i + 4].try_into().expect("f32"),
            ));
            ay.push(f32::from_le_bytes(
                raw[8 * i + 4..8 * i + 8].try_into().expect("f32"),
            ));
        }
        (ax, ay)
    }

    /// Autotune the tiled-force workgroup size over 64/128/256 on the
    /// running adapter. Measures the `fused` kernel over a few timed steps
    /// per candidate (fresh spawn each time for identical work) and keeps
    /// the fastest. Returns `(best, [(size, avg_fused_ns)])`.
    pub fn autotune(&mut self) -> (u32, Vec<(u32, f64)>) {
        let mut table = Vec::new();
        for &wg in &[64u32, 128, 256] {
            self.set_workgroup_size(wg);
            self.spawn();
            self.step();
            self.step();
            let mut acc = 0.0;
            let mut cnt = 0u32;
            for _ in 0..4 {
                for (name, ns) in self.step_timed() {
                    if name == "fused" {
                        acc += ns;
                        cnt += 1;
                    }
                }
            }
            table.push((wg, acc / cnt.max(1) as f64));
        }
        let best = table
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).expect("nan"))
            .map(|t| t.0)
            .unwrap_or(128);
        self.set_workgroup_size(best);
        self.spawn();
        (best, table)
    }
}
