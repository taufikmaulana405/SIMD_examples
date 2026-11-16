//! Wgpu device/queue context, simulation configuration, and small helpers.

use crate::cli::{Backend, GpuMode, MergeMode};
use crate::telemetry::AdapterInfo;

/// Simulation configuration (buffer capacity, mode switches).
#[derive(Debug, Clone)]
pub struct SimConfig {
    /// Buffer capacity = spawn count for this run.
    pub n_max: usize,
    /// RNG seed.
    pub seed: u64,
    /// Force kernel variant.
    pub mode: GpuMode,
    /// Merge path variant.
    pub merge_mode: MergeMode,
    /// Disable the cross-step acceleration cache (quirk-c A/B).
    pub no_accel_cache: bool,
    /// Spawn disk radius (sweep scales it with `sqrt(N)`).
    pub spawn_radius: f32,
    /// Tiled force workgroup size (autotuned when 0).
    pub workgroup_size: u32,
    /// Enable GPU shader-debug logging (force-kernel per-thread records).
    /// Pipelines compile with `SHADER_DEBUG=1`; the CPU reads the debug
    /// buffer back only on explicit request. Off by default; zero cost
    /// when off (the gate folds to `if (false)`).
    pub shader_debug: bool,
}

/// Wgpu device/queue plus adapter identity.
pub struct GpuContext {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    /// Logical device.
    pub device: wgpu::Device,
    /// Command queue.
    pub queue: wgpu::Queue,
    /// Adapter identity for telemetry.
    pub adapter_info: AdapterInfo,
}

impl GpuContext {
    /// Requested wgpu backends for a [`Backend`] choice.
    fn backends(choice: Backend) -> wgpu::Backends {
        match choice {
            Backend::Vulkan => wgpu::Backends::VULKAN,
            Backend::Dx12 => wgpu::Backends::DX12,
        }
    }

    /// List all adapters visible for the backend (for logging/selection).
    pub fn list_adapters(choice: Backend) -> Vec<AdapterInfo> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: Self::backends(choice),
            flags: wgpu::InstanceFlags::empty(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
        });
        pollster::block_on(instance.enumerate_adapters(Self::backends(choice)))
            .iter()
            .map(|a| AdapterInfo::from_wgpu(&a.get_info()))
            .collect()
    }

    /// The wgpu instance (for surface creation in windowed mode).
    pub fn instance(&self) -> &wgpu::Instance {
        &self.instance
    }

    /// The underlying adapter (for surface capability queries).
    pub fn adapter(&self) -> &wgpu::Adapter {
        &self.adapter
    }

    /// Open adapter `index` and request a device, preferring timestamp
    /// queries but falling back cleanly when unsupported.
    pub fn open(choice: Backend, index: usize) -> Result<Self, String> {
        let backends = Self::backends(choice);
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: Self::backends(choice),
            flags: wgpu::InstanceFlags::empty(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
        });
        let adapters = pollster::block_on(instance.enumerate_adapters(backends));
        if adapters.is_empty() {
            return Err(format!(
                "no adapters found for backend {choice:?} \
                 (on Linux only Vulkan exists; dx12 needs Windows)"
            ));
        }
        crate::log_debug!(
            "enumerated {} adapter(s) for backend {choice:?}",
            adapters.len()
        );
        let adapter = adapters
            .into_iter()
            .nth(index)
            .ok_or_else(|| format!("adapter index {index} out of range (see --adapter list)"))?;
        let adapter_info = AdapterInfo::from_wgpu(&adapter.get_info());
        crate::log_debug!(
            "adapter selected: {} [{}]",
            adapter_info.name,
            adapter_info.device_type
        );
        let desc = |features| wgpu::DeviceDescriptor {
            label: Some("gsim device"),
            required_features: features,
            required_limits: adapter.limits(),
            experimental_features: wgpu::ExperimentalFeatures::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        };
        // Prefer timestamp queries; fall back without them.
        let (device, queue, with_timestamps) = match pollster::block_on(
            adapter.request_device(&desc(crate::gpu::timing::TIMESTAMP_FEATURES)),
        ) {
            Ok(pair) => (pair.0, pair.1, true),
            Err(_) => {
                let (device, queue) =
                    pollster::block_on(adapter.request_device(&desc(wgpu::Features::empty())))
                        .map_err(|e| format!("request_device failed: {e:?}"))?;
                (device, queue, false)
            }
        };
        crate::log_debug!("device opened: timestamp_queries={with_timestamps}");
        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            adapter_info,
        })
    }
}
