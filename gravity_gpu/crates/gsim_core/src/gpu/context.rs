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
    /// Explicit path to `dxcompiler.dll` (or its folder) for DX12. `None`
    /// searches the exe folder, then PATH; no external DXC at all means
    /// the embedded DXC compiler is used.
    pub dxc_path: Option<String>,
    /// Force FXC on DX12, disabling DXC entirely (wins over `dxc_path`).
    pub no_dxc: bool,
}

/// Locate `dxcompiler.dll` for the DX12 backend.
///
/// An explicit path is honored strictly: it may name the DLL itself or a
/// folder containing it, and if neither exists there, DXC is not used at
/// all (no silent fallback to a different DXC than the one pointed at).
/// Without an explicit path, the folder next to the running exe is tried
/// first, then every PATH entry.
fn resolve_dxc_path(explicit: Option<&str>) -> Option<std::path::PathBuf> {
    const DLL: &str = "dxcompiler.dll";
    if let Some(p) = explicit {
        let p = std::path::Path::new(p);
        if p.is_file() {
            return Some(p.to_path_buf());
        }
        let joined = p.join(DLL);
        return joined.is_file().then_some(joined);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let cand = dir.join(DLL);
            if cand.is_file() {
                return Some(cand);
            }
        }
    }
    if let Ok(path_var) = std::env::var("PATH") {
        for dir in std::env::split_paths(&path_var) {
            let cand = dir.join(DLL);
            if cand.is_file() {
                return Some(cand);
            }
        }
    }
    None
}

/// Report the DXC outcome exactly once per process: Sims are created many
/// times per run (bench rungs, sweep points), and the compiler choice —
/// an input constant for the whole process — must not spam the log.
fn note_dxc_outcome(explicit: Option<&str>, no_dxc: bool, outcome: &DxcOutcome) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    let explicit = explicit.map(str::to_owned);
    let outcome = outcome.clone();
    ONCE.call_once(move || {
        if no_dxc {
            crate::log_info!(
                "DX12 shader compiler: FXC (DXC disabled by --no-dxc{})",
                if explicit.is_some() {
                    "; --dxc-path ignored"
                } else {
                    ""
                }
            );
            return;
        }
        match &outcome {
            DxcOutcome::External(p) => {
                crate::log_info!("DX12 shader compiler: external DXC ({})", p.display())
            }
            DxcOutcome::Embedded => match &explicit {
                Some(e) => crate::log_warn!(
                    "dxc not found at --dxc-path '{e}' (no dxcompiler.dll there); \
                     the DXC you specified is NOT being used — falling back to the embedded DXC compiler"
                ),
                None => crate::log_info!(
                    "DX12 shader compiler: embedded DXC (no external dxcompiler.dll found)"
                ),
            },
            DxcOutcome::Fxc => match &explicit {
                Some(e) => crate::log_warn!(
                    "dxc not found at --dxc-path '{e}' (no dxcompiler.dll there); \
                     DXC is NOT being used — falling back to FXC (slower shader compiles)"
                ),
                None => crate::log_warn!(
                    "dxc (dxcompiler.dll) not found next to the exe or in PATH; \
                     DXC is NOT being used — falling back to FXC (slower shader compiles)"
                ),
            },
        }
    });
}

/// Which DX12 shader compiler a run ends up with, in priority order.
#[derive(Debug, Clone)]
enum DxcOutcome {
    /// External `dxcompiler.dll` (explicit path, exe folder, or PATH).
    External(std::path::PathBuf),
    /// DXC statically linked into the exe (windows/msvc builds carry
    /// wgpu's `static-dxc` feature; see gsim_core's Cargo.toml).
    // Constructed only on windows/msvc builds; on other platforms the
    // outcome is External or Fxc.
    #[allow(dead_code)]
    Embedded,
    /// Legacy FXC (always set explicitly — wgpu's default is `Auto`,
    /// which would prefer the embedded DXC): reached via `--no-dxc`, or
    /// on platforms with no embedded compiler when no external DXC is
    /// found.
    Fxc,
}

/// Resolve the DX12 compiler: `--no-dxc` forces FXC; else external DXC
/// when found (see [`resolve_dxc_path`]), else the embedded DXC, else
/// FXC (platforms without an embedded compiler).
fn dxc_outcome(explicit: Option<&str>, no_dxc: bool) -> DxcOutcome {
    if no_dxc {
        return DxcOutcome::Fxc;
    }
    if let Some(p) = resolve_dxc_path(explicit) {
        return DxcOutcome::External(p);
    }
    // Gate mirrors the wgpu-hal static-dxc dependency exactly
    // (windows, msvc, not aarch64).
    #[cfg(all(
        target_os = "windows",
        target_env = "msvc",
        not(target_arch = "aarch64")
    ))]
    return DxcOutcome::Embedded;
    #[cfg(not(all(
        target_os = "windows",
        target_env = "msvc",
        not(target_arch = "aarch64")
    )))]
    return DxcOutcome::Fxc;
}

/// wgpu backend options for `choice`: on DX12, pick the shader compiler
/// per [`dxc_outcome`] and log the choice once.
fn backend_options(
    choice: Backend,
    dxc_explicit: Option<&str>,
    no_dxc: bool,
) -> wgpu::BackendOptions {
    let mut opts = wgpu::BackendOptions::default();
    if matches!(choice, Backend::Dx12) {
        let outcome = dxc_outcome(dxc_explicit, no_dxc);
        note_dxc_outcome(dxc_explicit, no_dxc, &outcome);
        match outcome {
            DxcOutcome::External(p) => {
                opts.dx12.shader_compiler = wgpu::Dx12Compiler::DynamicDxc {
                    dxc_path: p.to_string_lossy().into_owned(),
                };
            }
            DxcOutcome::Embedded => {
                opts.dx12.shader_compiler = wgpu::Dx12Compiler::StaticDxc;
            }
            // Set Fxc EXPLICITLY: wgpu's default Dx12Compiler is Auto
            // (despite the Fxc doc comment), and Auto prefers the
            // statically linked DXC whenever it is compiled in — leaving
            // this unset would silently keep using DXC, --no-dxc or not.
            DxcOutcome::Fxc => {
                opts.dx12.shader_compiler = wgpu::Dx12Compiler::Fxc;
            }
        }
    }
    opts
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
    /// queries but falling back cleanly when unsupported. `dxc_path` /
    /// `no_dxc` steer the DX12 shader compiler (see [`dxc_outcome`]).
    pub fn open(
        choice: Backend,
        index: usize,
        dxc_path: Option<&str>,
        no_dxc: bool,
    ) -> Result<Self, String> {
        let backends = Self::backends(choice);
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: Self::backends(choice),
            flags: wgpu::InstanceFlags::empty(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: backend_options(choice, dxc_path, no_dxc),
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
