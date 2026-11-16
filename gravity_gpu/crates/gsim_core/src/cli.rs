//! CLI configuration: `-h/--help`, `--bench/--headless/--seed/--n/--backend/--adapter/--steps/--dump`
//! plus GPU-port extras (`--mode`, `--merge-mode`, `--no-accel-cache`).
//!
//! Parsing is deliberately hand-rolled (like the stage-1 harness): unknown
//! flags panic so typos fail loudly instead of silently benchmarking the
//! wrong thing. `--help`/`-h` short-circuits everything: it is detected
//! before any other parsing so it always prints and never panics.

use crate::constants::DEFAULT_SEED;
use crate::log::LogLevel;
use std::str::FromStr;

/// Which wgpu backend to request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// Vulkan (default; the only backend available on Linux).
    #[default]
    Vulkan,
    /// DirectX 12 (Windows only; errors clearly on other platforms).
    Dx12,
}

impl FromStr for Backend {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "vulkan" => Ok(Backend::Vulkan),
            "dx12" | "d3d12" => Ok(Backend::Dx12),
            other => Err(format!("unknown backend '{other}'; want vulkan|dx12")),
        }
    }
}

/// Which force kernel the benchmark ladder runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GpuMode {
    /// One thread per particle, global memory, fixed ascending summation
    /// order, no tiling — the direct legacy-structure port.
    Naive,
    /// Shared-memory tiling, vec4 tile loads, fused collision detection,
    /// cross-step acceleration cache.
    #[default]
    Tiled,
}

impl FromStr for GpuMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "naive" => Ok(GpuMode::Naive),
            "tiled" => Ok(GpuMode::Tiled),
            other => Err(format!("unknown mode '{other}'; want naive|tiled")),
        }
    }
}

/// Which GPU merge path a step uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MergeMode {
    /// Collision graph → connected components → mass/momentum-conserving
    /// merge. Deliberately *not* order-equivalent to legacy (quirk b).
    #[default]
    Parallel,
    /// Single-threaded kernel replicating the legacy ascending scan with
    /// `swap_remove` exactly; used by Stage-3 parity tests.
    Serial,
}

impl FromStr for MergeMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "parallel" => Ok(MergeMode::Parallel),
            "serial" => Ok(MergeMode::Serial),
            other => Err(format!(
                "unknown merge-mode '{other}'; want parallel|serial"
            )),
        }
    }
}

/// Parsed command line for `gsim_app`.
#[derive(Debug, Clone)]
pub struct CliConfig {
    /// Print the help text and exit (`-h`/`--help`).
    pub help: bool,
    /// Run the benchmark ladder + N-sweep headlessly and exit.
    pub bench: bool,
    /// No window at all (still runs the sim loop for `--steps`).
    pub headless: bool,
    /// RNG seed (default 12345, as in stage 1).
    pub seed: u64,
    /// Spawn particle count (default 10 000).
    pub n: usize,
    /// GPU backend to request.
    pub backend: Backend,
    /// Adapter index among enumerated adapters.
    pub adapter: usize,
    /// Physics steps for headless/bench runs.
    pub steps: u64,
    /// Write the JSON summary here instead of stdout.
    pub dump: Option<String>,
    /// Force kernel variant.
    pub mode: GpuMode,
    /// Merge path variant.
    pub merge_mode: MergeMode,
    /// Disable the cross-step acceleration cache (quirk-c A/B parity).
    pub no_accel_cache: bool,
    /// Override the telemetry path (default `telemetry/stage2.jsonl`).
    pub telemetry: Option<String>,
    /// Rung-4 N-sweep list, e.g. `--sweep 1000,5000,10000`. Default is a
    /// small smoke pair; the Windows run passes the full 1k→200k list.
    pub sweep: Option<String>,
    /// Physics steps per sweep point (default 4).
    pub sweep_steps: u64,
    /// Log verbosity: error | warn | info | debug | trace (default info).
    pub log_level: LogLevel,
    /// Enable GPU shader-debug logging: force kernels append per-thread
    /// (kernel, thread, ax, ay) records to a debug buffer, read back on
    /// explicit request. Off by default; zero cost when off.
    pub shader_debug: bool,
    /// Explicit path to `dxcompiler.dll` (or a folder containing it) for
    /// the DX12 backend. Without it, the exe folder and PATH are searched.
    /// If no external DXC is found, the embedded DXC compiler is used;
    /// a `--dxc-path` that contains no DXC logs a warning.
    pub dxc_path: Option<String>,
    /// Force the legacy FXC shader compiler on DX12, disabling DXC
    /// entirely (external and embedded). Wins over `--dxc-path`.
    pub no_dxc: bool,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            help: false,
            bench: false,
            headless: false,
            seed: DEFAULT_SEED,
            n: crate::constants::PARTICLE_COUNT,
            backend: Backend::Vulkan,
            adapter: 0,
            steps: 600,
            dump: None,
            mode: GpuMode::Tiled,
            merge_mode: MergeMode::Parallel,
            no_accel_cache: false,
            telemetry: None,
            sweep: None,
            sweep_steps: 4,
            log_level: LogLevel::Info,
            shader_debug: false,
            dxc_path: None,
            no_dxc: false,
        }
    }
}

impl CliConfig {
    /// Parse `std::env::args()`. Panics on unknown flags or bad values.
    /// `-h`/`--help` short-circuits: returns a default config with
    /// [`Self::help`] set, without validating anything else.
    ///
    /// ```
    /// use gsim_core::cli::CliConfig;
    /// let cfg = CliConfig::parse_from(["gsim", "--seed", "7", "--n", "512", "--mode", "naive"]);
    /// assert_eq!(cfg.seed, 7);
    /// assert_eq!(cfg.n, 512);
    /// assert!(matches!(cfg.mode, gsim_core::cli::GpuMode::Naive));
    /// ```
    pub fn parse() -> Self {
        Self::parse_from(std::env::args())
    }

    /// Parse from an explicit argument iterator (test-friendly).
    pub fn parse_from<I, S>(args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let argv: Vec<String> = args.into_iter().map(|s| s.as_ref().to_owned()).collect();
        // Help wins over everything: detect it before any validation so
        // `--help` always prints, even next to typos or bad values.
        if argv.iter().skip(1).any(|a| a == "--help" || a == "-h") {
            return CliConfig {
                help: true,
                ..CliConfig::default()
            };
        }
        let mut cfg = CliConfig::default();
        let mut i = 1;
        while i < argv.len() {
            match argv[i].as_str() {
                "--bench" => cfg.bench = true,
                "--headless" => cfg.headless = true,
                "--no-accel-cache" => cfg.no_accel_cache = true,
                "--seed" => {
                    i += 1;
                    cfg.seed = value(&argv, i, "--seed");
                }
                "--n" => {
                    i += 1;
                    cfg.n = value(&argv, i, "--n");
                }
                "--backend" => {
                    i += 1;
                    cfg.backend = value(&argv, i, "--backend");
                }
                "--adapter" => {
                    i += 1;
                    cfg.adapter = value(&argv, i, "--adapter");
                }
                "--steps" => {
                    i += 1;
                    cfg.steps = value(&argv, i, "--steps");
                }
                "--dump" => {
                    i += 1;
                    cfg.dump = Some(value::<String>(&argv, i, "--dump"));
                }
                "--mode" => {
                    i += 1;
                    cfg.mode = value(&argv, i, "--mode");
                }
                "--merge-mode" => {
                    i += 1;
                    cfg.merge_mode = value(&argv, i, "--merge-mode");
                }
                "--telemetry" => {
                    i += 1;
                    cfg.telemetry = Some(value::<String>(&argv, i, "--telemetry"));
                }
                "--sweep" => {
                    i += 1;
                    cfg.sweep = Some(value::<String>(&argv, i, "--sweep"));
                }
                "--sweep-steps" => {
                    i += 1;
                    cfg.sweep_steps = value(&argv, i, "--sweep-steps");
                }
                "--log-level" => {
                    i += 1;
                    let s: String = value(&argv, i, "--log-level");
                    cfg.log_level = LogLevel::parse(&s).unwrap_or_else(|| {
                        panic!(
                            "--log-level: unknown level '{s}' (want error|warn|info|debug|trace)"
                        )
                    });
                }
                "--shader-debug" => cfg.shader_debug = true,
                "--dxc-path" => {
                    i += 1;
                    cfg.dxc_path = Some(value(&argv, i, "--dxc-path"));
                }
                "--no-dxc" => cfg.no_dxc = true,
                other => panic!("unknown argument: {other}"),
            }
            i += 1;
        }
        cfg
    }
    /// The `-h`/`--help` reference: every flag, its usage, window controls,
    /// and examples.
    ///
    /// ```
    /// use gsim_core::cli::CliConfig;
    /// let help = CliConfig::help_text();
    /// assert!(help.contains("--bench"));
    /// assert!(help.contains("--sweep-steps"));
    /// assert!(help.contains("--log-level"));
    /// assert!(help.contains("--shader-debug"));
    /// assert!(help.contains("--dxc-path"));
    /// assert!(help.contains("--no-dxc"));
    /// assert!(help.contains("EXAMPLES"));
    /// ```
    pub fn help_text() -> &'static str {
        "\
gsim_app — GPU-only 2-D all-pairs gravity sim (Rust + wgpu + WGSL compute/render)

USAGE:
    gsim_app [FLAGS]

MODES (default is the interactive window):
    (none)       Interactive winit window with egui HUD (needs a display)
    --bench      Headless 4-stage benchmark ladder + N-scaling sweep, then exit
    --headless   No window: run --steps physics steps, print a JSON summary

FLAGS:
    -h, --help            Print this help and exit
    --bench               Benchmark mode (see MODES)
    --headless            Headless mode (see MODES)
    --seed <u64>          RNG seed (default: 12345)
    --n <usize>           Spawn particle count (default: 10000)
    --backend <name>      vulkan | dx12 (default: vulkan; dx12 needs Windows)
    --adapter <usize>     Adapter index among enumerated adapters (default: 0)
    --steps <u64>         Physics steps for headless/bench runs (default: 600)
    --dump <path>         Write the JSON summary to <path> instead of stdout
    --mode <name>         naive | tiled — force kernel (default: tiled)
    --merge-mode <name>   parallel | serial — collision merge path
                          (default: parallel; serial replicates the legacy
                          ascending scan exactly, for parity tests)
    --no-accel-cache      Disable the cross-step acceleration cache
    --telemetry <path>    Telemetry JSONL path (default: telemetry/stage2.jsonl)
    --sweep <csv>         Rung-4 N-sweep list, e.g. 1000,5000,10000 (bench mode)
    --sweep-steps <u64>   Physics steps per sweep point (default: 4)
    --log-level <name>    error | warn | info | debug | trace (default: info)
                          Logs go to stderr; stdout stays clean for JSON output
    --shader-debug         Enable GPU shader-debug logging (force-kernel
                          per-thread records, read back on request; zero cost
                          when off)
    --dxc-path <path>     Path to dxcompiler.dll (or its folder) for the
                          DX12 backend. Default: search the exe folder, then
                          PATH; if no external DXC is found, the embedded
                          DXC compiler is used. A path with no DXC in it
                          logs a warning and falls back to the embedded one
    --no-dxc              Force the legacy FXC shader compiler on DX12,
                          disabling DXC entirely (wins over --dxc-path)

INTERACTIVE CONTROLS:
    Space         pause / resume
    R             reset (respawn, clear trails and accumulators)
    T             toggle trail visibility (data is kept)
    Up / Down     time scale x2 / x0.5 (range 0.125-4)
    Mouse wheel   zoom (x1.03 per notch, clamped 0.05-5)

EXAMPLES:
    gsim_app --help
    gsim_app                                        # interactive window
    gsim_app --headless --n 2000 --steps 60
    gsim_app --bench --backend dx12 --adapter 0
    gsim_app --seed 777 --n 5000 --mode naive --merge-mode serial
    gsim_app --bench --sweep 1000,5000,10000 --sweep-steps 30
"
    }
}

fn value<T: FromStr>(argv: &[String], i: usize, flag: &str) -> T
where
    T::Err: std::fmt::Display,
{
    let text = argv
        .get(i)
        .unwrap_or_else(|| panic!("{flag} needs a value"));
    text.parse::<T>()
        .unwrap_or_else(|e| panic!("{flag}: cannot parse '{text}': {e}"))
}
