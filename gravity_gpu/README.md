# gsim — GPU-only port of `jdvillal/SIMD_examples` gravity sims

A 2-D all-pairs gravitational N-body toy with **perfectly inelastic merging**,
fading motion trails, and an interactive HUD — ported from the CPU/SIMD
originals (`gravity_scalar`, `gravity_simd_avx2`, `gravity_simd_avx512`) to a
**GPU-only** Rust + wgpu + winit + WGSL implementation.

There is no central body: a seeded rotating particle disk collapses under its
own gravity, particles merge on contact (conserving mass and momentum), and
each particle leaves a fading trail. Everything per-frame — force summation,
semi-implicit Euler integration, collision/merge, trail storage and ageing,
and rendering — runs in WGSL compute/render shaders. The CPU only opens the
window, forwards input, and reads back a few statistics.

Three run modes (`gsim_app`):

| Mode | Flag | What it does |
|---|---|---|
| Interactive | _(none)_ | winit window, egui HUD (Space/R/T/Up/Down/wheel) |
| Headless | `--headless` | no window; runs `--steps` physics steps, prints a JSON summary |
| Benchmark | `--bench` | 4-stage benchmark ladder + N-scaling sweep, then exits |

Run `gsim_app --help` for every flag, its default, and examples.

## Requirements

- **Rust**: a recent stable toolchain (developed with 1.99.1, edition 2021).
- **A GPU with current drivers** exposing **Vulkan** (recommended,
  cross-platform) or **DirectX 12** (Windows, via `--backend dx12`).
  Headless/CI runs also work on a CPU through Mesa's lavapipe software
  Vulkan driver (timings from lavapipe are *not* GPU performance numbers).
- **For the Windows build** (from Linux/macOS): `cargo-xwin`
  (`cargo install cargo-xwin`, used here at 0.23.1). It fetches the MSVC
  CRT/SDK itself — no Visual Studio or Windows machine needed.

## Compiling

Native build (Linux/Windows/macOS with a Rust toolchain):

```sh
cargo build -p gsim_app --release
./target/release/gsim_app --help
```

Windows cross-compile from Linux/macOS (produces
`target/x86_64-pc-windows-msvc/release/gsim_app.exe`):

```sh
cargo xwin build --release --target x86_64-pc-windows-msvc -p gsim_app
```

or, to also install it into `dist/windows/` under a timestamped name
(`gsim_app_yyyyMMdd_HHmmss.exe`, so browsers never serve a cached copy):

```sh
./dist/release_windows.sh
```

Checks the project is held to:

```sh
cargo test                                  # unit + integration + doc tests
cargo clippy -p gsim_core -p gsim_app --all-targets -- -D warnings
cargo fmt --check
```

## Repository layout

```
crates/gsim_core/   headless library: constants, RNG/spawn, CLI, telemetry
crates/gsim_app/    thin frontend: winit window, wgpu context, egui HUD,
                    --bench / --headless / interactive entry points
shaders/            WGSL compute + render shaders (force, integrate, merge,
                    trails, particles, background)
legacy/             pristine upstream clone (jdvillal/SIMD_examples), untouched
legacy_work/        upstream copy with explicit RNG seeding + headless --steps
                    (Stage-1 audit baseline only; not part of the port)
dist/windows/       Windows deliverables: timestamped .exe, bench.ps1,
                    README_WINDOWS.md
dist/release_windows.sh   cross-compile + timestamped install helper
telemetry/          JSONL records + ingest_windows.py (merges bench.ps1 output)
docs/blueprint.md   Stage-1 architecture audit of the originals
report.html         self-contained benchmark/telemetry report
```

`bench.ps1` (in `dist/windows/`) runs the full benchmark on a Windows PC and
writes `telemetry/windows_<timestamp>.json`; see `README_WINDOWS.md` and
`telemetry_watch_spec.md` for the round-trip.
