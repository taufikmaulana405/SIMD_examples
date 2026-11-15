# gsim_app — Windows bundle

A GPU-accelerated 2-D all-pairs gravity simulation with perfectly inelastic
merging, fading motion trails, and an interactive HUD. This is a Rust + wgpu
port of the CPU-only [`jdvillal/SIMD_examples`](https://github.com/jdvillal/SIMD_examples)
gravity toys (`gravity_scalar`, `gravity_simd_avx2`, `gravity_simd_avx512`),
using the AVX-512 variant's constants as the defaults.

What to beat: the AVX-512 legacy on Ubuntu (AMD EPYC 9D25) managed
**26.18 steps/s @ N=10 000** — i.e. 0.22× real-time. A 60 fps interactive run
at time_scale=1 needs ≥120 steps/s at N=10 000, including merging and render.

## Requirements

- Windows 10 (1809+) or Windows 11, 64-bit
- A real GPU with current drivers (NVIDIA/AMD/Intel)
- One of: **Vulkan** drivers (recommended) or **DirectX 12** (backend chosen
  with `--backend vulkan|dx12`)
- No installation — this folder is self-contained: `gsim_app.exe`,
  `bench.ps1`, and this file

## Running

From a terminal in this folder:

```powershell
.\gsim_app.exe --help
```

**Windowed (interactive):**

```powershell
.\gsim_app.exe                # defaults: N=10000, seed 12345, tiled GPU kernel
.\gsim_app.exe --backend vulkan --adapter 0
```

Window controls (same as legacy):

| Key / input      | Action                                              |
|------------------|-----------------------------------------------------|
| Space            | pause / resume                                      |
| R                | reset (respawn, clear trails and accumulators)       |
| T                | toggle trail visibility (data is kept)              |
| Up / Down        | time scale ×2 / ×0.5 (range 0.125–4)                |
| Mouse wheel      | zoom ×1.03 per notch, clamped [0.05, 5]             |

HUD lines (egui, top-left): particle count / pause state / backend label /
FPS; total & largest mass / COM / time scale / zoom / trail points; control
help; plus a dropped-step counter (legacy silently dropped backlog after
8 substeps — this build counts it instead).

**Flags:**

```powershell
.\gsim_app.exe --bench                        # full 4-stage benchmark ladder
.\gsim_app.exe --headless --n 2000 --steps 60  # no window, physics only
.\gsim_app.exe --seed 777 --n 5000 --backend dx12 --adapter 1
```

`--help` lists everything: `--bench --headless --seed --n --backend
vulkan|dx12 --adapter N --mode naive|tiled --merge-mode parallel|serial
--sweep --steps --no-accel-cache`.

## Benchmarking

Run the bundled script (it invokes the full ladder: autotune, CPU-parity
check, naive rung, tiled rung, N-scaling sweep):

```powershell
powershell -ExecutionPolicy Bypass -File .\bench.ps1
```

Useful overrides:

```powershell
.\bench.ps1 -Backend vulkan -Adapter 0 -N 10000 -Steps 150
.\bench.ps1 -Sweep "1000,5000,10000,20000,50000" -SweepSteps 60
```

This writes **`telemetry/windows_<timestamp>.json`** next to the script.

### What the telemetry JSON contains

Every record carries its measurement context — adapter name, adapter type
(cpu/gpu), backend (vulkan/dx12), driver string — so numbers are never quoted
without provenance. The file holds:

- `legacy_citations`: Stage-1 CPU baselines (the 26.18 steps/s number to beat)
- `autotune`: workgroup-size sweep (64/128/256) and the picked size
- `parity`: CPU-vs-GPU bitwise parity check (`positions_hash` match)
- `rung_naive` / `rung_tiled`: steps/s + per-pass GPU-timestamp breakdowns
- `sweep`: steps/s at each N, `max_n_at_60fps` verdicts at 1× and 4×
- `headless`: wall-clock headless run + `vram_bytes` (VRAM budget)

**Please copy the resulting `windows_<timestamp>.json` back into the project's
`telemetry/` directory on the build machine.** An automated watch is configured
to ingest it: the report regenerates with real GPU numbers and a notification
is sent. See `telemetry_watch_spec.md` in the project root for details.

## Windows smoke-test checklist

Before trusting any number, run through these once:

1. **Windowed render executes.** Launch with no flags; a window appears with
   particles, trails, and the egui HUD (4 text areas, top-left). No validation
   errors, no crash on resize.
2. **Tiled-mode bitwise parity re-verified.** Run
   `.\gsim_app.exe --bench` and confirm the `parity` record reports
   `match: true` for **both** the naive and the tiled kernel (N=256, 20
   steps, serial merge). *Why this matters: on Mesa lavapipe (the software
   Vulkan used during development) the tiled kernel was nondeterministic with
   ≥2 workgroups — about a 50% per-run flake, always off by exactly
   1.53065e-4 when wrong, while the naive kernel was always bitwise-exact.
   This was judged a lavapipe driver quirk (the kernel is textbook-correct),
   but it MUST be confirmed on a real GPU before the tiled kernel is trusted.*
3. **≥60 FPS verdicts.** The bench sweep computes `max_n_at_60fps` at 1× and
   4× time scale. Record both verdicts.
4. **Adapter/backend/driver recorded.** Note the adapter name, `--backend`
   used, and driver version from the JSON — performance numbers are
   meaningless without them.

## Notes on behavior changes vs legacy

- Trail sampling runs on **sim time**, not wall time (legacy sampled on wall
  clock; at time_scale ≠ 1 the trails aged wrong). Deterministic and
  headless-friendly.
- The cross-step acceleration cache is on by default (proven bit-identical in
  Stage 1); disable with `--no-accel-cache` for A/B runs.
- The merge is parallel (union-find components) by default; `--merge-mode
  serial` reproduces the legacy order-dependent scan exactly for parity work.
- Trail storage is a fixed 4 M-point ring (~80 MB), not an unbounded
  `VecDeque` (legacy could reach ~360 MB at N=10 000).
