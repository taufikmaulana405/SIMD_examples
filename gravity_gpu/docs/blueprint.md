# GPU Port — Architectural Blueprint (Stage 1)

**Source:** `https://github.com/jdvillal/SIMD_examples`
**Crates audited:** `gravity_scalar`, `gravity_simd_avx2`, `gravity_simd_avx512`
**Date:** 2026-10-17 · **Stage:** 1 (Audit & Baseline)
**Working copies:** `legacy/` (pristine) · `legacy_work/` (instrumented, `headless` feature)
**GPU-port defaults = the AVX-512 variant's constants** (recorded verbatim in §8).

---

## 1. What the program is

A 2-D all-pairs gravitational N-body toy with **perfectly inelastic merging**,
fading motion trails, and a macroquad HUD. No central body; the cloud starts as
a rotating particle disk and collapses/merges over time. Three CPU variants differ
only in the force/collision kernels (scalar, AVX2×8, AVX-512×16) and in their
default constants (§8). All UI, timestepping, spawn, merge, trail and colour logic
is identical across variants.

## 2. Data-flow diagram (per frame)

```
                    ┌─────────────────────────────────────────────┐
                    │  CPU init (once): seeded PCG (§9), spawn     │
                    │  disk, center_reference_frame               │
                    └──────────────┬──────────────────────────────┘
                                   │ upload once
                    ┌──────────────▼──────────────────────────────┐
                    │ GPU storage buffers (SoA, §3)               │
                    │  pos, vel, mass, brightness, accel          │
                    └──────────────┬──────────────────────────────┘
                                   │
              ┌────────────────────▼────────────────────┐
              │ frame loop (winit + wgpu)               │
              │  accumulator += min(frame_dt,0.05)*ts    │
              │  repeat ≤ 8 while accumulator >= 1/120:  │
              │      ┌──────────────────────────────┐   │
              │      │ physics step (dt = 1/120)    │   │
              │      │  1. force_tiled → accel      │◄──┼── pass 1 may be
              │      │  2. integrate (kick+drift)   │   │   served from the
              │      │  3. detect_collisions        │   │   cached pass 2 of
              │      │  4. merge (components, or    │   │   the previous step
              │      │     serial-parity mode)      │   │   (§6, quirk c)
              │      │  5. force_tiled → accel      │───┼── cache for next
              │      │  6. finish_velocity (kick)   │   │   step (§6)
              │      └──────────────────────────────┘   │
              │  trail_append every 1/45 sim-second     │
              │  stats_reduce every K frames → HUD     │
              │  render: trails + particles + HUD      │
              └───────────────────────────────────────┘
```

Velocity Verlet, exactly as legacy `update_system`:

```
a  = F(x)            # pass 1 (cacheable across steps, quirk c)
v += a * dt/2 ;  x += v * dt
merge(x, v, m)
a  = F(x)            # pass 2
v += a * dt/2
```

`F` is exact all-pairs Newtonian gravity with Plummer-style softening:

```
dx = xj - xi ;  r2 = dx·dx + SOFTENING²          # SOFTENING = 3.0
inv = 1/sqrt(r2) ;  a_i += G * m_j * inv³ * dx   # G = 15.0
```

Equal-and-opposite is applied to both bodies (one pair evaluation updates two
accumulators), exactly like the scalar/AVX kernels.

## 3. Buffer list (SoA)

| # | name | type | bytes/elem | notes |
|---|------|------|-----------|-------|
| 1 | `pos` | `vec2<f32>` | 8 | world coords |
| 2 | `vel` | `vec2<f32>` | 8 | world velocity |
| 3 | `mass` | `f32` | 4 | |
| 4 | `brightness` | `f32` | 4 | spawn-time random in [0.55, 1.0] |
| 5 | `accel` | `vec2<f32>` | 8 | persistent across steps (quirk-c cache) |
| 6 | `parent` | `u32` | 4 | transient union-find for merge (§5) |
| 7 | `trail_ring` | struct | 20/pt | `pos:vec2<f32>`, `radius:f32`, `brightness:f32`, `birth:f32`; fixed capacity (§4) |
| 8 | `stats` | struct | 32 | `count:u32`, `total_mass:f32`, `max_mass:f32`, `com:vec2<f32>` (+pad) |
| 9 | `params` | uniform | ~64 | all constants in §8 |

`radius` is **derived**, never stored: `radius = 0.45 * sqrt(mass)`.
Colour is derived in the render shader from `mass`, `brightness`, `total_mass`
(§8, mass→colour mapping).

## 4. VRAM budget

Persistent per-particle cost: **32 B** (buffers 1–5).

| N | particles (32 B) | trail ring (cap 4M pts × 20 B) | total |
|---|-----------------|-------------------------------|-------|
| 2 000 | 64 KB | 80 MB | ~80 MB |
| 10 000 | 320 KB | 80 MB | ~80 MB |

Legacy comparison: unbounded `VecDeque<TrailPoint>` reaches
45 Hz × 40 s × 10 000 = **18 000 000 points × 20 B = 360 MB** (quirk f,
confirmed). The fixed 4 M-point ring (≈9 s of trails at N=10 000, 40 s at
N≈2 200) bounds this; when full, oldest points are overwritten, which is
visually equivalent to expiry in steady state. The cap is a named constant
(`TRAIL_MAX_POINTS`) so later stages can tune it.

Transient per-step: `parent` (4 B × N) + tile shared memory. Negligible.

## 5. Kernel list

| kernel | work | notes |
|--------|------|-------|
| `force_tiled` | O(N²) all-pairs | workgroup = 128/256 threads; each thread owns one row `i`; partner tiles streamed through `workgroup` shared memory; softening 3.0; Newton-3rd-law update needs care: write `a_i` per-thread (no atomics), accumulate partner contributions via a second tiled pass or accept 2× pair evaluations. **Decision:** evaluate each ordered pair once per thread-row (no cross-thread accumulation) — simplest correct; the N³… actually 2× pair evals vs legacy's 1×: document as 2× arithmetic, still memory-bound-friendly on GPU. |
| `integrate` | O(N) | `v += a*dt/2; x += v*dt` |
| `finish_velocity` | O(N) | `v += a*dt/2` |
| `detect_collisions` | O(N²) tiled | overlap test `dist² ≤ (0.45√mᵢ + 0.45√mⱼ)²`; emit union(i,j) via atomics on `parent` |
| `merge_components` | O(N α(N)) | path compression → canonical root per body; per-root parallel reduction of Σm, Σmx, Σmv, Σmb; compact survivors |
| `merge_serial_parity` | O(N²), 1 thread | **exact** replica of legacy scan for parity tests (below) |
| `trail_append` | O(N) | every 1/45 **sim**-second (quirk-e fix); ring write with atomic head |
| `trail_expire` | O(1) | implicit: ring overwrite + age check in render shader |
| `render_particles` | instanced | circle + glow layers + highlight, §8 colour mapping |
| `render_trails` | instanced | fading circles, §8 trail mapping |
| `stats_reduce` | O(N) | parallel reduction → `stats` buffer; read back every K frames for HUD |

### Merge design

**Parallel mode (default):** collision graph → connected components (union-find
with atomics, then path compression) → per-component mass/momentum-conserving
merge:

```
M   = Σ mᵢ
x   = Σ mᵢxᵢ / M          # centre of mass
v   = Σ mᵢvᵢ / M          # momentum conservation (perfectly inelastic)
b   = Σ mᵢbᵢ / M          # brightness, mass-weighted (legacy rule)
```

**Serial legacy-order mode (parity tests):** a single-threaded kernel (or CPU
fallback) replicating the legacy scan *exactly*, because the legacy merge is
order-dependent (quirk b, confirmed):

```
i = 0
while i < len:
    loop:
        j = first index > i with overlap(i, j)   # ascending scan, first hit wins
        if none: break
        merge j into i   # mass-weighted x/v/b, result stays at slot i
        swap_remove(j)   # last element moves into slot j  ← order dependence
    i += 1
```

`swap_remove` semantics and the restart-after-every-merge are the parity-critical
details. Note the parallel mode is *intentionally* not order-equivalent to
legacy; the serial mode exists so Stage-3 parity tests can assert exact merge
behaviour on small crafted inputs.

## 6. Cross-step acceleration cache (quirk c)

Verified **bit-identical** (`max abs diff = 0.0` over 200 steps, scalar harness
`--accel-check`): pass 2 of step k == pass 1 of step k+1, because no state
mutates between steps. GPU design: keep the `accel` buffer persistent; after a
step, its contents *are* next step's pass-1 input. A merge invalidates only the
rows of merged bodies — but since pass 2 already ran *after* the merge, the
cache is valid unconditionally. **This halves the dominant O(N²) cost** with
zero numerical difference. (Legacy recomputes it; we keep a `--no-accel-cache`
flag for A/B parity runs.)

## 7. Confirmed quirks table

| # | suspected quirk | verdict | evidence | GPU-port implication |
|---|----------------|---------|----------|----------------------|
| a | unseeded RNG → different run each launch | **REFUTED (inverted)** | quad-rand 0.2.3 `GLOBAL_STATE` starts at 0; neither macroquad 0.4.16 nor miniquad calls `srand` (vendored sources). Two `--raw-rng` runs: identical hash `9b619516311607f9` | Legacy is *accidentally deterministic*. GPU port seeds explicitly always (`--seed`, default 12345); keep `--raw-rng` equivalent out |
| b | order-dependent sequential merge via `swap_remove` | **CONFIRMED** | `--reverse` vs normal (200 steps, seed 12345): same count 798 & max mass, but permutation-invariant hash differs (`60fcd290…` vs `f3d014b3…`) | Parallel merge differs by design; provide serial legacy-order parity mode (§5) |
| c | two force passes/step, pass2(k)==pass1(k+1) | **CONFIRMED** | `--accel-check`: max abs diff `0.0` (bit-identical) over 200 steps | Cache `accel` across steps (§6): halves force cost, exact |
| d | third O(N²) collision pass | **CONFIRMED** | `--profile`: merge = 18–23% of force time on all three crates (scalar 0.84 s vs 4.21 s; avx2 0.45 vs 2.47; avx512 0.49 vs 2.17) | Tiled `detect_collisions` kernel, same complexity class |
| e | trail clock on wall time | **CONFIRMED** | `trail_clock += frame_time` (raw, clamp 0.1) vs physics `frame_time.min(0.05)*time_scale` | **Behaviour change:** trail clock on **sim time** (deterministic, headless-friendly); document |
| f | unbounded trail memory (~18 M pts @ N=10k) | **CONFIRMED** | 45 Hz × 40 s × 10 000 × 20 B = **360 MB** | Fixed ring buffer, `TRAIL_MAX_POINTS` (default 4 M ≈ 80 MB) |
| g | physics backlog silently dropped after 8 substeps | **CONFIRMED** | `if completed_steps == MAX { accumulated_time = 0.0; }` in all three mains | Keep the 8-substep cap; add a **dropped-step counter** to the HUD instead of silent drop |

## 8. Constants — GPU-port defaults (AVX-512 variant, verbatim)

Per-variant defaults differ; the GPU port uses the **AVX-512** column.

| constant | scalar | avx2 | **avx512 (GPU default)** |
|----------|--------|------|--------------------------|
| `PARTICLE_COUNT` | 1 000 | 5 000 | **10 000** |
| `GRAVITATIONAL_CONSTANT` | 15.0 | 15.0 | **15.0** |
| `MIN_PARTICLE_MASS` | 5.0 | 5.0 | **5.0** |
| `MAX_PARTICLE_MASS` | 12.0 | 20.0 | **20.0** |
| `PARTICLE_RADIUS_SCALE` | 0.45 | 0.45 | **0.45** |
| `SPAWN_RADIUS` | 500.0 | 1200.0 | **1500.0** |
| `CENTER_CONCENTRATION_STRENGTH` | 0.75 | 0.75 | **0.75** |
| `CENTER_CONCENTRATION_EXPONENT` | 1.40 | 1.40 | **1.40** |
| `CLOCKWISE_ROTATION_WEIGHT` | 1.0 | 1.0 | **1.0** |
| `COUNTERCLOCKWISE_ROTATION_WEIGHT` | 0.0 | 0.0 | **0.0** |
| `INITIAL_TANGENTIAL_SPEED_MIN` | 6.0 | 6.0 | **6.0** |
| `INITIAL_TANGENTIAL_SPEED_MAX` | 20.0 | 20.0 | **20.0** |
| `INITIAL_TANGENTIAL_SPEED_RADIUS_EXPONENT` | 0.50 | 0.50 | **0.50** |
| `INITIAL_RADIAL_SPEED_MAX` | 0.01 | 0.01 | **0.01** |
| `GRAVITY_SOFTENING` | 3.0 | 3.0 | **3.0** |
| `PHYSICS_DT` | 1/120 | 1/120 | **1/120** |
| `MAX_PHYSICS_STEPS_PER_FRAME` | 8 | 8 | **8** |
| `TRAIL_SAMPLE_INTERVAL_SECONDS` | 1/45 | 1/45 | **1/45** |
| `TRAIL_HALF_LIFE_SECONDS` | 10.0 | 10.0 | **10.0** |
| `TRAIL_MAX_AGE_SECONDS` | 40.0 | 40.0 | **40.0** |
| `TRAIL_INITIAL_ALPHA` | 0.35 | 0.35 | **0.35** |
| `TRAIL_MIN_WIDTH_PIXELS` | 1.0 | 1.0 | **1.0** |
| `TRAIL_MAX_WIDTH_PIXELS` | 2.0 | 2.0 | **2.0** |
| `TRAIL_WIDTH_SCALE` | 1.0 | 1.0 | **1.0** |
| `YELLOW_TRANSITION_START_MASS_FRACTION` | 0.25 | 0.25 | **0.25** |
| `CENTRAL_BODY_APPEARANCE_MASS_FRACTION` | 0.45 | 0.45 | **0.45** |
| `BACKGROUND_COLOR` | (0.005, 0.008, 0.025, 1.0) | same | **same** |
| window | 1280×800 | 1280×800 | **1280×800** |
| HUD backend label | "Scalar physics" | "AVX2: 8 x f32" | **"AVX-512: 16 x f32" → GPU: "WGSL compute"** |

Timestep scheme (verified in all three `main()` loops): fixed `dt = 1/120`;
per frame, `accumulated += min(frame_dt, 0.05) * time_scale`, then up to
`MAX_PHYSICS_STEPS_PER_FRAME = 8` substeps; leftover debt zeroed after 8
(quirk g). `time_scale ∈ [0.125, 4]` (×2 / ×0.5 on Up/Down).
Trail half-life 10 s, max age 40 s (quirk e: currently wall-clock).

### Spawn algorithm (exact, for CPU-side init)

1. `cw_count = round(N * 1.0/(1.0+0.0)) = N` (all clockwise by default).
2. Per particle: `a = rand(0, TAU)`; `u = rand(0,1)`;
   `r = (sqrt(u) + (u^1.40 − sqrt(u)) * 0.75) * 1500`.
3. Radial dir `(cos a, sin a)`; clockwise tangent `(-ry, rx)` (screen Y down).
4. `speed = 6 + 14 * clamp(r/1500, 0, 1)^0.5`; `vr = rand(−0.01, 0.01)`;
   `v = tangent*speed + radial*vr`.
5. `mass = rand(5, 20)`; `brightness = rand(0.55, 1.0)`.
6. `center_reference_frame`: subtract COM position and bulk velocity
   (so the isolated cloud neither drifts nor spins away).

### Mass → colour mapping (render shader spec)

```
r_disp   = clamp(0.45*sqrt(m) * zoom, 0.7, 50.0)
mb       = clamp(sqrt(m / 20.0), 0, 1)
normal   = (min(b*0.65 + mb*0.25, 1), min(b*0.75 + mb*0.20, 1), min(b,1), 0.95)
t        = smoothstep over mass_fraction in [0.25, 0.45]
           # t = clamp((f-0.25)/0.20,0,1); t*t*(3-2t)
body     = mix(normal, (1.0, 0.72, 0.15, 1.0), t)
if t > 0:
    circle(r_disp*2.7, (1.0, 0.45, 0.05, 0.035*t))     # outer glow
    circle(r_disp*1.8, (1.0, 0.55, 0.08, 0.08*t))      # inner glow
circle(r_disp, body)
if t > 0:
    circle at (-0.25*r_disp, -0.25*r_disp), r 0.55*r_disp,
           (1.0, 0.92, 0.55, 0.8*t)                    # highlight
```

### Trail rendering (render shader spec)

```
w_px  = clamp(radius * zoom * 2.0 * 1.0, 1.0, 2.0)   # full width, screen px
alpha = 0.35 * 0.5^(age_sim / 10.0)
col   = (min(b,1)*0.60, min(b,1)*0.75, min(b,1), alpha)
circle(pos_screen, w_px/2, col), culled off-screen
```

### HUD (exact legacy formats; GPU keeps all three lines)

- L1: `{n} particles | {RUNNING|PAUSED} | Initial rotation: {cw:.0}% CW / {ccw:.0}% CCW | {backend} | FPS: {fps}`
- L2: `Total mass: {m:.0} | Largest: {max:.1} ({pct:.1}%) | COM: ({x:.2}, {y:.2}) | Time: {ts:.2}x | Zoom: {z:.2}x | Trails: {VISIBLE|HIDDEN} ({trail_pts})`
- L3: `Space: pause | R: reset | T: trails | Up/Down: time speed | Mouse wheel: zoom`

Controls: `Space` pause · `R` reset (respawn, clear trails/accumulators) ·
`T` toggle trail visibility (data kept) · `Up/Down` time scale ×2/×0.5 ·
wheel zoom `*= 1.03^clamp(wheel,−1,1)`, clamped `[0.05, 5]`.
Initial zoom = `min(win_w, win_h) / (1500 * 2.25)`.

## 9. Legacy audit harness (`legacy_work/`, `headless` feature)

Each crate gained a cfg-gated headless entry point sharing the physics code
**verbatim** (only `create_particles` took a `count` parameter; windowed
behaviour unchanged — default build still compiles, verified §10).

CLI: `--seed <u64>` (default **12345** — arbitrary fixed default, logged) ·
`--steps <u64>` (default 600 = 5 sim-seconds) · `--count <usize>` (override
`PARTICLE_COUNT`) · `--dump <path>` (JSON summary; stdout if omitted) ·
`--profile` (per-phase ns: force/integrate/merge) · `--reverse` (reverse initial
order; quirk-b experiment) · `--raw-rng` (skip `srand`; quirk-a experiment) ·
`--accel-check` (scalar only; quirk-c verification).

Dump JSON: crate, seed, flags, steps, initial/final count, total & max mass,
COM, permutation-invariant FNV-1a `positions_hash` over sorted
`(x.to_bits, y.to_bits, mass.to_bits)`, wall seconds, steps/sec, optional
`accel_check_max_abs_diff` and `profile_ns`.

New code: 0 `unsafe`, rustfmt-clean, 0 new clippy warnings (7 pre-existing
warnings identical on the non-headless build).

## 10. Baseline benchmarks (this stage's reference numbers)

Ubuntu VM, **AMD EPYC 9D25 126-Core**, headless, release mode. SIMD crates built
with `RUSTFLAGS="-C target-cpu=native"` (repo README; host has avx512f+dq);
scalar with default flags. Metric: physics steps/sec (each step = 1/120
sim-second); 3 reps, median; fixed seed 12345. Steps scaled per config to keep
runs in the ~5–60 s range (steps/sec is step-count independent).

| crate | N | steps | median steps/s | reps | final N | sim × realtime |
|-------|---|-------|---------------|------|---------|----------------|
| scalar | 1 000 | 600 | **224.49** | 224.49, 224.14, 231.12 | 524 | 1.87× |
| scalar | 2 000 | 300 | **50.68** | 48.90, 53.11, 50.68 | 1121 | 0.42× |
| avx2 | 5 000 | 300 | **54.75** | 52.49, 55.52, 54.75 | 3415 | 0.46× |
| avx2 | 2 000 | 600 | **331.66** | 336.24, 331.66, 300.92 | 1360 | 2.76× |
| avx512 | 10 000 | 150 | **26.18** | 26.18, 26.20, 25.56 | 7923 | 0.22× |
| avx512 | 2 000 | 600 | **589.50** | 582.52, 589.50, 590.74 | 1515 | 4.91× |

At N=2000: AVX2 is **6.5×** scalar, AVX-512 **11.6×** scalar (8/16 lanes —
plausible). O(N²) scaling holds (4.4× slowdown for 2× N on scalar; 6.1× for
2.5× N on AVX2 vs 6.25 expected; 22.5× for 5× N on AVX-512 vs 25 expected —
merges shrink N mid-run). All reps bit-identical (`positions_hash` match):
the harness is fully deterministic.

**The number the GPU port must beat:** AVX-512 at N=10 000 does **26.18
steps/s** on this CPU — i.e. 0.22× real-time; a 60 fps interactive sim needs
120 steps/s at time_scale=1 (2 steps/frame). With the quirk-c cache the CPU
number would roughly double (~52 steps/s); the GPU target is ≥120 steps/s at
N=10 000 including merge and render.

Build/verify notes: `cargo` was at `~/.cargo/bin` (rustup finished during the
audit); `mesa-vulkan-drivers` has no install candidate on this image and
`apt-get` was lock-held by the parent's own provisioning — GPU work is Stage 2+
anyway. The AVX-512 binary was built **and executed** successfully (no illegal
instruction; `is_x86_feature_detected!("avx512f")` passes). Default
(non-headless) windowed build still compiles after instrumentation.
