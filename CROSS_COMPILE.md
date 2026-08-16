# Cross-Compiling Rust for Windows from Linux

This guide explains how to compile a Rust program on Linux to produce a Windows `.exe` binary, with AVX2 SIMD instruction support.

---

## Prerequisites

| Component | Version (used when this guide was written) |
|-----------|-------------------------------------------|
| `rustc` / `cargo` | 1.97.1 |
| `mingw-w64` (GCC) | 13-win32 |
| Rust Target | `x86_64-pc-windows-gnu` |

---

## Step 1 — Install Rust

If Rust is not yet installed, run:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
source "$HOME/.cargo/env"
```

Verify installation:

```bash
rustc --version
cargo --version
```

---

## Step 2 — Install `mingw-w64` (Windows Linker)

`mingw-w64` provides the linker and C runtime libraries needed to produce Windows binaries from Linux.

```bash
sudo apt-get install -y mingw-w64
```

Verify:

```bash
x86_64-w64-mingw32-gcc --version
```

---

## Step 3 — Add the Windows Target to Rust

```bash
rustup target add x86_64-pc-windows-gnu
```

Check installed targets:

```bash
rustup target list --installed
```

---

## Step 4 — Compile

Navigate to your Rust project directory, then run:

```bash
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release --target x86_64-pc-windows-gnu
```

### Flag Explanation

| Flag | Description |
|------|-------------|
| `--target x86_64-pc-windows-gnu` | Produces a 64-bit Windows binary using the GNU (MinGW) toolchain |
| `-C target-cpu=x86-64-v3` | Enables AVX2 and other modern instructions (equivalent to Intel/AMD 4th gen and newer) |
| `--release` | Full optimization mode (no debug info, code is optimized) |

---

## Step 5 — Output Location

Once the build is complete, the `.exe` file will be at:

```
<project-name>/target/x86_64-pc-windows-gnu/release/<project-name>.exe
```

Example for this project:

```
gravity_simd_avx2/target/x86_64-pc-windows-gnu/release/gravity_simd_avx2.exe
```

---

## Quick Reference

```bash
# 1. Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# 2. Install Windows linker
sudo apt-get install -y mingw-w64

# 3. Add Windows target
rustup target add x86_64-pc-windows-gnu

# 4. Compile
cd <project-directory>
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release --target x86_64-pc-windows-gnu
```

---

## `target-cpu` Levels for AVX2

Use one of the following values depending on your requirements:

| Value | Instructions Enabled | CPU Support |
|-------|---------------------|-------------|
| `x86-64` | SSE2 only (baseline) | All x86-64 CPUs |
| `x86-64-v2` | SSE4.2, POPCNT | Core 2 / Phenom II and newer |
| `x86-64-v3` | **AVX2**, BMI1, BMI2, FMA | Haswell (2013) / Excavator (2015) and newer |
| `x86-64-v4` | AVX-512 | Skylake-X / Zen 4 and newer |
| `native` | All features of local CPU | Only suitable for non-cross-compilation |

---

## Additional Notes

- The produced binary **only runs on Windows** (PE32+ format) and cannot be executed directly on Linux.
- If the program uses graphics/audio libraries (such as Macroquad), all dependencies are statically linked by `mingw-w64`, so the `.exe` can be distributed directly without any additional installation on the user's side.
- For projects requiring the Visual C++ Runtime (MSVC), use the `x86_64-pc-windows-msvc` target with a different toolchain (more complex, requires Wine + MSVC headers).
