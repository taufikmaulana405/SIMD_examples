# SIMD Video Examples

This repository contains the example programs used in the SIMD video.

## gravity_scalar

Contains the gravity simulation without explicit SIMD instructions.

Run with:

```bash
cargo run --release
```

## gravity_simd_avx2

Contains the gravity simulation using 256-bit SIMD vector instructions.

Your processor must support **AVX2**. AVX2 was introduced with Intel Haswell processors in 2013 and AMD Excavator processors in 2015, and is supported by most modern x86-64 desktop and laptop CPUs.

### Windows PowerShell

```powershell
$env:RUSTFLAGS="-C target-cpu=native"
cargo run --release
```

### Linux / macOS

```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release
```

## gravity_simd_avx512

Contains the gravity simulation using 512-bit SIMD vector instructions.

Your processor must support **AVX-512**.

Unlike AVX2, AVX-512 is not supported by all modern x86-64 processors. Support varies significantly between CPU generations and manufacturers, so make sure your processor supports AVX-512 before running this version.

### Windows PowerShell

```powershell
$env:RUSTFLAGS="-C target-cpu=native"
cargo run --release
```

### Linux / macOS

```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release
```

## gravity_wasm

Contains the gravity simulation compiled to **WebAssembly**, so it can run directly in any modern browser without installing Rust or native dependencies.

Because WebAssembly does not support x86-specific instructions such as AVX2, this version uses scalar arithmetic instead. The particle count is also reduced to 2,000 (from 10,000) to keep the simulation smooth inside the browser's single-threaded execution environment.

### Prerequisites

Add the WebAssembly target to your Rust toolchain (only needed once):

```bash
rustup target add wasm32-unknown-unknown
```

### Build

```bash
cd gravity_wasm
cargo build --target wasm32-unknown-unknown --release
```

Copy the compiled binary into the project directory alongside `index.html`:

```bash
cp target/wasm32-unknown-unknown/release/gravity_wasm.wasm .
```

### Run in the browser

Browsers require a local HTTP server to load `.wasm` files; opening `index.html` directly as a `file://` URL will not work.

**Python (recommended):**

```bash
python3 -m http.server 8080
```

Then open <http://localhost:8080> in your browser.

**Node.js / npx:**

```bash
npx serve .
```

### Controls

| Key / Input | Action |
|---|---|
| `Space` | Pause / Resume |
| `R` | Reset simulation |
| `T` | Toggle particle trails |
| `↑` / `↓` | Increase / decrease time scale |
| Mouse wheel | Zoom in / out |

---

## Checking CPU Support

### Linux

To check whether your CPU reports support for AVX2 or AVX-512:

```bash
lscpu | grep -E 'avx2|avx512'
```

You can also use:

```bash
grep -o -E 'avx2|avx512[^ ]*' /proc/cpuinfo | sort -u
```

### Windows

You can use tools such as CPU-Z or HWiNFO to check which instruction-set extensions your CPU supports.

## Notes

`-C target-cpu=native` tells the Rust compiler to optimize the program for the CPU on the machine where it is being compiled.

Because of this, binaries compiled with `target-cpu=native` may use instructions that are not available on older or different processors and therefore may not run correctly on another machine.
I'll explain this in a future video.
