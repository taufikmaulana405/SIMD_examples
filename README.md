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
