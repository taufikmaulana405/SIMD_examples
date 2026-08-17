# Cross-Compiling Rust for Windows from Linux

This guide explains how to compile Rust programs on Linux or WSL into Windows `.exe` binaries. The generic instructions apply to the CPU examples; `gravity_cuda` has additional CUDA-specific requirements documented below.

## Generic prerequisites

| Component | Value |
|---|---|
| Rust target | `x86_64-pc-windows-gnu` |
| Windows linker | `mingw-w64` |

Install MinGW and the Rust target:

```bash
sudo apt-get install -y mingw-w64
rustup target add x86_64-pc-windows-gnu
x86_64-w64-mingw32-gcc --version
rustup target list --installed
```

Build a regular project with:

```bash
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release --target x86_64-pc-windows-gnu
```

The output is placed at:

```text
<project>/target/x86_64-pc-windows-gnu/release/<project>.exe
```

The resulting PE32+ executable runs on Windows, not Linux. A Windows display environment is required for Macroquad applications.

## CUDA cross-compilation (`gravity_cuda`)

`gravity_cuda` uses the CUDA Driver API through `rustacuda`. Its `build.rs` invokes `nvcc` to compile the CUDA kernel into PTX. For the Windows GNU target, it automatically reads exports from `gravity_cuda/nvcuda.dll`, creates a temporary `.def` and `libcuda.dll.a` in Cargo's `OUT_DIR`, and links the generated import library.

Install the Rust target and MinGW as above, then build from the repository root:

```bash
export CUDA_HOME=/usr/local/cuda
export CUDA_ARCH=compute_86

cargo build \
  --manifest-path gravity_cuda/Cargo.toml \
  --release \
  --target x86_64-pc-windows-gnu
```

By default the build uses:

```text
gravity_cuda/nvcuda.dll
```

To use another Windows driver DLL, set:

```bash
export CUDA_DLL=/path/to/custom/nvcuda.dll
```

To bypass automatic generation and use a prebuilt GNU import library:

```bash
export CUDA_IMPORT_LIB=/path/to/libcuda.dll.a
# or:
export CUDA_WINDOWS_LIB_DIR=/path/to/directory-containing-libcuda.dll.a
```

Automatic generation requires an export inspection tool (`llvm-objdump` or MinGW `objdump`) and an import-library tool (`x86_64-w64-mingw32-dlltool`, `dlltool`, or `llvm-dlltool`). No Windows CUDA SDK is needed for this conversion; Linux `nvcc` is still needed to compile PTX.

Expected artifact:

```text
gravity_cuda/target/x86_64-pc-windows-gnu/release/gravity_cuda.exe
```

### DLL and runtime safety

The copied `nvcuda.dll` is used only as a build-time source of exported symbol names. Do not ship it with the executable or replace the Windows system driver DLL. At runtime, Windows must load the NVIDIA driver's own `nvcuda.dll` from `C:\Windows\System32`.

The Windows computer needs a supported NVIDIA GPU and driver, but not necessarily the CUDA Toolkit. Verify the driver before launching:

```powershell
nvidia-smi
.\gravity_cuda.exe
```

If no `nvcuda.dll` exists at the default project path and `CUDA_DLL`/`CUDA_IMPORT_LIB` is not set, the build fails with an actionable message. A Linux `libcuda.so` cannot satisfy a Windows GNU link. Native Windows MSVC with Visual Studio Build Tools and the Windows CUDA Toolkit is an alternative.

## CPU target levels

| Value | Instructions | Typical support |
|---|---|---|
| `x86-64` | SSE2 baseline | All x86-64 CPUs |
| `x86-64-v2` | SSE4.2, POPCNT | Newer x86-64 CPUs |
| `x86-64-v3` | **AVX2**, BMI1, BMI2, FMA | Haswell/Excavator and newer |
| `x86-64-v4` | AVX-512 | Selected newer CPUs |
| `native` | Host CPU features | Not suitable for cross-compilation |
