# gravity_cuda

CUDA Driver API implementation of the particle gravity simulation. This is a new standalone project; `gravity_simd_avx2` remains the CPU reference and `gravity_gpu` remains the portable `wgpu` implementation.

## Requirements

- Rust and Cargo
- NVIDIA GPU and compatible NVIDIA driver
- CUDA Toolkit with `nvcc` (CUDA 12 is known to provide the required compiler)
- A working desktop display for the Macroquad window

`build.rs` compiles `kernels/gravity.cu` to PTX with `nvcc`. The Rust host then loads that PTX through the CUDA Driver API using `rustacuda`. Generated PTX is kept in Cargo's `OUT_DIR` and is not committed.

## Build and run on Linux

```bash
cd gravity_cuda
cargo run --release
```

Set `CUDA_HOME` or `CUDA_PATH` when `nvcc` is not on `PATH`. The PTX target defaults to `compute_52`; override it for a known device:

```bash
CUDA_ARCH=compute_86 cargo run --release
```

The prototype uses synchronized host readback for both rendering and deterministic collision resolution. Physics integration and the final velocity-Verlet kick execute in CUDA; after the drift phase, Rust copies the active prefix to the host, applies the ordered AVX2-compatible merge/compaction routine, uploads the compacted prefix, and then launches the final CUDA kick. This correctness-first boundary is intentionally slower and can be replaced by a GPU-native collision pipeline after parity tests pass. CUDA/OpenGL interop can remove the rendering readback in a later optimization pass.

The project also contains a scalar CUDA-shaped reference oracle in `src/reference.rs`, compiled for CPU tests. It follows the same per-target acceleration loop and drift/merge/final-kick ordering. CUDA floating-point results are not expected to be bitwise identical to AVX2 because CUDA uses `rsqrtf`, fast math, and a different reduction order; future hardware tests should compare positions and velocities with documented tolerances while checking mass, momentum, center of mass, active count, and finite-value invariants.

Run the CPU parity and collision tests with:

```bash
cargo test --manifest-path Cargo.toml
```

The CUDA-backed parity smoke test is ignored by default because it requires an NVIDIA driver and device. Run it on the Windows NVIDIA machine with:

```bash
cargo test --manifest-path Cargo.toml -- --ignored cuda_one_particle_matches_reference
```

Trails, mass-based rendering, dominant-body glow, and the richer HUD now run on the host snapshot. Collision remains a synchronized deterministic Rust phase by design; GPU-native collision, CUDA/OpenGL interop, and asynchronous GPU residency remain subsequent optimization phases.

Runtime step, reset, and readback failures pause the simulation and are shown in the window. The last valid snapshot remains visible so an error is not mistaken for an empty simulation.

## Controls

| Key | Action |
|---|---|
| `Space` | Pause/resume |
| `R` | Reset |
| `Up` / `Down` | Increase/decrease time scale |
| `T` | Toggle trails |
| Mouse wheel | Zoom |

The all-pairs kernel is intentionally demanding. The current prototype is a CUDA bring-up baseline; hashed-grid collision and asynchronous GPU-resident rendering are follow-up work. GPU arithmetic is not expected to be bitwise identical to AVX2.

If no NVIDIA device is visible, the program shows an in-window error rather than silently running a different physics backend.

## Cross-compile a Windows `.exe` from Linux/WSL

Install the Windows GNU Rust target and MinGW linker:

```bash
rustup target add x86_64-pc-windows-gnu
sudo apt-get install -y mingw-w64
```

`nvcc` is used only to generate PTX. The Rust linker additionally needs a Windows-compatible GNU CUDA Driver API import library, normally named `libcuda.dll.a`. A Linux `libcuda.so`, `libcuda.so.1`, or `nvcc` executable is not a replacement.

### Create the GNU import library from Windows `nvcuda.dll`

On the Windows computer, copy `C:\Windows\System32\nvcuda.dll` to the Linux/WSL build environment using an authorized file-transfer method. Do not ship this DLL or copy it back to Windows; Windows must load the copy installed by its NVIDIA driver.

In a Visual Studio Developer Command Prompt on Windows, list the exports:

```powershell
dumpbin /exports C:\Windows\System32\nvcuda.dll > nvcuda-exports.txt
```

Create `nvcuda.def` using the exact exported names from that output. It must identify the real DLL and include every CUDA Driver API symbol needed by the selected `rustacuda` version. A minimal starting file looks like this, but the export list must be checked against the DLL:

```text
LIBRARY nvcuda.dll
EXPORTS
    cuInit
    cuDeviceGet
    cuDeviceGetCount
    cuDeviceGetName
    cuCtxCreate_v2
    cuCtxDestroy_v2
    cuCtxPushCurrent_v2
    cuCtxPopCurrent_v2
    cuModuleLoadData
    cuModuleGetFunction
    cuMemAlloc_v2
    cuMemFree_v2
    cuMemcpyHtoD_v2
    cuMemcpyDtoH_v2
    cuStreamCreate
    cuStreamDestroy_v2
    cuStreamSynchronize
    cuLaunchKernel
```

Copy the `.def` file to Linux/WSL and generate a MinGW import library:

```bash
x86_64-w64-mingw32-dlltool \
  --dllname nvcuda.dll \
  --def nvcuda.def \
  --output-lib libcuda.dll.a
```

`llvm-dlltool` can be used with equivalent options. Confirm that the generated file is a Windows import archive. Do not create a fake/stub library or rename `libcuda.so`; unresolved symbols or an invalid import library will only move the failure to link or runtime.

### Build the executable

Place the generated `libcuda.dll.a` in a dedicated directory and set the variables below:

```bash
export CUDA_HOME=/usr/local/cuda
export CUDA_WINDOWS_LIB_DIR=/path/to/windows-cuda-import-libs
export CUDA_LIBRARY_PATH="$CUDA_WINDOWS_LIB_DIR"
export CUDA_ARCH=compute_86

cargo clean --manifest-path gravity_cuda/Cargo.toml
cargo build \
  --manifest-path gravity_cuda/Cargo.toml \
  --release \
  --target x86_64-pc-windows-gnu
```

The build script accepts a prebuilt import library through `CUDA_IMPORT_LIB` or `CUDA_WINDOWS_LIB_DIR`. For convenience, when neither is set it automatically reads exports from `nvcuda.dll`, generates `nvcuda.def` and `libcuda.dll.a` inside Cargo's `OUT_DIR`, and links that generated library. By default it uses `gravity_cuda/nvcuda.dll`; provide a different DLL with `CUDA_DLL`:

```bash
# Optional: use a different Windows driver DLL.
export CUDA_DLL=/path/to/custom/nvcuda.dll

# Optional: bypass generation with a prebuilt GNU import library.
export CUDA_IMPORT_LIB=/path/to/windows-cuda-import-libs/libcuda.dll.a
```

The automatic path requires `x86_64-w64-mingw32-dlltool` (or `dlltool`/`llvm-dlltool`) and an export inspection tool (`llvm-objdump` or MinGW `objdump`). No Windows CUDA SDK is needed for this conversion.

Generated `.def` and `.dll.a` files are temporary build outputs and are not written into the project directory.

For a custom DLL, the path is optional; omitting `CUDA_DLL` uses `gravity_cuda/nvcuda.dll` automatically.

```bash
unset CUDA_IMPORT_LIB
unset CUDA_WINDOWS_LIB_DIR
cargo build --manifest-path gravity_cuda/Cargo.toml --release --target x86_64-pc-windows-gnu
```

Expected output:

```text
gravity_cuda/target/x86_64-pc-windows-gnu/release/gravity_cuda.exe
```

This Linux environment has `nvcc`, MinGW, and import-library generation tools, but it does not have the Windows `nvcuda.dll`; therefore the final import library must be supplied from the Windows computer. The `.exe` can be format/link checked on Linux but must be tested on Windows.

## Windows runtime

The executable is not self-contained. The Windows computer must have a supported NVIDIA GPU and driver providing `nvcuda.dll`. Verify it before launching:

```powershell
nvidia-smi
.\gravity_cuda.exe
```

CUDA Toolkit does not need to be installed on the runtime computer when the driver is present, because this program uses the Driver API. Native Windows MSVC (`x86_64-pc-windows-msvc`) with Visual Studio Build Tools and the Windows CUDA Toolkit is an alternative if producing a GNU import library is inconvenient.

For generic Rust cross-compilation details, see [`../CROSS_COMPILE.md`](../CROSS_COMPILE.md).
