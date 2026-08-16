# gravity_webgpu

This is the browser version of the gravity simulation using **WebGPU compute shaders**. The all-pairs gravity calculation is executed on the GPU; the JavaScript side only creates the initial particle data, schedules work, and handles controls.

## Run

WebGPU pages must be served from `localhost` or HTTPS. From this directory:

**Python:**

```bash
python3 -m http.server 8080
```

**Node.js / npx:**

```bash
npx serve -l 8080
```

Open <http://localhost:8080> in Chrome/Edge with WebGPU support enabled.

## Notes

- The default is 5,000 particles, matching `gravity_wasm`.
- Gravity is still an O(N²) calculation. GPU acceleration improves parallel throughput but does not change the asymptotic cost.
- Gravity and rendering remain GPU-resident; a throttled readback applies reference collision/merge ordering and captures independent trail samples so behavior follows `gravity_wasm` more closely.
- GPU floating-point reduction order can still produce small trajectory differences from scalar Rust.
- `gravity_wasm` remains the scalar CPU/WebAssembly fallback for browsers without WebGPU.
- The WGSL files are fetched at runtime, so the directory must be served over HTTP; opening `index.html` with `file://` will not work.

Controls: Space pauses/resumes, R resets, T toggles trails, Up/Down changes time scale, and the mouse wheel changes zoom.
