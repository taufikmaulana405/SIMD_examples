const SIMULATION_SHADER_URL = "./simulation.wgsl";
const RENDER_SHADER_URL = "./render.wgsl";
// Keep the default modest for browser GPUs: the solver is still O(N²).
// Start with a small count so the browser can present frames while the
// correctness path performs two all-pairs passes per physics step.
const COUNT = 5000; // Matches gravity_wasm; lower this only on weak GPUs.
const WORKGROUP_SIZE = 64;
const PHYSICS_DT = 1 / 120;
const MAX_STEPS = 4;
const G = 15;
const SOFTENING_SQUARED = 9;
const RADIUS_SCALE = 0.45;
const SPAWN_RADIUS = 1200;
const PARTICLE_STRIDE = 32;
const TRAIL_INTERVAL = 1 / 45;
const TRAIL_HALF_LIFE = 10;
const TRAIL_MAX_AGE = 40;
const TRAIL_INITIAL_ALPHA = 0.35;
const TRAIL_MAX_SAMPLES = 90;

const canvas = document.querySelector("#canvas");
const hud = document.querySelector("#hud");
const errorBox = document.querySelector("#error");

function showError(message) {
  errorBox.textContent = message;
  errorBox.style.display = "block";
  hud.textContent = "WebGPU not available";
}

function randomRange(min, max) { return min + Math.random() * (max - min); }
function sampleDistance() {
  const u = Math.random();
  const uniform = Math.sqrt(u);
  const biased = Math.pow(u, 1.4);
  return (uniform + (biased - uniform) * 0.75) * SPAWN_RADIUS;
}

function createParticles() {
  const data = new Float32Array(COUNT * 8);
  let totalMass = 0, weightedX = 0, weightedY = 0;
  let momentumX = 0, momentumY = 0;
  for (let i = 0; i < COUNT; i++) {
    const angle = Math.random() * Math.PI * 2;
    const distance = sampleDistance();
    const x = Math.cos(angle) * distance;
    const y = Math.sin(angle) * distance;
    const radius = Math.max(distance, 1e-6);
    const rx = x / radius, ry = y / radius;
    const speed = 6 + 14 * Math.pow(distance / SPAWN_RADIUS, 0.5);
    const radialSpeed = randomRange(-0.01, 0.01);
    const mass = randomRange(5, 20);
    const brightness = randomRange(0.55, 1);
    const base = i * 8;
    const vx = -ry * speed + rx * radialSpeed;
    const vy = rx * speed + ry * radialSpeed;
    data[base] = x; data[base + 1] = y;
    data[base + 2] = vx; data[base + 3] = vy;
    data[base + 4] = mass; data[base + 5] = brightness;
    data[base + 6] = 1; // alive
    totalMass += mass;
    weightedX += x * mass; weightedY += y * mass;
    momentumX += vx * mass; momentumY += vy * mass;
  }
  const cx = weightedX / totalMass, cy = weightedY / totalMass;
  const cvx = momentumX / totalMass, cvy = momentumY / totalMass;
  for (let i = 0; i < COUNT; i++) {
    const base = i * 8;
    data[base] -= cx; data[base + 1] -= cy;
    data[base + 2] -= cvx; data[base + 3] -= cvy;
  }
  return { data, totalMass };
}

function makeBuffer(device, size, usage, label) {
  return device.createBuffer({ size, usage, label });
}

async function readBuffer(state, source) {
  const encoder = state.device.createCommandEncoder({ label: "particle readback" });
  encoder.copyBufferToBuffer(source, 0, state.readback, 0, COUNT * PARTICLE_STRIDE);
  state.device.queue.submit([encoder.finish()]);
  await state.readback.mapAsync(GPUMapMode.READ);
  const copy = new Float32Array(state.readback.getMappedRange()).slice();
  state.readback.unmap();
  return copy;
}

function mergeCollisions(data, count) {
  const radius = mass => RADIUS_SCALE * Math.sqrt(Math.max(mass, 0));
  const index = (particle, field) => particle * 8 + field;
  let first = 0;
  while (first < count) {
    let second;
    while (true) {
      second = undefined;
      const firstX = data[index(first, 0)], firstY = data[index(first, 1)];
      const firstRadius = radius(data[index(first, 4)]);
      for (let candidate = first + 1; candidate < count; candidate++) {
        const dx = data[index(candidate, 0)] - firstX;
        const dy = data[index(candidate, 1)] - firstY;
        const distance = firstRadius + radius(data[index(candidate, 4)]);
        if (dx * dx + dy * dy <= distance * distance) {
          second = candidate;
          break;
        }
      }
      if (second === undefined) break;
      const firstMass = data[index(first, 4)];
      const secondMass = data[index(second, 4)];
      const combinedMass = firstMass + secondMass;
      for (const field of [0, 1]) {
        data[index(first, field)] =
          (data[index(first, field)] * firstMass + data[index(second, field)] * secondMass) / combinedMass;
      }
      for (const field of [2, 3]) {
        data[index(first, field)] =
          (data[index(first, field)] * firstMass + data[index(second, field)] * secondMass) / combinedMass;
      }
      data[index(first, 4)] = combinedMass;
      data[index(first, 5)] =
        (data[index(first, 5)] * firstMass + data[index(second, 5)] * secondMass) / combinedMass;
      data[index(first, 6)] = 1;
      const last = count - 1;
      if (second !== last) {
        data.copyWithin(index(second, 0), index(last, 0), index(last + 1, 0));
      }
      count--;
    }
    first++;
  }
  return count;
}

function makeSimBindGroup(device, layout, params, source, destination) {
  return device.createBindGroup({ layout, entries: [
    { binding: 0, resource: { buffer: params } },
    { binding: 1, resource: { buffer: source } },
    { binding: 2, resource: { buffer: destination } },
  ] });
}

function makeRenderPipeline(device, module, layout, format, vertexEntry, fragmentEntry) {
  return device.createRenderPipeline({
    layout,
    vertex: { module, entryPoint: vertexEntry },
    fragment: { module, entryPoint: fragmentEntry, targets: [{ format,
      blend: {
        color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
        alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
      },
    }] },
    primitive: { topology: "triangle-list" },
  });
}

async function main() {
  if (!navigator.gpu) {
    showError("WebGPU is not supported in this browser. Please use a recent Chrome or Edge release over HTTPS or http://localhost.");
    return;
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) { showError("No suitable WebGPU adapter found. Check GPU hardware support and browser flags."); return; }
  const device = await adapter.requestDevice();
  device.lost.then(info => showError(`WebGPU device lost: ${info.message || info.reason}`));
  const context = canvas.getContext("webgpu");
  if (!context) { showError("Unable to create WebGPU canvas context."); return; }
  const format = navigator.gpu.getPreferredCanvasFormat();
  const [simulationCode, renderCode] = await Promise.all([
    fetch(SIMULATION_SHADER_URL).then(response => response.text()),
    fetch(RENDER_SHADER_URL).then(response => response.text()),
  ]);
  const state = {
    device, context, format, paused: false, trailsVisible: false, zoom: 1, timeScale: 1,
    accumulator: 0, trailClock: 0, trailAccumulator: 0, lastTime: performance.now(), particleIndex: 0,
    width: 1, height: 1, fps: 0, frameCount: 0, fpsTime: performance.now(),
    totalMass: 0, activeCount: COUNT, trailCount: 0, resetGeneration: 0,
  };
  state.simParams = makeBuffer(device, 48, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, "simulation parameters");
  state.particles = [
    makeBuffer(device, COUNT * PARTICLE_STRIDE,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, "particles A"),
    makeBuffer(device, COUNT * PARTICLE_STRIDE,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, "particles B"),
  ];
  state.readback = makeBuffer(device, COUNT * PARTICLE_STRIDE,
    GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, "particle readback");
  state.trails = [];
  state.trailCapacity = TRAIL_MAX_SAMPLES * COUNT; // CPU history mirrors the reference until GPU snapshot pass is added.

  const initial = createParticles();
  state.totalMass = initial.totalMass;
  device.queue.writeBuffer(state.particles[0], 0, initial.data);
  device.queue.writeBuffer(state.particles[1], 0, initial.data);

  const simModule = device.createShaderModule({ label: "gravity compute shader", code: simulationCode });
  const renderModule = device.createShaderModule({ label: "particle render shader", code: renderCode });
  const simLayout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
  ] });
  const simPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [simLayout] });
  state.simPipeline = device.createComputePipeline({ layout: simPipelineLayout,
    compute: { module: simModule, entryPoint: "integrate" } });
  state.finishPipeline = device.createComputePipeline({ layout: simPipelineLayout,
    compute: { module: simModule, entryPoint: "finish" } });
  state.simBindGroups = [
    makeSimBindGroup(device, simLayout, state.simParams, state.particles[0], state.particles[1]),
    makeSimBindGroup(device, simLayout, state.simParams, state.particles[1], state.particles[0]),
  ];

  state.renderParams = makeBuffer(device, 32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, "render parameters");
  const renderLayout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
    { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
  ] });
  const renderPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [renderLayout] });
  state.particlePipeline = makeRenderPipeline(device, renderModule, renderPipelineLayout, format, "particle_vertex", "particle_fragment");
  state.renderBindGroups = state.particles.map(particleBuffer => device.createBindGroup({
    layout: renderLayout,
    entries: [{ binding: 0, resource: { buffer: state.renderParams } }, { binding: 1, resource: { buffer: particleBuffer } }],
  }));

  resize(state);
  installControls(state);
  requestAnimationFrame(now => frame(state, now));
}

function resize(state) {
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(canvas.clientWidth * ratio));
  const height = Math.max(1, Math.floor(canvas.clientHeight * ratio));
  if (canvas.width === width && canvas.height === height && state.configured) return;
  canvas.width = width; canvas.height = height;
  state.context.configure({ device: state.device, format: state.format, alphaMode: "opaque" });
  state.configured = true; state.width = width; state.height = height;
  if (!state.zoomInitialized) {
    state.zoom = Math.min(width, height) / (SPAWN_RADIUS * 2.25);
    state.zoomInitialized = true;
  }
}

function installControls(state) {
  addEventListener("resize", () => resize(state));
  addEventListener("keydown", event => {
    if (event.code === "Space") state.paused = !state.paused;
    if (event.code === "KeyT") state.trailsVisible = !state.trailsVisible;
    if (event.code === "ArrowUp") state.timeScale = Math.min(4, state.timeScale * 2);
    if (event.code === "ArrowDown") state.timeScale = Math.max(0.125, state.timeScale * 0.5);
    if (event.code === "KeyR") reset(state);
  });
  canvas.addEventListener("wheel", event => {
    event.preventDefault();
    const direction = Math.max(-1, Math.min(1, -event.deltaY / 100));
    state.zoom = Math.max(0.05, Math.min(5, state.zoom * Math.pow(1.03, direction)));
  }, { passive: false });
}

function reset(state) {
  const initial = createParticles();
  state.totalMass = initial.totalMass;
  state.activeCount = COUNT;
  state.accumulator = 0;
  state.trailClock = 0;
  state.trailAccumulator = 0;
  state.trails = [];
  state.trailCount = 0;
  state.particleIndex = 0;
  state.resetGeneration++;
  state.device.queue.writeBuffer(state.particles[0], 0, initial.data);
  state.device.queue.writeBuffer(state.particles[1], 0, initial.data);
}

function writeSimParams(state) {
  const data = new ArrayBuffer(48); const view = new DataView(data);
  view.setUint32(0, state.activeCount, true);
  view.setFloat32(8, PHYSICS_DT, true); view.setFloat32(12, G, true);
  view.setFloat32(16, SOFTENING_SQUARED, true); view.setFloat32(20, RADIUS_SCALE, true);
  view.setFloat32(24, state.width, true); view.setFloat32(28, state.height, true);
  view.setFloat32(32, state.zoom, true); view.setFloat32(36, state.totalMass, true);
  state.device.queue.writeBuffer(state.simParams, 0, data);
}

function simulateStep(state) {
  writeSimParams(state);
  const source = state.particleIndex;
  const destination = 1 - source;
  const commands = state.device.createCommandEncoder({ label: "gravity step" });
  let pass = commands.beginComputePass();
  pass.setPipeline(state.simPipeline);
  pass.setBindGroup(0, state.simBindGroups[source]);
  pass.dispatchWorkgroups(Math.ceil(state.activeCount / WORKGROUP_SIZE));
  pass.end();
  pass = commands.beginComputePass();
  pass.setPipeline(state.finishPipeline);
  pass.setBindGroup(0, state.simBindGroups[destination]);
  pass.dispatchWorkgroups(Math.ceil(state.activeCount / WORKGROUP_SIZE));
  pass.end();
  state.device.queue.submit([commands.finish()]);

  // Periodic non-blocking collision merging in the background (never blocks frame loop)
  if (state.frameCount % 30 === 0 && !state.readbackBusy) {
    triggerAsyncCollisionMerge(state, source, destination);
  }
}

function triggerAsyncCollisionMerge(state, source, destination) {
  state.readbackBusy = true;
  const generation = state.resetGeneration;
  readBuffer(state, state.particles[source]).then(data => {
    if (generation === state.resetGeneration) {
      const mergedCount = mergeCollisions(data, state.activeCount);
      if (mergedCount !== state.activeCount) {
        state.activeCount = mergedCount;
        state.totalMass = 0;
        for (let i = 0; i < mergedCount; i++) state.totalMass += data[i * 8 + 4];
        state.device.queue.writeBuffer(state.particles[source], 0, data);
        state.device.queue.writeBuffer(state.particles[destination], 0, data);
      }
    }
  }).catch(() => {}).finally(() => {
    state.readbackBusy = false;
  });
}

function sampleTrails(state) {
  if (!state.trailsVisible || state.readbackBusy) return;
  state.readbackBusy = true;
  readBuffer(state, state.particles[state.particleIndex]).then(data => {
    const sample = new Float32Array(state.activeCount * 4);
    for (let i = 0; i < state.activeCount; i++) {
      const base = i * 8;
      const trailBase = i * 4;
      sample[trailBase] = data[base];
      sample[trailBase + 1] = data[base + 1];
      sample[trailBase + 2] = RADIUS_SCALE * Math.sqrt(data[base + 4]);
      sample[trailBase + 3] = data[base + 5];
    }
    state.trails.push({ time: state.trailClock, sample });
    while (state.trails.length && state.trailClock - state.trails[0].time >= TRAIL_MAX_AGE) state.trails.shift();
    state.trailCount = state.trails.reduce((total, entry) => total + entry.sample.length / 4, 0);
  }).catch(() => {}).finally(() => {
    state.readbackBusy = false;
  });
}

function render(state) {
  const data = new ArrayBuffer(32); const view = new DataView(data);
  view.setUint32(0, state.activeCount, true); view.setUint32(4, state.trailsVisible ? state.trailCount : 0, true);
  view.setFloat32(8, state.width, true); view.setFloat32(12, state.height, true);
  view.setFloat32(16, state.zoom, true); view.setFloat32(20, state.totalMass, true);
  view.setFloat32(24, state.trailClock, true);
  state.device.queue.writeBuffer(state.renderParams, 0, data);
  const texture = state.context.getCurrentTexture();
  const commands = state.device.createCommandEncoder({ label: "render particles" });
  const pass = commands.beginRenderPass({ colorAttachments: [{
    view: texture.createView(), clearValue: { r: 0.005, g: 0.008, b: 0.025, a: 1 }, loadOp: "clear", storeOp: "store",
  }] });
  pass.setPipeline(state.particlePipeline);
  pass.setBindGroup(0, state.renderBindGroups[state.particleIndex]);
  pass.draw(6, state.activeCount);
  pass.end();
  state.device.queue.submit([commands.finish()]);
}

function frame(state, now) {
  resize(state);
  const elapsed = Math.min((now - state.lastTime) / 1000, 0.1);
  state.lastTime = now;
  if (!state.paused) {
    state.accumulator += Math.min(elapsed, 0.05) * state.timeScale;
    let steps = 0;
    while (state.accumulator >= PHYSICS_DT && steps < MAX_STEPS) {
      simulateStep(state);
      state.accumulator -= PHYSICS_DT;
      steps++;
    }
    if (steps === MAX_STEPS) state.accumulator = 0;
    state.trailClock += elapsed;
    if (state.trailsVisible) {
      state.trailAccumulator += elapsed;
      while (state.trailAccumulator >= TRAIL_INTERVAL) {
        sampleTrails(state);
        state.trailAccumulator -= TRAIL_INTERVAL;
      }
    }
  }
  render(state);
  state.frameCount++;
  if (now - state.fpsTime >= 500) {
    state.fps = state.frameCount * 1000 / (now - state.fpsTime);
    state.frameCount = 0; state.fpsTime = now;
  }
  hud.textContent = `${state.activeCount} particles · ${state.paused ? "PAUSED" : "RUNNING"} · WebGPU compute · FPS: ${state.fps.toFixed(0)} · Time: ${state.timeScale.toFixed(3)}x · Zoom: ${state.zoom.toFixed(2)}x`;
  requestAnimationFrame(next => frame(state, next));
}

main().catch(error => showError(`WebGPU initialization failed: ${error.message}`));
