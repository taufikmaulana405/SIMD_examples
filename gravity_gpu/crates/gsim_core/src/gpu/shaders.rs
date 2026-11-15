//! WGSL shader sources, embedded at compile time.
//!
//! `common.wgsl` (shared constants + the compute bind-group declarations) is
//! prepended to every compute shader. Render shaders are self-contained.

macro_rules! shader {
    ($name:literal) => {
        include_str!(concat!("../../../../shaders/", $name))
    };
}

const COMMON: &str = shader!("common.wgsl");

/// Compute shader source with [`COMMON`] prepended.
pub fn with_common(src: &str) -> String {
    let mut out = String::with_capacity(COMMON.len() + src.len() + 1);
    out.push_str(COMMON);
    out.push('\n');
    out.push_str(src);
    out
}

/// Raw source of one compute shader (without the common prelude).
pub fn source(name: &str) -> &'static str {
    match name {
        "force_naive" => shader!("force_naive.wgsl"),
        "force_tiled" => shader!("force_tiled.wgsl"),
        "integrate" => shader!("integrate.wgsl"),
        "detect_naive" => shader!("detect_naive.wgsl"),
        "merge" => shader!("merge.wgsl"),
        "trail" => shader!("trail.wgsl"),
        "stats" => shader!("stats.wgsl"),
        _ => panic!("unknown compute shader: {name}"),
    }
}

/// Render shader source (self-contained, no prelude).
pub fn render_source(name: &str) -> &'static str {
    match name {
        "particles" => shader!("render_particles.wgsl"),
        "trails" => shader!("render_trails.wgsl"),
        _ => panic!("unknown render shader: {name}"),
    }
}
