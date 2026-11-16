//! CLI parsing tests (Stage 3, task 1): every flag, the `FromStr`
//! implementations (including their error arms), and loud failure on
//! unknown flags or missing values.

use gsim_core::cli::{Backend, CliConfig, GpuMode, MergeMode};
use gsim_core::log::LogLevel;
use std::str::FromStr;

#[test]
fn defaults_match_stage1_harness() {
    let cfg = CliConfig::parse_from(["gsim"]);
    assert!(!cfg.help);
    assert!(!cfg.bench && !cfg.headless);
    assert_eq!(cfg.seed, 12345);
    assert_eq!(cfg.n, 10_000);
    assert_eq!(cfg.steps, 600);
    assert!(matches!(cfg.backend, Backend::Vulkan));
    assert!(matches!(cfg.mode, GpuMode::Tiled));
    assert!(matches!(cfg.merge_mode, MergeMode::Parallel));
    assert!(!cfg.no_accel_cache);
    assert_eq!(cfg.sweep_steps, 4);
    assert!(matches!(cfg.log_level, LogLevel::Info));
    assert!(!cfg.shader_debug);
    assert!(cfg.dump.is_none() && cfg.telemetry.is_none() && cfg.sweep.is_none());
}

#[test]
fn every_flag_parses() {
    let cfg = CliConfig::parse_from([
        "gsim",
        "--bench",
        "--headless",
        "--no-accel-cache",
        "--seed",
        "7",
        "--n",
        "512",
        "--backend",
        "vulkan",
        "--adapter",
        "1",
        "--steps",
        "33",
        "--dump",
        "/tmp/d.json",
        "--mode",
        "naive",
        "--merge-mode",
        "serial",
        "--telemetry",
        "/tmp/t.jsonl",
        "--sweep",
        "100,200",
        "--sweep-steps",
        "9",
        "--log-level",
        "debug",
        "--shader-debug",
    ]);
    assert!(cfg.bench && cfg.headless && cfg.no_accel_cache);
    assert_eq!(cfg.seed, 7);
    assert_eq!(cfg.n, 512);
    assert_eq!(cfg.adapter, 1);
    assert_eq!(cfg.steps, 33);
    assert_eq!(cfg.dump.as_deref(), Some("/tmp/d.json"));
    assert!(matches!(cfg.mode, GpuMode::Naive));
    assert!(matches!(cfg.merge_mode, MergeMode::Serial));
    assert_eq!(cfg.telemetry.as_deref(), Some("/tmp/t.jsonl"));
    assert_eq!(cfg.sweep.as_deref(), Some("100,200"));
    assert_eq!(cfg.sweep_steps, 9);
    assert!(matches!(cfg.log_level, LogLevel::Debug));
    assert!(cfg.shader_debug);
}

#[test]
fn log_level_flag_parses_all_levels() {
    for (text, level) in [
        ("error", LogLevel::Error),
        ("warn", LogLevel::Warn),
        ("warning", LogLevel::Warn),
        ("info", LogLevel::Info),
        ("debug", LogLevel::Debug),
        ("TRACE", LogLevel::Trace),
    ] {
        let cfg = CliConfig::parse_from(["gsim", "--log-level", text]);
        assert_eq!(cfg.log_level, level, "level '{text}'");
    }
}

#[test]
#[should_panic(expected = "--log-level")]
fn log_level_rejects_unknown() {
    let _ = CliConfig::parse_from(["gsim", "--log-level", "verbose"]);
}

#[test]
fn from_str_variants() {
    assert!(matches!(Backend::from_str("vulkan"), Ok(Backend::Vulkan)));
    assert!(matches!(Backend::from_str("DX12"), Ok(Backend::Dx12)));
    assert!(matches!(Backend::from_str("d3d12"), Ok(Backend::Dx12)));
    assert!(Backend::from_str("metal").is_err());
    assert!(matches!(GpuMode::from_str("naive"), Ok(GpuMode::Naive)));
    assert!(matches!(GpuMode::from_str("TILED"), Ok(GpuMode::Tiled)));
    assert!(GpuMode::from_str("fast").is_err());
    assert!(matches!(
        MergeMode::from_str("parallel"),
        Ok(MergeMode::Parallel)
    ));
    assert!(matches!(
        MergeMode::from_str("Serial"),
        Ok(MergeMode::Serial)
    ));
    assert!(MergeMode::from_str("chain").is_err());
    // Defaults are the GPU-port operating modes.
    assert!(matches!(Backend::default(), Backend::Vulkan));
    assert!(matches!(GpuMode::default(), GpuMode::Tiled));
    assert!(matches!(MergeMode::default(), MergeMode::Parallel));
}

#[test]
fn help_flag_short_circuits() {
    // Long and short forms both set `help` …
    assert!(CliConfig::parse_from(["gsim", "--help"]).help);
    assert!(CliConfig::parse_from(["gsim", "-h"]).help);
    // … anywhere in the argv …
    assert!(CliConfig::parse_from(["gsim", "--n", "512", "-h"]).help);
    // … and never panic on neighbouring typos or bad values.
    assert!(CliConfig::parse_from(["gsim", "--help", "--seed", "xyz"]).help);
    assert!(CliConfig::parse_from(["gsim", "--frobnicate", "-h"]).help);
    // Without it, `help` stays false.
    assert!(!CliConfig::parse_from(["gsim", "--bench"]).help);
}

#[test]
fn help_text_covers_every_flag() {
    let help = CliConfig::help_text();
    for flag in [
        "-h",
        "--help",
        "--bench",
        "--headless",
        "--seed",
        "--n",
        "--backend",
        "--adapter",
        "--steps",
        "--dump",
        "--mode",
        "--merge-mode",
        "--no-accel-cache",
        "--telemetry",
        "--sweep",
        "--sweep-steps",
        "--log-level",
        "--shader-debug",
    ] {
        assert!(help.contains(flag), "help text missing {flag}");
    }
    assert!(help.contains("USAGE"));
    assert!(help.contains("EXAMPLES"));
    assert!(help.contains("default: 12345"));
    assert!(help.contains("default: 10000"));
}

#[test]
#[should_panic(expected = "unknown argument")]
fn unknown_flag_panics() {
    let _ = CliConfig::parse_from(["gsim", "--frobnicate"]);
}

#[test]
#[should_panic(expected = "--seed needs a value")]
fn missing_value_panics() {
    let _ = CliConfig::parse_from(["gsim", "--seed"]);
}

#[test]
#[should_panic(expected = "cannot parse")]
fn bad_value_panics() {
    let _ = CliConfig::parse_from(["gsim", "--n", "many"]);
}

#[test]
#[should_panic(expected = "unknown backend")]
fn bad_backend_panics() {
    let _ = CliConfig::parse_from(["gsim", "--backend", "metal"]);
}
