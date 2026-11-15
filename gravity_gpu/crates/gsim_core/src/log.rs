//! Leveled logging: `ERROR`/`WARN`/`INFO`/`DEBUG`/`TRACE`, default `INFO`.
//!
//! Deliberately dependency-free (no `log`/`env_logger`): a process-global
//! verbosity ceiling plus thin macros. Human-readable lines go to
//! **stderr** so stdout stays clean for piped JSON (`--bench` without
//! `--dump`, `--help`). Nothing here touches the telemetry JSONL writer.
//!
//! Performance contract (measured, not hoped):
//! - `ERROR`..=`DEBUG` fire per-run or per-stage, outside the hot step
//!   loop: effectively free.
//! - `TRACE` is opt-in and stays CPU-side (dispatch labels, buffer sizes):
//!   a few thousand stderr lines per second at most, never a GPU readback.
//! - An option that exists but is *off* is a single predictable
//!   not-taken branch — unmeasurable. Shader-side verbosity, if ever
//!   needed, belongs behind a WGSL pipeline-creation `override` (like
//!   `WG`) so the off state compiles to nothing; no such override exists
//!   yet.

use std::sync::atomic::{AtomicU8, Ordering};

/// Log verbosity, ordered: a higher level includes everything below it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum LogLevel {
    Error = 0,
    Warn = 1,
    Info = 2,
    Debug = 3,
    Trace = 4,
}

impl LogLevel {
    /// Parse `error|warn|info|debug|trace` (case-insensitive; `warning`
    /// accepted as an alias).
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "error" => Some(Self::Error),
            "warn" | "warning" => Some(Self::Warn),
            "info" => Some(Self::Info),
            "debug" => Some(Self::Debug),
            "trace" => Some(Self::Trace),
            _ => None,
        }
    }

    /// Lowercase name used in the `[level]` line prefix and `--help`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Error => "error",
            Self::Warn => "warn",
            Self::Info => "info",
            Self::Debug => "debug",
            Self::Trace => "trace",
        }
    }
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

static MAX_LEVEL: AtomicU8 = AtomicU8::new(LogLevel::Info as u8);

/// Set the process-global verbosity ceiling. Call once at startup from the
/// parsed CLI config (`--log-level`).
pub fn set_max_level(level: LogLevel) {
    MAX_LEVEL.store(level as u8, Ordering::Relaxed);
}

/// Current verbosity ceiling.
pub fn max_level() -> LogLevel {
    match MAX_LEVEL.load(Ordering::Relaxed) {
        0 => LogLevel::Error,
        1 => LogLevel::Warn,
        2 => LogLevel::Info,
        3 => LogLevel::Debug,
        _ => LogLevel::Trace,
    }
}

/// Would a message at `level` be emitted right now?
pub fn enabled(level: LogLevel) -> bool {
    (level as u8) <= MAX_LEVEL.load(Ordering::Relaxed)
}

/// Emit one `[level] message` line to stderr if `level` is at or below the
/// ceiling. Prefer the `log_*` macros over calling this directly.
pub fn log(level: LogLevel, args: std::fmt::Arguments<'_>) {
    if enabled(level) {
        eprintln!("[{}] {}", level.name(), args);
    }
}

/// Emit at an explicit level: `gsim_log!(LogLevel::Debug, "x={}", x)`.
#[macro_export]
macro_rules! gsim_log {
    ($level:expr, $($arg:tt)*) => {
        $crate::log::log($level, format_args!($($arg)*))
    };
}

/// Emit at `ERROR`: the run cannot complete or a result is invalid.
#[macro_export]
macro_rules! log_error {
    ($($arg:tt)*) => {
        $crate::gsim_log!($crate::log::LogLevel::Error, $($arg)*)
    };
}

/// Emit at `WARN`: suspicious or degraded, but the run continues.
#[macro_export]
macro_rules! log_warn {
    ($($arg:tt)*) => {
        $crate::gsim_log!($crate::log::LogLevel::Warn, $($arg)*)
    };
}

/// Emit at `INFO`: headline results (default level).
#[macro_export]
macro_rules! log_info {
    ($($arg:tt)*) => {
        $crate::gsim_log!($crate::log::LogLevel::Info, $($arg)*)
    };
}

/// Emit at `DEBUG`: diagnostic detail (per-pass timings, tables, hashes).
#[macro_export]
macro_rules! log_debug {
    ($($arg:tt)*) => {
        $crate::gsim_log!($crate::log::LogLevel::Debug, $($arg)*)
    };
}

/// Emit at `TRACE`: firehose (per-dispatch labels, buffer sizes).
#[macro_export]
macro_rules! log_trace {
    ($($arg:tt)*) => {
        $crate::gsim_log!($crate::log::LogLevel::Trace, $($arg)*)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_levels() {
        assert_eq!(LogLevel::parse("error"), Some(LogLevel::Error));
        assert_eq!(LogLevel::parse("WARN"), Some(LogLevel::Warn));
        assert_eq!(LogLevel::parse("warning"), Some(LogLevel::Warn));
        assert_eq!(LogLevel::parse("Info"), Some(LogLevel::Info));
        assert_eq!(LogLevel::parse("debug"), Some(LogLevel::Debug));
        assert_eq!(LogLevel::parse("TRACE"), Some(LogLevel::Trace));
        assert_eq!(LogLevel::parse("verbose"), None);
        assert_eq!(LogLevel::parse(""), None);
    }

    #[test]
    fn ordering_includes_lower() {
        assert!(LogLevel::Error < LogLevel::Warn);
        assert!(LogLevel::Warn < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Debug);
        assert!(LogLevel::Debug < LogLevel::Trace);
    }

    #[test]
    fn ceiling_filters() {
        // Save/restore: tests run in parallel in one process.
        let prev = max_level();
        set_max_level(LogLevel::Warn);
        assert!(enabled(LogLevel::Error));
        assert!(enabled(LogLevel::Warn));
        assert!(!enabled(LogLevel::Info));
        assert!(!enabled(LogLevel::Debug));
        assert!(!enabled(LogLevel::Trace));
        set_max_level(prev);
    }
}
