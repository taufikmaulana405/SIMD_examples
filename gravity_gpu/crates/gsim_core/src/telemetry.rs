//! Stage-2/3 JSONL telemetry writer.
//!
//! Every record carries the adapter identity (`name`/`type`/`backend`/
//! `driver` from `wgpu::AdapterInfo`) so lavapipe-software timings can never
//! be mistaken for real GPU numbers downstream. Schema:
//! `{"ts","stage","kind","adapter":{...},"detail":{...}}`.

use serde::Serialize;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// Adapter identity attached to every telemetry record.
#[derive(Debug, Clone, Serialize, Default)]
pub struct AdapterInfo {
    /// `wgpu::AdapterInfo::name`.
    pub name: String,
    /// `wgpu::AdapterInfo::device_type` as a string.
    #[serde(rename = "type")]
    pub device_type: String,
    /// `wgpu::AdapterInfo::backend` as a string.
    pub backend: String,
    /// `wgpu::AdapterInfo::driver` + `driver_info`.
    pub driver: String,
}

impl AdapterInfo {
    /// Build from a live wgpu adapter.
    pub fn from_wgpu(info: &wgpu::AdapterInfo) -> Self {
        Self {
            name: info.name.clone(),
            device_type: format!("{:?}", info.device_type).to_lowercase(),
            backend: format!("{:?}", info.backend).to_lowercase(),
            driver: format!("{} {}", info.driver, info.driver_info),
        }
    }

    /// Placeholder for CPU-side records (benchmark citations, decisions).
    pub fn cpu_placeholder() -> Self {
        Self {
            name: "telemetry-writer (no adapter)".to_string(),
            device_type: "cpu".to_string(),
            backend: "none".to_string(),
            driver: "n/a".to_string(),
        }
    }
}

#[derive(Serialize)]
struct Record<'a> {
    ts: String,
    stage: u32,
    kind: &'a str,
    adapter: &'a AdapterInfo,
    detail: &'a serde_json::Value,
}

/// Current UTC time as `YYYY-MM-DDTHH:MM:SSZ` (no chrono dependency).
fn utc_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Days since epoch → civil date (Howard Hinnant's algorithm).
    let days = (secs / 86_400) as i64;
    let sod = (secs % 86_400) as i64;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    format!(
        "{year:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        sod / 3_600,
        (sod % 3_600) / 60,
        sod % 60
    )
}

/// Append one telemetry record to `path` (created with parent dirs as needed).
///
/// ```
/// use gsim_core::telemetry::{append_telemetry, AdapterInfo};
/// use serde_json::json;
/// let dir = std::env::temp_dir().join("gsim_tel_test");
/// let path = dir.join("t.jsonl");
/// append_telemetry(&path, "phase", &AdapterInfo::cpu_placeholder(), &json!({"phase": "x"}));
/// let text = std::fs::read_to_string(&path).unwrap();
/// assert!(text.contains("\"stage\":2"));
/// std::fs::remove_dir_all(&dir).ok();
/// ```
pub fn append_telemetry(
    path: &Path,
    kind: &str,
    adapter: &AdapterInfo,
    detail: &serde_json::Value,
) {
    append_telemetry_stage(path, 2, kind, adapter, detail);
}

/// Stage-parameterized variant (stage 2 used the hardcoded form above;
/// stage 3 passes `3`). Schema: `{"ts","stage","kind","adapter":{...},
/// "detail":{...}}`.
pub fn append_telemetry_stage(
    path: &Path,
    stage: u32,
    kind: &str,
    adapter: &AdapterInfo,
    detail: &serde_json::Value,
) {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("telemetry dir");
        }
    }
    let record = Record {
        ts: utc_now(),
        stage,
        kind,
        adapter,
        detail,
    };
    let mut line = serde_json::to_string(&record).expect("telemetry serialize");
    line.push('\n');
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("telemetry open");
    file.write_all(line.as_bytes()).expect("telemetry write");
}

/// Default stage-2 telemetry path under the workspace root.
pub fn default_path() -> std::path::PathBuf {
    std::path::PathBuf::from("telemetry/stage2.jsonl")
}

/// Default stage-3 telemetry path under the workspace root.
pub fn stage3_path() -> std::path::PathBuf {
    std::path::PathBuf::from("telemetry/stage3.jsonl")
}
