#!/usr/bin/env python3
"""Auto-ingest watch action for Windows GPU telemetry.

Trigger: a new ~/workspace/gravity_gpu/telemetry/windows_*.json whose mtime is
> 10 minutes old and which is not listed in telemetry/.ingested.

Action (one-shot per file):
  1. Parse the bench.ps1 summary JSON (adapter identity, parity_ok, rungs,
     sweep, max_n_at_60fps_1x/4x).
  2. Append a `windows_ingested` record to telemetry/stage4.jsonl.
  3. Surgically replace the "pending" placeholders in report.html with the
     measured Windows numbers (header badge, rung table, N-scaling chart
     series + annotation, legend).
  4. Record the filename in telemetry/.ingested.
  5. Print a one-line NOTIFY headline for the scheduler to surface to the user.

Env overrides (for dry-run testing only): GSIM_ROOT (default
~/workspace/gravity_gpu).

Exit codes: 0 = nothing to do OR success; 2 = a candidate file failed to parse
(ingest aborted for that file, not marked ingested).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("GSIM_ROOT", str(Path.home() / "workspace" / "gravity_gpu")))
TELEMETRY = ROOT / "telemetry"
REPORT = ROOT / "report.html"
INGESTED = TELEMETRY / ".ingested"
MIN_AGE_S = 600  # 10 minutes: file must be fully written


def log(msg: str) -> None:
    print(f"[ingest_windows] {msg}", flush=True)


def read_ingested() -> set[str]:
    if INGESTED.exists():
        return {l.strip() for l in INGESTED.read_text().splitlines() if l.strip()}
    return set()


def candidates() -> list[Path]:
    done = read_ingested()
    now = time.time()
    out = []
    for p in sorted(TELEMETRY.glob("windows_*.json")):
        if p.name in done:
            continue
        try:
            if now - p.stat().st_mtime < MIN_AGE_S:
                continue
        except OSError:
            continue
        out.append(p)
    return out


def parse_summary(p: Path) -> dict:
    data = json.loads(p.read_text(encoding="utf-8"))
    adapter = data.get("adapter") or {}
    for field in ("adapter", "parity_ok", "rungs", "sweep"):
        if field not in data:
            raise ValueError(f"summary JSON missing '{field}'")
    if not adapter.get("name"):
        raise ValueError("summary JSON missing adapter.name")
    rungs = {r.get("mode"): r for r in data["rungs"] if isinstance(r, dict)}
    return {
        "adapter_name": str(adapter.get("name")),
        "adapter_type": str(adapter.get("type", "?")),
        "backend": str(adapter.get("backend", "?")),
        "driver": str(adapter.get("driver", "?")),
        "parity_ok": bool(data["parity_ok"]),
        "rungs": rungs,
        "sweep": data["sweep"],
        "max_n_1x": data.get("max_n_at_60fps_1x"),
        "max_n_4x": data.get("max_n_at_60fps_4x"),
        "collected_at": str(data.get("collected_at", "?")),
        "host": str(data.get("host", "?")),
        "os": str(data.get("os", "?")),
        "autotune": data.get("autotune") or data.get("workgroup") or "?",
    }


def append_telemetry(summary: dict, fname: str) -> None:
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": 4,
        "kind": "windows_ingested",
        "adapter": {
            "name": summary["adapter_name"],
            "type": summary["adapter_type"],
            "backend": summary["backend"],
            "driver": summary["driver"],
        },
        "detail": {
            "file": fname,
            "parity_ok": summary["parity_ok"],
            "rungs": {
                m: {
                    "steps_per_sec": (r.get("steps_per_sec")),
                    "n": r.get("n"),
                    "steps": r.get("steps"),
                    "final_count": r.get("final_count"),
                }
                for m, r in summary["rungs"].items()
            },
            "max_n_at_60fps_1x": summary["max_n_1x"],
            "max_n_at_60fps_4x": summary["max_n_4x"],
            "autotune": summary["autotune"],
            "host": summary["host"],
            "os": summary["os"],
        },
    }
    with (TELEMETRY / "stage4.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def rung_sps(summary: dict, mode: str) -> str:
    r = summary["rungs"].get(mode)
    if not r or r.get("steps_per_sec") is None:
        return "n/a"
    try:
        return f"{float(r['steps_per_sec']):.2f}"
    except (TypeError, ValueError):
        return "n/a"


def rung_n(summary: dict, mode: str) -> str:
    r = summary["rungs"].get(mode)
    return str(r.get("n", "?")) if r else "?"


def patch_report(summary: dict) -> bool:
    """Surgically replace 'pending' Windows placeholders. Returns True if the
    report was modified."""
    if not REPORT.exists():
        log("report.html missing; skipping HTML regeneration")
        return False
    html = REPORT.read_text(encoding="utf-8")
    orig = html
    date = summary["collected_at"]
    a = summary["adapter_name"]
    be = summary["backend"]
    short = (a[:28] + "…") if len(a) > 29 else a
    ident = f"{short} [{be}]"

    # 1. Header badge.
    html = html.replace(
        '<span class="badge b-pend">WINDOWS GPU: PENDING — user has not run bench.ps1 yet</span>',
        f'<span class="badge b-ok">WINDOWS GPU: MEASURED — {ident} · {date}</span>',
    )
    # 2. Rung-2 section badge.
    html = html.replace(
        "<h3>Rung 2 — real Windows GPU "
        '<span class="badge b-pend">pending: user has not run bench.ps1</span></h3>',
        f"<h3>Rung 2 — real Windows GPU "
        f'<span class="badge b-ok">measured {date} — {ident}</span></h3>',
    )
    # 3. Rung table cells (order-sensitive; replace each full row cell).
    repl = [
        (
            "<tr><td>Autotune (WG 64/128/256)</td><td>Windows real GPU, dx12 or vulkan</td>"
            '<td class="mono">pending</td></tr>',
            f"<tr><td>Autotune (WG 64/128/256)</td><td>{ident}</td>"
            f'<td class="mono">{summary["autotune"]}</td></tr>',
        ),
        (
            "<tr><td>CPU-parity (bitwise)</td><td>Windows real GPU, N=256, 20 steps, serial merge</td>"
            '<td class="mono">pending</td></tr>',
            f"<tr><td>CPU-parity (bitwise)</td><td>{ident}</td>"
            f'<td class="mono">{"match=true" if summary["parity_ok"] else "MISMATCH"}</td></tr>',
        ),
        (
            "<tr><td>Naive rung steps/s @ N=10 000</td><td>Windows real GPU</td>"
            '<td class="mono">pending</td></tr>',
            f"<tr><td>Naive rung steps/s @ N={rung_n(summary, 'naive')}</td><td>{ident}</td>"
            f'<td class="mono">{rung_sps(summary, "naive")}</td></tr>',
        ),
        (
            "<tr><td>Tiled rung steps/s @ N=10 000</td><td>Windows real GPU</td>"
            '<td class="mono">pending — target ≥120 steps/s</td></tr>',
            f"<tr><td>Tiled rung steps/s @ N={rung_n(summary, 'tiled')}</td><td>{ident}</td>"
            f'<td class="mono">{rung_sps(summary, "tiled")} (target ≥120 steps/s)</td></tr>',
        ),
        (
            "<tr><td>N-scaling sweep 1k→200k</td><td>Windows real GPU</td>"
            '<td class="mono">pending</td></tr>',
            f"<tr><td>N-scaling sweep 1k→200k</td><td>{ident}</td>"
            '<td class="mono">measured — see chart</td></tr>',
        ),
        (
            "<tr><td>max N @ 60 fps (1× and 4×)</td><td>Windows real GPU</td>"
            '<td class="mono">pending</td></tr>',
            f"<tr><td>max N @ 60 fps (1× and 4×)</td><td>{ident}</td>"
            f'<td class="mono">1×={summary["max_n_1x"]} 4×={summary["max_n_4x"]}</td></tr>',
        ),
    ]
    for old, new in repl:
        html = html.replace(old, new)

    # 4. Chart legend + caption.
    html = html.replace(
        "<span>Windows real GPU — pending</span>",
        f"<span>Windows real GPU — measured {date}</span>",
    )
    html = html.replace(
        "Lavapipe sweep points: N=256 → 684.5, N=512 → 296.1 steps/s. "
        "Real GPU curve: pending.",
        "Lavapipe sweep points: N=256 → 684.5, N=512 → 296.1 steps/s "
        "(software, NOT GPU perf). Windows curve below — measured.",
    )
    # 5. Chart JS: add a Windows series + replace the pending annotation.
    sweep_pts = summary["sweep"]
    pts: list[list[float]] = []
    if isinstance(sweep_pts, list):
        for p in sweep_pts:
            try:
                if isinstance(p, (list, tuple)) and len(p) == 2:
                    pts.append([float(p[0]), float(p[1])])
                elif isinstance(p, dict) and "n" in p:
                    v = p.get("steps_per_sec", p.get("sps"))
                    pts.append([float(p["n"]), float(v)])
            except (TypeError, ValueError):
                continue
    if pts:
        series_js = (
            f"  series({json.dumps([[int(n), v] for n, v in pts])}, "
            f"'#58a6ff', 'Windows real GPU ({ident})');\n"
        )
        html = html.replace(
            "  // pending zone annotation",
            series_js + "  // Windows measured (auto-ingested)",
        )
    html = html.replace(
        "ctx.fillText('Windows real GPU: pending (bench.ps1 not run)', X(9000), padT+18);",
        f"ctx.fillText('Windows real GPU: measured {date} — {ident}', X(9000), padT+18);",
    )
    # 6. Open-items list entry about pending Windows numbers.
    html = html.replace(
        "<b>Windows numbers:</b> marked <b>pending</b>",
        "<b>Windows numbers:</b> <b>measured</b> (auto-ingested)",
    )
    html = html.replace(
        "Windows GPU numbers pending — auto-ingest watch will regenerate this report when",
        "Windows GPU numbers were pending at Stage 4; auto-ingest merged them on",
    )

    if html == orig:
        log("no pending placeholders matched; report left unchanged")
        return False
    REPORT.write_text(html, encoding="utf-8")
    log("report.html regenerated with Windows numbers")
    return True


def main() -> int:
    files = candidates()
    if not files:
        log("no new windows_*.json to ingest")
        return 0
    rc = 0
    for p in files:
        try:
            summary = parse_summary(p)
        except (ValueError, json.JSONDecodeError, OSError) as e:
            log(f"SKIP {p.name}: unparseable ({e}); not marking ingested")
            rc = 2
            continue
        append_telemetry(summary, p.name)
        patch_report(summary)
        with INGESTED.open("a", encoding="utf-8") as f:
            f.write(p.name + "\n")
        a = summary
        print(
            f"NOTIFY: Windows GPU results ingested from {p.name} — "
            f"adapter {a['adapter_name']} [{a['adapter_type']}/{a['backend']}] "
            f"driver {a['driver']}; parity_ok={a['parity_ok']}; "
            f"naive={rung_sps(a, 'naive')} steps/s; tiled={rung_sps(a, 'tiled')} steps/s; "
            f"maxN@60fps 1x={a['max_n_1x']} 4x={a['max_n_4x']}. "
            f"report.html regenerated."
        )
        log(f"ingested {p.name} (one-shot; watch should now self-disable)")
        break  # one-shot: a single new file per run
    return rc


if __name__ == "__main__":
    sys.exit(main())
