# Telemetry auto-ingest watch — install spec (APPROVED, DEFERRED)

User approved 2026-10-17. **Do not install before stage 4 completes** — the
merge/regenerate logic (stage-4 telemetry merge + report.html generation) must
exist first, otherwise the watch races the active run.

## Purpose
When the user runs `dist/windows/bench.ps1` on their Windows GPU machine and
copies back `telemetry/windows_<timestamp>.json`, automatically ingest it
without requiring the user to announce its arrival.

## Behavior
1. Trigger: new file matching `~/workspace/gravity_gpu/telemetry/windows_*.json`
   whose mtime is > 10 minutes old (fully written) and which has not been
   ingested before (track ingested filenames in
   `~/workspace/gravity_gpu/telemetry/.ingested`).
2. Action: run the stage-4 merge procedure (fold Windows results into telemetry,
   regenerate `~/workspace/gravity_gpu/report.html`), log the run to
   `telemetry/stage4.jsonl`.
3. Notify the user once with the headline Windows numbers (labeled with adapter
   name/type/backend/driver from the file).
4. Self-disable after the first successful merge (one-shot).
5. Expiry/silence rule: if no new file appears within 30 days of install, stop
   checking quietly (log the retirement to telemetry, no user message). Never
   message the user about "still waiting".

## Mechanism (at install time)
- Prefer a `hooks`-based file watcher if the runtime supports file events on
  `~/workspace/gravity_gpu/telemetry/`; otherwise a cron job hourly
  (`0 * * * *`) running a small shell script implementing the trigger above.
- Before creating, check `cron.list` / `hooks.list` to avoid duplicates.
- Install in the main chat's context so the notification reaches the user.

## Stop conditions
First successful merge+notify, user decline, or 30-day expiry — whichever first.
