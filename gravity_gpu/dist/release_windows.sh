#!/bin/bash
# Release helper: cross-compile gsim_app for Windows and install it into
# dist/windows/ with a timestamped filename (gsim_app_yyyyMMdd_HHmmss.exe).
# The timestamp defeats browser caching so a fresh download never collides
# with a previously downloaded build. bench.ps1 auto-selects the newest
# timestamped binary, so nothing else needs updating.
set -euo pipefail
export PATH="$HOME/.cargo/bin:$PATH"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
cargo xwin build --release --target x86_64-pc-windows-msvc -p gsim_app
STAMP="$(date '+%Y%m%d_%H%M%S')"
cp "target/x86_64-pc-windows-msvc/release/gsim_app.exe" \
   "dist/windows/gsim_app_${STAMP}.exe"
echo "installed dist/windows/gsim_app_${STAMP}.exe"
