#!/bin/bash
# Release helper: cross-compile gsim_app for Windows and install it into
# dist/windows/ with a timestamped filename (gsim_app_yyyyMMdd_HHmmss.exe).
# The timestamp defeats browser caching so a fresh download never collides
# with a previously downloaded build. bench.ps1 auto-selects the newest
# timestamped binary, so nothing else needs updating.
set -euo pipefail
export PATH="$HOME/.cargo/bin:$PATH"
# Static-DXC build requirements (embedded DXC, windows/msvc):
# - TAR_OPTIONS=--no-same-owner: mach-dxcompiler-rs's build script untars
#   its prebuilt libs; this sandbox's tar dies on ownership preservation.
# - XWIN_INCLUDE_ATL=true: DXC's static lib needs atls.lib (ATL). The
#   xwin cache in ~/.cache/cargo-xwin already contains ATL (grafted in
#   2026-10-20); if the cache is ever rebuilt from scratch, pre-cache
#   with this env set so ATL is included from the start.
export TAR_OPTIONS=--no-same-owner
export XWIN_INCLUDE_ATL=true
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
cargo xwin build --release --target x86_64-pc-windows-msvc -p gsim_app
STAMP="$(date '+%Y%m%d_%H%M%S')"
cp "target/x86_64-pc-windows-msvc/release/gsim_app.exe" \
   "dist/windows/gsim_app_${STAMP}.exe"
echo "installed dist/windows/gsim_app_${STAMP}.exe"
