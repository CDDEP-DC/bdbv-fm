#!/usr/bin/env bash
# Weekly run for macOS and Linux. Usage: scripts/weekly.sh [--submit]
# Without --submit it builds the submission and review report only.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p logs
log="logs/$(date +%F).log"
{
  git pull --ff-only || echo "git pull failed; continuing with local copy"
  bdbv forecast
  if [[ "${1:-}" == "--submit" ]]; then
    bdbv submit --yes
  fi
} >>"$log" 2>&1
