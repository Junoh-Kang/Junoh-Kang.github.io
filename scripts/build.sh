#!/usr/bin/env bash
# Regenerate the CV outputs (site data JSON and PDF) from cv/, then build the site into dist/.
# Usage: scripts/build.sh [--skip-cv]
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" != "--skip-cv" ]]; then
  make -C cv publish-site
fi
npm run build
