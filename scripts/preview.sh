#!/usr/bin/env bash
# Serve the already-built site in dist/ locally; run `npm run build` first to see new changes.
# Usage: scripts/preview.sh
set -euo pipefail
cd "$(dirname "$0")/.."

npx astro preview
