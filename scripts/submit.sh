#!/usr/bin/env bash
set -euo pipefail
RESULTS_DIR=${1:-./results}
OUT_ZIP=${2:-./result.zip}
cd "$(dirname "$0")/.."
rm -f "$OUT_ZIP"
cd "$RESULTS_DIR" && zip -r "../$(basename "$OUT_ZIP")" shapenet
echo "Created $(basename "$OUT_ZIP")"
