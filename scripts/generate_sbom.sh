#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT/dist}"
mkdir -p "$OUT_DIR"

python -m pip install --quiet cyclonedx-bom

cyclonedx-py environment \
  --output-format json \
  --output-file "$OUT_DIR/sbom-cat-agent.cdx.json"

echo "SBOM written to $OUT_DIR/sbom-cat-agent.cdx.json"
