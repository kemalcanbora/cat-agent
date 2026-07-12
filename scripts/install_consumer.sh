#!/usr/bin/env bash
set -euo pipefail

#
# Install cat-agent the same way end users will: build a native wheel, then
# pip install cat-agent[extra] from that wheel. No editable install, no
# separate maturin develop step.
#
# Usage:
#   ./scripts/install_consumer.sh
#   ./scripts/install_consumer.sh rag
#   ./scripts/install_consumer.sh 'rag,test' examples/rag_keyword/rust_keyword_search_demo.py
#

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3.10}"
EXTRA="${1:-rag}"
EXAMPLE="${2:-}"

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "Error: ${PYTHON} not found. Set PYTHON=python3.11 (or similar)." >&2
  exit 1
fi

cd "${ROOT}"

echo "Using interpreter: $("${PYTHON}" --version)"
"${PYTHON}" -m pip install --upgrade pip maturin

rm -rf dist
"${PYTHON}" -m maturin build --release --manifest-path native/Cargo.toml --out dist

WHEEL="$(ls -1t dist/*.whl | head -1)"
if [[ -z "${WHEEL}" ]]; then
  echo "Error: no wheel produced under dist/" >&2
  exit 1
fi

echo "Installing ${WHEEL}[${EXTRA}]"
"${PYTHON}" -m pip uninstall -y cat-agent 2>/dev/null || true
"${PYTHON}" -m pip install --force-reinstall "${WHEEL}[${EXTRA}]"

"${ROOT}/scripts/sync_native_from_wheel.sh" "${WHEEL}"

"${PYTHON}" -c "import cat_agent._native as native; print(f'Installed cat-agent with native {native.__version__}')"

if [[ -n "${EXAMPLE}" ]]; then
  echo "Running example: ${EXAMPLE}"
  "${PYTHON}" "${EXAMPLE}"
fi
