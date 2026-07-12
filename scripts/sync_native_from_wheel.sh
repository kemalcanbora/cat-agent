#!/usr/bin/env bash
set -euo pipefail

# Copy the Rust extension from a built wheel into ./cat_agent/ so imports work
# when running examples/tests from the repo root (sys.path prefers ./cat_agent).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEEL="${1:-}"
if [[ -z "${WHEEL}" ]]; then
  WHEEL="$(ls -1t "${ROOT}"/dist/*.whl | head -1)"
fi
if [[ ! -f "${WHEEL}" ]]; then
  echo "Error: wheel not found: ${WHEEL}" >&2
  exit 1
fi

NATIVE_DST="${ROOT}/cat_agent/_native.abi3.so"
unzip -p "${WHEEL}" cat_agent/_native.abi3.so > "${NATIVE_DST}"
echo "Synced $(basename "${WHEEL}") -> cat_agent/_native.abi3.so"
