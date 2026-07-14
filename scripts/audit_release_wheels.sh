#!/usr/bin/env bash
set -euo pipefail

# Verify that release artifacts include wheels for all supported platforms.
# abi3-py310 wheels: one per OS/CPU tag is enough for Python 3.10+.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="${ROOT}/dist"

if [[ ! -d "${DIST}" ]]; then
  echo "Error: dist/ not found at ${DIST}" >&2
  exit 1
fi

shopt -s nullglob
wheels=("${DIST}"/*.whl)
if [[ ${#wheels[@]} -eq 0 ]]; then
  echo "Error: no wheels found in ${DIST}" >&2
  exit 1
fi

missing=0
check_tag() {
  local label="$1"
  local pattern="$2"
  if ! ls "${DIST}"/*.whl 2>/dev/null | grep -Eq "${pattern}"; then
    echo "Missing wheel: ${label} (expected filename matching /${pattern}/)"
    missing=1
  fi
}

check_tag 'manylinux x86_64' 'manylinux.*x86_64'
check_tag 'manylinux aarch64' 'manylinux.*aarch64'
check_tag 'macOS arm64' 'macosx.*arm64'
check_tag 'macOS x86_64' 'macosx.*x86_64'
check_tag 'Windows amd64' 'win_amd64'

echo "Found wheels:"
for wheel in "${wheels[@]}"; do
  basename "${wheel}"
done

if [[ "${missing}" -ne 0 ]]; then
  echo "Wheel audit failed." >&2
  exit 1
fi

echo "Wheel audit passed."
