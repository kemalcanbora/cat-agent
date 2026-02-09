#!/usr/bin/env bash
set -euo pipefail

#
# Simple helper script to cut a new release:
# - updates cat_agent/__init__.py __version__
# - commits the change
# - creates a git tag v<version>
# - pushes commit and tag to origin (which triggers the GitHub release workflow)
#
# Usage:
#   ./release.sh 0.1.2
#

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 NEW_VERSION" >&2
  exit 1
fi

NEW_VERSION="$1"

if ! [[ "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Error: version must be in the form X.Y.Z (e.g. 0.1.2)" >&2
  exit 1
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Error: working tree is not clean. Commit or stash changes first." >&2
  exit 1
fi

if git rev-parse "v${NEW_VERSION}" >/dev/null 2>&1; then
  echo "Error: tag v${NEW_VERSION} already exists. Choose a new version." >&2
  exit 1
fi

echo "Bumping version to ${NEW_VERSION} in cat_agent/__init__.py"

python - <<PY
from pathlib import Path
import re

path = Path("cat_agent/__init__.py")
text = path.read_text(encoding="utf-8")
new_text, count = re.subn(
    r'__version__\s*=\s*["\'][^"\']*["\']',
    f'__version__ = "{NEW_VERSION}"',
    text,
    count=1,
)
if count == 0:
    raise SystemExit("Could not find __version__ assignment in cat_agent/__init__.py")
path.write_text(new_text, encoding="utf-8")
PY

git add cat_agent/__init__.py
git commit -m "chore: release ${NEW_VERSION}"

git tag "v${NEW_VERSION}"

git push
git push origin "v${NEW_VERSION}"

echo "Release ${NEW_VERSION} created and pushed."

