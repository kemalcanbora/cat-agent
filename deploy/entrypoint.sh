#!/bin/sh
set -eu

if [ -n "${CAT_AGENT_ENCRYPTION_KEY:-}" ]; then
  echo "Encryption key configured via CAT_AGENT_ENCRYPTION_KEY"
else
  echo "WARNING: CAT_AGENT_ENCRYPTION_KEY is not set. Configure a key for production deployments."
fi

exec cat-agent "$@"
