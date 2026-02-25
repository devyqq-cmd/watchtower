#!/bin/bash
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

# Install all dependencies using uv
uv sync

# Ensure PYTHONPATH includes project root
echo 'export PYTHONPATH="."' >> "$CLAUDE_ENV_FILE"
