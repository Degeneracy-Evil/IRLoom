#!/usr/bin/env bash
# CMake's Python helper: keep Python execution inside the existing uv project.
set -euo pipefail
exec uv run --project "$(dirname "$0")/.." python "$@"
