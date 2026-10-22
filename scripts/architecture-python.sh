#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export TMPDIR="$PWD/tmp"
cd experiments/architecture
exec uv run --frozen python "$@"
