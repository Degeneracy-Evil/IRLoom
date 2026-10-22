#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../experiments/architecture"
exec uv run --frozen python "$@"
