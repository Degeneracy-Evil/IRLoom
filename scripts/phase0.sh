#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p tmp logs
export TMPDIR="$PWD/tmp"
if [[ ${1:-} == setup ]]; then
  exec bash scripts/setup-toolchain.sh
fi
exec uv run python -m irloom.collect "$@"
