#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p tmp logs
export TMPDIR="$PWD/tmp"
if [[ ${1:-} == setup ]]; then
  exec bash scripts/setup-toolchain.sh
fi
if [[ ${1:-collect} =~ ^(collect|compile|m0)$ ]] && \
   [[ -z ${MLIR_OPT:-} ]] && ! command -v mlir-opt >/dev/null && \
   [[ ! -x tmp/toolchain/usr/lib/llvm-18/bin/mlir-opt ]]; then
  bash scripts/setup-toolchain.sh
fi
exec uv run python -m irloom "$@"
