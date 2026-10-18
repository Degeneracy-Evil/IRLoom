#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p tmp logs
export TMPDIR="$PWD/tmp"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ ${1:-} == setup ]]; then
  exec bash scripts/setup-toolchain.sh
fi
if [[ ${1:-collect} =~ ^(collect|compile|m0|m4|m5|m6|m7)$ ]] && \
   [[ -z ${MLIR_OPT:-} ]] && ! command -v mlir-opt >/dev/null && \
   [[ ! -x tmp/toolchain/usr/lib/llvm-18/bin/mlir-opt ]]; then
  bash scripts/setup-toolchain.sh
fi
exec uv run python -m irloom "$@"
