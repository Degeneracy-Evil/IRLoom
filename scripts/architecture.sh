#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export TMPDIR="$PWD/tmp"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ulimit -c 0
mkdir -p logs/architecture tmp
bash scripts/setup-architecture.sh > logs/architecture/setup.log 2>&1
cmake -S compiler/mlir_bridge -B benchmark-data/architecture/build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++-18 \
  -DMLIR_SDK="$PWD/benchmark-data/architecture/sdk/usr/lib/llvm-18" \
  > logs/architecture/configure.log 2>&1
cmake --build benchmark-data/architecture/build -j 2 > logs/architecture/build.log 2>&1
for argument in "$@"; do
  if [[ "$argument" == --program || "$argument" == --program=* || "$argument" == --pilot || "$argument" == --compare-pilot ]]; then
    frontend=benchmark-data/architecture/frontends/build/bin/irloom-polygeist-bridge
    ready=benchmark-data/architecture/frontends/.ready
    missing_frontend=false
    for tool in cgeist polygeist-opt mlir-opt mlir-translate irloom-polygeist-bridge; do
      [[ -f "benchmark-data/architecture/frontends/build/bin/$tool" ]] || missing_frontend=true
    done
    if [[ "$missing_frontend" == true || ! -f "$ready" || compiler/mlir_bridge/main.cpp -nt "$ready" || compiler/mlir_bridge/CMakeLists.txt -nt "$ready" || scripts/prepare-polygeist.sh -nt "$ready" ]]; then
      bash scripts/prepare-polygeist.sh --build > logs/architecture/prepare-frontend.log 2>&1
    fi
    break
  fi
done
cd experiments/architecture
exec uv run --frozen python -m irloom.neural.driver "$@"
