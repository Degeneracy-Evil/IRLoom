#!/usr/bin/env bash
# Download/extract only; never installs into system directories.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p tmp/packages tmp/toolchain logs
if [[ -x tmp/toolchain/usr/lib/llvm-18/bin/mlir-opt ]]; then
  tmp/toolchain/usr/lib/llvm-18/bin/mlir-opt --version
  exit 0
fi
if [[ $(uname -m) != x86_64 ]] || [[ ! -x /usr/lib/llvm-18/bin/clang ]]; then
  echo 'This local package recipe needs Ubuntu amd64 + LLVM 18; see docs/RUNNING.md.' >&2
  exit 1
fi
curl -fL --retry 2 \
  https://archive.ubuntu.com/ubuntu/pool/universe/l/llvm-toolchain-18/mlir-18-tools_18.1.3-1ubuntu1_amd64.deb \
  -o tmp/packages/mlir-18-tools.deb > logs/toolchain-download.log 2>&1
dpkg-deb -x tmp/packages/mlir-18-tools.deb tmp/toolchain > logs/toolchain-extract.log 2>&1
tmp/toolchain/usr/lib/llvm-18/bin/mlir-opt --version
