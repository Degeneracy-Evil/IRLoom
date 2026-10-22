#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -e benchmark-data && ! -L benchmark-data ]]; then
  mkdir -p /mnt/data/chp/irloom-benchmarks
  ln -s /mnt/data/chp/irloom-benchmarks benchmark-data
fi
if [[ ! -L benchmark-data ]]; then
  echo 'benchmark-data must link to the external project cache' >&2; exit 2
fi
mkdir -p logs/architecture tmp benchmark-data/architecture/packages benchmark-data/architecture/sdk
arch_root="$PWD/benchmark-data/architecture"
for package in libmlir-18 libmlir-18-dev mlir-18-tools; do
  archive="$arch_root/packages/${package}_18.1.3-1ubuntu1_amd64.deb"
  if [[ ! -f "$archive" ]]; then
    curl -fL --retry 2 "https://archive.ubuntu.com/ubuntu/pool/universe/l/llvm-toolchain-18/$(basename "$archive")" \
      -o "$archive.partial" > "logs/architecture/download-$package.log" 2>&1
    mv "$archive.partial" "$archive"
  fi
  if [[ ! -f "$arch_root/sdk/.$package-extracted" ]]; then
    dpkg-deb -x "$archive" "$arch_root/sdk" > "logs/architecture/extract-$package.log" 2>&1
    touch "$arch_root/sdk/.$package-extracted"
  fi
done
"$arch_root/sdk/usr/lib/llvm-18/bin/mlir-opt" --version
