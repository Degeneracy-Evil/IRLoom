#!/usr/bin/env bash
# Independent, revision-matched frontend. No system install; no LLVM bootstrap.
set -euo pipefail
cd "$(dirname "$0")/.."
export TMPDIR="$PWD/tmp"
base="$PWD/benchmark-data/architecture/frontends"
polygeist_revision=77c04bb2a7a2406ca9480bcc9e729b07d2c8d077
llvm_revision=26eb4285b56edd8c897642078d91f16ff0fd3472
mkdir -p "$base" logs/architecture tmp
if [[ ! -d "$base/Polygeist/.git" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/llvm/Polygeist.git "$base/Polygeist" \
    > logs/architecture/polygeist-clone.log 2>&1
fi
if [[ "$(git -C "$base/Polygeist" rev-parse HEAD)" != "$polygeist_revision" ]]; then
  if [[ -n "$(git -C "$base/Polygeist" status --porcelain)" ]]; then
    echo 'Frontend source has local changes; refusing checkout' >&2; exit 2
  fi
  git -C "$base/Polygeist" checkout --detach "$polygeist_revision" > logs/architecture/polygeist-checkout.log 2>&1
fi
archive="$base/llvm-$llvm_revision.tar.gz"
if [[ ! -f "$archive" ]]; then
  curl -fL --retry 2 "https://codeload.github.com/llvm/llvm-project/tar.gz/$llvm_revision" \
    -o "$archive.partial" > logs/architecture/polygeist-llvm-download.log 2>&1
  mv "$archive.partial" "$archive"
fi
source_dir="$base/llvm-$llvm_revision"
if [[ ! -f "$source_dir/.extracted" ]]; then
  mkdir -p "$source_dir"
  tar -xzf "$archive" -C "$source_dir" --strip-components=1 \
    "llvm-project-$llvm_revision/llvm" "llvm-project-$llvm_revision/mlir" \
    "llvm-project-$llvm_revision/clang" "llvm-project-$llvm_revision/cmake" \
    "llvm-project-$llvm_revision/third-party" "llvm-project-$llvm_revision/LICENSE.TXT" \
    > logs/architecture/polygeist-llvm-extract.log 2>&1
  touch "$source_dir/.extracted"
fi
if [[ "${1:-}" == --build ]]; then
  cmake -S "$source_dir/llvm" -B "$base/build" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18 \
    -DPython3_EXECUTABLE="$PWD/scripts/architecture-python.sh" \
    -DLLVM_ENABLE_PROJECTS='clang;mlir' -DLLVM_TARGETS_TO_BUILD=X86 \
    -DLLVM_EXTERNAL_PROJECTS=polygeist -DLLVM_EXTERNAL_POLYGEIST_SOURCE_DIR="$base/Polygeist" \
    -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_ENABLE_BINDINGS=OFF \
    -DLLVM_ENABLE_ASSERTIONS=ON -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
    -DPOLYGEIST_ENABLE_CUDA=OFF -DPOLYGEIST_ENABLE_ROCM=OFF -DPOLYGEIST_ENABLE_POLYMER=OFF \
    -DPOLYGEIST_PGO_DEFAULT_DATA_DIR="$base/pgo-unused" \
    > logs/architecture/polygeist-configure.log 2>&1
  cmake --build "$base/build" --target cgeist mlir-opt mlir-translate -j "${IRLOOM_FRONTEND_JOBS:-8}" \
    > logs/architecture/polygeist-build.log 2>&1
fi
echo "Polygeist $polygeist_revision; LLVM $llvm_revision; cache $base"
