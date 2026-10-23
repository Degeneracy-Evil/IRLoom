#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export TMPDIR="$PWD/tmp"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p tmp logs/architecture
exec bash scripts/architecture-python.sh "$PWD/irloom/neural/correctness_build.py"
