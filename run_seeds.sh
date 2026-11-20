#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_PYTHON="${WARPX_PYTHON:-python3}"
exec "$PROJECT_PYTHON" "$PROJECT_ROOT/scripts/run_seeds.py" "$@"
