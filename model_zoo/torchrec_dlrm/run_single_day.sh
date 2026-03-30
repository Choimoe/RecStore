#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "Error: Neither python nor python3 is available in PATH" >&2
    exit 1
fi

show_help() {
    cat <<'EOF'
Single-day launcher wrapper.

Usage:
  ./run_single_day.sh [OPTIONS]

This wrapper forwards all options to:
  python -m model_zoo.torchrec_dlrm.launch_single_day

Common options:
  --gin_config PATH        Repeatable layered gin config
  --gin_binding EXPR       Repeatable gin override
  --torchrec               Use TorchRec backend
  --custom                 Use RecStore backend
  --dataset-path PATH      Override processed dataset path
  --batch-size N           Override batch size
  --learning-rate V        Override learning rate
  --epochs N               Override epochs
  --dry-run                Print command without running training
  --print-config           Print merged config and exit
  -h, --help               Show this help
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    show_help
    exit 0
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

exec "$PYTHON_BIN" -m model_zoo.torchrec_dlrm.launch_single_day "$@"
