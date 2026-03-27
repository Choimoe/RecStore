#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DO_RUN=0
DEEP_CLEAN=0

usage() {
  cat <<'EOF'
Usage:
  ./clean.sh [--run] [--deep] [--help]

Options:
  --run    Actually delete matched files/directories (default is dry-run)
  --deep   Also clean large output dirs: binary/, logs/, third_party/*-install
  --help   Show this help

Examples:
  ./clean.sh
  ./clean.sh --run
  ./clean.sh --run --deep
EOF
}

for arg in "$@"; do
  case "$arg" in
    --run) DO_RUN=1 ;;
    --deep) DEEP_CLEAN=1 ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "$ROOT_DIR/CMakeLists.txt" ]]; then
  echo "Error: script must run in RecStore repo root." >&2
  exit 1
fi

declare -a targets=()
declare -A seen=()

add_target() {
  local path="$1"
  [[ -e "$path" ]] || return 0

  case "$path" in
    "$ROOT_DIR/.git"|"$ROOT_DIR/.git/"*|"$ROOT_DIR/third_party/folly/build"|"$ROOT_DIR/third_party/folly/build/"*)
      return 0
      ;;
  esac

  if [[ -z "${seen["$path"]+x}" ]]; then
    seen["$path"]=1
    targets+=("$path")
  fi
}

# Common temporary build/cache artifacts.
# Prune very large vendor trees and virtualenv internals to keep scans fast and output useful.
while IFS= read -r path; do add_target "$path"; done < <(
  find "$ROOT_DIR" \
    \( -path "$ROOT_DIR/.git" \
      -o -path "$ROOT_DIR/third_party/pytorch" \
      -o -path "$ROOT_DIR/model_zoo/torchrec_dlrm/dlrm_venv" \
      -o -path "*/site-packages" \) -prune -o \
    \( -type d \
      \( -name "_build" \
        -o -name "build" \
        -o -name "cmake-build-*" \
        -o -name "CMakeFiles" \
        -o -name "__pycache__" \
        -o -name ".pytest_cache" \
        -o -name ".mypy_cache" \) \) -print
)

# Large but usually safe local cache.
add_target "$ROOT_DIR/.cache"
add_target "$ROOT_DIR/third_party/pytorch/build"

if [[ "$DEEP_CLEAN" -eq 1 ]]; then
  add_target "$ROOT_DIR/binary"
  add_target "$ROOT_DIR/logs"
  while IFS= read -r path; do add_target "$path"; done < <(find "$ROOT_DIR/third_party" -maxdepth 1 -type d -name "*-install")
fi

if [[ "${#targets[@]}" -eq 0 ]]; then
  echo "No temporary directories found."
  exit 0
fi

echo "Found ${#targets[@]} removable paths:"
for path in "${targets[@]}"; do
  du -sh "$path" 2>/dev/null || echo "0B  $path"
done | sort -hr

if [[ "$DO_RUN" -eq 0 ]]; then
  echo
  echo "Dry-run only. Re-run with --run to delete."
  echo "Tip: add --deep to also clean binary/, logs/, third_party/*-install."
  exit 0
fi

echo
echo "Deleting..."
for path in "${targets[@]}"; do
  rm -rf -- "$path"
done
echo "Done."
