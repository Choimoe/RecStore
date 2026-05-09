#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ci/format/clang_format.sh [--check] [--all | --staged | --changed]

Options:
  --check    Verify formatting without modifying files.
  --all      Format all tracked RecStore C++ source files under src/.
  --staged   Format staged RecStore C++ source files under src/.
  --changed  Format changed RecStore C++ source files under src/.
EOF
}

mode="fix"
scope="changed"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check)
      mode="check"
      ;;
    --all)
      scope="all"
      ;;
    --staged)
      scope="staged"
      ;;
    --changed)
      scope="changed"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if ! command -v clang-format >/dev/null 2>&1; then
  echo "clang-format is required but was not found in PATH." >&2
  exit 127
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

if [[ "$scope" == "staged" ]]; then
  mapfile -t files < <(
    git diff --cached --name-only --diff-filter=ACMR \
      | grep -E '^src/.*\.(cc|cpp|h)$' || true
  )
elif [[ "$scope" == "changed" ]]; then
  base_ref="${FORMAT_BASE_REF:-}"
  if [[ -n "$base_ref" && ! "$base_ref" =~ ^0+$ ]]; then
    mapfile -t files < <(
      git diff --name-only --diff-filter=ACMR "$base_ref"...HEAD \
        | grep -E '^src/.*\.(cc|cpp|h)$' || true
    )
  elif git rev-parse --verify HEAD^ >/dev/null 2>&1; then
    mapfile -t files < <(
      git diff --name-only --diff-filter=ACMR HEAD^..HEAD \
        | grep -E '^src/.*\.(cc|cpp|h)$' || true
    )
  else
    mapfile -t files < <(
      {
        git diff --name-only --diff-filter=ACMR
        git diff --cached --name-only --diff-filter=ACMR
      } | sort -u | grep -E '^src/.*\.(cc|cpp|h)$' || true
    )
  fi
else
  mapfile -t files < <(
    git ls-files 'src/*.cc' 'src/*.cpp' 'src/*.h' \
      'src/**/*.cc' 'src/**/*.cpp' 'src/**/*.h'
  )
fi

if [[ ${#files[@]} -eq 0 ]]; then
  echo "No C++ source files to format."
  exit 0
fi

if [[ "$mode" == "check" ]]; then
  clang-format --dry-run --Werror "${files[@]}"
else
  clang-format -i "${files[@]}"
fi
