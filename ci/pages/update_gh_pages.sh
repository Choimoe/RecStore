#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash ci/pages/update_gh_pages.sh --source <dir> --mode <docs|coverage|ycsb> [options]

Update the gh-pages branch without clobbering independently published content.

Modes:
  docs      Replace the documentation site while preserving /coverage and /ycsb.
  coverage  Replace only /coverage while preserving the documentation site.
  ycsb      Replace only /ycsb while preserving the documentation site and /coverage.

Options:
  --branch <name>       Pages branch to update (default: gh-pages)
  --remote <name>       Git remote to push to (default: origin)
  --worktree <dir>      Temporary worktree path (default: .gh-pages-worktree)
  --message <message>   Commit message
  -h, --help            Show this help
EOF
}

SOURCE_DIR=""
MODE=""
BRANCH="gh-pages"
REMOTE="origin"
WORKTREE=".gh-pages-worktree"
MESSAGE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE_DIR="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --branch)
      BRANCH="$2"
      shift 2
      ;;
    --remote)
      REMOTE="$2"
      shift 2
      ;;
    --worktree)
      WORKTREE="$2"
      shift 2
      ;;
    --message)
      MESSAGE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${SOURCE_DIR}" || -z "${MODE}" ]]; then
  usage >&2
  exit 2
fi

if [[ "${MODE}" != "docs" && "${MODE}" != "coverage" && "${MODE}" != "ycsb" ]]; then
  echo "--mode must be docs, coverage, or ycsb" >&2
  exit 2
fi

if [[ ! -d "${SOURCE_DIR}" ]]; then
  echo "Source directory does not exist: ${SOURCE_DIR}" >&2
  exit 1
fi

if [[ -z "${MESSAGE}" ]]; then
  if [[ "${MODE}" == "docs" ]]; then
    MESSAGE="docs: deploy documentation"
  elif [[ "${MODE}" == "coverage" ]]; then
    MESSAGE="ci: deploy coverage reports"
  else
    MESSAGE="ci: deploy ycsb reports"
  fi
fi

git config user.name "${GIT_COMMITTER_NAME:-github-actions[bot]}"
git config user.email "${GIT_COMMITTER_EMAIL:-41898282+github-actions[bot]@users.noreply.github.com}"

rm -rf "${WORKTREE}"

if git ls-remote --exit-code --heads "${REMOTE}" "${BRANCH}" >/dev/null 2>&1; then
  git fetch "${REMOTE}" "${BRANCH}" --depth=1
  git worktree add -B "${BRANCH}" "${WORKTREE}" "${REMOTE}/${BRANCH}"
else
  git worktree add --detach "${WORKTREE}"
  git -C "${WORKTREE}" checkout --orphan "${BRANCH}"
  git -C "${WORKTREE}" rm -rf . >/dev/null 2>&1 || true
fi

if [[ "${MODE}" == "docs" ]]; then
  find "${WORKTREE}" -mindepth 1 -maxdepth 1 \
    ! -name '.git' \
    ! -name 'coverage' \
    ! -name 'ycsb' \
    -exec rm -rf {} +

  shopt -s dotglob nullglob
  for path in "${SOURCE_DIR%/}"/*; do
    cp -a "${path}" "${WORKTREE%/}/"
  done
  shopt -u dotglob nullglob
elif [[ "${MODE}" == "coverage" ]]; then
  rm -rf "${WORKTREE%/}/coverage"
  mkdir -p "${WORKTREE%/}/coverage"
  shopt -s dotglob nullglob
  for path in "${SOURCE_DIR%/}"/*; do
    cp -a "${path}" "${WORKTREE%/}/coverage/"
  done
  shopt -u dotglob nullglob
else
  rm -rf "${WORKTREE%/}/ycsb"
  mkdir -p "${WORKTREE%/}/ycsb"
  shopt -s dotglob nullglob
  for path in "${SOURCE_DIR%/}"/*; do
    cp -a "${path}" "${WORKTREE%/}/ycsb/"
  done
  shopt -u dotglob nullglob
fi

touch "${WORKTREE%/}/.nojekyll"

git -C "${WORKTREE}" add -A
if git -C "${WORKTREE}" diff --cached --quiet; then
  echo "No gh-pages changes to publish."
  git worktree remove "${WORKTREE}" --force
  exit 0
fi

git -C "${WORKTREE}" commit -m "${MESSAGE}"
git -C "${WORKTREE}" push "${REMOTE}" "HEAD:${BRANCH}"
git worktree remove "${WORKTREE}" --force
