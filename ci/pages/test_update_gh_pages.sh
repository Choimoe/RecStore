#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

REMOTE="${TMP_DIR}/remote.git"
WORK="${TMP_DIR}/work"
CALLER="${TMP_DIR}/caller"
SOURCE="${TMP_DIR}/source"
git init --bare "${REMOTE}" >/dev/null
git clone "${REMOTE}" "${WORK}" >/dev/null 2>&1

(
  cd "${WORK}"
  git config user.name "test"
  git config user.email "test@example.invalid"
  git checkout --orphan gh-pages >/dev/null 2>&1
  mkdir -p coverage ycsb
  echo "docs" > index.html
  echo "coverage" > coverage/index.html
  echo "old-ycsb" > ycsb/index.html
  git add -A
  git commit -m "seed pages" >/dev/null
  git push origin gh-pages >/dev/null 2>&1
)

git clone "${REMOTE}" "${CALLER}" >/dev/null 2>&1
(
  cd "${CALLER}"
  git checkout -b main >/dev/null 2>&1
)

mkdir -p "${SOURCE}"
echo "new-ycsb" > "${SOURCE}/index.html"

(
  cd "${CALLER}"
  bash "${ROOT_DIR}/ci/pages/update_gh_pages.sh" \
    --source "${SOURCE}" \
    --mode ycsb \
    --remote origin \
    --worktree "${TMP_DIR}/gh-pages-worktree" \
    --message "test: deploy ycsb" >/dev/null
)

CHECKOUT="${TMP_DIR}/checkout"
git clone --branch gh-pages "${REMOTE}" "${CHECKOUT}" >/dev/null 2>&1
grep -q "docs" "${CHECKOUT}/index.html"
grep -q "coverage" "${CHECKOUT}/coverage/index.html"
grep -q "new-ycsb" "${CHECKOUT}/ycsb/index.html"

echo "update_gh_pages ycsb mode test passed"
