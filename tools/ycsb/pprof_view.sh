#!/usr/bin/env bash
set -euo pipefail
BIN=${1:-./build/ycsb}
PROF=${2:-}
PORT=${3:-18080}
current_time=$(date "+%Y-%m-%d_%H-%M-%S")
if [[ -z "${PROF}" ]]; then
  echo "usage: $0 <bin> <profile> [port]"
  exit 1
fi

mkdir -p ./profiles

if command -v pprof >/dev/null 2>&1; then
  echo "[*] Using 'pprof' (Go), http://localhost:${PORT}"
  exec pprof -http=:${PORT} "${BIN}" "${PROF}"
elif command -v go >/dev/null 2>&1; then
  echo "[*] Using 'go tool pprof', http://localhost:${PORT}"
  exec go tool pprof -http=:${PORT} "${BIN}" "${PROF}"
elif command -v google-pprof >/dev/null 2>&1; then
  echo "[*] Using 'google-pprof' (Perl) -> exporting SVG/PDF"
  # 避免浏览器问题，直接导出文件
  svg="./profiles/$(basename "${PROF%.*}")${current_time}.svg"
  pdf="./profiles/$(basename "${PROF%.*}")${current_time}.pdf"
  google-pprof --svg "${BIN}" "${PROF}" > "${svg}"
  google-pprof --pdf "${BIN}" "${PROF}" > "${pdf}"
  echo "[*] Wrote:"
  echo "    ${svg}"
  echo "    ${pdf}"
  exit 0
else
  echo "No pprof found. Install 'pprof' or Go, or google-perftools."
  exit 1
fi