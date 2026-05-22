#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

MEMCACHED_HOST="${RECSTORE_MEMCACHED_HOST:-127.0.0.1}"
MEMCACHED_PORT="${RECSTORE_MEMCACHED_PORT:-21211}"
MEMCACHED_PIDFILE="${RECSTORE_MEMCACHED_PIDFILE:-/tmp/memcached.recstore.pid}"

SERVER_COUNT="${SERVER_COUNT:-1}"
CLIENT_COUNT="${CLIENT_COUNT:-1}"
TEST_BINARY="${TEST_BINARY:-./build/bin/petps_integration_test}"
GTEST_FILTER="${GTEST_FILTER:-PetPSIntegrationTest.PutGetRoundTripSingleShard:PetPSIntegrationTest.MissingKeysReturnZeroSlots}"
BUILD_TARGETS="${BUILD_TARGETS:-petps_server petps_integration_test}"
BUILD_JOBS="${BUILD_JOBS:-8}"

started_memcached=0

log() {
  printf '[petps-local] %s\n' "$*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

check_memcached_ready() {
  local out
  out="$(printf 'get serverNum\r\nquit\r\n' | nc -w 1 "${MEMCACHED_HOST}" "${MEMCACHED_PORT}" 2>/dev/null || true)"
  [[ "${out}" == *"END"* || "${out}" == *"VALUE serverNum"* ]]
}

start_memcached() {
  if check_memcached_ready; then
    log "memcached already reachable at ${MEMCACHED_HOST}:${MEMCACHED_PORT}"
    return
  fi

  require_cmd memcached
  require_cmd nc

  if [[ -f "${MEMCACHED_PIDFILE}" ]]; then
    rm -f "${MEMCACHED_PIDFILE}"
  fi

  log "starting memcached at ${MEMCACHED_HOST}:${MEMCACHED_PORT}"
  memcached -u root -l "${MEMCACHED_HOST}" -p "${MEMCACHED_PORT}" -c 10000 -d -P "${MEMCACHED_PIDFILE}"
  started_memcached=1

  for _ in $(seq 1 20); do
    if check_memcached_ready; then
      log "memcached is ready"
      return
    fi
    sleep 0.2
  done

  echo "memcached failed to become ready at ${MEMCACHED_HOST}:${MEMCACHED_PORT}" >&2
  exit 1
}

init_memcached_keys() {
  require_cmd nc
  log "initializing Mayfly coordination keys"
  printf 'set serverNum 0 0 1\r\n0\r\nset clientNum 0 0 1\r\n0\r\nset xmh-consistent-dsm 0 0 1\r\n0\r\nquit\r\n' \
    | nc -w 1 "${MEMCACHED_HOST}" "${MEMCACHED_PORT}"
}

show_memcached_keys() {
  require_cmd nc
  log "current memcached coordination keys"
  printf 'get serverNum\r\nget clientNum\r\nget xmh-consistent-dsm\r\nquit\r\n' \
    | nc -w 1 "${MEMCACHED_HOST}" "${MEMCACHED_PORT}"
}

cleanup() {
  if [[ "${started_memcached}" -eq 1 && -f "${MEMCACHED_PIDFILE}" ]]; then
    log "stopping memcached pid $(cat "${MEMCACHED_PIDFILE}")"
    kill "$(cat "${MEMCACHED_PIDFILE}")" >/dev/null 2>&1 || true
    rm -f "${MEMCACHED_PIDFILE}"
  fi
}

trap cleanup EXIT

require_cmd cmake
require_cmd python3

log "building targets: ${BUILD_TARGETS}"
cmake --build build --target ${BUILD_TARGETS} -j"${BUILD_JOBS}"

start_memcached
init_memcached_keys
show_memcached_keys

export RECSTORE_MEMCACHED_HOST="${MEMCACHED_HOST}"
export RECSTORE_MEMCACHED_PORT="${MEMCACHED_PORT}"
export RECSTORE_MEMCACHED_TEXT_PROTOCOL=1

log "running integration test"
python3 src/test/scripts/run_petps_integration.py \
  --server-count "${SERVER_COUNT}" \
  --client-count "${CLIENT_COUNT}" \
  --test-binary "${TEST_BINARY}" \
  --gtest-filter="${GTEST_FILTER}"
