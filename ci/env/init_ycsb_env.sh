#!/bin/bash
# ci/env/init_ycsb_env.sh
# Purpose: Initialize environment variables (paths, LD_LIBRARY_PATH) for running YCSB.
# Usage: source ci/env/init_ycsb_env.sh [install_prefix]
#
# Arguments:
#   install_prefix (optional): Path to the directory containing bin/ycsb and lib/.
#                              If not provided, attempts to auto-detect "build/" or "runner/extracted-ycsb/package".

# Note: This script is meant to be sourced, not executed, so that it exports variables to the current shell.

# Detect Project Root
if [ -n "${BASH_SOURCE[0]}" ]; then
    SCRIPT_PATH="${BASH_SOURCE[0]}"
else
    SCRIPT_PATH="$0"
fi
PROJECT_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/../../" && pwd)"

# 1. Determine Installation Prefix
INSTALL_PREFIX="${1:-}"

if [ -z "$INSTALL_PREFIX" ]; then
    # CI Environment (from artifact extraction)
    if [ -d "${PROJECT_ROOT}/runner/extracted-ycsb/package" ]; then
        INSTALL_PREFIX="${PROJECT_ROOT}/runner/extracted-ycsb/package"
    # Local Development Environment
    elif [ -d "${PROJECT_ROOT}/build/bin" ] && [ -f "${PROJECT_ROOT}/build/bin/ycsb" ]; then
        INSTALL_PREFIX="${PROJECT_ROOT}/build"
    else
        echo "Error: Could not automatically find YCSB installation."
        echo "  Checked: ${PROJECT_ROOT}/runner/extracted-ycsb/package"
        echo "  Checked: ${PROJECT_ROOT}/build"
        echo "Please build YCSB first (make ycsb) or provide the install prefix as an argument."
        return 1 2>/dev/null || exit 1
    fi
fi

echo "Using YCSB Prefix: ${INSTALL_PREFIX}"

# 2. Locate Binary
export YCSB_BIN="${INSTALL_PREFIX}/bin/ycsb"
if [ ! -f "$YCSB_BIN" ]; then
    echo "Error: YCSB binary not found at $YCSB_BIN"
    return 1 2>/dev/null || exit 1
fi

# 3. Setup LD_LIBRARY_PATH
#   CI Layout:    package/lib, package/deps/lib
#   Local Layout: build/lib
LIB_PATH="${INSTALL_PREFIX}/lib"
DEPS_LIB_PATH="${INSTALL_PREFIX}/deps/lib"

if [ -d "$DEPS_LIB_PATH" ]; then
    export LD_LIBRARY_PATH="${LIB_PATH}:${DEPS_LIB_PATH}:${LD_LIBRARY_PATH:-}"
else
    export LD_LIBRARY_PATH="${LIB_PATH}:${LD_LIBRARY_PATH:-}"
fi

echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"

# 4. Prepare Data & Log Directories
export YCSB_DATA_DIR="${PROJECT_ROOT}/tools/ycsb/data-store"
export YCSB_LOG_DIR="${PROJECT_ROOT}/dbtestlog"

mkdir -p "$YCSB_DATA_DIR"
mkdir -p "$YCSB_LOG_DIR"

echo "YCSB_DATA_DIR: ${YCSB_DATA_DIR}"
echo "YCSB_LOG_DIR:  ${YCSB_LOG_DIR}"

# 5. Common Configuration Paths (Relative to Project Root when running from Root)
export YCSB_WORKLOAD_DIR="tools/ycsb/workloads"
export YCSB_PROP_DIR="tools/ycsb/db"

echo "Environment Initialized."
