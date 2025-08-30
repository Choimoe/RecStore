#!/bin/bash

set -euo pipefail

OUTPUT_DIR="./dataset"
# MIRROR_BASE="https://huggingface.co/datasets/criteo/CriteoClickLogs/resolve/main/"
MIRROR_BASE="https://hf-mirror.com/datasets/criteo/CriteoClickLogs/resolve/main"
MAX_RETRIES=3

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

if ! command -v aria2c &> /dev/null; then
    echo "aria2c not found. Please install: sudo apt install aria2" >&2
    exit 1
fi

download_file() {
    local day=$1
    local file="day_${day}.gz"
    local url="${MIRROR_BASE}/${file}"

    [ -f "$file" ] && gzip -t "$file" 2>/dev/null && return 0

    rm -f "$file"

    for i in $(seq 1 $MAX_RETRIES); do
        if aria2c -x 16 -s 16 -k 1M --file-allocation=trunc \
                  --connect-timeout=60 --timeout=300 --check-certificate=false \
                  -o "$file" "$url" && gzip -t "$file" 2>/dev/null; then
            return 0
        fi
        sleep $((2 ** i))
    done

    return 1
}

failed=()
for day in $(seq 0 23); do
    download_file "$day" || failed+=("day_${day}.gz")
done

if [ ${#failed[@]} -ne 0 ]; then
    printf 'Failed: %s\n' "${failed[@]}"
    exit 1
fi