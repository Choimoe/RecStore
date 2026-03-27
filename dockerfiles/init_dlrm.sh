#!/bin/bash
cd "$(dirname "$0")"
set -euo pipefail
set -x

PYTHON_BIN="${PYTHON_BIN:-python3}"
MIRROR=1
WORKSPACE_ROOT="${DLRM_BUILD_ROOT:-/tmp/recstore-dlrm-deps}"
FBGEMM_REPO="${FBGEMM_REPO:-https://github.com/pytorch/FBGEMM.git}"
FBGEMM_REF="${FBGEMM_REF:-v1.1.2}"
TORCHREC_REPO="${TORCHREC_REPO:-https://github.com/pytorch/torchrec.git}"
TORCHREC_REF="${TORCHREC_REF:-v1.1.0}"

usage() {
    cat <<'EOF'
Usage: bash init_dlrm.sh [--mirror=0] [--workspace-root PATH] [--fbgemm-ref REF] [--torchrec-ref REF]

This script keeps the preinstalled torch untouched and builds fbgemm_gpu + torchrec
from source against the current torch environment.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --mirror=0)
            MIRROR=0
            ;;
        --workspace-root)
            WORKSPACE_ROOT="$2"
            shift
            ;;
        --fbgemm-ref)
            FBGEMM_REF="$2"
            shift
            ;;
        --torchrec-ref)
            TORCHREC_REF="$2"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
    shift
done

PIP_INDEX_ARGS=()
if [ "$MIRROR" = "1" ]; then
    PIP_INDEX_ARGS=(--index-url https://pypi.tuna.tsinghua.edu.cn/simple)
fi

pip_install() {
    "$PYTHON_BIN" -m pip install "$@" "${PIP_INDEX_ARGS[@]}"
}

user_base_bin() {
    "$PYTHON_BIN" -m site --user-base
}

ensure_repo() {
    local repo_url="$1"
    local repo_ref="$2"
    local target_dir="$3"

    if [ ! -d "$target_dir/.git" ]; then
        git clone --recursive "$repo_url" "$target_dir"
    fi

    git -C "$target_dir" fetch --tags origin
    git -C "$target_dir" checkout "$repo_ref"
    git -C "$target_dir" submodule sync --recursive
    git -C "$target_dir" submodule update --init --recursive
}

readarray -t TORCH_INFO < <("$PYTHON_BIN" - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda or "cpu")
print(int(torch.compiled_with_cxx11_abi()))
PY
)

TORCH_VERSION="${TORCH_INFO[0]}"
TORCH_CUDA_VERSION="${TORCH_INFO[1]}"
TORCH_CXX11_ABI="${TORCH_INFO[2]}"

echo "Detected torch version: ${TORCH_VERSION}"
echo "Detected torch cuda: ${TORCH_CUDA_VERSION}"
echo "Detected torch cxx11abi: ${TORCH_CXX11_ABI}"

if [ "${TORCH_CUDA_VERSION}" = "cpu" ]; then
    echo "Detected CPU-only torch install; skipping TorchRec/FBGEMM source build"
    exit 0
fi

if ! command -v nvcc >/dev/null 2>&1; then
    echo "Skipping TorchRec/FBGEMM source build because CUDA toolkit is not available"
    exit 0
fi

echo "Building TorchRec stack from source against the current torch install"
echo "Aligning extension builds with torch cxx11abi=${TORCH_CXX11_ABI}"
mkdir -p "$WORKSPACE_ROOT"
FBGEMM_SRC_DIR="$WORKSPACE_ROOT/FBGEMM"
TORCHREC_SRC_DIR="$WORKSPACE_ROOT/torchrec"

pip_install --upgrade pip setuptools wheel cmake ninja scikit-build packaging "numpy<2" setuptools-git-versioning tabulate
pip_install torchmetrics==1.0.3 tqdm iopath pyre-extensions
export PATH="$(user_base_bin)/bin:$PATH"
NINJA_BIN="$(command -v ninja)"
NINJA_SHIM_ROOT="$WORKSPACE_ROOT/python_shims"
mkdir -p "$NINJA_SHIM_ROOT/ninja"
cat > "$NINJA_SHIM_ROOT/ninja/__init__.py" <<EOF
BIN_DIR = "${NINJA_BIN%/*}"
EOF
export PYTHONPATH="$NINJA_SHIM_ROOT:${PYTHONPATH:-}"
echo "Using ninja binary: ${NINJA_BIN}"
export USE_CXX11_ABI="${TORCH_CXX11_ABI}"
export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI} ${CXXFLAGS:-}"
export CMAKE_ARGS="-DCMAKE_MAKE_PROGRAM:FILEPATH=${NINJA_BIN} -D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI} ${CMAKE_ARGS:-}"
"$PYTHON_BIN" -m pip uninstall -y torchrec fbgemm-gpu fbgemm_gpu || true

ensure_repo "$FBGEMM_REPO" "$FBGEMM_REF" "$FBGEMM_SRC_DIR"
ensure_repo "$TORCHREC_REPO" "$TORCHREC_REF" "$TORCHREC_SRC_DIR"

export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"
export MAX_JOBS="${MAX_JOBS:-$CMAKE_BUILD_PARALLEL_LEVEL}"

pushd "$FBGEMM_SRC_DIR/fbgemm_gpu"
rm -rf _skbuild build dist
"$PYTHON_BIN" setup.py bdist_wheel
"$PYTHON_BIN" -m pip install --force-reinstall --no-deps dist/*.whl
"$PYTHON_BIN" -m pip install --force-reinstall "numpy<2"
if ! command -v patchelf >/dev/null 2>&1; then
    echo "patchelf is required to patch fbgemm_gpu dependencies" >&2
    exit 1
fi
FBGEMM_SO_PATH="$("$PYTHON_BIN" - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("fbgemm_gpu")
if spec is None or spec.origin is None:
    raise SystemExit("Unable to locate installed fbgemm_gpu package")

pkg_dir = Path(spec.origin).resolve().parent
so_path = pkg_dir / "fbgemm_gpu_py.so"
if not so_path.exists():
    raise SystemExit(f"Unable to locate fbgemm_gpu_py.so under {pkg_dir}")

print(so_path)
PY
)"
if ! patchelf --print-needed "$FBGEMM_SO_PATH" | grep -Fx 'libtbb.so.12' >/dev/null 2>&1; then
    patchelf --add-needed libtbb.so.12 "$FBGEMM_SO_PATH"
fi
popd

pushd "$TORCHREC_SRC_DIR"
"$PYTHON_BIN" -m pip install -e . --no-deps
popd

"$PYTHON_BIN" - <<'PY'
import torch
import fbgemm_gpu
import torchrec

required_ops = [
    'asynchronous_complete_cumsum',
    'bounds_check_indices',
]
missing = [name for name in required_ops if not hasattr(torch.ops.fbgemm, name)]
if missing:
    raise SystemExit(f"Missing registered fbgemm ops after build: {missing}")

print('Torch version:', torch.__version__)
print('TorchRec import path:', torchrec.__file__)
print('FBGEMM import path:', fbgemm_gpu.__file__)
print('Verified fbgemm ops:', ', '.join(required_ops))
PY

# Usage
# bash init_dlrm.sh [--mirror=0]
# bash init_dlrm.sh --workspace-root /tmp/recstore-dlrm-deps --fbgemm-ref main --torchrec-ref main
# torchrun --nnodes 1 --nproc_per_node 2 --rdzv_backend c10d --rdzv_endpoint localhost --rdzv_id 54321 --role trainer dlrm_main.py
