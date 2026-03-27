#!/bin/bash
SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
cd "$SCRIPT_DIR"
set -x
set -e
git config --global --add safe.directory '*'

USER="$(whoami)"
PROJECT_PATH="$(cd .. && pwd)"

LOG_TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${PROJECT_PATH}/init_env_${LOG_TS}.log"
echo "Init env log: ${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

sudo service ssh start

CMAKE_REQUIRE="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
GPU_ARCH="80"

if [ "${CI_THROTTLE_MAKE:-0}" = "1" ]; then
    MAKE_OPTS="${MAKE_OPTS:--j4 -O}"
else
    MAKE_OPTS="${MAKE_OPTS:--j20 -O}"
fi
export MAKEFLAGS="${MAKE_OPTS}"

TORCH_VERSION="2.5.1"
CUDA_VERSION="cu118"
LIBTORCH_VARIANT="${LIBTORCH_VARIANT:-${CUDA_VERSION}}"  # default to CUDA variant (e.g., cu118); set to cpu to force CPU libtorch

MARKER_DIR="/tmp/env_setup_markers"

if [ "$USER" = "root" ]; then
    target_dir="/root"
else
    target_dir="/home/$USER"
fi

ln -sf "${PROJECT_PATH}/dockerfiles/docker_config/.bashrc" "${target_dir}/.bashrc"
source "${target_dir}/.bashrc"

# ===============================================

step_base() {
    sudo apt update
    sudo apt install -y libmemcached-dev ca-certificates lsb-release wget python3-dev pybind11-dev
    pip3 install pymemcache
}

step_liburing() {
    cd ${PROJECT_PATH}/third_party/liburing
    ./configure --cc=gcc --cxx=g++
    make ${MAKE_OPTS}
    make liburing.pc
    sudo make install
}

step_glog() {
    # git submodule add https://github.com/google/glog third_party/glog
    sudo rm -f /usr/lib/x86_64-linux-gnu/libglog.so.0*

    cd ${PROJECT_PATH}/third_party/glog/
    git checkout v0.5.0
    rm -rf _build
    mkdir -p _build
    cd _build
    CXXFLAGS="-fPIC" cmake .. -DCMAKE_BUILD_TYPE=Release ${CMAKE_REQUIRE} && make ${MAKE_OPTS} && make ${MAKE_OPTS} DESTDIR=${PROJECT_PATH}/third_party/glog/glog-install-fPIC install
    sudo make install
}

step_fmt() {
    # git submodule add https://github.com/fmtlib/fmt third_party/fmt
    cd ${PROJECT_PATH}/third_party/fmt/
    rm -rf _build
    mkdir -p _build
    cd _build
    CXXFLAGS="-fPIC" cmake .. -DCMAKE_BUILD_TYPE=Release ${CMAKE_REQUIRE}
    make ${MAKE_OPTS}
    sudo make install
}

step_folly() {
    # git submodule add https://github.com/facebook/folly third_party/folly
    export CC=`which gcc`
    export CXX=`which g++`
    cd ${PROJECT_PATH}/third_party/folly
    # git checkout v2021.01.04.00
    git checkout v2023.09.11.00
    rm -rf _build
    mkdir -p _build
    cd _build
    CFLAGS='-fPIC' CXXFLAGS='-fPIC -Wl,-lrt' cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INCLUDE_PATH=${PROJECT_PATH}/third_party/glog/glog-install-fPIC/usr/local/include -DCMAKE_LIBRARY_PATH=${PROJECT_PATH}/third_party/glog/glog-install-fPIC/usr/local/lib ${CMAKE_REQUIRE}
    make ${MAKE_OPTS}
    make ${MAKE_OPTS} DESTDIR=${PROJECT_PATH}/third_party/folly/folly-install-fPIC install
}

# step_gtest() {
#     # git submodule add https://github.com/google/googletest third_party/googletest
# }

step_gperftools() {
    # cd ${PROJECT_PATH}/third_party/gperftools/ && ./autogen.sh && ./configure && make -j20 && sudo make install
    cd ${PROJECT_PATH}/third_party/gperftools
    rm -rf _build
    mkdir -p _build
    cd _build
    CFLAGS='-fPIC' CXXFLAGS='-fPIC -Wl,-lrt' CC=/usr/bin/gcc CXX=/usr/bin/g++ cmake .. -DCMAKE_BUILD_TYPE=Release ${CMAKE_REQUIRE}
    make ${MAKE_OPTS}
    sudo make install
}

step_cityhash() {
    cd ${PROJECT_PATH}/third_party/cityhash/
    ./configure
    make ${MAKE_OPTS}
    sudo make install
}

# cd ${PROJECT_PATH}/third_party/rocksdb/ && rm -rf _build && mkdir _build && cd _build && cmake .. && make -j20 && sudo make install

# "#############################SPDK#############################
# cd ${PROJECT_PATH}/
# sudo apt install -y ca-certificates
# # sudo cp docker_config/ubuntu20.04.apt.ustc /etc/apt/sources.list
# sudo sed -i "s@http://.*archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
# sudo sed -i "s@http://.*security.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
# sudo -E apt-get update

step_spdk() {
    sudo apt-get install -y libfuse-dev kmod
    cd ${PROJECT_PATH}/third_party/spdk
    rm -rf build
    sudo PATH=$PATH which pip3
    sudo apt-get update --fix-missing

    # # if failed, sudo su, and execute in root;
    # # the key is that which pip3 == /opt/bin/pip3
    export GO111MODULE=on
    export GOPROXY=https://goproxy.cn,direct
    sudo -E PATH=$PATH scripts/pkgdep.sh --all
    # # exit sudo su

    ./configure
    sudo make clean
    export PATH=$PATH:/var/spdk/dependencies/pip/bin
    sudo pip3 install pyelftools
    make ${MAKE_OPTS}
    sudo env "PATH=/var/spdk/dependencies/pip/bin:$PATH" make ${MAKE_OPTS} install
}
# #############################SPDK#############################

# sudo rm /opt/conda/lib/libtinfo.so.6
# "

step_torch() {
    if [ "${CI:-}" = "1" ] || [ "${CI:-}" = "true" ] || [ "${GITHUB_ACTIONS:-}" = "true" ] || [ -n "${GITLAB_CI:-}" ] || [ -n "${JENKINS_URL:-}" ] || [ -n "${BUILD_BUILDID:-}" ]; then
        echo "Skipping torch install because CI environment detected"
        return 0
    fi

    mkdir -p ${PROJECT_PATH}/binary
    cd ${PROJECT_PATH}/binary
    ################################################################################
    #     Manually compile torch from source, specifically, enable the CXX11 ABI.
    #     (binary/pytorch/dist/torch-2.5.0a0+gita8d6afb-cp310-cp310-linux_x86_64.whl)
    #     TODO: Manually compile and install torch commands or share wheel files.
    ################################################################################
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple torch-2.6.0a0+git2236df1-cp310-cp310-linux_x86_64.whl
    # pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cu118
}

step_arrow() {
    mkdir -p ${PROJECT_PATH}/build
    cd ${PROJECT_PATH}/build
    wget https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
    sudo apt install -y -V ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
    sudo apt update
    sudo apt install -y -V libarrow-dev libparquet-dev --fix-missing
}

step_cpptrace() {
    cd ${PROJECT_PATH}/third_party/cpptrace
    git checkout v0.3.1
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release ${CMAKE_REQUIRE}
    make ${MAKE_OPTS}
    sudo make install
}

step_libtorch_abi() {
    local libtorch_dir="${PROJECT_PATH}/third_party/libtorch"
    local zip_file="${libtorch_dir}/libtorch.zip"
    local extracted_marker="${libtorch_dir}/libtorch"

    mkdir -p "${libtorch_dir}"
    cd "${libtorch_dir}"

    if [ -f "${zip_file}" ] && [ -d "${extracted_marker}" ]; then
        return 0
    fi

    local url
    if [ "${LIBTORCH_VARIANT}" = "cpu" ]; then
        url="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2Bcpu.zip"
    else
        url="https://download.pytorch.org/libtorch/${LIBTORCH_VARIANT}/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2B${LIBTORCH_VARIANT}.zip"
    fi

    wget -q "${url}" -O libtorch.zip

    if [ $? -ne 0 ]; then
        return 1
    fi

    unzip -o libtorch.zip -d . > /dev/null

    if [ $? -ne 0 ]; then
        return 1
    fi
}

step_HugeCTR() {
    if [ "${SKIP_HUGECTR:-}" = "1" ]; then
        echo "Skipping HugeCTR build because SKIP_HUGECTR=1"
        return 0
    fi
    if ! command -v nvcc >/dev/null 2>&1; then
        echo "Skipping HugeCTR build because nvcc (CUDA toolkit) is not available"
        return 0
    fi
    # find /usr -name "libparquet.so"
    # find /usr -name "properties.h" | grep "parquet/properties.h"
    cd ${PROJECT_PATH}/third_party/HugeCTR
    rm -rf _build
    mkdir -p _build
    cd _build
    cmake -DCMAKE_BUILD_TYPE=Release \
        ${CMAKE_REQUIRE} \
        ..
    make ${MAKE_OPTS} embedding
    sudo mkdir -p /usr/local/hugectr/lib/
    sudo find . -name "*.so" -exec cp {} /usr/local/hugectr/lib/ \;
    make clean
}


# GRPC
step_GRPC() {
    cd ${PROJECT_PATH}/third_party/grpc
    export MY_INSTALL_DIR=${PROJECT_PATH}/third_party/grpc-install
    rm -rf cmake/build
    mkdir -p cmake/build
    pushd cmake/build
    cmake -DgRPC_INSTALL=ON \
        -DgRPC_BUILD_TESTS=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=$MY_INSTALL_DIR \
        $CMAKE_REQUIRE \
        ../..
    make ${MAKE_OPTS}
    make ${MAKE_OPTS} install
    popd
}

step_ssh() {
    sudo apt install -y sshpass
    yes y | ssh-keygen -t rsa -q -f "$HOME/.ssh/id_rsa" -N ""
}

step_set_coredump() {
    cd ${PROJECT_PATH}/dockerfiles
    source set_coredump.sh
}

# step_dgl() {
#     cd ${PROJECT_PATH}/src/kg/kg
#     bash install_dgl.sh
# }

step_libibverbs() {
    if [ "${SKIP_LIBIBVERBS:-0}" = "1" ]; then
        echo "Skipping libibverbs relink because SKIP_LIBIBVERBS=1"
        return 0
    fi
    cd /usr/lib/x86_64-linux-gnu
    local target
    target=$(ls libibverbs.so.1.* 2>/dev/null | head -n 1)
    if [ -z "${target}" ]; then
        echo "Skipping libibverbs relink because libibverbs.so.1.* not found"
        return 0
    fi
    sudo unlink libibverbs.so || true
    sudo cp -f "${target}" libibverbs.so
}

step_brpc() {

    sudo apt install -y libleveldb-dev
    # protobuf
    cd ${PROJECT_PATH}/third_party/grpc/third_party/protobuf
    if [ ! -d "_build" ]; then
        mkdir _build
    fi
    cd _build
    if [ ! -f "Makefile" ]; then
        cmake .. -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
            -DBUILD_SHARED_LIBS=ON \
            ${CMAKE_REQUIRE} \
            -DCMAKE_INSTALL_PREFIX=${PROJECT_PATH}/third_party/protobuf-install
    fi
    make ${MAKE_OPTS}
    make ${MAKE_OPTS} install
    
    # brpc
    cd ${PROJECT_PATH}/third_party/brpc
    if [ ! -d "_build" ]; then
        mkdir -p _build
    fi
    cd _build
    if [ ! -f "Makefile" ]; then
        cmake .. -DProtobuf_INCLUDE_DIR=${PROJECT_PATH}/third_party/protobuf-install/include \
            -DProtobuf_LIBRARIES=${PROJECT_PATH}/third_party/protobuf-install/lib/libprotobuf.so   \
            -DProtobuf_PROTOC_EXECUTABLE=${PROJECT_PATH}/third_party/protobuf-install/bin/protoc \
            -DCMAKE_BUILD_TYPE=Release \
            ${CMAKE_REQUIRE} \
            -DCMAKE_INSTALL_PREFIX=${PROJECT_PATH}/third_party/brpc-install \
            -DWITH_GLOG=ON
    fi
    make ${MAKE_OPTS}
    make ${MAKE_OPTS} install
}

mkdir -p "${MARKER_DIR}"
[ "$1" = "--clean" ]&&{ echo "Cleaning all markers..."; rm -rf "${MARKER_DIR:?}"; exit 0; };
marker_path(){ echo "${MARKER_DIR}/${1}.done"; }
steps=($(grep -oE '^step_[a-zA-Z0-9_]+\(\)' "$SCRIPT_PATH" | cut -d '(' -f1))
declare -a step_names=()
declare -A step_status=()
declare -A step_duration_sec=()

for STEP in "${steps[@]}"; do
    step_names+=("$STEP")
    step_start_ts=$(date +%s)
    MARKER=$(marker_path "$STEP")

    if [ -f "$MARKER" ]; then
        echo "Step $STEP: Skipping (already completed)"
        step_status["$STEP"]="SKIPPED"
    else
        echo "Step $STEP: Running..."
        $STEP
        touch "$MARKER"
        step_status["$STEP"]="DONE"
    fi

    step_end_ts=$(date +%s)
    step_duration_sec["$STEP"]=$((step_end_ts - step_start_ts))
done

echo "Environment setup completed successfully."
echo "Step duration summary:"
for STEP in "${step_names[@]}"; do
    printf "  - %-24s [%7s] %6ss\n" "$STEP" "${step_status[$STEP]}" "${step_duration_sec[$STEP]}"
done
