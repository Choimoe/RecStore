cd "$(dirname "$0")"
set -x
set -e

sudo service ssh start

USER="$(whoami)"
PROJECT_PATH="$(cd .. && pwd)"

CMAKE_REQUIRE="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
GPU_ARCH="80"

TORCH_VERSION="2.5.1"
CUDA_VERSION="cu118"

MARKER_DIR="/tmp/env_setup_markers"

step_base() {
    sudo apt install -y libmemcached-dev ca-certificates lsb-release wget python3-dev
    pip3 install pymemcache
}

step_recover_bash() {
    ln -sf ${PROJECT_PATH}/dockerfiles/docker_config/.bashrc /home/${USER}/.bashrc
    source /home/${USER}/.bashrc
}

step_glog() {
    # git submodule add https://github.com/google/glog third_party/glog
    sudo rm -f /usr/lib/x86_64-linux-gnu/libglog.so.0*

    cd ${PROJECT_PATH}/third_party/glog/
    git checkout v0.5.0
    rm -rf _build
    mkdir -p _build
    cd _build
    CXXFLAGS="-fPIC" cmake .. ${CMAKE_REQUIRE} && make -j20 && make DESTDIR=${PROJECT_PATH}/third_party/glog/glog-install-fPIC install
    sudo make install
    make clean
}

step_fmt() {
    # git submodule add https://github.com/fmtlib/fmt third_party/fmt
    cd ${PROJECT_PATH}/third_party/fmt/
    rm -rf _build
    mkdir -p _build
    cd _build
    CXXFLAGS="-fPIC" cmake .. ${CMAKE_REQUIRE}
    make -j20
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
    CFLAGS='-fPIC' CXXFLAGS='-fPIC -Wl,-lrt' cmake .. -DCMAKE_INCLUDE_PATH=${PROJECT_PATH}/third_party/glog/glog-install-fPIC/usr/local/include -DCMAKE_LIBRARY_PATH=${PROJECT_PATH}/third_party/glog/glog-install-fPIC/usr/local/lib ${CMAKE_REQUIRE}
    make -j20
    make DESTDIR=${PROJECT_PATH}/third_party/folly/folly-install-fPIC install
    make clean
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
    CFLAGS='-fPIC' CXXFLAGS='-fPIC -Wl,-lrt' CC=/usr/bin/gcc CXX=/usr/bin/g++ cmake .. ${CMAKE_REQUIRE}
    make -j20
    sudo make install
    make clean
}

step_cityhash() {
    cd ${PROJECT_PATH}/third_party/cityhash/
    ./configure
    make -j20
    sudo make install
    make clean
}

# cd ${PROJECT_PATH}/third_party/rocksdb/ && rm -rf _build && mkdir _build && cd _build && cmake .. && make -j20 && sudo make install

# "#############################SPDK#############################
# cd ${PROJECT_PATH}/
# sudo apt install -y ca-certificates
# # sudo cp docker_config/ubuntu20.04.apt.ustc /etc/apt/sources.list
# sudo sed -i "s@http://.*archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
# sudo sed -i "s@http://.*security.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
# sudo -E apt-get update

# cd third_party/spdk
# sudo PATH=$PATH which pip3

# # if failed, sudo su, and execute in root;
# # the key is that which pip3 == /opt/bin/pip3
# sudo -E PATH=$PATH scripts/pkgdep.sh --all
# # exit sudo su

# ./configure
# sudo make clean
# make -j20
# sudo make install
# # make clean
# #############################SPDK#############################

# sudo rm /opt/conda/lib/libtinfo.so.6
# "

step_torch() {
    mkdir -p ${PROJECT_PATH}/binary
    cd ${PROJECT_PATH}/binary
    ################################################################################
    #     Manually compile torch from source, specifically, enable the CXX11 ABI.
    #     (binary/pytorch/dist/torch-2.5.0a0+gita8d6afb-cp310-cp310-linux_x86_64.whl)
    #     TODO: Manually compile and install torch commands or share wheel files.
    ################################################################################
    # pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple torch-2.5.0a0+git*.whl
    # pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cu118
}

step_arrow() {
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
    make -j
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

    wget -q https://download.pytorch.org/libtorch/${CUDA_VERSION}/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2B${CUDA_VERSION}.zip -O libtorch.zip

    if [ $? -ne 0 ]; then
        return 1
    fi

    unzip -o libtorch.zip -d . > /dev/null

    if [ $? -ne 0 ]; then
        return 1
    fi
}

step_HugeCTR() {
    # find /usr -name "libparquet.so"
    # find /usr -name "properties.h" | grep "parquet/properties.h"
    cd ${PROJECT_PATH}/third_party/HugeCTR
    rm -rf _build
    mkdir -p _build
    cd _build
    cmake -DCMAKE_BUILD_TYPE=Release \
        ${CMAKE_REQUIRE} \
        ..
    make embedding -j20
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
        -DCMAKE_INSTALL_PREFIX=$MY_INSTALL_DIR \
        $CMAKE_REQUIRE \
        ../..
    make -j
    make install -j
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
    cd /usr/lib/x86_64-linux-gnu
    sudo unlink libibverbs.so
    sudo cp -f libibverbs.so.1.14.39.0 libibverbs.so
}

mkdir -p "${MARKER_DIR}"
[ "$1" = "--clean" ]&&{ echo "Cleaning all markers..."; rm -rf "${MARKER_DIR:?}"; exit 0; };
marker_path(){ echo "${MARKER_DIR}/${1}.done"; }
steps=($(grep -oE '^step_[a-zA-Z0-9_]+\(\)' "$(readlink -f "$0")" | cut -d '(' -f1))
for STEP in "${steps[@]}"; do MARKER=$(marker_path "$STEP"); [ -f "$MARKER" ]&&{ echo "Step $STEP: Skipping (already completed)"; }||{ echo "Step $STEP: Running..."; $STEP; touch "$MARKER"; }; done

echo "Environment setup completed successfully."