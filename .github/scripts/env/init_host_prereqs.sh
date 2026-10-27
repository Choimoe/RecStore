#!/bin/bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

sudo apt-get update -y
sudo apt-get remove --auto-remove -y cmake || true
sudo rm -rf /usr/local/mpi /usr/local/ucx || true

sudo apt-get install -y --no-install-recommends \
  vim gdb git wget tar unzip curl clang-format \
  libboost-all-dev \
  libevent-dev libdouble-conversion-dev libgflags-dev \
  libiberty-dev liblz4-dev liblzma-dev libsnappy-dev zlib1g-dev \
  binutils-dev libjemalloc-dev libssl-dev pkg-config \
  libunwind-dev libunwind8-dev libelf-dev libdwarf-dev \
  cloc check sudo libtbb-dev

sudo apt-get remove -y libgoogle-glog-dev || true

sudo apt-get install -y --no-install-recommends \
  zsh fzf google-perftools openssh-server software-properties-common \
  kmod libaio-dev

sudo apt-get install -y --no-install-recommends \
  pax-utils patchelf binutils chrpath

pip3 install --no-cache-dir cmake numpy tqdm
