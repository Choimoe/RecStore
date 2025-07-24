cd "$(dirname "$0")"
set -x
set -e

sudo service ssh start

USER="$(whoami)"
PROJECT_PATH="$(cd .. && pwd)"

CMAKE_REQUIRE="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
GPU_ARCH="80"

sudo apt install -y libmemcached-dev ca-certificates lsb-release wget python3-dev


ln -sf ${PROJECT_PATH}/dockerfiles/docker_config/.bashrc /home/${USER}/.bashrc
source /home/${USER}/.bashrc


# git submodule add https://github.com/google/glog third_party/glog
sudo rm -f /usr/lib/x86_64-linux-gnu/libglog.so.0*

cd ${PROJECT_PATH}/third_party/glog/ && git checkout v0.5.0 && rm -rf _build &&  mkdir -p _build && cd _build && CXXFLAGS="-fPIC" cmake .. ${CMAKE_REQUIRE} && make -j20 && make DESTDIR=${PROJECT_PATH}/third_party/glog/glog-install-fPIC install
sudo make install
make clean


# git submodule add https://github.com/fmtlib/fmt third_party/fmt
cd ${PROJECT_PATH}/third_party/fmt/ && rm -rf _build && mkdir -p _build && cd _build && CXXFLAGS="-fPIC" cmake .. ${CMAKE_REQUIRE} && make -j20 && sudo make install


# git submodule add https://github.com/facebook/folly third_party/folly
export CC=`which gcc`
export CXX=`which g++`
cd ${PROJECT_PATH}/third_party/folly && \
# git checkout v2021.01.04.00 && \
git checkout v2023.09.11.00 && \
rm -rf _build && \
mkdir -p _build && cd _build \
&& CFLAGS='-fPIC' CXXFLAGS='-fPIC -Wl,-lrt' cmake .. -DCMAKE_INCLUDE_PATH=${PROJECT_PATH}/third_party/glog/glog-install-fPIC/usr/local/include -DCMAKE_LIBRARY_PATH=${PROJECT_PATH}/third_party/glog/glog-install-fPIC/usr/local/lib ${CMAKE_REQUIRE} \
&& make -j20 && make DESTDIR=${PROJECT_PATH}/third_party/folly/folly-install-fPIC install && make clean

# git submodule add https://github.com/google/googletest third_party/googletest


cd ${PROJECT_PATH}/third_party/gperftools && rm -rf _build &&  mkdir -p _build && cd _build && CFLAGS='-fPIC' CXXFLAGS='-fPIC -Wl,-lrt' CC=/usr/bin/gcc CXX=/usr/bin/g++ cmake .. ${CMAKE_REQUIRE} && make -j20 && sudo  make install && make clean


# cd ${PROJECT_PATH}/third_party/gperftools/ && ./autogen.sh && ./configure && make -j20 && sudo make install

cd ${PROJECT_PATH}/third_party/cityhash/ && ./configure && make -j20 && sudo make install && make clean

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

TORCH_VERSION="2.5.1"
CUDA_VERSION="cu118"

mkdir -p ${PROJECT_PATH}/binary
cd ${PROJECT_PATH}/binary
# pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple torch-2.0.0a0+git*.whl
pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cu118

# HugeCTR
cd ${PROJECT_PATH}/build
wget https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
sudo apt install -y -V ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
sudo apt update
sudo apt install -y -V libarrow-dev libparquet-dev

cd ${PROJECT_PATH}/third_party/cpptrace
git checkout v0.3.1
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release ${CMAKE_REQUIRE} && make -j && sudo make install

mkdir -p ${PROJECT_PATH}/third_party/libtorch
cd ${PROJECT_PATH}/third_party/libtorch
wget https://download.pytorch.org/libtorch/${CUDA_VERSION}/libtorch-shared-with-deps-${TORCH_VERSION}%2B${CUDA_VERSION}.zip -O libtorch.zip \
&& unzip libtorch.zip -d . > /dev/null \
&& rm libtorch.zip

# find /usr -name "libparquet.so"
# find /usr -name "properties.h" | grep "parquet/properties.h"
cd ${PROJECT_PATH}/third_party/HugeCTR && rm -rf _build && mkdir -p _build && cd _build && \
cmake -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=OFF \
      -DENABLE_SAMPLES=OFF \
      -DSM=${GPU_ARCH} \
      -DCMAKE_INSTALL_PREFIX=/usr/local/hugectr \
      -DPARQUET_LIB_PATH=/usr/lib/x86_64-linux-gnu/libparquet.so \
      -DPARQUET_INCLUDE_DIR=/usr/include \
      ${CMAKE_REQUIRE} \
      ..
make embedding -j20
sudo find . -name "*.so" -exec cp {} /usr/local/hugectr/lib/ \;
make clean


# GRPC
cd ${PROJECT_PATH}/
cd third_party/grpc
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

sudo apt install -y sshpass
yes y | ssh-keygen -t rsa -q -f "$HOME/.ssh/id_rsa" -N ""

cd ${PROJECT_PATH}/dockerfiles
source set_coredump.sh


cd ${PROJECT_PATH}/src/kg/kg
bash install_dgl.sh



pip3 install pymemcache


cd /usr/lib/x86_64-linux-gnu
sudo unlink libibverbs.so
sudo cp -f libibverbs.so.1.14.39.0  libibverbs.so
