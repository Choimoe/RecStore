export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890
set -x
set -e

USER=xieminhui
PROJECT_PATH="/home/${USER}/RecStore"

sudo apt install -y libmemcached-dev 


# git submodule add https://github.com/google/glog third_party/glog
# sudo rm -f /usr/lib/x86_64-linux-gnu/libglog.so.0*

cd ${PROJECT_PATH}/third_party/glog/ && rm -rf _build && mkdir _build && cd _build && CXXFLAGS="-fPIC" cmake .. && make -j20 && make DESTDIR=${PROJECT_PATH}/third_party/glog/glog-install-fPIC install


# git submodule add https://github.com/fmtlib/fmt third_party/fmt
cd ${PROJECT_PATH}/third_party/fmt/ && rm -rf _build && mkdir _build && cd _build && CXXFLAGS="-fPIC" cmake .. && make -j20 && sudo make install


# git submodule add https://github.com/facebook/folly third_party/folly
export CC=`which gcc`
export CXX=`which g++`
cd ${PROJECT_PATH}/third_party/folly && \
rm -rf _build && mkdir -p _build && cd _build \
&& CFLAGS='-fPIC' CXXFLAGS='-fPIC -Wl,-lrt' cmake .. -DCMAKE_INCLUDE_PATH=${PROJECT_PATH}/third_party/glog/glog-install-fPIC/usr/local/include -DCMAKE_LIBRARY_PATH=${PROJECT_PATH}/third_party/glog/glog-install-fPIC/usr/local/lib \
&& make -j20 && make DESTDIR=${PROJECT_PATH}/third_party/folly/folly-install-fPIC install && make clean

# git submodule add https://github.com/google/googletest third_party/googletest


cd ${PROJECT_PATH}/third_party/gperftools && rm -rf _build && mkdir -p _build && cd _build && CFLAGS='-fPIC' CXXFLAGS='-fPIC -Wl,-lrt' CC=/usr/bin/gcc CXX=/usr/bin/g++ cmake .. && make -j20 && sudo make install && make clean


cd ${PROJECT_PATH}/third_party/cityhash/ && ./configure && make -j20 && sudo make install

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

cd ${PROJECT_PATH}/binary
pip3 install  -i https://pypi.tuna.tsinghua.edu.cn/simple torch-2.0.0a0+gitunknown-cp310-cp310-linux_x86_64.whl

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
      ../..
make -j
make install -j
popd



cd ${PROJECT_PATH}/src/kg/kg
bash install_dgl.sh


pip3 install pymemcache
