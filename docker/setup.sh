#!/bin/sh

SUDO="sudo -H"

# install most dependencies via apt-get
${SUDO} apt-get -y update
${SUDO} apt-get -y upgrade
# We explicitly set the C++ compiler to g++, the default GNU g++ compiler. This is
# needed because we depend on system-installed libraries built with g++ and linked
# against libstdc++. In case `c++` corresponds to `clang++`, code will not build, even
# if we would pass the flag `-stdlib=libstdc++` to `clang++`.
${SUDO} apt-get -y g++ install pkg-config libboost-serialization-dev libboost-filesystem-dev libboost-system-dev libboost-program-options-dev libboost-test-dev wget libyaml-cpp-dev
# install GCC10.3.0
cd
wget https://ftp.gnu.org/gnu/gcc/gcc-10.3.0/gcc-10.3.0.tar.gz
tar -xvzf gcc-10.3.0.tar.gz
cd gcc-10.3.0
./contrib/download_prerequisites
mkdir build
cd build
../configure --prefix=/usr/local/gcc-10.3.0 --enable-languages=c,c++ --disable-multilib
make -j$(nproc)
sudo make install

# install cmake == 3.19.0
cd
wget https://github.com/Kitware/CMake/releases/download/v3.19.0/cmake-3.19.0.tar.gz
tar -xzvf cmake-3.19.0.tar.gz
cd cmake-3.19.0
./bootstrap
make -j$(nproc)
${SUDO} make install
echo "export PATH=/usr/local/cmake-3.19.0/bin:$PATH" >> ~/.bashrc
cd
${SUDO} rm -rf cmake-3.19.0.tar.gz gcc-10.3.0.tar.gz

export CXX=g++
export CUDACXX=/usr/local/cuda/bin/nvcc
export MAKEFLAGS="-j `nproc`"

${SUDO} apt-get -y install python3-dev python3-pip
${SUDO} apt-get -y install libboost-all-dev
# install additional python dependencies via pip
${SUDO} pip3 install  pyplusplus
${SUDO} pip3 install  pygccxml==2.2.1
# install castxml
${SUDO} apt-get -y install castxml
${SUDO} apt-get -y install libboost-python-dev
${SUDO} apt-get -y install libboost-numpy-dev python${PYTHONV}-numpy
${SUDO} pip3 install -vU pygccxml pyplusplus
#install pypy3
${SUDO} add-apt-repository ppa:pypy/ppa
${SUDO} apt-get update
${SUDO} apt-get -y install pypy3

# install KNN_CUDA
pip3 install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

# install cutlass
cd
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass
mkdir build && cd build
cmake -DCMAKE_C_COMPILER=/usr/local/gcc-10.3.0/bin/gcc -DCMAKE_CXX_COMPILER=/usr/local/gcc-10.3.0/bin/g++ .. -DCUTLASS_NVCC_ARCHS=89
export CUTLASS_PATH=/home/user/cutlass/build
