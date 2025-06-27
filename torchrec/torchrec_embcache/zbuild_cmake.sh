#!/bin/bash

set -e
warn() { echo >&2 -e "\033[1;31m[WARN ][Depend  ] $1\033[1;37m" ; }

script_path=$(cd $(dirname $0); pwd)

#---------------------------------------
# build
#---------------------------------------
mkdir -p src/cmake_build
cd src/cmake_build
cmake ../ \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=${script_path}/cmake_build/install

make -j8
make install