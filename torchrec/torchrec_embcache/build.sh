#!/usr/bin/env bash
# Copyright (c) huawei Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e

SCRIPT_PATH=$(cd $(dirname $0); pwd)

function check_ret_fn()
{
    if [ $? -ne 0 ]; then
        echo "[FAIL] $@ failed" 1>&2
        exit 1
    else
        echo "[SUCCESS] $@ successful" 
    fi 
}

function build_with_cmake_func()
{
    mkdir -p ${SCRIPT_PATH}/src/cmake_build
    cd ${SCRIPT_PATH}/src/cmake_build
    cmake ../ \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${SCRIPT_PATH}/cmake_build/install

    make -j8
    make install

    check_ret_fn "build torchrec_embcache"
}

function build_whl_pkg_with_setup_func()
{
    rm -rf build
    rm -rf dist
    rm -rf *.egg-info

    rm -f src/torchrec_embcache/*.so*
    cp cmake_build/install/embcache_pybind.so src/torchrec_embcache/

    python3 setup.py bdist_wheel
    check_ret_fn "python3 setup.py bdist_wheel"
}


build_with_cmake_func
build_whl_pkg_with_setup_func
