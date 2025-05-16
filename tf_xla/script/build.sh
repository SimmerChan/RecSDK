#!/bin/bash
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

set -e

# check Inference Graph output path
if [ -z "$1" ]; then
    echo "Please input Inference Graph output path"
    exit 1
fi

ig_path=$1
if [ ! -d "$ig_path" ]; then
    echo "Inference Graph output path not exist: $ig_path"
    exit 1
fi

ig_bin="$ig_path"/bin
ig_lib="$ig_path"/lib

clang_path=$(which clang)
clangxx_path=$(which clang++)

export CC=$clang_path
export CXX=$clangxx_path

cur_dir=$(dirname $(readlink -f "$0"))
root_dir=$cur_dir/../

cd $root_dir/tf_community
yes "" | ./configure

export TF_MAJOR_VERSION=2
export TF_MINOR_VERSION=18

cd $root_dir/npu_xla

cp -f $ig_lib/libruntime_pipeline.so ./tf_bridge/kernels/inference_graph/
cp -f $ig_lib/libsitere_grt.so ./tf_bridge/kernels/inference_graph/

# 获取tensorflow安装路径
TF_PATH=( $(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include()); print(tf.sysconfig.get_lib())') )
export TF_HEADER_DIR="${TF_PATH[0]}" 
export TF_SHARED_LIBRARY_DIR="${TF_PATH[1]}"
export TF_SHARED_LIBRARY_NAME="libtensorflow_framework.so.2"

bazel build //tf_mlir:tf-mlir-opt
bazel build //:libnpu_xla.so
bazel build //:libnpu_device.so

build_out_dir=$root_dir/npu_xla/bazel-bin
package_dir=$root_dir/npu_xla/python/tf_npu_xla

cp -f $build_out_dir/tf_mlir/tf-mlir-opt $package_dir/
cp -f $build_out_dir/libnpu_xla.so $package_dir/
cp -f $build_out_dir/libnpu_device.so $package_dir/
cp -f $ig_lib/*.so $package_dir/
cp -f $ig_bin/inference-mlir-opt $package_dir/
cp -f $ig_bin/inference-mlir-compiler $package_dir/

cd $root_dir/npu_xla && python3 setup.py bdist_wheel

rm -f $package_dir/*.so
rm -f $package_dir/inference-mlir-opt
rm -f $package_dir/inference-mlir-compiler
rm -f $package_dir/tf-mlir-opt

rm -rf $root_dir/npu_xla/build

mkdir -p $root_dir/build
cp -f $root_dir/npu_xla/dist/*.whl $root_dir/build/