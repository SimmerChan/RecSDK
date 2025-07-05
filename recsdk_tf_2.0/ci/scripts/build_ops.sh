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

##################################################################
#   build.sh 编译自定义算子
# 编译环境：Python3.9.6 GCC 10.2.1 CMake 3.20.6
##################################################################

ARCH="$(uname -m)"
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
MxRec_DIR=$(dirname "$(dirname "${SCRIPT_DIR}")")

ops_path="${MxRec_DIR}"/cust_ops

cd "${MxRec_DIR}"

function make_output_dir() {
    opp_output_path="${MxRec_DIR}"/mxrec_ops
    mkdir -p "${opp_output_path}"
}

function compile_ops() {
    for dir in "$ops_path"/*; do
        cd "$ops_path"
        dir_name=$(basename "$dir")
        if [ -d "$dir" ] && [ "$dir_name" != "tf_cpu_op" ]; then
            echo "Entering directory: $dir_name"
            cd "$dir_name"
            bash ./run.sh ai_core-Ascend910_95
            new_op_name=mxrec_opp_"${dir_name}_91095".run
            cd "$dir_name"
            cp ./build_out/custom_opp*.run  "${new_op_name}"
            mv "${new_op_name}" "${opp_output_path}"
        fi
    done
}

# start to build mxrec-add-ons
make_output_dir
compile_ops
