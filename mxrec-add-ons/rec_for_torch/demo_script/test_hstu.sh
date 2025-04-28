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

help_flag=0
test_type=0
npu_type=0

script_dir=$(dirname "$(readlink -f "$0" || realpath "$0")")

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            help_flag=1
            shift
            ;;
        -t|--test_type)
            test_type="$2"
            shift 2
            ;;
        -n|--npu_type)
            npu_type="$2"
            shift 2
            ;;
    esac
done

if [[ $help_flag -eq 1 ]]; then
    echo "usage: $0 [test_type] [npu_type]"
    echo "options:"
    echo "-h, --help:       Displaying help information."
    echo "-t, --test_type:  Test framework type, only support 0: TensorFlow; 1: pytorch. The default value is 0."
    echo "-n, --npu_type:   Test npu type, only support 0: 910B; 1: 310P. The default value is 0."
    echo "Example:          bash test_hstu.sh -t 0 -n 0"
    exit 0
fi

if [[ $test_type -ne 0 && $test_type -ne 1 ]]; then
    echo "Test framework type error, please check test_type. use -h to view Help."
    exit 1
fi

if [[ $npu_type -ne 0 && $npu_type -ne 1 ]]; then
    echo "Test npu type error, please check npu_type. use -h to view Help."
    exit 1
fi

if [[ $test_type -eq 0 && $npu_type -eq 0 ]]; then
    if ! npu-smi info | grep -q "910B"; then
        echo "Please check whether the npu_type is the same as that in the current environment."
        exit 1
    fi

    cd $script_dir/../mxrec_ops
    ./mxrec_opp_hstu_dense_forward.run
    ./mxrec_opp_hstu_dense_backward.run

    cd $script_dir/../tf_ops
    bash build_ops.sh

    cd $script_dir/tf_demo/hstu_dense
    pytest -s hstu_dense_forward_demo.py
    pytest -s hstu_dense_backward_demo.py
elif [[ $test_type -eq 1 && $npu_type -eq 0 ]]; then
    if ! npu-smi info | grep -q "910B"; then
        echo "Please check whether the npu_type is the same as that in the current environment."
        exit 1
    fi

    cd $script_dir/../mxrec_ops
    ./mxrec_opp_hstu_dense_forward.run
    ./mxrec_opp_hstu_dense_backward.run

    cd $script_dir/../torch_library/hstu
    bash build_ops.sh

    cd $script_dir/torch_demo/hstu_dense
    pytest -s hstu_dense_forward_demo.py
    pytest -s hstu_dense_backward_demo.py

    pytest -s hstu_dense_autograd_demo.py
elif [[ $test_type -eq 1 && $npu_type -eq 1 ]]; then
    if ! npu-smi info | grep -q "310P"; then
        echo "Please check whether the npu_type is the same as that in the current environment."
        exit 1
    fi

    cd $script_dir/../mxrec_ops
    ./mxrec_opp_hstu_dense_forward_310p.run

    cd $script_dir/../torch_library/hstu
    bash build_ops.sh

    cd $script_dir/torch_demo/hstu_dense
    pytest -s hstu_dense_forward_demo.py
else
    echo "This combination test is not supported currently."
fi