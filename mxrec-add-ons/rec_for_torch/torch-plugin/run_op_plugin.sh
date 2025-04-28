#!/bin/bash
# Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.
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

export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH

CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)
cd $CURRENT_DIR

# 导出环境变量
PTA_DIR=op-plugin

if [ ! $ASCEND_HOME_DIR ]; then
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        export ASCEND_HOME_DIR=$HOME/Ascend/ascend-toolkit/latest
    else
        export ASCEND_HOME_DIR=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi
source $ASCEND_HOME_DIR/bin/setenv.bash

# 当前示例使用Python-3.11版本
PYTHON_VERSION=$(python3 -V 2>&1 | awk '{print $2}' | awk -F '.' '{print $1"."$2}')
if [ "$PYTHON_VERSION" != "3.11" ]; then
    echo "Error: Python3 version is not 3.11"
    exit 1
fi
# 当前示例使用Pytorch-2.1.0版本
PYTORCH_VESION=$(pip3 show torch | grep "Version:" | awk '{print $2}' | awk -F '.' '{print $1"."$2"."$3}' | awk -F '+' '{print $1}')

export HI_PYTHON=python${PYTHON_VERSION}
export PYTHONPATH=$ASCEND_HOME_DIR/python/site-packages:$PYTHONPATH
export PATH=$ASCEND_HOME_DIR/python/site-packages/bin:$PATH

function main() {
    # 1. 清除遗留生成文件和日志文件
    rm -rf $HOME/ascend/log/*

    # 2. 下载PTA源码仓，必须要git下载
    cd ${CURRENT_DIR}
    git config --global http.postBuffer 2M
    git clone https://gitee.com/ascend/op-plugin.git -b 6.0.rc3

    # 3. PTA自定义算子注册
    FUNCTION_REGISTE_FIELD="op_plugin_patch/op_plugin_functions.yaml"
    FUNCTION_REGISTE_FILE="${PTA_DIR}/op_plugin/config/op_plugin_functions.yaml"
    line="  - func: index_select_for_rank1_backward(Tensor gradY, Tensor x, Tensor index) -> (Tensor, Tensor)"
    if ! grep -q "\  $line" $FUNCTION_REGISTE_FILE; then
        sed -i "/custom:/r   $FUNCTION_REGISTE_FIELD" $FUNCTION_REGISTE_FILE
    fi

    # 4. 编译PTA插件并安装
    cp -rf op_plugin_patch/*.cpp ${PTA_DIR}/op_plugin/ops/opapi
    (
        cd ${PTA_DIR}
        bash ci/build.sh --python=${PYTHON_VERSION} --pytorch=v${PYTORCH_VESION}-6.0.rc3
    )
}
main
