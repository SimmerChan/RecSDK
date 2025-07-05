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


msopgen_path=$(find /usr/local/Ascend/ -name msopgen | grep bin)
parent_dir=$(dirname "$msopgen_path")
export PATH=$parent_dir:$PATH

rm -rf ./custom_op
msopgen gen -i all2allvc_uss.json -f tf -c ai_core-ascend910_95 -lan cpp -out ./custom_op -m 0 -op All2allvcUssFm
msopgen gen -i all2allvc_uss.json -f tf -c ai_core-ascend910_95 -lan cpp -out ./custom_op -m 1 -op All2allvcUss
msopgen gen -i all2allvc_uss.json -f tf -c ai_core-ascend910_95 -lan cpp -out ./custom_op -m 1 -op All2allvcUssCf

cp -rf op_kernel custom_op/
cp -rf op_host custom_op/

cd custom_op


if [ ! -f "CMakePresets.json" ]; then
  echo "当前目录下不存在cmake.json文件"
  exit 1
fi


sed -i 's/--nomd5/--nomd5 --nocrc/g' ./cmake/makeself.cmake

if [ -d "/usr/local/Ascend/ascend-toolkit/latest/" ]; then
    sed -i 's:"/usr/local/Ascend/latest":"/usr/local/Ascend/ascend-toolkit/latest/":g' CMakePresets.json
else
    sed -i 's:"/usr/local/Ascend/latest":"/usr/local/Ascend/latest":g' CMakePresets.json
fi

cd cmake


if [ ! -f "config.cmake" ]; then
  echo "当前目录下不存在cmake.json文件"
  exit 1
fi

cd ..
sed -i 's:"customize":"lccl":g' CMakePresets.json

bash build.sh


bash ./build_out/custom_opp*.run

cd ..

rm -rf ./custom_op
