#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

CUR_PATH=$(cd "$(dirname "$0")" || { warn "Failed to check path/to/run_python_ut.sh" ; exit ; } ; pwd)
TOP_PATH="${CUR_PATH}"/../../../

local_rank_size=1
num_server=1
num_process=$((num_server * local_rank_size))

# Please configure according to the actual situation.
toml_path="ut_test.toml"
use_ranktable=$(grep -oP '^use_ranktable\s*=\s*\Ktrue' ${toml_path})
if [ "$use_ranktable" == "true" ]; then
  ranktable_path=${TOP_PATH}/examples/ranktable_samples/ranktable_${num_process}p.json
  topo_path=${TOP_PATH}/examples/ranktable_samples/topo_${num_process}p.json
  export RANK_TABLE_FILE=${ranktable_path}
  export HCCL_TOPO_FILE_PATH=${topo_path}
  echo "RANK_TABLE_FILE=$RANK_TABLE_FILE"
fi

export PYTHONPATH="${TOP_PATH}":$PYTHONPATH

if [ -d "result" ]; then
    rm -rf result
fi

mkdir result

interface="lo"
mpi_args='-x BIND_INFO="0:12 12:48 60:48" -bind-to none -x NCCL_SOCKET_IFNAME=docker0 -mca btl_tcp_if_exclude docker0'
# In `pytest`, the `--loadfile` option specifies that each test file should be executed in its own process,
# while the `-n` option indicates the total number of processes to be executed.
horovodrun --network-interface ${interface} -np ${num_process} --mpi-args "${mpi_args}" \
  --mpi -H localhost:${local_rank_size} pytest --cov="${CUR_PATH}"/../ \
  --cov-config ./.coveragerc --cov-report=html --cov-report=xml --junit-xml=./final.xml \
  --html=./final.html --self-contained-html --durations=5 -vv --cov-branch
coverage xml -i --omit="algo/*,ci/*,cust_ops/*,examples/*,third_party/*,tools/*"
cp coverage.xml final.xml final.html ./result
cp -r htmlcov ./result
rm -rf coverage.xml final.xml final.html htmlcov
