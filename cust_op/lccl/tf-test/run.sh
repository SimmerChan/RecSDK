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
local_rank_size=$1
host=localhost
py=$2
rm -rf /root/atc_data/
rm -rf /root/ascend/*
rm -rf kernel_meta_*

mpi_path=/usr/local/openmpi/bin/
interface="enp61s0f0"
ulimit -c 0
export ASCEND_GLOBAL_LOG_LEVEL=0
export TF_CPP_MIN_LOG_LEVEL=3
export ASCEND_INSTALL_PATH=/usr/local/Ascend/ascned-toolkit/latest/
export ASCEND_HOME_PATH=${ASCEND_INSTALL_PATH}
export PATH=${mpi_path}/bin:$PATH
export PYTHONPATH=/usr/local/python3.7.5/lib/python3.7/site-packages/mx_rec/libasc:/usr/local/python3.7.5/lib/python3.7/site-packages/:${PYTHONPATH}:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages
export LD_PRELOAD=/lib64/libgomp.so.1
export TOOLCHAIN_HOME=${ASCEND_HOME_PATH}/toolkit
export HCCL_BUFFSIZE=1

export BETTER_EXCEPTIONS=1
mpi_args='-x BIND_INFO="0:48 48:48 96:48" -x MXREC_LOG_LEVEL=DEBUG -bind-to none'

rm *txt > /dev/null
rm -rf /root/ascend/log/*


for i in $(ipcs -m | tail -n +4 | awk {'print $2'}); do
    ipcrm -m $i
done

num_process=${local_rank_size}
host_string=${host//_/:${local_rank_size},node}:${local_rank_size}
echo run in $host_string

interface="lo"
env
horovodrun --network-interface ${interface} -np ${num_process} --mpi-args "${mpi_rags}" --mpi -H localhost:${local_rank_size} \
    python3.7 ${py} --local_rank_size ${local_rank_size} --hccl_json hccl_json_${local_rank_size}p.json | tee temp.log
