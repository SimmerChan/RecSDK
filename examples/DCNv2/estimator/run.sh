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

cur_path=$(dirname "$(readlink -f "$0")")

so_path=$1
mx_rec_package_path=$2
hccl_cfg_json=$3
dlrm_criteo_data_path=$4
ip=$5  # 仅no ranktabke时使用，传入当前节点ip

interface="lo"
local_rank_size=8
num_server=1
num_process=$((num_server * local_rank_size))
export TRAIN_RANK_SIZE=${num_process}

################# 参数配置 ######################
export USE_DYNAMIC=0            # 0：静态shape；1：动态shape
export CACHE_MODE="HBM"         # HBM；DDR；SSD  注意：DDR、SSD需要USE_DYNAMIC=1，暂未适配静态send count
export USE_MODIFY_GRAPH=0       # 0：feature spec模式；1：自动改图模式
export USE_HOT=1                # 0：关闭hot emb；1: 开启hot emb
export USE_DYNAMIC_EXPANSION=1  # 0：关闭动态扩容；1: 开启动态扩容
export USE_ONE_SHOT=0           # 0：不使用oneshot；1：使用oneshot
################################################

echo "CACHE_MODE:${CACHE_MODE}"
OUTPUT_PATH=$cur_path/model_dir
export DLRM_CRITEO_DATA_PATH=${dlrm_criteo_data_path}
export HCCL_CONNECT_TIMEOUT=1200

export CUSTOMIZED_OPS_LIB_PATH=${so_path}/libcust_ops.so
export HOST_PIPELINE_OPS_LIB_PATH=${so_path}/libasc_ops.so
export PYTHONPATH=${mx_rec_package_path}:${so_path}:$PYTHONPATH
export LD_PRELOAD=/usr/lib64/libgomp.so.1
#export LD_PRELOAD=/usr/local/python3.7.5/lib/python3.7/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0:/usr/local/gcc7.3.0/lib64/libgomp.so.1
export LD_LIBRARY_PATH=${so_path}:/usr/local/lib:$LD_LIBRARY_PATH
export ASCEND_DEVICE_ID=0
export RANK_ID_START=0
export JOB_ID=10086
export MXREC_LOG_LEVEL="INFO"
export TF_CPP_MIN_LOG_LEVEL=3
export ENABLE_FORCE_V2_CONTROL=1
export GE_USE_STATIC_MEMORY=2     # 配置后，HBM占用是最大graph的大小
#export ASCEND_GLOBAL_LOG_LEVEL=1
#export ASCEND_SLOG_PRINT_TO_STDOUT=1 #可以打屏cann日志
export PROFILING_OPTIONS='{"output":"/home/yz/profiling",
                           "training_trace":"on",
                           "task_trace":"on",
                           "aicpu":"on",
                           "fp_point":"",
                           "bp_point":"",
                           "aic_metrics":"PipeUtilization"}'

RANK_ID_START=0

export MXREC_MODE="ASC"
echo "MXREC_MODE is $MXREC_MODE"

export USE_MPI=1
echo "USE_MPI is $USE_MPI"

py=main.py
echo "py is $py"

# 区分ranktable和no ranktable
if [ -n "$ip" ]; then
    # no ranktable
    echo "Current is no ranktable solution."
    export CM_CHIEF_IP=$ip  # 主节点ip
    export CM_CHIEF_PORT=6000  # 主节点监听端口
    export CM_CHIEF_DEVICE=0  # 主节点device id
    export CM_WORKER_IP=$ip  # 当前节点ip
    export CM_WORKER_SIZE=$num_process  # 参与集群训练的device数量
    echo "CM_CHIEF_IP=$CM_CHIEF_IP"
    echo "CM_CHIEF_PORT=$CM_CHIEF_PORT"
    echo "CM_CHIEF_DEVICE=$CM_CHIEF_DEVICE"
    echo "CM_WORKER_IP=$CM_WORKER_IP"
    echo "CM_WORKER_SIZE=$CM_WORKER_SIZE"
    echo "ASCEND_VISIBLE_DEVICES=$ASCEND_VISIBLE_DEVICES"
else
    # ranktable
    echo "Current is ranktable solution, hccl json file:${hccl_cfg_json}"
    export RANK_SIZE=$num_process
    echo "RANK_SIZE=${RANK_SIZE}, please make sure hccl configuration json file match this parameter"
    export RANK_TABLE_FILE=${hccl_cfg_json}
fi

if [ $USE_MPI -eq 0 ]; then
  echo "use for loop to start tasks"
  for((RANK_ID=$RANK_ID_START;RANK_ID<$((RANK_SIZE+RANK_ID_START));RANK_ID++));
  do
      #设置环境变量，不需要修改
      echo "Device ID: $RANK_ID"
      export RANK_ID=$RANK_ID
      export ASCEND_DEVICE_ID=$RANK_ID
      ASCEND_DEVICE_ID=$RANK_ID
    if [   -d $cur_path/output/${ASCEND_DEVICE_ID} ];then
       rm -rf $cur_path/output/${ASCEND_DEVICE_ID}
       mkdir -p $cur_path/output/${ASCEND_DEVICE_ID}
    else
       mkdir -p $cur_path/output/${ASCEND_DEVICE_ID}
    fi
      nohup python3 ${py} > $cur_path/output/$ASCEND_DEVICE_ID/test_$ASCEND_DEVICE_ID.log 2>&1 &
  done
else
  echo "use horovod to start tasks"
  # GLOG日志级别
  # GLOG_stderrthreshold -2:TRACE -1:DEBUG 0:INFO 1:WARN 2.ERROR, 默认为INFO
  mpi_args='-x GLOG_stderrthreshold=2 -x GLOG_logtostderr=true -x GLOG_v=0 -bind-to none'

  horovodrun --network-interface ${interface} -np ${num_process} --mpi-args "${mpi_args}" --mpi -H localhost:${local_rank_size} \
    python3.7 ${py} --run_mode=train_and_evaluate \
                    --model_ckpt_dir=${OUTPUT_PATH} \
                    --batch_size=8192 \
                    --epoch_num=1 \
                    --line_per_sample=1024 \
                    --train_steps=100000 \
                    --save_checkpoints_steps=100000 \
                    --eval_steps=1360 \
                    --modify_graph=${USE_MODIFY_GRAPH} \
                    --use_one_shot=${USE_ONE_SHOT} \
                    --iterations_per_loop=10 \
                    --data_path=${DLRM_CRITEO_DATA_PATH} 2>&1 | tee temp.log
fi
