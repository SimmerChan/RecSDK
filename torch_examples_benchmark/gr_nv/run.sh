#!/bin/bash
# --------------------------------------------
# lib relate
#---------------------------------------------
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_PRELOAD=/usr/lib64/libgomp.so.1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1


RECSYS_DIR=$(realpath ../)
HSTU_DIR=$RECSYS_DIR/hstu
# 根据实际情况设置python引用路径
MEGATRON_DIR=$RECSYS_DIR/megatron-lm/
MINDSPEED_DIR=$RECSYS_DIR/MindSpeed/
export PYTHONPATH=${PYTHONPATH}:${HSTU_DIR}:${MEGATRON_DIR}:${MINDSPEED_DIR}

#根据实际情况设置算子适配so文件
export LIB_FBGEMM_NPU_API_SO_PATH="/path/to/libfbgemm_npu_api.so"
#---------------------------------------------
# speedup
#---------------------------------------------
export TASK_QUEUE_ENABLE=2

# cpu-binding
NPU_NUM=$(npu-smi info|grep 910B|wc -l)
CPU_CORES=$(nproc --all)
CORES_PER_NPU=$((CPU_CORES / NPU_NUM))
CPU_AFFINITY_CONF_TMP=1
if [ "$NPU_NUM" -gt 0 ]; then
  for (( i=0; i<NPU_NUM; i++)); do
    start_core=$(( i * CORES_PER_NPU))
    end_core=$((start_core + CORES_PER_NPU -1))
    CPU_AFFINITY_CONF_TMP+=",npu${i}:${start_core}-${end_core}"
  done
fi
export CPU_AFFINITY_CONF=$CPU_AFFINITY_CONF_TMP
echo "CPU_AFFINITY_CONF="$CPU_AFFINITY_CONF

#---------------------------------------------
# embcache related
#---------------------------------------------
export EMBCACHE_SIZE_ON_HBM=$((128*1024*1024))
export WITH_EMBCACHE=0

export ENABLE_FAST_HASHMAP=1
export FAST_HASHMAP_RESERVE_BUCKET_NUM=$((2*1024*1024))

# FAST_HASHMAP will use EMB_MEMORY_POOL
export EMB_MEMORY_POOL_THREAD_NUM=1 #x86下设置1，arm下设置2
export EMB_MEMORY_POOL_SIZE=102400

# thread number for parallel_for
export OMP_NUM_THREADS=1 #x86下设置1，arm下设置2


export SPARSE_OPTIM_NUM=1
export ENABLE_GLOBAL_UNIQUE=0
# ENABLE_FAST_HASHMAP=false时,默认适用unordered_map
export ENABLE_FAST_HASHMAP=1
export ENABLE_PARALLEL_GLOBAL_UNIQUE=1 #x86下设置1，arm下设置2

# 0:INFO; 1:WARNINH; 2:ERROR; 3:FATAL; (on profiling, we set 0; otherwise we set 1)
export GLOG_MIN_LOG_LEVEL=1


export USE_EC=1
export DO_EC_LOCAL_UNIQUE=1
# 同时做local unique的batch数
export LOCAL_UNIQUE_PARALLEL_BATCH_NUM=2 #大batchsize适当调高

#---------------------------------------------
# train job related
#---------------------------------------------
py_file=pretrain_gr_ranking.py
config_file=movielen_ranking.gin
TOKENIZER_MODEL=$RECSYS_DIR/llama2-tokenizer.model

# 根据实际情况修改
export WORLD_SIZE=4
export ASCEND_RT_VISIBLE_DEVICE=4,5,6,7

MICRO_BATCH_SIIZE=8
GLOBAL_BATCH_SIZE=$((MICRO_BATCH_SIIZE * WORLD_SIZE))

GPT_ARGS="
    --mico-batch-size ${MICRO_BATCH_SIIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --hidden-size 128 \
    --num-attention-heads 4 \
    --seq-length 8000 \
    --max-position-embeddings 8000 \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
"

torchrun \
    --nproc_per_node ${WORLD_SIZE} \
    --master_addr localhost \
    --master_port 6000 \
    ${py_file} \
    --gin-config-file ${config_file} \
    ${GPT_ARGS} \
    2>&1 |tee temp_$(date '+%Y%m%d_%H%M%S').log