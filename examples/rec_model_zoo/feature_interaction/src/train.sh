#!/bin/bash

source /usr/local/Ascend/driver/bin/setenv.bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/tfplugin/set_env.sh
export JOB_ID=10087
export ASCEND_DEVICE_ID=0
export RANK_ID=0
export DEVICE_ID=0

export PREPROCESSED_DATASET=/home/ma-user/work/ydz/EffBench/feature_interaction/data/criteo/

models=("LR" "FM" "WideDeep" "IPNN" "OPNN" "PNN" "DeepFM" "AutoInt" "AutoInt_plus" "FiBiNet" "DCNv2")


for model in ${models[@]}; do
python -u ./models/${model}.py --data_dir=$PREPROCESSED_DATASET --model_dir=../checkpoint/criteo/${model}/ \
        --task_type=train --field_size=39 --feature_size=2100000 --train_size=33003326 --embedding_size=10
done

python -u ./models/FFM.py --data_dir=$PREPROCESSED_DATASET --model_dir=../checkpoint/criteo/FFM/ \
        --task_type=train --field_size=39 --feature_size=2100000 --train_size=33003326 --embedding_size=2

python -u ./models/AFN.py --data_dir=$PREPROCESSED_DATASET --model_dir=../checkpoint/criteo/AFN/ \
        --task_type=train --field_size=39 --feature_size=2100000 --train_size=33003326 --embedding_size=10 --hidden_size=1500

python -u ./models/AFN_plus.py --data_dir=$PREPROCESSED_DATASET --model_dir=../checkpoint/criteo/AFN_plus/ \
        --task_type=train --field_size=39 --feature_size=2100000 --train_size=33003326 --embedding_size=10 --hidden_size=1500
    
