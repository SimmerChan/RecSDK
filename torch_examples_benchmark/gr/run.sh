export USE_NPU_HSTU=1
export ENABLE_RAB=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
ASCEND_RT_VISIBLE_DEVICES=0
python3 main.py --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin --master_port=12345 | tee temp.log
