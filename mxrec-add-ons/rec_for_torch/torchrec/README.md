# Torchrec模型迁移样例
## 迁移说明
本样例以DLRM模型为例,适配torchrec框架并在NPU上进行训练。
模型参考的开源链接为:https://github.com/facebookresearch/dlrm/tree/main/torchrec_dlrm/
克隆源码并固定版本为:Commits on Jun 7 , 2024，提交的SHA-1 hash值（提交ID）：b631a99 

## 启动容器
镜像地址：https://www.hiascend.com/developer/ascendhub/detail/rec_sdk-torch 
创建启动脚本run_docker.sh,参考如下：
```shell
docker run \
-u root \
-it \
--name ${container_name} \
--net=host \
--shm-size="300g" \
--privileged \
-v /etc/localtime:/etc/localtime \
-e ASCEND_VISIBLE_DEVICES=0-7 \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /home:/home \
-v /root/ascend:/root/ascend \
-v /root/.ssh:/root/.ssh \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
${image_name} \
/bin/bash
```
执行如下命令新建容器：
```shell
bash run_docker.sh 容器名 {镜像名称}:{版本名称}
```

进入容器后，执行：
```shell
sudo su
```
切换为root用户


## 下载配套软件
```shell
# torch版本
pip3 install torch==2.1.0+cpu  --index-url https://download.pytorch.org/whl/cpu # x86
pip3 install torch==2.1.0 # arm
# fbgemm_gpu版本
pip3 install fbgemm_gpu==0.5.0+cpu -i https://download.pytorch.org/whl/cpu
```
下载mindxsdk-mxrec-add-ons软件包,在mindxsdk-mxrec-add-ons/torch_plugin/ 目录下使用 pip3 安装 .whl 文件
```shell
pip3 install /mindxsdk-mxrec-add-ons/torch_plugin/torch_npu-2.1.0.*.whl
```
下载torchrec适配npu的软件包,安装.whl文件及其依赖
```shell
pip3 install torchrec-0.5.0-npu-*.whl
pip3 install -r requirements.txt
```
## 安装算子
进入mindxsdk-mxrec-add-ons/mxrec_ops/目录，并执行：
```shell
cd mindxsdk-mxrec-add-ons/mxrec_ops/
bash mxrec_opp_backward_codegen_adagrad_unweighted_exact.run
bash mxrec_opp_permute2d_sparse_data.run
bash mxrec_opp_split_embedding_codegen_forward_unweighted.run
bash mxrec_opp_bounds_check_indices.run
bash mxrec_opp_asynchronous_complete_cumsum.run
```

## 设置环境变量

```shell
export LD_PRELOAD=/usr/lib64/libgomp.so.1
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
Q&A

1."libgomp.so.1: cannot allocate memory in static TLS block" 这个错误通常发生在动态加载库时，系统无法为线程本地存储（Thread Local Storage, TLS）分配足够的内存，这通常与 OpenMP 库的加载有关。在某些 Arm 环境下，特别是在运行 PyTorch 或 NPU 相关任务时，系统可能无法正确加载 libgomp.so.1。使用 LD_PRELOAD 可以强制系统在程序启动时预先加载这个库，从而避免动态加载时出现的 TLS 分配问题。通过 LD_PRELOAD 预加载 libgomp.so.1，系统可以在主程序启动之前就分配必要的 TLS 内存。这样可以确保 OpenMP 库正确加载，避免在后续动态加载时出现内存分配失败的问题。

2.CANN软件提供进程级环境变量设置脚本，供用户在进程中引用，以自动完成环境变量设置。用户进程结束后自动失效。可在程序启动的Shell脚本中使用以上命令设置CANN的相关环境变量，也可通过命令行执行以上命令（以root用户默认安装路径“/usr/local/Ascend”为例）。

## 修改fbgemm-gpu代码
进入/usr/local/python3.11.0/lib/python3.11/site-packages/fbgemm_gpu/
### split_table_batched_embeddings_ops_training.py:57
将
```python
    class ComputeDevice(enum.IntEnum):
        CPU = 0
        CUDA = 1
```
改为
```python
    class ComputeDevice(enum.IntEnum):
        CPU = 0
        CUDA = 1
        NPU = 2
```


## dlrm 模型适配
### dlrm-main/torchrec_dlrm/dlrm_main.py:523
将
```python
torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
```
改为
```python
import torch_npu
torch_npu.npu.matmul.allow_hf32 = args.allow_tf32
```

### dlrm-main/torchrec_dlrm/dlrm_main.py:545
将
```python
    if torch.cuda.is_available():
        device: torch.device = torch.device(f"cuda:{rank}")
        backend = "nccl"
        torch.cuda.set_device(device)
    else:
        device: torch.device = torch.device("cpu")
        backend = "gloo"
```
改为
```python
    if torch.cuda.is_available():
        device: torch.device = torch.device(f"cuda:{rank}")
        backend = "nccl"
        torch.cuda.set_device(device)
    elif torch_npu.npu.is_available():
        device: torch.device = torch.device(f"npu:{rank}")
        backend = "hccl"
        torch_npu.npu.set_device(device)
    else:
        device: torch.device = torch.device("cpu")
        backend = "gloo"
```

### dlrm-main/torchrec_dlrm/dlrm_main.py:337
将
```python
    auroc = metrics.AUROC(compute_on_step=False, num_classes=2).to(device)
```
改为
```python
    auroc = metrics.AUROC("binary").to(device)
```

### dlrm-main/torchrec_dlrm/multi_hot.py:110
将
```python
        multi_hot_tables_l = [
            torch.from_numpy(multi_hot_table).int()
            for multi_hot_table in multi_hot_tables_l
        ]
```
改为
```python
        multi_hot_tables_l = [
            torch.from_numpy(multi_hot_table).long()
            for multi_hot_table in multi_hot_tables_l
        ]
```

## 运行命令
根据torchrec dlrm官方文档 https://github.com/facebookresearch/dlrm/tree/main/torchrec_dlrm/  配置数据集，运行
```shell
export PREPROCESSED_DATASET=$insert_your_path_here
export TOTAL_TRAINING_SAMPLES=4195197692 ;
export GLOBAL_BATCH_SIZE=16384;
export WORLD_SIZE=8;
torchx run -s local_cwd dist.ddp -j 1x8 --script dlrm_main.py -- \
    --embedding_dim 128 \
    --dense_arch_layer_sizes 512,256,128 \
    --over_arch_layer_sizes 1024,1024,512,256,1 \
    --in_memory_binary_criteo_path $PREPROCESSED_DATASET \
    --num_embeddings_per_feature 40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36 \
    --validation_freq_within_epoch $((TOTAL_TRAINING_SAMPLES / (GLOBAL_BATCH_SIZE * 20))) \
    --epochs 1 \
    --pin_memory \
    --mmap_mode \
    --batch_size $((GLOBAL_BATCH_SIZE / WORLD_SIZE)) \
    --interaction_type=dcn \
    --dcn_num_layers=3 \
    --dcn_low_rank_dim=512 \
    --adagrad \
    --learning_rate 0.005 \
    --multi_hot_distribution_type uniform \
    --multi_hot_sizes=3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1
```