# dlrm 300IDUO 推理适配说明

## 迁移说明
1. 源码地址：[DLRM](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/DLRM)
2. 提交的SHA-1 hash值 729963dd47e7c8bd462ad10bfac7a7b0b604e6dd

## 启动容器
1. 镜像地址： https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f
2. 创建启动脚本如下：
```bash
docker run \
-u root \
-it \
--name ${container_name} \
--net=host \
--shm-size="300g" \
--privileged \
-v /etc/localtime:/etc/localtime \
-e ASCEND_VISIBLE_DEVICES=0-3 \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /home:/home \
-v /root/ascend:/root/ascend \
-v /root/.ssh:/root/.ssh \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
${image_name} \
/bin/bash
```

## python、pytorch、CANN版本
1. python 3.11.6
2. pytorch 2.1.0
3. CANN 8.0.0

## python 安装依赖
1. [dllogger](https://github.com/NVIDIA/dllogger) 1.0.0
2. tqdm 4.66.5
3. torch_npu 2.1.0.post10
4. [Apex Patch](https://gitee.com/luweizheng/ascend-apex) 0.1+ascend

## 设置环境变量
`source /usr/local/Ascend/ascend-toolkit/set_env.sh`

## 适配
1. dlrm/cuda_ext/dot_based_interact.py
删除19-22行
2. dlrm/cuda_ext/fused_gather_embedding.py
删除23行，添加如下代码
```python
import torch_npu
from torch_npu.contrib import transfer_to_npu


def fusedEmbeddingNpu(embedding, categorical_inputs, offsets, amp_train):
    offsets = offsets[:-1]
    indices = categorical_inputs + offsets.unsqueeze(0)
    return embedding[indices][:]

if hasattr(torch, "npu"):
    fused_embedding_fwd = fusedEmbeddingNpu
else:
    from dlrm.cuda_ext import fused_embedding
    fused_embedding_fwd = fused_embedding.gather_gpu_fused_fwd
```
原31行替换成`output = fused_embedding_fwd(embedding, indices, offsets, amp_train)`
3. dlrm/cuda_ext/sparse_embedding.py
删除19行，69行替换成如下
```python
if hasattr(torch, "npu"):
    def embedding_gather(weights: torch.Tensor, indices: torch.Tensor):
        embedding = weights.index_select(0, indices.flatten())
        return embedding.reshape(*indices.shape, -1)
else:
    embedding_gather = EmbeddingGatherFunction.apply
```
4. dlrm/model/distributed.py(分布式需要用到，若不使用分布式可忽略)
64-68行替换成如下
```python
        gathered_outputs = [torch.empty(
            batch_sizes_per_gpu[rank], n, vector_dim,
            device=local_bottom_outputs.device, dtype=local_bottom_outputs.dtype) for n in vectors_per_gpu]

        torch.distributed.all_gather(gathered_outputs, local_bottom_outputs)

        bottom_output_buffer = []

        for i in range(ctx.world_size):
            split_tensors = list(gathered_outputs[i].split(batch_sizes_per_gpu, dim=0))
            bottom_output_buffer.append(split_tensors[rank])
```
310p目前不支持all to all算子，但all gather算子目前也存在问题，还在处理中
5. dlrm/nn/factories.py
删除22行的CppMlp
添加import torch
27行修改成`return TorchMlp(input_dim, sizes)`
58-60替换成
```python
        if hasattr(torch, "npu"):
            return DotInteraction(embedding_num, embedding_dim)
        else:
            return CudaDotInteraction(
                DotInteraction(embedding_num, embedding_dim)
            )
```
6. /dlrm/nn/mlps.py
删除18、23-44、117-140行
7. /dlrm/nn/parts.py
删除59行
8. 添加文件/dlrm/scripts/caculate_loss_auc.py(不在推理阶段进行AUC和loss计算，单独脚本计算)
内容详见caculate_loss_auc.py
9. dlrm/scripts/main.py
添加
```python
import re
import torch
import torch_npu
```
添加flag参数`flags.DEFINE_integer("total_devices", 1, "Total number of devices to run the test on")`
validate_flags函数添加如下判断
```python
    if "npu" in FLAGS.base_device:
        if FLAGS.amp:
            print('WARNING: Automatic mixed precision not supported on NPU')
            FLAGS.amp = False
```
原389行即main函数一开始，添加一行`torch.npu.set_compile_mode(jit_compile=False)`
原468-471行修改成
```python
    if hasattr(torch, "npu"):
        if FLAGS.Adam_MLP_optimizer:
            mlp_optimizer = apex_optim.NpuFusedAdam(mlp_params)
        else:
            mlp_optimizer = apex_optim.NpuFusedSGD(mlp_params)
    else:
        if FLAGS.Adam_MLP_optimizer:
            mlp_optimizer = apex_optim.FusedAdam(mlp_params)
        else:
            mlp_optimizer = apex_optim.FusedSGD(mlp_params)
```
原508行修改成
```python
        device_id = re.findall(r'\d+', device)
        if device_id:
            device_id = int(device_id[0])
            total_devices = FLAGS.total_devices
        else:
            device_id = 0
            total_devices = 1
        auc, valid_loss = dist_evaluate(model, data_loader_test, device_id, total_devices)
```
原733行修改成`def dist_evaluate(model, data_loader, device_id=0, total_devices=1):`
原755行修改成`    steps_per_epoch = len(data_loader)//total_devices + int(bool(len(data_loader) % total_devices > device_id))`
原772-773行修改成
```python
        for step in range(len(data_loader)//total_devices + int(bool(len(data_loader) % total_devices > device_id))):
            try:
                for i in range(device_id):
                    if i == device_id:
                        numerical_features, categorical_features, click = next(batch_iter)
                    else:
                        next(batch_iter)
            except StopIteration:
                break
```
原820-835行修改成
```python
        output_click = [y_true, y_score]

        if not os.path.exists('output'):
            os.makedirs('output')
        
        np.save(f'output/output_click_{device_id}.pt', output_click)

        if world_size > 1:
            torch.distributed.barrier()

    model.train()

    return None, None
```
10. 添加文件dlrm/scripts/test_main.py(替代分布式，用多进程开启多个芯片跑数据)
添加如下
```python
import os
import subprocess
import multiprocessing
import re
from absl import app, flags

flags.DEFINE_string("args", None, "Arguments to pass to the script")

def run_script(base_device, total_devices, base_args):
    script_directory = os.path.join(os.path.dirname(__file__))

    main_script_path = os.path.join(script_directory, "main.py")

    cmd = ["python3", main_script_path, f"--device={base_device}", f"--total_devices={total_devices}"]
    cmd.extend(base_args)
    print(f"Running {' '.join(cmd)}")
    subprocess.run(cmd)

def parse_args(args_string):
    base_args = []
    total_devices = 1
    base_device = "npu"
    for arg_v in re.split(r"\s+--", args_string.strip()):
        arg_v = arg_v.strip("--")
        arg, v = re.split(r"\s+|=", arg_v)
        if arg == "base_device":
            base_device = v
        elif arg == "total_devices":
            total_devices = int(v)
        base_args.append(f"--{arg}={v}")
    return base_device, total_devices, base_args

def start_processes(base_device, total_devices, base_args):
    FLAGS = flags.FLAGS

    base_args, total_devices, base_args = parse_args(FLAGS.args)

    processes = []

    for i in range(total_devices):
        sub_base_device = f"{base_device}:{i}"
        p = multiprocessing.Process(target=run_script, args=(sub_base_device, total_devices, base_args))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

if __name__ == "__main__":
    app.run(start_processes)
```
## 启动命令
### 在线推理
单芯片
```bash
cd /workspace/DeepLeaningExamples/PyTorch/Recommendation/DLRM
python3 dlrm/scripts/main.py --mode test --dataset /your/path/to/data --load_checkpoint_path /your/path/to/checkpoint --base_device npu --test_batch_size 65000
```
多芯片
```bash
python3 dlrm/scripts/test_main.py --args "--mode test --dataset /your/path/to/data --load_checkpoint_path /your/path/to/checkpoint --base_device npu --embedding_type joint --test_batch_size 40000 --total_devices=2"
```