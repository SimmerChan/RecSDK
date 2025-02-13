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
```python
from absl import app, flags

import torch
import torch_npu

import dlrm.scripts.utils as utils

FLAGS = flags.FLAGS

flags.DEFINE_string("path", None, "Path to the model")


loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")


def caculate(argv):
    path = FLAGS.path
    output_click_lst = torch.load(path)
    for i, output_click in enumerate(output_click_lst):
        y_true, y_score = output_click
        y_score = torch.sigmoid(y_score).float()
        auc = utils.roc_auc_score(y_true, y_score)
        loss = loss_fn(y_score, y_true).item()
        print(f"step {i}, loss: {loss}, auc: {auc}")

if __name__ == "__main__":
    app.run(caculate)
```
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

11. dlrm/preproc/run_spark_cpu.sh
将64行改成 export MASTER=local[1]
这一步是在CPU使用spark处理数据时可能需要的步骤

12. dlrm/data/factories.py
若启动在线推理且不准备train数据，则需要将105-112注释，且123行改成`return None dataset_test`

13. dlrm/scripts/generate_data_for_om.py
创建新文件，添加以下内容
```
import yaml
import json
import os
import numpy as np
from absl import app, flags, logging
from apex import optimizers as apex_optim

from ais_bench.infer.interface import InferSession, MemorySummary
from ais_bench.infer.summary import summary

from dlrm.data.feature_spec import FeatureSpec
from dlrm.utils import distributed as dist
from dlrm.utils.distributed import get_gpu_batch_sizes, get_device_mapping, is_main_process, is_distributed
from dlrm.data.data_loader import get_data_loaders
from dlrm.data.utils import prefetcher, get_embedding_sizes

import re
import torch
import torch_npu

FLAGS = flags.FLAGS

# Basic run settings
flags.DEFINE_enum("mode", default='train', enum_values=['train', 'test', 'inference_benchmark'],
                  help="Select task to be performed")
flags.DEFINE_integer("seed", 12345, "Random seed")

# Training flags
flags.DEFINE_integer("batch_size", 65536, "Batch size used for training")
flags.DEFINE_integer("test_batch_size", 65536, "Batch size used for testing/validation")
flags.DEFINE_float("lr", 24, "Base learning rate")
flags.DEFINE_integer("epochs", 1, "Number of epochs to train for")
flags.DEFINE_integer("max_steps", None, "Stop training after doing this many optimization steps")

# Learning rate schedule flags
flags.DEFINE_integer("warmup_factor", 0, "Learning rate warmup factor. Must be a non-negative integer")
flags.DEFINE_integer("warmup_steps", 8000, "Number of warmup optimization steps")
flags.DEFINE_integer("decay_steps", 24000,
                     "Polynomial learning rate decay steps. If equal to 0 will not do any decaying")
flags.DEFINE_integer("decay_start_step", 48000,
                     "Optimization step after which to start decaying the learning rate, "
                     "if None will start decaying right after the warmup phase is completed")
flags.DEFINE_integer("decay_power", 2, "Polynomial learning rate decay power")
flags.DEFINE_float("decay_end_lr", 0, "LR after the decay ends")

# Model configuration
flags.DEFINE_enum("embedding_type", "custom_cuda",
                  ["joint", "custom_cuda", "multi_table", "joint_sparse", "joint_fused"],
                  help="The type of the embedding operation to use")
flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of embedding space for categorical features")
flags.DEFINE_list("top_mlp_sizes", [1024, 1024, 512, 256, 1], "Linear layer sizes for the top MLP")
flags.DEFINE_list("bottom_mlp_sizes", [512, 256, 128], "Linear layer sizes for the bottom MLP")
flags.DEFINE_enum("interaction_op", default="cuda_dot", enum_values=["cuda_dot", "dot", "cat"],
                  help="Type of interaction operation to perform.")

# Data configuration
flags.DEFINE_string("dataset", None, "Path to dataset directory")
flags.DEFINE_string("feature_spec", default="feature_spec.yaml",
                    help="Name of the feature spec file in the dataset directory")
flags.DEFINE_enum("dataset_type", default="parametric", enum_values=['synthetic_gpu', 'parametric'],
                  help='The type of the dataset to use')
flags.DEFINE_boolean("shuffle_batch_order", False, "Read batch in train dataset by random order", short_name="shuffle")

flags.DEFINE_integer("max_table_size", None,
                     "Maximum number of rows per embedding table, "
                     "by default equal to the number of unique values for each categorical variable")
flags.DEFINE_boolean("hash_indices", False,
                     "If True the model will compute `index := index % table size` "
                     "to ensure that the indices match table sizes")

# Synthetic data configuration
flags.DEFINE_integer("synthetic_dataset_num_entries", default=int(2 ** 15 * 1024),
                     help="Number of samples per epoch for the synthetic dataset")
flags.DEFINE_list("synthetic_dataset_table_sizes", default=','.join(26 * [str(10 ** 5)]),
                  help="Cardinalities of variables to use with the synthetic dataset.")
flags.DEFINE_integer("synthetic_dataset_numerical_features", default='13',
                     help="Number of numerical features to use with the synthetic dataset")
flags.DEFINE_boolean("synthetic_dataset_use_feature_spec", default=False,
                     help="Create a temporary synthetic dataset based on a real one. "
                          "Uses --dataset and --feature_spec"
                          "Overrides synthetic_dataset_table_sizes and synthetic_dataset_numerical_features."
                          "--synthetic_dataset_num_entries is still required")

# Checkpointing
flags.DEFINE_string("load_checkpoint_path", None, "Path from which to load a checkpoint")
flags.DEFINE_string("save_checkpoint_path", None, "Path to which to save the training checkpoints")

# Saving and logging flags
flags.DEFINE_string("log_path", "./log.json", "Destination for the log file with various results and statistics")
flags.DEFINE_integer("test_freq", None,
                     "Number of optimization steps between validations. If None will test after each epoch")
flags.DEFINE_float("test_after", 0, "Don't test the model unless this many epochs has been completed")
flags.DEFINE_integer("print_freq", 200, "Number of optimizations steps between printing training status to stdout")
flags.DEFINE_integer("benchmark_warmup_steps", 0,
                     "Number of initial iterations to exclude from throughput measurements")

# Machine setting flags
flags.DEFINE_string("base_device", "cuda", "Device to run the majority of the model operations")
flags.DEFINE_boolean("amp", False, "If True the script will use Automatic Mixed Precision")

flags.DEFINE_boolean("cuda_graphs", False, "Use CUDA Graphs")

# inference benchmark
flags.DEFINE_list("inference_benchmark_batch_sizes", default=[1, 64, 4096],
                  help="Batch sizes for inference throughput and latency measurements")
flags.DEFINE_integer("inference_benchmark_steps", 200,
                     "Number of steps for measuring inference latency and throughput")

# Miscellaneous
flags.DEFINE_float("auc_threshold", None, "Stop the training after achieving this AUC")
flags.DEFINE_boolean("optimized_mlp", True, "Use an optimized implementation of MLP from apex")
flags.DEFINE_enum("auc_device", default="GPU", enum_values=['GPU', 'CPU'],
                  help="Specifies where ROC AUC metric is calculated")

flags.DEFINE_string("backend", "nccl", "Backend to use for distributed training. Default nccl")
flags.DEFINE_boolean("bottom_features_ordered", False,
                     "Sort features from the bottom model, useful when using saved "
                     "checkpoint in different device configurations")
flags.DEFINE_boolean("freeze_mlps", False,
                     "For debug and benchmarking. Don't perform the weight update for MLPs.")
flags.DEFINE_boolean("freeze_embeddings", False,
                     "For debug and benchmarking. Don't perform the weight update for the embeddings.")
flags.DEFINE_boolean("Adam_embedding_optimizer", False, "Swaps embedding optimizer to Adam")
flags.DEFINE_boolean("Adam_MLP_optimizer", False, "Swaps MLP optimizer to Adam")

# test mode settings flags
flags.DEFINE_integer("total_devices", 1, "Total number of devices to run the test on")

flags.DEFINE_string("output_dir", None, "Output directory for test mode")

def load_feature_spec(flags):
    fspec_path = os.path.join(flags.dataset, flags.feature_spec)
    return FeatureSpec.from_yaml(fspec_path)


def BatchDataloader():
    feature_spec = load_feature_spec(FLAGS)
    use_gpu = "cpu" not in FLAGS.base_device.lower()
    rank, world_size, gpu = dist.init_distributed(backend=FLAGS.backend, use_gpu=use_gpu)
    world_embedding_sizes = get_embedding_sizes(feature_spec, FLAGS.max_table_size)
    device_mapping = get_device_mapping(world_embedding_sizes, num_gpus=world_size)
    _, data_loader_test = get_data_loaders(FLAGS, device_mapping=device_mapping,
                                                    feature_spec=feature_spec)
    return data_loader_test, world_size


def main(argv):
    torch.npu.set_compile_mode(jit_compile=False)

    dataloader, world_size = BatchDataloader()

    batch_size_per_gpu = [FLAGS.test_batch_size // world_size for _ in range(world_size)]
    test_batch_sizes = sum(batch_size_per_gpu)

    if FLAGS.test_batch_size != test_batch_sizes:
        logging.error("Batch size must be divisible by the number of GPUs")
        return
    
    data_stream = torch.cuda.Stream()
    batch_iter = prefetcher(iter(dataloader), data_stream)

    for count, step in enumerate(batch_iter):
        numerical_features, categorical_features, click = next(batch_iter)
        torch.cuda.synchronize()

        if click.shape[0] != test_batch_sizes:
            last_batch_size = click.shape[0]
            padding_size = test_batch_sizes - last_batch_size

            if numerical_features is not None:
                padding_numerical = torch.empty(
                    padding_size, numerical_features.shape[1], device=numerical_features.device, dtype=numerical_features.dtype)
                numerical_features = torch.cat([numerical_features, padding_numerical], dim=0)

            if categorical_features is not None:
                padding_categorical = torch.ones(
                    padding_size, categorical_features.shape[1], 
                    device=categorical_features.device, dtype=categorical_features.dtype)
                categorical_features = torch.cat([categorical_features, padding_categorical], dim=0)

        output_dir = FLAGS.output_dir
        if not os.path.exists(os.path.join(output_dir, "numerical_features")):
            os.makedirs(os.path.join(output_dir, "numerical_features"))
        if not os.path.exists(os.path.join(output_dir, "categorical_features")):
            os.makedirs(os.path.join(output_dir, "categorical_features"))
        if not os.path.exists(os.path.join(output_dir, "click")):
            os.makedirs(os.path.join(output_dir, "click"))
        numerical_features_array = numerical_features.numpy()
        categorical_features_array = categorical_features.numpy()
        click_array = click.numpy()
        np.save(os.path.join(output_dir, "numerical_features", f"numerical_features_{count}.npy"), numerical_features_array)
        np.save(os.path.join(output_dir, "categorical_features", f"categorical_features_{count}.npy"), categorical_features_array)
        np.save(os.path.join(output_dir, "click", f"click_{count}.npy"), click_array)

if __name__ == '__main__':
    app.run(main)
```
该文件为生成离线推理数据脚本

14. dlrm/scripts/pth2onnx.py
新建该文件，添加如下内容
```
import itertools
import os
import sys
from absl import app, flags, logging
from apex import optimizers as apex_optim

from dlrm.data.feature_spec import FeatureSpec
from dlrm.model.distributed import DistributedDlrm
from dlrm.utils import distributed as dist
from dlrm.utils.checkpointing.distributed import make_distributed_checkpoint_writer, make_distributed_checkpoint_loader
from dlrm.utils.distributed import get_gpu_batch_sizes, get_device_mapping, is_main_process, is_distributed
from dlrm.scripts.main import load_feature_spec, validate_flags

import datetime
from time import time

import dllogger
import numpy as np
import torch
from absl import app, flags

import dlrm.scripts.utils as utils
from dlrm.data.data_loader import get_data_loaders
from dlrm.data.utils import prefetcher, get_embedding_sizes

import re
import torch
import torch_npu

FLAGS = flags.FLAGS

# Basic run settings
flags.DEFINE_enum("mode", default='train', enum_values=['train', 'test', 'inference_benchmark'],
                  help="Select task to be performed")
flags.DEFINE_integer("seed", 12345, "Random seed")

# Training flags
flags.DEFINE_integer("batch_size", 65536, "Batch size used for training")
flags.DEFINE_integer("test_batch_size", 65536, "Batch size used for testing/validation")
flags.DEFINE_float("lr", 24, "Base learning rate")
flags.DEFINE_integer("epochs", 1, "Number of epochs to train for")
flags.DEFINE_integer("max_steps", None, "Stop training after doing this many optimization steps")

# Learning rate schedule flags
flags.DEFINE_integer("warmup_factor", 0, "Learning rate warmup factor. Must be a non-negative integer")
flags.DEFINE_integer("warmup_steps", 8000, "Number of warmup optimization steps")
flags.DEFINE_integer("decay_steps", 24000,
                     "Polynomial learning rate decay steps. If equal to 0 will not do any decaying")
flags.DEFINE_integer("decay_start_step", 48000,
                     "Optimization step after which to start decaying the learning rate, "
                     "if None will start decaying right after the warmup phase is completed")
flags.DEFINE_integer("decay_power", 2, "Polynomial learning rate decay power")
flags.DEFINE_float("decay_end_lr", 0, "LR after the decay ends")

# Model configuration
flags.DEFINE_enum("embedding_type", "custom_cuda",
                  ["joint", "custom_cuda", "multi_table", "joint_sparse", "joint_fused"],
                  help="The type of the embedding operation to use")
flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of embedding space for categorical features")
flags.DEFINE_list("top_mlp_sizes", [1024, 1024, 512, 256, 1], "Linear layer sizes for the top MLP")
flags.DEFINE_list("bottom_mlp_sizes", [512, 256, 128], "Linear layer sizes for the bottom MLP")
flags.DEFINE_enum("interaction_op", default="cuda_dot", enum_values=["cuda_dot", "dot", "cat"],
                  help="Type of interaction operation to perform.")

# Data configuration
flags.DEFINE_string("dataset", None, "Path to dataset directory")
flags.DEFINE_string("feature_spec", default="feature_spec.yaml",
                    help="Name of the feature spec file in the dataset directory")
flags.DEFINE_enum("dataset_type", default="parametric", enum_values=['synthetic_gpu', 'parametric'],
                  help='The type of the dataset to use')
flags.DEFINE_boolean("shuffle_batch_order", False, "Read batch in train dataset by random order", short_name="shuffle")

flags.DEFINE_integer("max_table_size", None,
                     "Maximum number of rows per embedding table, "
                     "by default equal to the number of unique values for each categorical variable")
flags.DEFINE_boolean("hash_indices", False,
                     "If True the model will compute `index := index % table size` "
                     "to ensure that the indices match table sizes")

# Synthetic data configuration
flags.DEFINE_integer("synthetic_dataset_num_entries", default=int(2 ** 15 * 1024),
                     help="Number of samples per epoch for the synthetic dataset")
flags.DEFINE_list("synthetic_dataset_table_sizes", default=','.join(26 * [str(10 ** 5)]),
                  help="Cardinalities of variables to use with the synthetic dataset.")
flags.DEFINE_integer("synthetic_dataset_numerical_features", default='13',
                     help="Number of numerical features to use with the synthetic dataset")
flags.DEFINE_boolean("synthetic_dataset_use_feature_spec", default=False,
                     help="Create a temporary synthetic dataset based on a real one. "
                          "Uses --dataset and --feature_spec"
                          "Overrides synthetic_dataset_table_sizes and synthetic_dataset_numerical_features."
                          "--synthetic_dataset_num_entries is still required")

# Checkpointing
flags.DEFINE_string("load_checkpoint_path", None, "Path from which to load a checkpoint")
flags.DEFINE_string("save_checkpoint_path", None, "Path to which to save the training checkpoints")

# Saving and logging flags
flags.DEFINE_string("log_path", "./log.json", "Destination for the log file with various results and statistics")
flags.DEFINE_integer("test_freq", None,
                     "Number of optimization steps between validations. If None will test after each epoch")
flags.DEFINE_float("test_after", 0, "Don't test the model unless this many epochs has been completed")
flags.DEFINE_integer("print_freq", 200, "Number of optimizations steps between printing training status to stdout")
flags.DEFINE_integer("benchmark_warmup_steps", 0,
                     "Number of initial iterations to exclude from throughput measurements")

# Machine setting flags
flags.DEFINE_string("base_device", "cuda", "Device to run the majority of the model operations")
flags.DEFINE_boolean("amp", False, "If True the script will use Automatic Mixed Precision")

flags.DEFINE_boolean("cuda_graphs", False, "Use CUDA Graphs")

# inference benchmark
flags.DEFINE_list("inference_benchmark_batch_sizes", default=[1, 64, 4096],
                  help="Batch sizes for inference throughput and latency measurements")
flags.DEFINE_integer("inference_benchmark_steps", 200,
                     "Number of steps for measuring inference latency and throughput")

# Miscellaneous
flags.DEFINE_float("auc_threshold", None, "Stop the training after achieving this AUC")
flags.DEFINE_boolean("optimized_mlp", True, "Use an optimized implementation of MLP from apex")
flags.DEFINE_enum("auc_device", default="GPU", enum_values=['GPU', 'CPU'],
                  help="Specifies where ROC AUC metric is calculated")

flags.DEFINE_string("backend", "nccl", "Backend to use for distributed training. Default nccl")
flags.DEFINE_boolean("bottom_features_ordered", False,
                     "Sort features from the bottom model, useful when using saved "
                     "checkpoint in different device configurations")
flags.DEFINE_boolean("freeze_mlps", False,
                     "For debug and benchmarking. Don't perform the weight update for MLPs.")
flags.DEFINE_boolean("freeze_embeddings", False,
                     "For debug and benchmarking. Don't perform the weight update for the embeddings.")
flags.DEFINE_boolean("Adam_embedding_optimizer", False, "Swaps embedding optimizer to Adam")
flags.DEFINE_boolean("Adam_MLP_optimizer", False, "Swaps MLP optimizer to Adam")

# test mode settings flags
flags.DEFINE_integer("total_devices", 1, "Total number of devices to run the test on")


def main(argv):
    torch.manual_seed(FLAGS.seed)
    torch.npu.set_compile_mode(jit_mode=False)

    use_gpu = "cpu" not in FLAGS.base_device.lower()
    rank, world_size, gpu = dist.init_distributed(backed=FLAGS.backend, use_gpu=use_gpu)
    device = FLAGS.base_device

    feature_spec = load_feature_spec(FLAGS)

    cat_feature_count = len(get_embedding_sizes(feature_spec, None))
    validate_flags(cat_feature_count)

    world_embedding_size = get_embedding_sizes(feature_spec, max_table_size=FLAGS.max_table_size)
    world_categorical_feature_sizes = np.asarray(world_embedding_size)
    device_mapping = get_device_mapping(world_embedding_size, num_gpus=world_size)

    categorical_feature_sizes = world_categorical_feature_sizes[device_mapping["embedding"][rank]].tolist()
    num_numerical_features = feature_spec.get_number_of_numerical_features()

    bottom_mlp_sizes = FLAGS.bottom_mlp_sizes if rank == device_mapping["bottom_mlp"] else None


    model = DistributedDlrm(
        vectors_per_gpu=device_mapping['vectors_per_gpu'],
        embedding_device_mapping=device_mapping['embedding'],
        embedding_type=FLAGS.embedding_type,
        embedding_dim=FLAGS.embedding_dim,
        world_num_categorical_features=len(world_categorical_feature_sizes),
        categorical_feature_sizes=categorical_feature_sizes,
        num_numerical_features=num_numerical_features,
        hash_indices=FLAGS.hash_indices,
        bottom_mlp_sizes=bottom_mlp_sizes,
        top_mlp_sizes=FLAGS.top_mlp_sizes,
        interaction_op=FLAGS.interaction_op,
        fp16=FLAGS.amp,
        use_cpp_mlp=FLAGS.optimized_mlp,
        bottom_features_ordered=FLAGS.bottom_features_ordered,
        device=device
    )

    dist.setup_distributed_print(is_main_process())

    checkpoint_loader = make_distributed_checkpoint_loader(device_mapping=device_mapping, rank=rank)

    if FLAGS.load_checkpoint_path:
        checkpoint_loader.load_checkpoint(model, FLAGS.load_checkpoint_path)
        model.to(device)


    def parallelize(model):
        if world_size <= 1:
            return model
        
        model.top_model = torch.nn.parallel.DistributedDataParallel(model.top_model)
        return model
    
    model = parallelize(model)
    model = model.eval()

    numerical_features_example = torch.zeros(1, 13, dtype=torch.float32)
    numerical_features_example = numerical_features_example.to(device)

    categorical_features_example = torch.zeros(1, 26, dtype=torch.int64)
    categorical_features_example = categorical_features_example.to(device)

    input_names = ["numerical_features", "categorical_features"]
    output_names = ["output"]

    dynamic_axes = {
        "numerical_features": {0: "batch_size"},
        "categorical_features": {0: "batch_size"},
        "output": {0: "batch_size"}
    }

    x = (numerical_features_example, categorical_features_example)

    torch.onnx.export(model, 
                      x, 
                      "output/dlrm.onnx", 
                      input_names=input_names, 
                      output_names=output_names, 
                      dynamic_axes=dynamic_axes,
                      opset_version=11
                    )
    

if __name__ == "__main__":
    app.run(main)
```

## 数据预处理
需安装spark及相关依赖
- spark 3.4.4-with-hadop3
- java 1.8.0
- scala 2.13.0
启动命令
```bash
cd dlrm/preproc
bash prepare_dataset.sh 15 CPU Spark
```
预估需要一天时间处理数据

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
### 离线推理
1. 先生成onnx文件
`python dlrm/scripts/pth2onnx.py --mode test --dataset /your/path/to/data --load_checkpoint_path /your/path/to/checkpoint --base_device npu --test_batch_size 65000`
2. onnx转om
由于pytorch1.10之后的特性，onnx文件会被分割成多个文件，转om时只需要加载onnx单个文件即可，其他文件atc会自动加载
`atc --model output/dlrm.onnx --framework=5 --input_shape="numerical_features:65000,13;categorical_features:65000,26 --soc_version=Ascend310P3 --output=dlrm_output`
3. 生成离线推理数据
`python dlrm/scripts/generate_data_for_om.py --mode test --dataset /your/path/to/data --base_device cpu --base_device npu --test_batch_size 65000`
4. 离线推理
`python -m ais_bench --model path/to/om --input "path/to/numerical_features,path/to/categorical_features" --output path/to/output`

## note
- main.py的step time含义：单step消耗时间

  