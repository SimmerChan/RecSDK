# Generative Recommendations迁移样例
## 迁移说明
本样例以Generative Recommendations模型为例,迁移至NPU侧训练, 模型参考的开源链接为 https://github.com/facebookresearch/generative-recommenders
克隆源码并固定版本为:Commits on Jul 3, 2024，提交的SHA-1 hash值（提交ID）：9e3d3103af3bd17416cb0c4fbe6c4d98948877ce

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
切换为root用户。

## CANN和驱动使用版本

| 软件包简称              | 配套版本           |
|--------------------|----------------|
| CANN-toolkit版本     | 8.0.0.alpha001 |
| CANN-Kernels二进制算子包 | 8.0.0.alpha001 |
| NPU-driver驱动包      | 24.1.rc2       |
| NPU-firmware固件包    | 7.3.0.1.231    |

软件包参考地址:

1. https://www.hiascend.com/developer/download/community/result?module=pt+cann&pt=6.0.0.alpha001&cann=8.0.0.alpha001&product=4&model=26  

2. https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=26&cann=8.0.0.alpha001&driver=1.0.25.alpha

## 安装算子
下载mindxsdk-mxrec-add-ons软件包,在mindxsdk-mxrec-add-ons/mxrec_ops/目录下安装 jagged_to_padded_dense、IndexSelect优化、
dense_to_jagged、asynchronous_complete_cumsum、gather_for_rank1昇腾算子,执行：
```shell
cd mindxsdk-mxrec-add-ons/mxrec_ops/
bash mxrec_opp_asynchronous_complete_cumsum.run
bash mxrec_opp_dense_to_jagged.run
bash mxrec_opp_index_select_for_rank1_backward.run
bash mxrec_opp_jagged_to_padded_dense.run
bash mxrec_opp_gather_for_rank1.run
```

## 使用pytorch调用的方式调用算子工程
该样例脚本基于Pytorch2.1.0、python3.11.0、gcc version 10.2.0运行，执行如下命令：
```shell
source /etc/profile
```
在联网环境下，在 mindxsdk-mxrec-add-ons/torch_plugin/ 目录下使用 pip3 安装 .whl 文件，例如：
```shell
pip3 install /mindxsdk-mxrec-add-ons/torch_plugin/torch_npu-2.1.0.post8-cp311-cp311-linux_aarch64.whl
```

## 设置环境变量
```shell
export LD_PRELOAD=/usr/lib64/libgomp.so.1
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/driver/bin/setenv.bash
```
Q&A

1."libgomp.so.1: cannot allocate memory in static TLS block" 这个错误通常发生在动态加载库时，系统无法为线程本地存储（Thread Local Storage, TLS）分配足够的内存，这通常与 OpenMP 库的加载有关。在某些 Arm 环境下，特别是在运行 PyTorch 或 NPU 相关任务时，系统可能无法正确加载 libgomp.so.1。使用 LD_PRELOAD 可以强制系统在程序启动时预先加载这个库，从而避免动态加载时出现的 TLS 分配问题。通过 LD_PRELOAD 预加载 libgomp.so.1，系统可以在主程序启动之前就分配必要的 TLS 内存。这样可以确保 OpenMP 库正确加载，避免在后续动态加载时出现内存分配失败的问题。

2.CANN软件提供进程级环境变量设置脚本，供用户在进程中引用，以自动完成环境变量设置。用户进程结束后自动失效。可在程序启动的Shell脚本中使用以上命令设置CANN的相关环境变量，也可通过命令行执行上面命令（以root用户默认安装路径“/usr/local/Ascend”为例）。



## 修改 Generative Recommendations 样例
### modeling/sequential/autoregressive_losses.py
修改modeling/sequential/autoregressive_losses.py。在第23行添加

```python
class MinClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, min):
        result =  torch.clamp(x, min)
        ctx.save_for_backward(x)
        ctx.min = min
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        min = ctx.min
        zeros = torch.zeros_like(grad_output)
        commpare = torch.full_like(x, min)
        grad_output = torch.where(x < commpare, zeros, grad_output)

        return grad_output, None
```
这个自定义的 MinClamp 操作是因为通过重写反向算子，可以自定义梯度计算的方式。在NPU上进行计算时，有时需要同步不同的操作流。去掉反向算子里的流同步的item可以提高计算效率。forward方法实现了前向传播，使用torch.clamp对输入进行截断。backward方法实现了反向传播，计算梯度。在backward方法中，通过比较输入x和最小值min，决定是否传递梯度。当x小于min时，梯度被设置为0，否则保持原梯度不变。这个实现避免了使用item()方法，which可能会导致NPU和CPU之间的同步。相反，它使用torch.where进行全张量操作，这通常更高效。
在第47行添加

```python
self.clamp_op = MinClamp()
```

所以第42~47行代码为：

```python
def __init__(self, l2_norm: bool, l2_norm_eps: float) -> None:
    super().__init__()

    self._l2_norm: bool = l2_norm
    self._l2_norm_eps: float = l2_norm_eps
    self.clamp_op = MinClamp()
```

把第52~57行

```python
def _maybe_l2_norm(self, x: torch.Tensor) -> torch.Tensor:
    if self._l2_norm:
        x = x / torch.clamp(
            torch.linalg.norm(x, ord=2, dim=-1, keepdim=True),
            min=self._l2_norm_eps,
        )
```
改为：
```python
def _maybe_l2_norm(self, x: torch.Tensor) -> torch.Tensor:
    if self._l2_norm:
        x = x / self.clamp_op.apply(
            torch.sqrt(self.clamp_op.apply(torch.sum(x**2, dim=-1, keepdim=True), 0.0)+1e-10),
            self._l2_norm_eps
        )
```
原代码使用的torch.linalg.norm函数是因为NPU的这个算子存在精度问题。新的实现使用了一种更稳定的方法来计算L2范数（欧几里得范数）。计算每个向量元素的平方和；再使用自定义的MinClamp操作确保平方和不小于0；对结果开平方根，并加上一个很小的值（1e-10）以避免除以零；再次使用MinClamp操作，确保最终结果不小于self._l2_norm_eps。这种方法通过避免直接使用torch.linalg.norm，并引入一些数值稳定性的技巧（如添加小常数、使用自定义的截断操作），来提高计算的精度和稳定性。总的来说，这个修改旨在解决存在的数值精度问题，同时保持计算的高效性。通过使用更稳定的计算方法和自定义操作，可以在保证精度的同时可能还提高了性能。
把第581~641行

```python
assert output_embeddings.size() == supervision_embeddings.size()
assert supervision_ids.size() == supervision_embeddings.size()[:-1]
jagged_id_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
jagged_supervision_ids = (
    torch.ops.fbgemm.dense_to_jagged(
        supervision_ids.unsqueeze(-1).float(), [jagged_id_offsets]
    )[0]
    .squeeze(1)
    .long()
)

args = OrderedDict(
    [
        (
            "output_embeddings",
            torch.ops.fbgemm.dense_to_jagged(
                output_embeddings,
                [jagged_id_offsets],
            )[0],
        ),
        ("supervision_ids", jagged_supervision_ids),
        (
            "supervision_embeddings",
            torch.ops.fbgemm.dense_to_jagged(
                supervision_embeddings,
                [jagged_id_offsets],
            )[0],
        ),
        (
            "supervision_weights",
            torch.ops.fbgemm.dense_to_jagged(
                supervision_weights.unsqueeze(-1),
                [jagged_id_offsets],
            )[0].squeeze(1),
        ),
        ("negatives_sampler", negatives_sampler),
    ]
)
if self._activation_checkpoint:
    return checkpoint(
        self.jagged_forward,
        *args.values(),
        use_reentrant=False,
    )
else:
    return self.jagged_forward(
        output_embeddings=torch.ops.fbgemm.dense_to_jagged(
            output_embeddings,
            [jagged_id_offsets],
        )[0],
        supervision_ids=jagged_supervision_ids,
        supervision_embeddings=torch.ops.fbgemm.dense_to_jagged(
            supervision_embeddings,
            [jagged_id_offsets],
        )[0],
        supervision_weights=torch.ops.fbgemm.dense_to_jagged(
            supervision_weights.unsqueeze(-1),
            [jagged_id_offsets],
        )[0].squeeze(1),
        negatives_sampler=negatives_sampler,
    )
```
改为：
```python
import prefetch_shape
prefetch_offset_shape = prefetch_shape.args["offset"]

assert output_embeddings.size() == supervision_embeddings.size()
assert supervision_ids.size() == supervision_embeddings.size()[:-1]
jagged_id_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
jagged_supervision_ids = (
    torch.ops.fbgemm.dense_to_jagged(
        supervision_ids.unsqueeze(-1).float(), [jagged_id_offsets], prefetch_offset_shape
    )[0]
    .squeeze(1)
    .long()
)

args = OrderedDict(
    [
        (
            "output_embeddings",
            torch.ops.fbgemm.dense_to_jagged(
                output_embeddings,
                [jagged_id_offsets],
                prefetch_offset_shape
            )[0],
        ),
        ("supervision_ids", jagged_supervision_ids),
        (
            "supervision_embeddings",
            torch.ops.fbgemm.dense_to_jagged(
                supervision_embeddings,
                [jagged_id_offsets],
                prefetch_offset_shape
            )[0],
        ),
        (
            "supervision_weights",
            torch.ops.fbgemm.dense_to_jagged(
                supervision_weights.unsqueeze(-1),
                [jagged_id_offsets],
                prefetch_offset_shape
            )[0].squeeze(1),
        ),
        ("negatives_sampler", negatives_sampler),
    ]
)
if self._activation_checkpoint:
    return checkpoint(
        self.jagged_forward,
        *args.values(),
        use_reentrant=False,
    )
else:
    return self.jagged_forward(
        output_embeddings=torch.ops.fbgemm.dense_to_jagged(
            output_embeddings,
            [jagged_id_offsets],
            prefetch_offset_shape
        )[0],
        supervision_ids=jagged_supervision_ids,
        supervision_embeddings=torch.ops.fbgemm.dense_to_jagged(
            supervision_embeddings,
            [jagged_id_offsets],
            prefetch_offset_shape
        )[0],
        supervision_weights=torch.ops.fbgemm.dense_to_jagged(
            supervision_weights.unsqueeze(-1),
            [jagged_id_offsets],
            prefetch_offset_shape
        )[0].squeeze(1),
        negatives_sampler=negatives_sampler,
    )
```
这个修改是为了优化数据预取过程。通过预先计算和存储 offset 信息,我们可以减少数据加载过程中的延迟,提高训练效率。这对于处理大规模稀疏数据的推荐系统来说尤其重要。
### modeling/sequential/features.py
修改modeling/sequential/features.py。把第37~39行

```python
historical_lengths = row["history_lengths"].to(device)  # [B]
historical_ids = row["historical_ids"].to(device)  # [B, N]
historical_ratings = row["historical_ratings"].to(device)
```
改为：
```python
import prefetch_shape
prefetch_shape.args["offset"] = row["history_lengths"].sum().item()
historical_lengths = row["history_lengths"].to(device)  # [B]
historical_ids = row["historical_ids"].to(device)  # [B, N]
historical_ratings = row["historical_ratings"].to(device)
```
### modeling/sequential/hstu.py
修改modeling/sequential/hstu.py。在第77行添加：

```python
import torch_npu
class EmbedRank1Select(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, index):
        result =  torch_npu.gather_for_rank1(x, index=index)
        ctx.save_for_backward(x, index)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, index = ctx.saved_tensors
        gradX, gradIndex = torch_npu.index_select_for_rank1_backward(grad_output, x, index)
        return gradX, gradIndex
```
这段代码是为了优化index_select操作的反向传播性能。这里定义了一个名为EmbedRank1Select的自定义PyTorch操作。它继承自torch.autograd.Function，允许我们自定义前向和反向传播的行为。使用标准的torch.index_select函数进行前向计算。保存输入张量x和索引index以供反向传播使用。使用torch_npu.index_select_for_rank1_backward函数来计算梯度。标准的index_select操作在反向传播时可能不够高效，特别是在处理大型嵌入表时。使用专门的NPU函数可能会显著提高反向传播的性能。torch_npu模块这种优化可以利用硬件特性来加速计算。gradX是对输入张量x的梯度；gradIndex是对索引index的梯度。这个修改旨在通过使用硬件特定的优化函数来提高index_select操作在反向传播时的性能。这对于涉及大量嵌入查找的模型（如大语言模型）可能会带来显著的性能提升。 把第112~114行

```python
self._bucketization_fn: Callable[[torch.Tensor], torch.Tensor] = (
    bucketization_fn
)
```
改为：
```python
self._bucketization_fn: Callable[[torch.Tensor], torch.Tensor] = (
    bucketization_fn
)
self.tw_elect_op = EmbedRank1Select()
```

把第137~147行

```python
bucketed_timestamps = torch.clamp(
    self._bucketization_fn(
        ext_timestamps[:, 1:].unsqueeze(2) - ext_timestamps[:, :-1].unsqueeze(1)
    ),
    min=0,
    max=self._num_buckets,
).detach()
rel_pos_bias = t[:, :, r:-r]
rel_ts_bias = torch.index_select(
    self._ts_w, dim=0, index=bucketed_timestamps.view(-1)
).view(B, N, N)
```
改为：
```python
bucketed_timestamps = torch.clamp(
    self._bucketization_fn(
        (ext_timestamps[:, 1:].unsqueeze(2) - ext_timestamps[:, :-1].unsqueeze(1)).to(torch.float64)
    ),
    min=0,
    max=self._num_buckets,
).detach()
rel_pos_bias = t[:, :, r:-r]
rel_ts_bias = self.tw_elect_op.apply(self._ts_w, bucketed_timestamps.view(-1)).view(B, N, N)
```

把第204~223行

```python
qk_attn = torch.einsum(
    "bnhd,bmhd->bhnm",
    padded_q.view(B, n, num_heads, attention_dim),
    padded_k.view(B, n, num_heads, attention_dim),
)
if all_timestamps is not None:
    qk_attn = qk_attn + rel_attn_bias(all_timestamps).unsqueeze(1)
qk_attn = F.silu(qk_attn) / n
qk_attn = qk_attn * invalid_attn_mask.unsqueeze(0).unsqueeze(0)
attn_output = torch.ops.fbgemm.dense_to_jagged(
    torch.einsum(
        "bhnm,bmhd->bnhd",
        qk_attn,
        torch.ops.fbgemm.jagged_to_padded_dense(v, [x_offsets], [n]).reshape(
            B, n, num_heads, linear_dim
        ),
    ).reshape(B, n, num_heads * linear_dim),
    [x_offsets],
)[0]
return attn_output, padded_q, padded_k
```
改为：
```python
import prefetch_shape
prefetch_offset_shape = prefetch_shape.args["offset"]

qk_attn = torch.einsum(
    "bnhd,bmhd->bhnm",
    padded_q.view(B, n, num_heads, attention_dim),
    padded_k.view(B, n, num_heads, attention_dim),
)
if all_timestamps is not None:
    qk_attn = qk_attn + rel_attn_bias(all_timestamps).unsqueeze(1)
qk_attn = F.silu(qk_attn) / n
qk_attn = qk_attn * invalid_attn_mask.unsqueeze(0).unsqueeze(0)
attn_output = torch.ops.fbgemm.dense_to_jagged(
    torch.einsum(
        "bhnm,bmhd->bnhd",
        qk_attn,
        torch.ops.fbgemm.jagged_to_padded_dense(v, [x_offsets], [n]).reshape(
            B, n, num_heads, linear_dim
        ),
    ).reshape(B, n, num_heads * linear_dim),
    [x_offsets],
    prefetch_offset_shape
)[0]
return attn_output, padded_q, padded_k
```

把第405~411行

```python
attn_output = torch.ops.fbgemm.dense_to_jagged(
    torch.bmm(
        qk_attn,
        torch.ops.fbgemm.jagged_to_padded_dense(v, [x_offsets], [n]),
    ),
    [x_offsets],
)[0]
```
改为：
```python
import prefetch_shape
prefetch_offset_shape = prefetch_shape.args["offset"]

attn_output = torch.ops.fbgemm.dense_to_jagged(
    torch.bmm(
        qk_attn,
        torch.ops.fbgemm.jagged_to_padded_dense(v, [x_offsets], [n]),
    ),
    [x_offsets],
    prefetch_offset_shape
)[0]
```

把第489~493行

```python
with torch.autocast(
        "cuda",
        enabled=self._autocast_dtype is not None,
        dtype=self._autocast_dtype or torch.float16,
):
```
改为：
```python
with torch.autocast(
        "npu",
        enabled=self._autocast_dtype is not None,
        dtype=self._autocast_dtype or torch.float16,
):
```

把第528~529行

```python
if len(x.size()) == 3:
    x = torch.ops.fbgemm.dense_to_jagged(x, [x_offsets])[0]
```
改为：
```python
import prefetch_shape
prefetch_offset_shape = prefetch_shape.args["offset"]

if len(x.size()) == 3:
    x = torch.ops.fbgemm.dense_to_jagged(x, [x_offsets], prefetch_offset_shape)[0]
```
### modeling/sequential/output_postprocessors.py
修改modeling/sequential/output_postprocessors.py。在第34行添加：

```python
class MinClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, min):
        result =  torch.clamp(x, min)
        ctx.save_for_backward(x)
        ctx.min = min
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        min = ctx.min
        zeros = torch.zeros_like(grad_output)
        commpare = torch.full_like(x, min)
        grad_output = torch.where(x < commpare, zeros, grad_output)

        return grad_output, None
```

把第61~74行

```python
    self._eps: float = eps

def debug_str(self) -> str:
    return "l2"

def forward(
        self,
        output_embeddings: torch.Tensor,
) -> torch.Tensor:
    output_embeddings = output_embeddings[..., : self._embedding_dim]
    return output_embeddings / torch.clamp(
        torch.linalg.norm(output_embeddings, ord=None, dim=-1, keepdim=True),
        min=self._eps,
    )
```
改为：
```python
    self._eps: float = eps
    self.clamp_op = MinClamp()

def debug_str(self) -> str:
    return "l2"

def forward(
        self,
        output_embeddings: torch.Tensor,
) -> torch.Tensor:
    output_embeddings = output_embeddings[..., : self._embedding_dim]
    return output_embeddings / self.clamp_op.apply(
        torch.sqrt(self.clamp_op.apply(torch.sum(output_embeddings**2, dim=-1, keepdim=True), 0.0)+1e-10),
        self._eps
    )   
```
### modeling/similarity/mol.py
修改modeling/similarity/mol.py。把第443、466、495行

```python
with torch.autocast(enabled=self._bf16_training, dtype=torch.bfloat16, device_type='cuda'):
```
改为：
```python
with torch.autocast(enabled=self._bf16_training, dtype=torch.bfloat16, device_type='npu'):
```
### modeling/similarity_module.py
修改modeling/similarity_module.py。把第76行

```python
with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
```
改为：
```python
with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="npu"):
```
### prefetch_shape.py
在 trainer 同级目录下添加 prefetch_shape.py ，里面代码为：

```python
args = {"offset": 0}
```
### run.sh
在 trainer 同级目录下添加 run.sh ，里面代码为：

```shell
python3 train.py --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin --master_port=12345 | tee temp.log
```
### train.py
修改train.py ，在第36行添加代码：

```python
import torch_npu
```
把第75行

```python
FLAGS = flags.FLAGS
```
改为：
```python
import numpy as np
FLAGS = flags.FLAGS
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.npu.manual_seed(seed) # GPU
    torch.npu.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
set_seed(42)
```

把第91行

```python
dist.init_process_group("nccl", rank=rank, world_size=world_size)
```
改为：
```python
dist.init_process_group("hccl", rank=rank, world_size=world_size)
```

把第138、139行

```python
torch.backends.cuda.matmul.allow_tf32 = enable_tf32
torch.backends.cudnn.allow_tf32 = enable_tf32
```
改为：
```python
torch_npu.npu.set_compile_mode(jit_compile=False)
torch_npu.npu.matmul.allow_hf32 = True
torch_npu.npu.conv.allow_hf32 = True
```

把第265行

```python
model = model.to(device)
```
改为：
```python
device = f"npu:{device}"
model = model.to(device)
```
注释掉第303行：
```python
torch.autograd.set_detect_anomaly(True)
```
把第551、553行

```python
world_size = torch.cuda.device_count()

mp.set_start_method("forkserver")
```
改为：
```python
world_size = 1 #torch.cuda.device_count()

# mp.set_start_method("forkserver")
```

## 运行命令
```shell
bash run.sh
```

## FAQ

### GCC版本问题导致的编译错误

如果遇到以下报错:

```log
/fbgemm/PytorchInvocation/op-plugin/build/pytorch/torch_npu/csrc/distributed/ProcessGroupHCCL.cpp:28:0:
/fbgemm/PytorchInvocation/op-plugin/build/pytorch/torch_npu/csrc/distributed/HCCLUtils.hpp:9:10: fatal error: filesystem: No such file or directory
 #include <filesystem>
          ^~~~~~~~~~~~
compilation terminated.
make[2]: *** [CMakeFiles/torch_npu.dir/build.make:13502: CMakeFiles/torch_npu.dir/torch_npu/csrc/distributed/ProcessGroupHCCL.cpp.o] Error 1
make[1]: *** [CMakeFiles/Makefile2:277: CMakeFiles/torch_npu.dir/all] Error 2
make: *** [Makefile:136: all] Error 2
Traceback (most recent call last):
  File "/fbgemm/PytorchInvocation/op-plugin/build/pytorch/setup.py", line 579, in <module>
    setup(
  File "/usr/local/python3.9.6/lib/python3.9/site-packages/setuptools/__init__.py", line 153, in setup
    return distutils.core.setup(**attrs)
  File "/usr/local/python3.9.6/lib/python3.9/distutils/core.py", line 148, in setup
    dist.run_commands()
  File "/usr/local/python3.9.6/lib/python3.9/distutils/dist.py", line 966, in run_commands
    self.run_command(cmd)
  File "/usr/local/python3.9.6/lib/python3.9/distutils/dist.py", line 985, in run_command
    cmd_obj.run()
  File "/usr/local/python3.9.6/lib/python3.9/distutils/command/build.py", line 135, in run
    self.run_command(cmd_name)
  File "/usr/local/python3.9.6/lib/python3.9/distutils/cmd.py", line 313, in run_command
    self.distribution.run_command(command)
  File "/usr/local/python3.9.6/lib/python3.9/distutils/dist.py", line 985, in run_command
    cmd_obj.run()
  File "/gr-npu/PytorchInvocation/op-plugin/build/pytorch/setup.py", line 352, in run
    subprocess.check_call(['make'] + build_args, cwd=build_type_dir, env=os.environ)
  File "/usr/local/python3.9.6/lib/python3.9/subprocess.py", line 373, in check_call
    raise CalledProcessError(retcode, cmd)
subprocess.CalledProcessError: Command '['make', '-j', '192']' returned non-zero exit status 2.
```
解决方法:
需要删除 PytorchInvocation 目录下的 op-plugin 目录，然后重新执行以下步骤:
1. 安装算子
2. 使用pytorch调用的方式调用算子工程
