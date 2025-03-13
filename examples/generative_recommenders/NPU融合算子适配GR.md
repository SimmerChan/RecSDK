# NPU融合算子适配GR

## 适配说明

本样例的适配对象为Generative Recommendations模型, 将其迁移至NPU侧训练，并使用NPU的HSTU融合算子来实现性能的优化。

模型参考的开源链接为 https://github.com/facebookresearch/generative-recommenders

克隆源码并固定版本为:Commits on Dec 16, 2024，提交的SHA-1 hash值（提交ID）：bb389f9539b054e7268528efcd35457a6ad52439

## 启动容器

镜像下载地址： https://www.hiascend.com/developer/ascendhub/detail/9faeb4847b3e419f81b78a4d0ed574b5

该镜像中部分配套版本说明：

| 软件包简称   | 配套版本   |
| ------- | ------ |
| Pytorch | 2.1.0  |
| Python  | 3.11.0 |
| Fbgemm  | 0.5.0  |

启动容器命令参考：

```python
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

## 安装依赖

### CANN、驱动、Kernels包

| 软件           | 版本           | 下载链接                                                                                                                 |
| ------------ | ------------ | -------------------------------------------------------------------------------------------------------------------- |
| CANN-toolkit | 8.0.0.beta1  | https://www.hiascend.com/developer/download/community/result?module=pt+cann                                          |
| CANN-kernels | 8.0.0.beta1  | https://www.hiascend.com/developer/download/community/result?module=pt+cann                                          |
| driver       | 1.0.28.alpha | https://www.hiascend.com/hardware/firmware-drivers/community?product=1&model=30&cann=8.0.0.beta1&driver=1.0.28.alpha |

请根据机器架构、机器型号在下载链接中选择合适的安装包进行安装。

上面是参考安装版本，也可以安装其他版本，只要满足版本之间配套关系即可。

### 安装torch_npu

下载最新版本的mindxsdk-mxec-add-ons安装包

解压之后，进入文件夹mindxsdk-mxec-add-ons：

`cd mindxsdk-mxec-add-ons`

`cd torch_plugin`

执行命令安装：

`pip3 install torch_npu-2.1.0.post9-cp311-cp311-linux_aarch64.whl`

### 安装算子

重新进入文件夹 mindxsdk-mxec-add-ons, 安装需要的昇腾适配算子： jagged_to_padded_dense、IndexSelect优化、 dense_to_jagged、asynchronous_complete_cumsum、gather_for_rank1

```shell
cd mindxsdk-mxec-add-ons/mxrec_ops
bash mxrec_opp_asynchronous_complete_cumsum.run
bash mxrec_opp_dense_to_jagged.run
bash mxrec_opp_index_select_for_rank1_backward.run
bash mxrec_opp_jagged_to_padded_dense.run
bash mxrec_opp_gather_for_rank1.run
```

### 编译融合算子依赖的lib

进入 torch_library 文件夹：

```shell
cd mindxsdk-mxec-add-ons/torch_library
cd hstu
bash build_ops.sh
```

执行完以上命令之后，融合算子的依赖包libhstu_dense_ops.so会生成在同目录下的build文件夹下，可将该so包拷贝到某固定目录下。示例如下：

`cp ./build/libhstu_dense_ops.so /home/torch_ops/`

## 代码修改

将 Generative Recommendations 模型迁移到NPU上并适配NPU融合HSTU算子，需要进行以下适配工作：

### NPU适配

#### autoregressive_losses.py

修改 `generative-recommenders/generative-recommenders/modeling/sequential/autoregressive_losses.py`

将 BCELossWithRatings(AutoregressiveLoss) 中的forward函数替换为以下代码：

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
jagged_supervision_weights = torch.ops.fbgemm.dense_to_jagged(
    supervision_weights.unsqueeze(-1),
    [jagged_id_offsets],
    prefetch_offset_shape
)[0].squeeze(1)

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
    supervision_weights = jagged_supervision_weights,
    supervision_ratings=torch.ops.fbgemm.dense_to_jagged(
        supervision_ratings.unsqueeze(-1),
        [jagged_id_offsets],
        prefetch_offset_shape
    )[0].squeeze(1),
    negatives_sampler=negatives_sampler,
)
```

新增一个`MinClamp`类

```python
class MinClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, min):
        result = torch.clamp(x, min)
        ctx.save_for_backward(x)
        ctx.min = min
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        min = ctx.min
        zeros = torch.zeros_like(grad_output)
        compare = torch.full_like(x, min)
        grad_output = torch.where(x < compare, zeros, grad_output)

        return grad_output, None
```

将 `NegativesSampler`类的`__init__`函数做如下修改:

```python
def __init__(self, l2_norm: bool, l2_norm_eps: float) -> None:
    super().__init__()

    self._l2_norm: bool = l2_norm
    self._l2_norm_eps: float = l2_norm_eps
    self.clamp_op = MinClamp()
```

将 `NegativesSampler`类的`_maybe_l2_norm`函数做如下修改:

```python
def _maybe_l2_norm(self, x: torch.Tensor) -> torch.Tensor:
    if self._l2_norm:
        x = x / self.clamp_op.apply(
            torch.sqrt(self.clamp_op.apply(torch.sum(x**2, dim=-1, keepdim=True), 0.0) + 1e-10),
            self._l2_norm_eps
        )
    return x
```

#### output_postprocessor.py

修改 `generative-recommenders/generative-recommenders/modeling/sequential/output_postprocessor.py`

在文件中新增一个`MinClamp`类

```python
class MinClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, min):
        result = torch.clamp(x, min)
        ctx.save_for_backward(x)
        ctx.min = min
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        min = ctx.min
        zeros = torch.zeros_like(grad_output)
        compare = torch.full_like(x, min)
        grad_output = torch.where(x < compare, zeros, grad_output)

        return grad_output, None
```

将`L2NormEmbeddingPostprocessor`类中的`__init__`函数做如下修改：

```python
def __init__(
    self,
    embedding_dim: int,
    eps: float = 1e-6
) -> None:
    super().__init__()
    self._embedding_dim: int = embedding_dim
    self._eps: float = eps
    self.clamp_op = MinClamp()
```

将`L2NormEmbeddingPostprocessor`类中的`forward`函数做如下修改：

```python
def forward(
    self,
    output_embeddings: torch.Tensor
) -> torch.Tensor:
    output_embeddings = output_embeddings[..., self._embedding_dim]
    return output_embeddings / self.clamp_op.apply(
        torch.sqrt(self.clamp_op.apply(torch.sum(output_embeddings**2, dim=-1, keepdim=True), 0.0) + 1e-10),
        self._eps
    )
```

#### features.py

修改 `generative-recommenders/generative-recommenders/modeling/sequential/features.py`

修改 `movielens_seq_features_from_row` 函数：

在代码38行后添加代码：

```python
import prefetch_shape
prefetch_shape.args["offset"] = row["history_lengths"].sum().item()
lengths = row["history_lengths"].tolist()
cumulative_sum = [sum(lengths[:i]) for i in range(len(lengths) + 1)]
prefetch_shape.args["lengths"] = cumulative_sum
```

#### main.py

修改`generative-recommenders/main.py`

在第34行增加代码：

```python
import torch_npu
```

修改 `_main` 函数
将原代码：

```python
world_size = torch.cuda.device_count()
mp.set_start_method("forkserver")
```

修改为：

```python
world_size = torch_npu.npu.device_count()
# mp.set_start_method("forkserver")
```

#### similarity_fn.py

修改 `generative-recommenders/generative-recommenders/rails/similarities/mol/similarity_fn.py`

将原代码第330-332行:

```python
with torch.autocast(
        enabled=self._autocast_bf16, dtype=torch.bfloat16, device_type='cuda'
):
```

修改为:

```python
with torch.autocast(
        enabled=self._autocast_bf16, dtype=torch.bfloat16, device_type='npu'
):
```

#### train.py

修改 `generative-recommenders/generative-recommenders/trainer/train.py`

在原代码第30行， 增加代码：

```python
import torch_npu
import numpy as np
```

在原代码第70行，增加代码：

```python
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    torch.npu.manual_seed(seed) 
    torch.npu.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化

set_seed(42)
```

新增用于重复数据集迭代的类`RepeatDataset` ， 类的代码放在原代码函数 `setup` 前：

```python
from torch.utils.data import Dataset

class RepeatDataset(Dataset):
    def __init__(self, dataset, num_repeats):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return len(self.dataset) * self.num_repeats

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]
```

修改 `setup` 函数:  
将原始代码第76行

```python
dist.init_process_group("nccl", rank=rank, world_size=world_size)
```

改为：

```python
dist.init_process_group("hccl", rank=rank, world_size=world_size)
```

修改 `train_fn` 函数:
将原代码第137-138行:

```python
torch.backends.cuda.matmul.allow_tf32 = enable_tf32
torch.backends.cudnn.allow_tf32 = enable_tf32
```

修改为：

```python
torch_npu.npu.set_device(rank)
torch_npu.npu.set_compile_mode(jit_compile=False)
torch_npu.npu.matmul.allow_hf32 = True
torch_npu.npu.conv.allow_hf32 = True
```

将原代码第151-158行：

```python
train_data_sampler, train_data_loader = create_data_loader(
    dataset.train_dataset,
    batch_size=local_batch_size,
    world_size=world_size,
    rank=rank,
    shuffle=True,
    drop_last=world_size>1,
)
```

修改为:

```python
train_dataset = RepeatDataset(dataset.train_dataset, num_repeats=300)
train_data_sampler, train_data_loader = create_data_loader(
    train_dataset,
    batch_size=local_batch_size,
    world_size=world_size,
    rank=rank,
    shuffle=True,
    drop_last=world_size>1,
)
```

在原代码第264行：

```python
model = model.to(device)
```

前面添加代码：

```python
device = f"npu:{device}"
```

注释原代码第300行：

```python
# torch.autograd.set_detect_anomaly(True)
```

#### 新增 prefetch_shape.py

在 main.py 同级目录下添加 prefetch_shape.py ，里面代码为：

```python
args = {"offset": 0}
```

### 融合算子适配

#### 修改适配hstu.py

修改 `generative-recommenders/generative-recommenders/modeling/sequential/hstu.py`

在原代码第22行添加代码:

```python
import os
import numpy
import torch.distributed as dist
import torch_npu
```

在原代码路径第45行添加代码:

```python
torch.ops.load_library("/home/torch_ops/libhstu_dense_ops.so")
```

该so包路径采用上文示例的路径，用户可根据该包实际路径更改代码中的路径

在原代码第48行新增类EmbedRank1Select的实现：

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

继续新增类HstuFusion的实现:

```python
class HstuFusion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask, bias, mask_type, mode, seq_offset, max_seq_len):
        silu_value = 1 / max_seq_len
        out = torch.ops.mxrec.hstu_dense(q, k, v, mask, bias, mask_type, max_seq_len, silu_value, mode, seq_offset)
        ctx.save_for_backward(q, k ,v, mask, bias)
        ctx.max_seq_len = max_seq_len
        ctx.silu_scale = silu_value
        ctx.seq_offset = seq_offset
        ctx.mask_type = mask_type
        ctx.mode = mode
        return out

    @staticmethod
    def backward(ctx, grad_output):
        q, k ,v, mask, bias = ctx.saved_tensors
        q_grad, k_grad, v_grad, bias_grad = torch.ops.mxrec.hstu_dense_backward(
            grad_output, q, k, v, mask, bias, ctx.mode, ctx.mask_type, ctx.max_seq_len, ctx.silu_scale, ctx.seq_offset)
        if bias is None:
            bias_grad = None
        return q_grad, k_grad, v_grad, None, bias_grad, None, None, None, None
```

在原代码108行后加入：

```python
self.tw_elect_op = EmbedRank1Select()
```

把原代码第131-141行

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

修改为：

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

修改`_hstu_attention_maybe_from_cache` 函数：

将原代码第202-221行代码:

修改为：

```python
import prefetch_shape
prefetch_offset_shape = prefetch_shape.args["offset"]
use_npu_hstu = int(os.getenv("USE_NPU_HSTU", 0))

if use_npu_hstu:
    hstu_fusion_op = HstuFusion()
    q_ = q.reshape(-1, num_heads, attention_dim)
    k_ = k.reshape(-1, num_heads, attention_dim)
    v_ = v.reshape(-1, num_heads, attention_dim)
    prefetch_lengths = numpy.array(prefetch_shape.args["lengths"])
    mask_type = 0
    attn_output = hstu_fusion_op.apply(q_, k_, v_, None, None, mask_type, "jagged", prefetch_lengths, n)
    attn_output = attn_output.reshape(-1, num_heads * linear_dim)
    return attn_output, padded_q, padded_k
else:
    qk_attn = torch.einsum(
        "bnhd,bmhd->bhnm",
        padded_q.view(B, n, num_heads, attention_dim),
        padded_k.view(B, n, num_heads, attention_dim),
    )
    if all_timestamps is not None and real_attn_bias:
        qk_attn = qk_attn + rel_attn_bias(all_timestamps).unsqueeze(1)

    qk_attn = F.silu(qk_attn) / n
    qk_attn = qk_attn * invalid_attn_mask.unsqueeze(0).unsqueeze(0)
    attn_output = torch.ops.fbgemm.dense_to_jagged(
        torch.einsum(
            "bhnm,bmhd -> bnhd",
            qk_attn,
            torch.ops.fbgemm.jagged_to_padded_dense(v, [x_offsets], [n]).reshape(
                B, n, num_heads, linear_dim
            ),
        ).reshape(B,n, num_heads * linear_dim),
        [x_offsets],
        prefetch_offset_shape
    )[0]

    return attn_output, padded_q, padded_k

```

将原代码第342行注释：

```python
# assert self._rel_attn_bias is not None
```

将原代码402-408行：

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

将原代码第481-484行：

```python
with torch.autocast(
        "cuda",
        enabled=self._autocast_dtype is not None,
        dtype=self._autocast_dtype or torch.float16,
):
```

修改为：

```python
with torch.autocast(
        "npu",
        enabled=self._autocast_dtype is not None,
        dtype=self._autocast_dtype or torch.float16,
):
```

将原代码520-521行：

```python
if len(x.size()) == 3:
    x = torch.ops.fbgemm.dense_to_jagged(x, [x_offsets])[0]
```

修改为：

```python
import prefetch_shape
prefetch_offset_shape = prefetch_shape.args["offset"]

if len(x.size()) == 3:
    x = torch.ops.fbgemm.dense_to_jagged(x, [x_offsets], prefetch_offset_shape)[0]
```

注释掉原代码第603-614行， 不传入relative_attention_bias_module参数， 本次示例修改代码为去rab操作，用户可根据需要选择是否传入

### 新增测试运行脚本run.sh

在 main.py 同级目录下添加 run.sh ，里面代码为：

```shell
export USE_NPU_HSTU = 1
export PYTORCH_NPU_ALLOC_CONF = expandable_segments:True
python3 train.py --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin --master_port=12345 | tee temp.log
```

测试的gin文件请根据实际测试配置选择更改gin_config_file参数

## 测试示例

本次测试基于ml-1m数据集，使用NPU的HSTU融合算子(去rab, 下三角mask)， 基于以下配置config文件进行测试：

### config文件：

创建一个hstu-mt-3400.gin文件，文件内容如下。将该gin文件放置在 `generative-recommenders/configs/ml-1m/` 目录下

```gin
train_fn.dataset_name = "ml-1m"
train_fn.max_sequence_length = 3389
train_fn.local_batch_size = 32

train_fn.main_module = "HSTU"
train_fn.dropout_rate = 0.2
train_fn.user_embedding_norm = "l2_norm"
train_fn.num_epochs = 1
train_fn.item_embedding_dim = 512

hstu_encoder.num_blocks = 3
hstu_encoder.num_heads = 2
hstu_encoder.dqk = 256
hstu_encoder.dv = 256
hstu_encoder.linear_dropout_rate = 0.2

train_fn.learning_rate = 1e-3
train_fn.weight_decay = 0
train_fn.num_warmup_steps = 0

train_fn.interaction_module_type = "DotProduct"
train_fn.top_k_method = "MIPSBruteForceTopK"

train_fn.loss_module = "SampledSoftmaxLoss"
train_fn.num_negatives = 128
train_fn.eval_interval = 50
train_fn.sampling_strategy = "local"
train_fn.temperature = 0.05
train_fn.item_l2_norm = True
train_fn.l2_norm_eps = 1e-6

train_fn.enable_tf32 = True

create_data_loader.prefetch_factor = 128
create_data_loader.num_workers = 8
```

### 运行命令：

修改run.sh 脚本，使用上面的配置文件：

```shell
export USE_NPU_HSTU = 1
export PYTORCH_NPU_ALLOC_CONF = expandable_segments:True
python3 train.py --gin_config_file=configs/ml-1m/hstu-mt-3400.gin --master_port=12345 | tee temp.log
```

执行命令：
`bash run.sh`

### 性能测试结果

| 数据集   | seq_len | num_block | num_heads | dqk、dv | 端到端耗时  | GPU triton耗时 |
| ----- | ------- | --------- | --------- | ------ | ------ | ------------ |
| ml-1m | 3400    | 3         | 2         | 256    | 54.8ms | 75ms         |

### 精度loss比对

## FAQ


