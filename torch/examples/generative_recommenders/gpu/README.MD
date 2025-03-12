# triton适配说明
## 适配说明
1. 源码地址：[generative-recommenders](https://github.com/facebookresearch/generative-recommenders)
2. commit: bb389f9539b054e7268528efcd35457a6ad52439
## Python、pytorch、fbgemm-gpu、CUDA版本
| 名称 | 版本 |
| ----- | ----- |
| python | 3.11.6 |
| pytorch | 2.4.1 |
| CUDA | 12.1 |
| fbgemm-gpu | 0.8.0 |
## 依赖安装
修改requirements.txt
```
torch==2.4.0
torch_tensorrt==2.4.0
torchaudio==2.4.0
torchvison==0.19
fbgemm-gpu==0.6.0
```
执行命令
`pip install -r requirements.txt`
安装 [mindstudio-probe](https://gitee.com/ascend/mstt/blob/master/debug/accuracy_tools/msprobe/docs/01.installation.md)
## 适配
1. generative_recommenders/modeling/sequential/hstu.py
原46行替换成如下代码
```python
from generative_recommenders.ops.triton.triton_hstu_attention import (
    triton_hstu_mha,
    triton_hstu_attention_fwd,
    triton_hstu_attention_bwd,
)
from generative_recommenders.ops.pytorch.pt_hstu_attention import (
    pytorch_hstu_mha
)
from generative_recommenders.ops.pytorch.pt_hstu_attention import (
    generate_sparse_seq_len,
    gpu_unavailable,
    HammerKernel,
    set_dev_mode,
)
from generative_recommenders.ops.hstu_attention import hstu_mha
TIMESTAMPS_KEY = "timestamps"
ref_kernel=HammerKernel.PYTORCH,
real_kernel=HammerKernel.TRITON,

class AttentionFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        N: int,
        alpha: float,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_offsets: torch.Tensor,
        causal: bool,
        num_targets: Optional[torch.Tensor],
        max_attn_len: Optional[int],
        contextual_seq_len: int,
        sort_by_length: bool,
    ) -> torch.Tensor:
        sort_by_length_indices = None
        if sort_by_length:
            seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
            _, sort_by_length_indices = torch.sort(
                seq_lengths, descending=True, stable=False
            )
        save_tensors = [q, k, v, seq_offsets]
        if num_targets is not None:
            save_tensors.append(num_targets)
        max_attn_len = max_attn_len or 0
        if sort_by_length_indices is not None:
            save_tensors.append(sort_by_length_indices)
        ctx.save_for_backward(*save_tensors)
        ctx.alpha = alpha
        ctx.causal = causal
        ctx.has_multiple_targets = num_targets is not None
        ctx.max_attn_len = max_attn_len
        ctx.N = N
        ctx.contextual_seq_len = contextual_seq_len
        ctx.sort_by_length = sort_by_length
        return triton_hstu_attention_bwd(
            N=N,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            causal=causal,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            sort_by_length_indices=sort_by_length_indices,
        )
    
    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, dout: torch.Tensor
    ) -> Tuple[
        None,
        None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        with torch.inference_mode():
            q, k, v, seq_offsets = ctx.saved_tensors[:4]
            idx = 4
            if ctx.has_multiple_targets:
                num_targets = ctx.saved_tensors[idx]
                idx += 1
            else:
                num_targets = None
            if ctx.sort_by_length:
                sort_by_length_indices = ctx.saved_tensors[idx]
                idx += 1
            else:
                sort_by_length_indices = None
            
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            dq, dk, dv = triton_hstu_attention_bwd(
                dout=dout,
                q=q,
                k=k,
                v=v,
                dq=dq,
                dk=dk,
                dv=dv,
                seq_offsets=seq_offsets,
                num_targets=num_targets,
                N=ctx.N,
                alpha=ctx.alpha,
                max_attn_len=ctx.max_attn_len,
                causal=ctx.causal,
                contextual_seq_len=ctx.contextual_seq_len,
                sort_by_length_indices=sort_by_length_indices,
            )
            return (
                None,
                None,
                dq,
                dk,
                dv,
                None,
                None,
                None,
                None,
                None,
                None,
            )
```
SequentialTransductionUnitJagged类里，原226行下添加`max_seq_len: int,`，原241行下添加`self._max_seq_len: int = max_seq_len`
SequentialTransductionUnitJagged类里，forward函数里，原308行下添加
```python
        alpha = 1.0
        B: int = x_offsets.shape(0) - 1
```
将原340-410行替换成
```python
        bs = q.size(0)
        q = q.contiguous().view(bs, self._num_heads, self._attention_dim)
        k = k.contiguous().view(bs, self._num_heads, self._attention_dim)
        v = v.contiguous().view(bs, self._num_heads, self._linear_dim)

        attn_op = AttentionFunction()
        attn_output = attn_op.apply(
            self._max_seq_len,
            alpha,
            q,
            k,
            v,
            x_offsets,
            True,
            None,
            None,
            0,
            False,
        )

        attn_output = attn_output.contiguous().view(bs, self._num_heads*self._attention_dim)
        B: int = x_offsets.shape(0) - 1
```
最后替换return值，原442行改成`return new_outputs, (v, cached_q, cached_k, new_outputs)`
HSTU类里，原594行下添加`max_seq_len=max_sequence_len+max_output_len,`
删除原603-614行，去除rab

2. generative_recommenders/common.py
注释83-84行
```
# pre_hook=None
# post_hook=None
```

### precision_mode模式适配
1. generative_recommenders/modeling/sequential/embedding_modules.py
LocaEmbeddingModule类__init__函数下
添加参数precision_mode: bool = False
把`self.reset_params`替换成
```
        if precision_mode:
            self._item_emb.weight.data.fill_(1)
            self._item_emb.weight.data[0] = 0
        else:
            self.reset_params()
```
2. generative_recommenders/modeling/sequential/hstu.py
SequentialTransductionUnitJagged类__init__函数下
添加参数precision_mode: bool = False
将
```
            self._uvqk: torch.nn.Parameter = torch.nn.Parameter(
              torch.empty(
                     (
                         embedding_dim,
                         linear_hidden_dim * 2 * num_heads
                         + attention_dim * num_heads * 2,
                     )
                 ).normal_(mean=0, std=0.02),
            )
```
替换成
```
            if not precision_mode:
                self._uvqk: torch.nn.Parameter = torch.nn.Parameter(
                    torch.empty(
                        (
                            embedding_dim,
                            linear_hidden_dim * 2 * num_heads
                            + attention_dim * num_heads * 2,
                        )
                    ).normal_(mean=0, std=0.02),
                )
            else:
                self._uvqk = torch.nn.Parameter(
                    torch.empty(
                        (
                            embedding_dim,
                            linear_hidden_dim * 2 * num_heads
                            + attention_dim * num_heads * 2,
                        )
                    ).fill_(0.01),
                )
```
HSTU __init__添加参数precision_mode: bool = False
初始化self._hstu下的SequentialTransductionUnitJagged里添加precision_mode=precision_mode参数
__init__初始化函数添加`self.precision_mode = precision_mode`
HSTU reset_params函数下
将`torch.nn.init.xavier_normal_(params.data)`替换成
```
                if not self.precision_mode:
                    torch.nn.init.xavier_normal_(params.data)
                else:
                    fixed_value = 1
                    params.data.fill_(fixed_value)
```

3. trainer/train.py
原69行添加
```
from nsprobe.pytorch import seed_all
seed_all(mode=True, rm_dropout=True, seed=1234)
```
train_fn函数
添加参数precision_mode: bool = False
将
```
    if embedding_module_type == "local":
        embedding_module: EmbeddingModule = LocalEmbeddingModule(
            num_items=dataset.max_item_id,
            item_embedding_dim=item_embedding_dim,
        )
```
替换成
```
    if embedding_module_type == "local":
        embedding_module: EmbeddingModule = LocalEmbeddingModule(
            num_items=dataset.max_item_id,
            item_embedding_dim=item_embedding_dim,
            precision_mode=precision_mode,
        )
```
model = get_sequential_encoder下添加参数precision_mode=precision_mode
input_preproc_module = LearnablePositionalEmbeddingInputFeaturesPreprocessor下添加参数precision_mode=precision_mode

4. generative_recommenders/modeling/sequential/encoder_utils.py
hstu_encoder函数添加参数precision_mode: bool = False
return HSTU下添加precision_mode=precision_mode参数
get_sequential_encoder函数添加参数precision_mode: bool = False
model = hstu_encoder下添加参数precision_mode=precision_mode

5. generative_recommenders/modeling/sequential/input_features_preprocessors.py
LearnablePositionalEmbeddingInputFeaturesPreprocessor类添加参数precision_mode: bool = False
并将`self.reset_state()`替换成
```
        if precision_mode:
            self._pos_emb.weight.data.fill_(1)
        else:
            self.reset_state()
```
通过开启precision_mode，开启精度模式，可以消除随机性对模型精度进行对比定位