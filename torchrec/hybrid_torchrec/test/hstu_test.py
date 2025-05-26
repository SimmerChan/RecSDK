import os
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch_npu

torch.ops.load_library("/home/lh/torch-plugin/torch_library/2.1.0/hstu/build/libhstu_dense_ops.so")
device = "npu:0"

# class HstuFusion(torch.autograd.Function):
#     # q_, k_, v_, None, None, mask_type, "jagged", prefetch_lengths, n
#
#     @staticmethod
#     def forward(ctx, q, k, v, mask, bias, mask_type, mode, seq_offset, max_seq_len):
#         silu_value = 1 / max_seq_len
#         out = torch.ops.mxrec.hstu_dense(q, k, v, mask, bias, mask_type, max_seq_len, silu_value, mode, seq_offset)
#         ctx.save_for_backward(q, k ,v, mask, bias)
#         ctx.max_seq_len = max_seq_len
#         ctx.silu_scale = silu_value
#         ctx.seq_offset = seq_offset
#         ctx.mask_type = mask_type
#         ctx.mode = mode
#         return out
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         q, k ,v, mask, bias = ctx.saved_tensors
#         q_grad, k_grad, v_grad, bias_grad = torch.ops.mxrec.hstu_dense_backward(
#             grad_output, q, k, v, mask, bias, ctx.mode, ctx.mask_type, ctx.max_seq_len, ctx.silu_scale, ctx.seq_offset)
#         if bias is None:
#             bias_grad = None
#         # return q_grad, k_grad, v_grad, None, bias_grad, None, None, None, None
#         return q_grad, k_grad, v_grad, bias_grad

def jagged_to_dense(jagged_tensor, seq_lens, head_nums, atten_dim):
    need_pad_seq = []
    offset = 0
    for batch_id, seq_len in enumerate(seq_lens):
        src_tensor = jagged_tensor[offset: offset + seq_len, :, :].reshape(seq_len, head_nums, atten_dim)
        need_pad_seq.append(src_tensor)
        offset = offset + seq_len
    
    dense_tensor = torch.nn.utils.rnn.pad_sequence(need_pad_seq, batch_first=True)
    return dense_tensor
    

def dense_to_jagged(q, dense_tensor, seq_lens):
    tensor = torch.zeros_like(q).cpu()

    offset = 0
    for batch_id, seq_len in enumerate(seq_lens):
        tensor[offset : offset + seq_len, :, :] = dense_tensor[batch_id, 0: seq_len, :, :]
        offset = offset + seq_len

    return tensor

def gloden_op_exec(q, k, v, seq_offset, bias, mask, max_seq_len, enableBias, maskType, siluScale, dataType):
    head_nums = q.shape[1]
    head_dim = q.shape[2]
    batch_size = 32

    seq_lens = np.zeros((batch_size, )).astype(np.int64)
    for batch_id in range(batch_size):
        seq_lens[batch_id] = seq_offset[batch_id + 1] - seq_offset[batch_id]

    siluScale = 1 / max_seq_len if siluScale == 0 else siluScale

    q_dens = jagged_to_dense(q, seq_lens, head_nums, head_dim).to(dataType).to(device=device)
    k_dens = jagged_to_dense(k, seq_lens, head_nums, head_dim).to(dataType).to(device=device)
    v_dens = jagged_to_dense(v, seq_lens, head_nums, head_dim).to(dataType).to(device=device)
    mask = mask.reshape(batch_size, head_nums, max_seq_len, max_seq_len).to(dataType).to(device=device)
    attnBias = bias.reshape(batch_size, head_nums, max_seq_len, max_seq_len).to(dataType).to(device=device)

    q_dens = q_dens.permute(0, 2, 1, 3)
    k_dens = k_dens.permute(0, 2, 3, 1)
    qk_attn = torch.matmul(q_dens, k_dens)

    qk_attn = qk_attn.to(torch.float32)
    attnBias = attnBias.to(torch.float32)
    mask = mask.to(torch.float32)
    if enableBias:
        qk_attn = qk_attn + attnBias

    qk_attn = F.silu(qk_attn) * siluScale

    qk_attn = qk_attn * mask

    v_dens = v_dens.permute(0, 2, 1, 3)

    qk_attn = qk_attn.to(dataType)
    atten_output = torch.matmul(qk_attn, v_dens)
    atten_output = atten_output.permute(0, 2, 1, 3).cpu()
    atten_output = dense_to_jagged(q, atten_output, seq_lens)

    torch.npu.synchronize()
    return atten_output.to(dataType).reshape(-1)


def _hstu_attention_maybe_from_cache(
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    invalid_attn_mask: torch.Tensor,
    seq_offset: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n: int = invalid_attn_mask.size(-1)    # max_seq_len
    torch.npu.set_device(device)

    print(q.dtype)
    print(f"max_seq_len", n)
    gloden_out = gloden_op_exec(q, k, v, seq_offset, None, invalid_attn_mask, 2408, False, 3, 1 / n, q.dtype)

    q_ = q.reshape(-1, num_heads, attention_dim).to(device=device).to(torch.bfloat16)
    k_ = k.reshape(-1, num_heads, attention_dim).to(device=device).to(torch.bfloat16)
    v_ = v.reshape(-1, num_heads, attention_dim).to(device=device).to(torch.bfloat16)
    seq_offset = seq_offset.to(device=device).tolist()
    #print("seq_offset", seq_offset)
    #print("invalid_attn_mask shape berfor", invalid_attn_mask.shape)
    # 96,2408,2408 -> 32,3,2408,2408
    invalid_attn_mask = invalid_attn_mask.unsqueeze(1).repeat(1,num_heads,1,1).to(torch.bfloat16).to(device=device)

    print("invalid_attn_mask.shape", invalid_attn_mask.shape)
    print("q_.shape", q_.shape)
    print("k_.shape", k_.shape)
    print("v_.shape", v_.shape)
    mask_type = 3

    # 前向
    silu_value = 1 / n
    # torch.ops.mxrec.hstu_dense(q, k, v, mask, bias, mask_type, max_seq_len, silu_value, mode, seq_offset)
    grad_output = torch.ops.mxrec.hstu_dense(q_, k_, v_, invalid_attn_mask, None, mask_type, n, silu_value, "jagged", seq_offset)
    attn_output = grad_output
    print("attn_output.shape", attn_output.shape)
    grad_output = grad_output.to(device=device)

    # 反向
    # torch.ops.mxrec.hstu_dense_backward(grad_output, q, k, v, mask, bias, mode, mask_type, max_seq_len, silu_scale, seq_offset)
    q_grad, k_grad, v_grad, bias_grad = torch.ops.mxrec.hstu_dense_backward(grad_output.to(), q_, k_, v_, invalid_attn_mask, None, "jagged", mask_type, n, silu_value, seq_offset)
    # torch.npu.synchronize()
    bias_grad = None

    attn_output = attn_output.reshape(-1, num_heads * linear_dim)
    # return attn_output, padded_q, padded_k
    return attn_output, q_grad, k_grad, v_grad, bias_grad


def pad_with_zerospad2d(tensor, num_heads):
    b_head, s, _ = tensor.shape
    bs = b_head // num_heads
    new_s = ((s + 15) // 16) * 16
    pad_total = new_s - s
    tensor = tensor.unsqueeze(1)
    pad_layer = torch.nn.ZeroPad2d((0, pad_total, 0, pad_total))
    padded_tensor = pad_layer(tensor).squeeze(1)
    return padded_tensor


if __name__ == "__main__":

    cycle_nums = 10
    q = np.load("./online_input/q_d.npy")
    k = np.load("./online_input/k_d.npy")
    v = np.load("./online_input/v_d.npy")
    seq_offset = np.load("./online_input/offset.npy")
    max_seq_len = np.load("./online_input/input_max_length.npy")  # 2408
    invalid_attn_mask = np.load("./online_input/invalid_attn_mask.npy")  # 2432

    print(seq_offset)

    invalid_attn_mask = invalid_attn_mask[:, :max_seq_len, :max_seq_len]   # 96*2408*2408
    avg_factor = np.load("./online_input/input_avg_factor.npy")    # 1172

    # num_heads = q.shape[1]  # 3
    # attention_dim = linear_dim = q.shape[2]  # 256
    num_heads = 3  # 3
    attention_dim = linear_dim = 256  # 256

    pad_mask = pad_with_zerospad2d(torch.tensor(invalid_attn_mask), num_heads)
    print(f"pad_mask_shape", pad_mask.shape)

    for ii in range(cycle_nums):
        result = _hstu_attention_maybe_from_cache(
            num_heads=num_heads,
            attention_dim=attention_dim,
            linear_dim=linear_dim,
            q=torch.tensor(q),
            k=torch.tensor(k),
            v=torch.tensor(v),
            #invalid_attn_mask=torch.tensor(invalid_attn_mask),
            invalid_attn_mask=pad_mask,
            seq_offset=torch.tensor(seq_offset),
        )
        #print(f"==== 第{ii}次单算子测试结束 ====")

    # print("==== result:", result)
    # print("==== len:", len(result))
    # print(result[-1])

# 使用msprof计算算子性能耗时
# msprof op 会默认返回第一个算子的性能数据
# msprof op --application="python3 hstu_test.py" --output=prof

# msprof --application="python3 hstu_test.py" --output=prof
