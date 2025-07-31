#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import math
import os
import traceback
from statistics import mean
from typing import Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange
from pynvml import (
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlInit,
    nvmlShutdown,
)


from test_read_benchmark import (
    DATASETS,
    init_result_csv_index,
    logger,
    read_and_validate_parameters,
    result_csv,
)

sm_major_version = torch.cuda.get_device_properties(0).major
sm_minor_version = torch.cuda.get_device_properties(0).minor
if sm_major_version == 9 and sm_minor_version == 0:
    from hstu_attn_interface import hstu_attn_varlen_func
elif sm_major_version == 8:
    from hstu_attn import hstu_attn_varlen_func


PERFORMANCE = True
if PERFORMANCE:
    g_iterations = 100
    g_profiler_step_start = 20
    loop = 3
else:
    g_iterations = 1
    g_profiler_step_start = 0
    loop = 3


def get_gpu_memory_info():
    nvmlInit()
    device_count = nvmlDeviceGetCount()
    memory_info = []
    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        memory_info.append((i, info.total, info.used))
    nvmlShutdown()
    return memory_info


def auto_select_gpu():
    memory_info = get_gpu_memory_info()
    min_memory_used = float('inf')
    best_gpu_index = None
    for i, total, used in memory_info:
        logger.info(f"GPU {i}: Total Memory =  {total / 1024**2} MiB, Used Memory = {used / 1024**2} MiB")
        if used < min_memory_used:
            min_memory_used = used
            best_gpu_index = i
    return best_gpu_index

gdevice = auto_select_gpu()
logger.info(f"Selected GPU device index: {gdevice}")
torch.cuda.set_device(gdevice)


def pad_input(unpadded_input, cu_seqlen, batch, seqlen):
    indices = []
    for i in range(batch):
        indices.append(
            torch.arange(seqlen * i, seqlen * i + cu_seqlen[i + 1] - cu_seqlen[i])
        )
    indices = torch.cat(indices)
    output = torch.zeros(
        (batch * seqlen),
        *unpadded_input.shape[1:],
        device=unpadded_input.device,
        dtype=unpadded_input.dtype,
    )
    output[indices] = unpadded_input
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


def pad_input_delta_q(unpadded_input, cu_seqlen_q, cu_seqlen_k, batch, seqlen):
    indices = []
    for i in range(batch):
        act_seqlen_q = (cu_seqlen_q[i + 1] - cu_seqlen_q[i]).item()
        act_seqlen_k = (cu_seqlen_k[i + 1] - cu_seqlen_k[i]).item()
        indices.append(
            torch.arange(
                seqlen * i + act_seqlen_k - act_seqlen_q, seqlen * i + act_seqlen_k
            )
        )
    indices = torch.cat(indices)
    output = torch.zeros(
        (batch * seqlen),
        *unpadded_input.shape[1:],
        device=unpadded_input.device,
        dtype=unpadded_input.dtype,
    )
    output[indices] = unpadded_input
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


def unpad_input(padded_input, cu_seqlen):
    padded_input.reshape(padded_input.size(0), padded_input.size(1), -1)
    output = []
    for i in range(len(cu_seqlen) - 1):
        output.append(padded_input[i, : (cu_seqlen[i + 1] - cu_seqlen[i]), :])
    return torch.cat(output, dim=0)


def unpad_input_delta_q(padded_input, cu_seqlen_q, cu_seqlen_k, batch, seqlen):
    padded_input.reshape(padded_input.size(0), padded_input.size(1), -1)
    output = []
    for i in range(batch):
        act_seqlen_q = (cu_seqlen_q[i + 1] - cu_seqlen_q[i]).item()
        act_seqlen_k = (cu_seqlen_k[i + 1] - cu_seqlen_k[i]).item()
        output.append(padded_input[i, act_seqlen_k - act_seqlen_q : act_seqlen_k, :])
    return torch.cat(output, dim=0)


def construct_mask(
    seqlen_c,
    seqlen,
    seqlen_t=0,
    target_group_size=1,
    window_size=(-1, -1),  # -1 means infinite window size
    seq_offsets=None,
    num_contexts=None,
    device=None,
):
    seqlen = seqlen_c + seqlen + seqlen_t
    bs = seq_offsets.size(0) - 1

    mask = torch.zeros((seqlen, seqlen), device=device, dtype=torch.bool)
    if window_size[0] < 0 and window_size[1] == 0:
        # causal mask
        for i in range(seqlen):
            mask[i, : i + 1] = True

        # context mask
        if seqlen_c != 0:
            mask = mask.unsqueeze(0).unsqueeze(0).repeat(bs, 1, 1, 1)
            for i in range(bs):
                target_start = (
                    num_contexts[i] + seq_offsets[i + 1] - seq_offsets[i]
                ).item()
                mask[i, 0, : num_contexts[i], :target_start] = True

        # target mask
        if seqlen_t != 0:
            mask = (
                mask.unsqueeze(0).unsqueeze(0).repeat(bs, 1, 1, 1)
                if mask.ndim == 2
                else mask
            )
            for i in range(bs):
                target_start = (
                    num_contexts[i] + seq_offsets[i + 1] - seq_offsets[i]
                ).item()
                # target group mask
                if target_group_size > 1:
                    group_num = math.ceil((seqlen - target_start) / target_group_size)
                    for j in range(group_num):
                        for k in range(
                            min(
                                target_group_size,
                                seqlen - target_start - j * target_group_size,
                            )
                        ):
                            mask[
                                i,
                                0,
                                target_start + j * target_group_size + k,
                                target_start : target_start + j * target_group_size,
                            ] = False
                else:
                    for j in range(target_start, seqlen):
                        mask[i, 0, j, target_start:j] = False

    # local mask
    else:
        window_size_0 = window_size[0] if window_size[0] > 0 else seqlen
        window_size_1 = window_size[1] if window_size[1] > 0 else seqlen
        for i in range(seqlen):
            mask[
                i, max(0, i - window_size_0) : min(seqlen, i + window_size_1 + 1)
            ] = True
    return mask


def gen_seq(length, mean_value, max_value, total_sum=None):
    if total_sum is None:
        total_sum = mean_value * length
        remaining_sum = total_sum - max_value
        random_ratios = np.random.random(length - 1)
        random_ratios /= np.sum(random_ratios)
        other_values = np.ceil(random_ratios * remaining_sum)
        max_value = total_sum - sum(other_values)
        sequence = np.append(other_values, max_value)
        return sequence
    else:
        logger.info(f"gen_seq with total_sum {total_sum}")
        if length == 1:
            return np.array([total_sum])
        if length == 2:
            return np.array([max_value, total_sum - max_value])
        remaining_sum = total_sum - max_value
        mean_value = (remaining_sum - 20) // (length - 2)
        min_val = remaining_sum - mean_value * (length - 2)
        sequence = [mean_value] * (length - 2)
        sequence.extend([max_value, min_val])
        return np.array(sequence)


def generate_input(
    total_len: int,
    batch_size: int,
    heads: int,
    heads_rab: Optional[int],
    max_seq_len_q: int,
    max_seq_len_k: int,
    max_context_len: int,
    max_target_len: int,
    target_group_size: int,
    attn_dim: int,
    hidden_dim: int,
    window_size: Tuple[int, int],
    dtype: torch.dtype,
    full_batch: bool,
    has_drab: bool,
    is_delta_q: bool,
):
    device_str = "cuda"
    has_context = max_context_len > 0
    has_target = max_target_len > 0
    target_group_size > 1
    num_contexts_all = gen_seq(batch_size, 673, max_context_len + max_seq_len_k + max_target_len, total_len)
    num_contexts_all = torch.from_numpy(num_contexts_all).to(device_str).int()
    # Generate lengths for context
    if max_context_len > 0:
        num_contexts = (
            torch.ones(
                (batch_size,), device=torch.device(device_str), dtype=torch.int32
            )
            * max_context_len
        )
    else:
        num_contexts = torch.zeros(
            (batch_size,), dtype=torch.int32, device=torch.device(device_str)
        )
    seq_offsets_c = torch.zeros(
        (batch_size + 1,), dtype=torch.int32, device=torch.device(device_str)
    )
    seq_offsets_c[1:] = torch.cumsum(num_contexts, dim=0)

    # Generate lengths for historial qkv
    if full_batch or (full_batch is False and max_seq_len_k == 0):
        lengths_k = (
            torch.ones((batch_size,), device=torch.device(device_str), dtype=torch.int32)
            * max_seq_len_k
        )
    else:
        lengths_k = torch.randint(
            1, max_seq_len_k + 1, size=(batch_size,), device=torch.device(device_str)
        )
    seq_offsets_k = torch.zeros(
        (batch_size + 1,), dtype=torch.int32, device=torch.device(device_str)
    )
    seq_offsets_k[1:] = torch.cumsum(lengths_k, dim=0)

    # Generate lengths for target qkv
    if has_target:
        if full_batch:
            num_targets = (
                torch.ones(
                    (batch_size,), device=torch.device(device_str), dtype=torch.int32
                )
                * max_target_len
            )
        else:
            num_targets = num_contexts_all - num_contexts
    else:
        num_targets = torch.zeros(
            (batch_size,), dtype=torch.int32, device=torch.device(device_str)
        )
    seq_offsets_t = torch.zeros(
        (batch_size + 1,), dtype=torch.int32, device=torch.device(device_str)
    )
    seq_offsets_t[1:] = torch.cumsum(num_targets, dim=0)

    # Generate lengths for delta q
    if is_delta_q:
        if full_batch:
            lengths_q = (
                torch.ones(
                    (batch_size,), device=torch.device(device_str), dtype=torch.int32
                )
                * max_seq_len_q
            )
        else:
            # lengths_q[i] is an integer between 1 and min(max_seq_len_q, lengths_k[i])
            lengths_q = torch.zeros(
                (batch_size,), device=torch.device(device_str), dtype=torch.int32
            )
            for i in range(batch_size):
                lengths_q[i] = torch.randint(
                    1,
                    min(max_seq_len_q, lengths_k[i]) + 1,
                    size=(1,),
                    device=torch.device(device_str),
                )
        seq_offsets_q = torch.zeros(
            (batch_size + 1,), dtype=torch.int32, device=torch.device(device_str)
        )
        seq_offsets_q[1:] = torch.cumsum(lengths_q, dim=0)
    else:
        seq_offsets_q = seq_offsets_k

    # Lengths for whole q, kv
    seq_offsets_q_wt = torch.zeros(
        (batch_size + 1,), dtype=torch.int32, device=torch.device(device_str)
    )
    seq_offsets_q_wt = seq_offsets_c + seq_offsets_q + seq_offsets_t
    seq_offsets_k_wt = torch.zeros(
        (batch_size + 1,), dtype=torch.int32, device=torch.device(device_str)
    )
    seq_offsets_k_wt = seq_offsets_c + seq_offsets_k + seq_offsets_t

    l_q = int(seq_offsets_q_wt[-1].item())
    l_k = int(seq_offsets_k_wt[-1].item())
    if dtype == torch.float8_e4m3fn:
        dtype_init = torch.float16
    else:
        dtype_init = dtype

    # Generate q, k, v for history + target
    q = (
        torch.empty(
            (l_q, heads, attn_dim), dtype=dtype_init, device=torch.device(device_str)
        )
        .uniform_(-1, 1)
        .requires_grad_()
    ).to(dtype)
    k = (
        torch.empty(
            (l_k, heads, attn_dim), dtype=dtype_init, device=torch.device(device_str)
        )
        .uniform_(-1, 1)
        .requires_grad_()
    ).to(dtype)
    v = (
        torch.empty(
            (l_k, heads, hidden_dim), dtype=dtype_init, device=torch.device(device_str)
        )
        .uniform_(-1, 1)
        .requires_grad_()
    ).to(dtype)
    rab = None
    if has_drab:
        rab = torch.empty(
            (
                batch_size,
                heads if heads_rab is None else heads_rab,
                max_context_len + max_seq_len_k + max_target_len,
                max_context_len + max_seq_len_k + max_target_len,
            ),
            dtype=dtype_init,
            device=torch.device(device_str),
        ).uniform_(-1, 1)
        rab = rab.requires_grad_()

    if window_size[0] == -1 and window_size[1] == -1:
        attn_mask = None
    else:
        attn_mask = (
            construct_mask(
                seqlen_c=max_context_len,
                seqlen=max_seq_len_k,
                seqlen_t=max_target_len,
                target_group_size=target_group_size,
                window_size=window_size,
                num_contexts=num_contexts,
                seq_offsets=seq_offsets_k,
            )
            .cuda()
            .to(torch.float32)
        )
    grad = torch.rand_like(v)
    return (
        l_q,
        l_k,
        num_contexts if has_context else None,
        seq_offsets_q_wt,
        seq_offsets_k_wt,
        num_targets if has_target else None,
        q,
        k,
        v,
        rab,
        attn_mask,
        grad
    )


def _hstu_attention_maybe_from_cache(
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    seqlen_q: int,
    seqlen_k: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_offsets: torch.Tensor,
    k_offsets: torch.Tensor,
    rab: Optional[torch.Tensor],
    invalid_attn_mask: torch.Tensor,
    alpha: float,
    upcast: bool = True,
    reorder_op: bool = False,
    is_delta_q: bool = False,
):
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    batch: int = q_offsets.size(0) - 1
    dtype_out = q.dtype
    if is_delta_q:
        padded_q = pad_input_delta_q(q, q_offsets, k_offsets, batch, seqlen_k)
    else:
        padded_q = pad_input(q, q_offsets, batch, seqlen_q)
    padded_k = pad_input(k, k_offsets, batch, seqlen_k)
    padded_v = pad_input(v, k_offsets, batch, seqlen_k)

    padded_q = padded_q.view(batch, seqlen_k, num_heads, attention_dim)
    padded_k = padded_k.view(batch, seqlen_k, num_heads, attention_dim)
    padded_v = padded_v.view(batch, seqlen_k, num_heads, linear_dim)
    if upcast:
        padded_q, padded_k, padded_v = (
            padded_q.float(),
            padded_k.float(),
            padded_v.float(),
        )
        if rab is not None:
            rab = rab.float()
    qk_attn = torch.einsum(
        "bnhd,bmhd->bhnm",
        padded_q,
        padded_k,
    )

    if rab is not None:
        padding = (
            0,
            qk_attn.shape[-1] - rab.shape[-1],
            0,
            qk_attn.shape[-2] - rab.shape[-2],
        )
        rab = F.pad(rab, padding, value=0)
        masked_qk_attn = qk_attn + rab
    else:
        masked_qk_attn = qk_attn
    masked_qk_attn = masked_qk_attn * alpha
    masked_qk_attn = F.silu(masked_qk_attn)
    masked_qk_attn = masked_qk_attn / seqlen_q
    if invalid_attn_mask is not None:
        if invalid_attn_mask.ndim == 2:
            invalid_attn_mask = invalid_attn_mask.unsqueeze(0).unsqueeze(0)
        masked_qk_attn = (
            masked_qk_attn * invalid_attn_mask.type(masked_qk_attn.dtype)[:, :, :, :]
        )

    attn_output = torch.einsum(
        "bhnm,bmhd->bnhd",
        masked_qk_attn,
        padded_v,
    )

    attn_output = attn_output.reshape(batch, seqlen_k, num_heads * linear_dim)
    if is_delta_q:
        attn_output = unpad_input_delta_q(
            attn_output, q_offsets, k_offsets, batch, seqlen_k
        )
    else:
        attn_output = unpad_input(attn_output, q_offsets)
    attn_output = attn_output.reshape(-1, num_heads * linear_dim)

    return attn_output.to(dtype_out)


def test_fused_attn(
    total_len: int,
    batch_size: int,
    heads: int,
    heads_rab: Optional[int],
    max_seq_len_q: int,
    max_seq_len_k: int,
    max_context_len: int,
    max_target_len: int,
    target_group_size: int,
    attn_dim: int,
    hidden_dim: int,
    alpha: float,
    has_rab: bool,
    has_drab: bool,
    window_size: Tuple[int, int],
    run_benchmark: Optional[int],
    dtype: torch.dtype,
    full_batch: bool,
    is_delta_q: bool,
    image_name: str,
):
    has_context = max_context_len > 0
    has_target = max_target_len > 0
    group_target = target_group_size > 1
    is_causal = window_size[0] == -1 and window_size[1] == 0
    if dtype == torch.float8_e4m3fn:
        raise ValueError(
            "float8_e4m3fn is not supported, please use test_fused_attn_fp8 instead"
        )
    if has_drab and not has_rab:
        raise ValueError("has_drab is True but has_rab is False")
    if (has_target and not is_causal) or (
        has_target and (window_size[0] > 0 or window_size[1] > 0)
    ):
        raise ValueError(
            "has_target is True but is_causal is False or window_size is not (-1, -1)"
        )
    if (max_seq_len_q != max_seq_len_k) and not is_delta_q:
        raise ValueError("max_seq_len_q != max_seq_len_k but is_delta_q is False")
    if is_delta_q and max_seq_len_q > max_seq_len_k:
        raise ValueError("is_delta_q is True but max_seq_len_q > max_seq_len_k")
    if group_target and not has_target:
        raise ValueError("group_target is True but has_target is False")
    # TODO: find a better way to avoid these combinations
    if is_delta_q and has_target:
        return
    if is_delta_q and has_context:
        return
    if not is_causal and has_context:
        return
    if (window_size[0] > 0 or window_size[1] > 0) and has_context:
        return

    torch.cuda.synchronize()
    if run_benchmark is not None:
        if run_benchmark not in [
            0b01,
            0b10,
            0b11
        ]: # 0b01 is run hstu benchmark and 0b10 is run torch benchmark, 0b11 is run both and compare precision
            raise ValueError(
                "run_benchmark should be in [0b01, 0b10, 0b11]"
            )
        iterations = g_iterations
        profiler_step_start = g_profiler_step_start

        
        input_data = generate_input(
            total_len=total_len,
            batch_size=batch_size,
            heads=heads,
            heads_rab=heads_rab,
            max_seq_len_q=max_seq_len_q,
            max_seq_len_k=max_seq_len_k,
            max_context_len=max_context_len,
            max_target_len=max_target_len,
            target_group_size=target_group_size,
            attn_dim=attn_dim,
            hidden_dim=hidden_dim,
            window_size=window_size,
            dtype=dtype,
            full_batch=full_batch,
            has_drab=has_drab,
            is_delta_q=is_delta_q,
        )
        
        (lq, lk, num_contexts, seq_offsets_q, seq_offsets_k, num_targets, q, k, v, rab, attn_mask, grad) = input_data
        
        logger.info(f"max_context_len: {max_context_len}, max_seq_len_q: {max_seq_len_q}, "
                    f"max_target_len: {max_target_len}")
        logger.info(f"q.shape: {q.shape}")
        logger.info(f"k.shape: {k.shape}")
        logger.info(f"v.shape: {v.shape}")
        logger.info(f"grad.shape: {grad.shape}")
        logger.info(f"attn_mask.shape: {attn_mask.shape}")
        logger.info(f"seq_offsets_q.shape: {seq_offsets_q.shape}")
        logger.info(f"seq_offsets_k.shape: {seq_offsets_k.shape}")
        logger.info(f"total_max_seq_len_q: {max_seq_len_q + max_context_len + max_target_len}")
        logger.info(f"total_max_seq_len_k: {max_seq_len_k + max_context_len + max_target_len}")
        logger.info(f"num_contexts.shape: {num_contexts.shape if (has_context and run_benchmark & 0b01) else None}")
        logger.info(f"num_targets.shape: {num_targets.shape if (has_target and run_benchmark & 0b01) else None}")
        logger.info(f"target_group_size: {target_group_size}")
        logger.info(f"window_size: {window_size}")
        logger.info(f"alpha: {alpha}")
        logger.info(f"rab.shape: {rab.shape if has_rab else None}")
        logger.info(f"has_drab: {has_drab}")
        logger.info(f"is_delta_q: {is_delta_q}")


        fwd_event_start = torch.cuda.Event(enable_timing=True)
        fwd_event_stop = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        for i in range(iterations):
            if i == profiler_step_start:
                fwd_event_start.record()
           
            if run_benchmark & 0b01:
                out_hstu = hstu_attn_varlen_func(
                    q=q,
                    k=k,
                    v=v,
                    seq_offsets_q=seq_offsets_q,
                    seq_offsets_k=seq_offsets_k,
                    max_seqlen_q=max_context_len + max_seq_len_q + max_target_len,
                    max_seqlen_k=max_context_len + max_seq_len_k + max_target_len,
                    num_contexts=num_contexts if has_context else None,
                    num_targets=num_targets if has_target else None,
                    target_group_size=target_group_size,
                    window_size=window_size,
                    alpha=alpha,
                    rab=rab if has_rab else None,
                    has_drab=has_drab,
                    is_delta_q=is_delta_q,
                )
            if run_benchmark & 0b10:
                out_torch = _hstu_attention_maybe_from_cache(
                    num_heads=heads,
                    attention_dim=attn_dim,
                    linear_dim=hidden_dim,
                    seqlen_q=max_context_len + max_seq_len_q + max_target_len,
                    seqlen_k=max_context_len + max_seq_len_k + max_target_len,
                    q=q.view(lq, -1),
                    k=k.view(lk, -1),
                    v=v.view(lk, -1),
                    q_offsets=seq_offsets_q,
                    k_offsets=seq_offsets_k,
                    rab=rab if has_rab else None,
                    invalid_attn_mask=attn_mask.to(torch.float32)
                    if attn_mask is not None
                    else None,
                    alpha=alpha,
                    upcast=False,
                    reorder_op=True,
                    is_delta_q=is_delta_q,
                )
                out_torch = out_torch.reshape(-1, heads, attn_dim)
        fwd_event_stop.record()
        torch.cuda.synchronize()
        fwd_time = fwd_event_start.elapsed_time(fwd_event_stop) / (
            iterations - profiler_step_start
        )


        bwd_event_start = torch.cuda.Event(enable_timing=True)
        bwd_event_stop = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        for i in range(iterations):
            if i == profiler_step_start:
                bwd_event_start.record()

            autograd_input = (q, k, v, rab) if has_rab else (q, k, v)
            if run_benchmark & 0b01:
                dq_hstu, dk_hstu, dv_hstu = torch.autograd.grad(out_hstu, autograd_input, grad, retain_graph=True)
            if run_benchmark & 0b10:
                dq_torch, dk_torch, dv_torch = torch.autograd.grad(out_torch, autograd_input, grad, retain_graph=True)
        bwd_event_stop.record()
        torch.cuda.synchronize()
        bwd_time = bwd_event_start.elapsed_time(bwd_event_stop) / (
            iterations - profiler_step_start
        )

        if run_benchmark & 0b11 == 0b11:
            if dtype == torch.bfloat16:
                eps = 1e-2
            elif dtype == torch.float16:
                eps = 1e-3
            else:
                eps = 1e-4
        
            out_close = torch.allclose(out_hstu, out_torch, eps, eps)
            q_close = torch.allclose(dq_hstu, dq_torch, eps, eps)
            k_close = torch.allclose(dk_hstu, dk_torch, eps, eps)
            v_close = torch.allclose(dv_hstu, dv_torch, eps, eps)

            logger.info(f"cpu_out vs gpu_out: {out_close}")
            logger.info(f"cpu_q vs gpu_q: {q_close}")
            logger.info(f"cpu_k vs gpu_k: {k_close}")
            logger.info(f"cpu_v vs gpu_v: {v_close}")
            logger.info(f"all pass: {(out_close and q_close and k_close and v_close)}")

        save_dir = DATASETS
        prefix = "gpu_"
        save_data(attn_mask, dk_hstu, dq_hstu, dtype, dv_hstu, out_hstu, grad, k, max_context_len, max_seq_len_q,
            max_target_len, num_contexts, num_targets, prefix, q, rab, alpha, save_dir, seq_offsets_q, v, image_name)       


        return fwd_time, bwd_time


def save_mask(matrix, title='matrix'):
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "CustomMap", [(1, 1, 1), (0.8, 0.902, 0.8)], N=256
    )

    max_pixels = 65536
    dpi = 100

    width_inches = min(12, max(6, matrix.shape[1] * 0.05)) 
    height_inches = min(12, max(6, matrix.shape[0] * 0.05))

    width_pixels = width_inches * dpi
    height_pixels = height_inches * dpi

    if width_pixels > max_pixels or height_pixels > max_pixels:
        scale_factor = min(max_pixels / width_pixels, max_pixels / height_pixels)
        width_inches *= scale_factor
        height_inches *= scale_factor
    
    fig, ax = plt.subplots(figsize=(width_inches, height_inches))

    img = ax.imshow(matrix, cmap=cmap, origin='lower', vmin=0, vmax=1, interpolation="nearest")

    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    ax.invert_yaxis()

    plt.colorbar(img)
    plt.title(title)
    plt.xlabel('seq_len')
    plt.ylabel('seq_len')
    plt.savefig(f"{title}.png", bbox_inches='tight', dpi=dpi)
    plt.close(fig)


def save_data(attn_mask, dk_hstu, dq_hstu, dtype, dv_hstu, out_hstu, g, k, max_context_len, max_seq_len_q,
            max_target_len, num_contexts, num_targets, prefix, q, rab, alpha, save_dir, seq_offsets_q, v, image_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    torch.save(g, os.path.join(save_dir, "grad.pth"))
    torch.save(q, os.path.join(save_dir, "q.pth"))
    torch.save(k, os.path.join(save_dir, "k.pth"))
    torch.save(v, os.path.join(save_dir, "v.pth"))
    torch.save(rab, os.path.join(save_dir, "bias.pth"))
    torch.save(alpha, os.path.join(save_dir, "alpha.pth"))
    torch.save(attn_mask, os.path.join(save_dir, "invalid_attn_mask.pth"))
    torch.save(torch.tensor(max_context_len + max_seq_len_q + max_target_len),
               os.path.join(save_dir, "input_max_length.pth"))
    torch.save(seq_offsets_q, os.path.join(save_dir, "offset.pth"))
    torch.save(dtype, os.path.join(save_dir, "data_type.pth"))
    torch.save(max_context_len, os.path.join(save_dir, "max_context_len.pth"))
    torch.save(max_seq_len_q, os.path.join(save_dir, "max_seq_len_q.pth"))
    torch.save(max_target_len, os.path.join(save_dir, "max_target_len.pth"))
    torch.save(num_contexts, os.path.join(save_dir, "num_contexts.pth"))
    torch.save(num_targets, os.path.join(save_dir, "num_targets.pth"))
    
    logger.info(f"prefix: {prefix}")
    

    torch.save(out_hstu.to("cpu"), os.path.join(save_dir, f"{prefix}out.pth"))
    torch.save(dq_hstu.to("cpu"), os.path.join(save_dir, f"{prefix}q.pth"))
    torch.save(dk_hstu.to("cpu"), os.path.join(save_dir, f"{prefix}k.pth"))
    torch.save(dv_hstu.to("cpu"), os.path.join(save_dir, f"{prefix}v.pth"))
    
    if len(attn_mask.shape) == 2:
        save_matrix = attn_mask.to("cpu")
    elif len(attn_mask.shape) == 3:
        save_matrix = attn_mask.to("cpu")[0]
    elif len(attn_mask.shape) == 4:
        save_matrix = attn_mask.to("cpu")[0, 0]
    
    save_mask(save_matrix, os.path.join(save_dir, image_name))
    
    torch.save(torch.tensor(1), os.path.join(save_dir, "complete.flag"))
    logger.info("save complete")


def main():
    parser = argparse.ArgumentParser(description="Read CSV file and run a specific index benchmark")
    parser.add_argument("--index", type=int, required=True, help="index of the benchmark to run")
    args = parser.parse_args()

    _, params = read_and_validate_parameters(args.index)

    init_result_csv_index(args.index)
    df_res = pd.read_csv(result_csv)
    select_mask = df_res['index'] == args.index
    if df_res.loc[select_mask, 'gpu_fw_time'].notna().all() and \
        df_res.loc[select_mask, 'gpu_bw_time'].notna().all():
        logger.info(f"bemchmark {args.index} already")
        exit(0)
    
    if params is None:
        logger.error("Invalid data, benchmark failed")
        return
    
    fwd_time_list = []
    bwd_time_list = []
    
    for i in range(loop):
        try:
            fwd_time, bwd_time = test_fused_attn(**params)
            logger.info(f"iter: {i}, fwd time: {fwd_time}, bwd time: {bwd_time}")
            fwd_time_list.append(fwd_time)
            bwd_time_list.append(bwd_time)
        except Exception as e:
            logger.error(e)
            logger.error(traceback.format_exc())
            return
        
    df_res.loc[select_mask, "gpu_fw_time"] = mean(fwd_time_list[1:])
    df_res.loc[select_mask, "gpu_bw_time"] = mean(bwd_time_list[1:])
    df_res.to_csv(result_csv, index=False)

    logger.info(f"Forward gpu time = {mean(fwd_time_list[1:])} ms")
    logger.info(f"Backward gpu time = {mean(bwd_time_list[1:])} ms") 
        

if __name__ == "__main__":
    main()