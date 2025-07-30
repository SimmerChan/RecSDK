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
import logging
import os
import subprocess
import config
import torch
import torch_npu
import sysconfig

from test_read_benchmark import logger, DATASETS

torch.npu.config.allow_internal_format = False
torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

# def find_ration_in_string(input_str):
#     separator = "+===========================+===============+====================================================+"
#     parts = input_str.split(separator)

#     npu_pattern = re.complie('r')

def read_data_from_path(save_dir, device='cpu'):
    logger.info("start read...")
    grad = torch.load(os.path.join(save_dir, "grad.pth"), map_location=device)
    q = torch.load(os.path.join(save_dir, "q.pth"), map_location=device)
    k = torch.load(os.path.join(save_dir, "k.pth"), map_location=device)
    v = torch.load(os.path.join(save_dir, "v.pth"), map_location=device)
    bias = torch.load(os.path.join(save_dir, "bias.pth"), map_location=device)
    seq_offset =  torch.load(os.path.join(save_dir, "offset.pth"), map_location=device)
    mask = torch.load(os.path.join(save_dir, "invalid_attn_mask.pth"), map_location=device)
    max_seq_len = torch.load(os.path.join(save_dir, "input_max_length.pth"), map_location=device).item()
    alpha = torch.load(os.path.join(save_dir, "alpha.pth"), map_location=device)
    data_type = torch.load(os.path.join(save_dir, "data_type.pth"))
    logger.info(f"grad_shape: {grad.size()} grad_dtype:{grad.dtype}")
    logger.info(f"q_shape: {q.size()} q_dtype:{q.dtype}")
    logger.info(f"k_shape: {k.size()} k_dtype:{k.dtype}")
    logger.info(f"v_shape: {v.size()} v_dtype:{v.dtype}")
    logger.info(f"bias_shape: {bias.size() if bias else None} bias_dtype:{bias.dtype if bias else None}")
    logger.info(f"seq_offset_shape: {seq_offset.size()} seq_offset_dtype:{seq_offset.dtype}")
    logger.info(f"mask_shape: {mask.size()} mask_dtype:{mask.dtype}")
    logger.info(f"max_seq_len:{max_seq_len}")
    logger.info(f"data_type:{data_type}")
    logger.info(f"alpha:{alpha}")


    return grad, q, k, v, bias, mask, max_seq_len, seq_offset, q.shape[2], q.shape[1], data_type, alpha

def _hstu_attention_maybe_from_cache(
        num_heads: int,
        attention_dim: int,
        linear_dim: int,
        silu_value: float,
        grad: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        invalid_attn_mask: torch.Tensor,
        seq_offset: torch.Tensor,
        data_type: torch.dtype,
        device: str
        ):
        n: int = invalid_attn_mask.size(-1)
        torch.npu.set_device(device)

        q_ = q.reshape(-1, num_heads, attention_dim).to(device=device).to(data_type)
        k_ = k.reshape(-1, num_heads, attention_dim).to(device=device).to(data_type)
        v_ = v.reshape(-1, num_heads, attention_dim).to(device=device).to(data_type)
        grad = grad.to(device=device).to(data_type)

        seq_offset = seq_offset.to(device=device).tolist()

        if len(invalid_attn_mask.shape) == 2:
            invalid_attn_mask = invalid_attn_mask.repeat(len(seq_offset) - 1, num_heads, 1, 1)
        if len(invalid_attn_mask.shape) == 4 and invalid_attn_mask.shape[1] == 1:
            invalid_attn_mask = invalid_attn_mask.repeat(1, num_heads, 1, 1)

        logger.info(f"invalid_attn_mask shape: {invalid_attn_mask.shape}")

        invalid_attn_mask = invalid_attn_mask.to(device=device).to(data_type)
        mask_type = 3
        silu_value = silu_value / n
        local_cycle_nums = 100
        for _ in range(local_cycle_nums):
            grad_output = torch.ops.mxrec.hstu_dense(q_, k_, v_, invalid_attn_mask, None, mask_type, n, silu_value, 
                                                     "jagged", seq_offset)
            q_grad, k_grad, v_grad, _ = torch.ops.mxrec.hstu_dense_backward(grad, q_, k_, v_, invalid_attn_mask, None, 
                                                "jagged", mask_type, n, silu_value, seq_offset)

            torch.npu.synchronize()
            grad_output = grad_output.reshape(-1, num_heads * linear_dim)

        save_dir = DATASETS
        
        torch.save(grad_output, os.path.join(save_dir, "npu_out.pth"))
        torch.save(q_grad, os.path.join(save_dir, "npu_q.pth"))
        torch.save(k_grad, os.path.join(save_dir, "npu_k.pth"))
        torch.save(v_grad, os.path.join(save_dir, "npu_v.pth"))
                   

  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read CSV file and run a specific index benchmark")
    parser.add_argument("--index", type=int, required=True, help="index of the benchmark to run")
    args = parser.parse_args()
    
    devicex = 0
    device = f"npu:{devicex}"
    logger.info(f"device: {device}")
    read_dir = os.path.join(config.NFS_DIR, DATASETS)
    grad, q, k, v, bias, mask, max_seq_len, seq_offset, attention_dim, num_heads, data_type, alpha = \
        read_data_from_path(read_dir)

    _hstu_attention_maybe_from_cache(
         num_heads=num_heads, 
         attention_dim=attention_dim, 
         linear_dim=attention_dim, 
         silu_value=alpha,
         grad=grad,
         q=q, 
         k=k, 
         v=v, 
         invalid_attn_mask=mask,
         seq_offset=seq_offset,
         data_type=data_type,
         device=device
    )