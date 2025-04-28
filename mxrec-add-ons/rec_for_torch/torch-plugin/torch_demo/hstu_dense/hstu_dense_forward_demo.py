#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.
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

import os
import shutil
import unittest
import sysconfig

import pytest
import torch
import numpy as np
import torch.nn.functional as F
import torch_npu
import subprocess

torch.npu.config.allow_internal_format = False

torch.ops.load_library(f"{sysconfig.get_path('purelib')}/libfbgemm_npu_api.so")

device_id:int = 0
mask_tril:int = 0
mask_triu:int = 1
mask_none:int = 2
mask_custom:int = 3

def get_chip():
    result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if "310P" in line:
            return True
    return False

def skip_seq_len(seq_len):
    block_len = 128
    if (get_chip() and seq_len % block_len) :
        return True
    return False

def jagged_data_gen(batch_size, max_seq_len, num_heads, attention_dim, dataType, maskType):
    seq_lens = np.random.randint(1, max_seq_len + 1, (batch_size))

    seq_offset = torch.concat((torch.zeros((1, ), dtype=torch.int64), \
        torch.cumsum(torch.from_numpy(seq_lens), axis=0))).to(torch.int64).numpy()
    
    max_seq_len = np.max(seq_lens)
    total_seqs = np.sum(seq_lens)

    q = torch.rand(total_seqs, num_heads, attention_dim).to(torch.float32)
    q = q.uniform_(-1, 1)
    k = torch.rand(total_seqs, num_heads, attention_dim).to(torch.float32)
    k = k.uniform_(-1, 1)
    v = torch.rand(total_seqs, num_heads, attention_dim).to(torch.float32)
    v = v.uniform_(-1, 1)

    rel_attn_bias = torch.zeros(batch_size, num_heads, max_seq_len, max_seq_len).to(torch.float32)
    for batch_id in range(batch_size):
        seq_len = seq_lens[batch_id]
        rel_attn_bias[batch_id, :, 0:seq_len, 0:seq_len] = torch.rand(seq_len, seq_len).to(torch.float32)

    if maskType == mask_tril:
        invalid_attn_mask = 1 - torch.triu(torch.ones(batch_size, num_heads, max_seq_len, max_seq_len), diagonal=1)
    else:
        invalid_attn_mask = torch.randint(0, 2, size=(batch_size, num_heads, max_seq_len, max_seq_len))
    invalid_attn_mask = invalid_attn_mask.cpu().to(torch.float32)

    return q, k, v, seq_offset, rel_attn_bias, invalid_attn_mask, max_seq_len


def generate_tensor(batch_size, max_seq_len, num_heads, attention_dim, dataType, maskType):
    total_num = batch_size*max_seq_len*num_heads*attention_dim

    q = torch.rand(total_num).reshape(batch_size, max_seq_len, num_heads, attention_dim)
    k = torch.rand(total_num).reshape(batch_size, max_seq_len, num_heads, attention_dim)
    v = torch.rand(total_num).reshape(batch_size, max_seq_len, num_heads, attention_dim)
    rel_attn_bias = torch.rand(batch_size, num_heads, max_seq_len, max_seq_len)
    if get_chip():
        invalid_attn_mask = torch.randint(0, 2, (max_seq_len, max_seq_len))
        invalid_attn_mask = torch.tril(invalid_attn_mask)
        invalid_attn_mask = invalid_attn_mask.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1)
    elif maskType == mask_tril:
        invalid_attn_mask = 1 - torch.triu(torch.ones(batch_size, num_heads, max_seq_len, max_seq_len), diagonal=1)
    else:
        invalid_attn_mask = torch.randint(0, 2, size=(batch_size, num_heads, max_seq_len, max_seq_len))
    return q.to(dataType).to(f"npu:{device_id}"), k.to(dataType).to(f"npu:{device_id}"), v.to(dataType).to(f"npu:{device_id}"), rel_attn_bias.to(dataType).to(f"npu:{device_id}"), invalid_attn_mask.to(dataType).to(f"npu:{device_id}")

torch.npu.set_device(device_id)

class TestHstuJaggedDemo:
    def jagged_to_dense(self, jagged_tensor, seq_lens, head_nums, atten_dim):
        need_pad_seq = []
        offset = 0
        for batch_id, seq_len in enumerate(seq_lens):
            src_tensor = jagged_tensor[offset: offset + seq_len, :, :].reshape(seq_len, head_nums, atten_dim)
            need_pad_seq.append(src_tensor)
            offset = offset + seq_len
        
        dense_tensor = torch.nn.utils.rnn.pad_sequence(need_pad_seq, batch_first=True)
        return dense_tensor
    

    def dense_to_jagged(self, q, dense_tensor, seq_lens):
        tensor = torch.zeros_like(q).cpu()

        offset = 0
        for batch_id, seq_len in enumerate(seq_lens):
            tensor[offset : offset + seq_len, :, :] = dense_tensor[batch_id, 0: seq_len, :, :]
            offset = offset + seq_len

        return tensor
    

    def gloden_op_exec(self, q, k, v, seq_offset, bias, mask, max_seq_len, enableBias, maskType, siluScale, dataType):
        head_nums = q.shape[1]
        head_dim = q.shape[2]
        batch_size = bias.shape[0]

        seq_lens = np.zeros((batch_size, )).astype(np.int64)
        for batch_id in range(batch_size):
            seq_lens[batch_id] = seq_offset[batch_id + 1] - seq_offset[batch_id]

        siluScale = 1 / max_seq_len if siluScale == 0 else siluScale

        q_dens = self.jagged_to_dense(q, seq_lens, head_nums, head_dim).to(dataType).to(f"npu:{device_id}")
        k_dens = self.jagged_to_dense(k, seq_lens, head_nums, head_dim).to(dataType).to(f"npu:{device_id}")
        v_dens = self.jagged_to_dense(v, seq_lens, head_nums, head_dim).to(dataType).to(f"npu:{device_id}")
        mask = mask.reshape(batch_size, head_nums, max_seq_len, max_seq_len).to(dataType).to(f"npu:{device_id}")
        attnBias = bias.reshape(batch_size, head_nums, max_seq_len, max_seq_len).to(dataType).to(f"npu:{device_id}")

        q_dens = q_dens.permute(0, 2, 1, 3)
        k_dens = k_dens.permute(0, 2, 3, 1)
        qk_attn = torch.matmul(q_dens, k_dens)

        qk_attn = qk_attn.to(torch.float32)
        attnBias = attnBias.to(torch.float32)
        mask = mask.to(torch.float32)
        if enableBias:
            qk_attn = qk_attn + attnBias

        qk_attn = F.silu(qk_attn) * siluScale

        if maskType != mask_none:
            qk_attn = qk_attn * mask

        v_dens = v_dens.permute(0, 2, 1, 3)

        qk_attn = qk_attn.to(dataType)
        atten_output = torch.matmul(qk_attn, v_dens)
        atten_output = atten_output.permute(0, 2, 1, 3).cpu()
        atten_output = self.dense_to_jagged(q, atten_output, seq_lens)

        torch.npu.synchronize()
        return atten_output.to(dataType).reshape(-1)
    
    def custom_op_exec(self, q, k, v, seq_offset, bias, mask, max_seq_len, enableBias, maskType, siluScale, dataType):
        q_npu = q.to(f"npu:{device_id}").to(dataType)
        k_npu = k.to(f"npu:{device_id}").to(dataType)
        v_npu = v.to(f"npu:{device_id}").to(dataType)
        bias_npu = bias.to(f"npu:{device_id}").to(dataType)
        mask_npu = mask.to(f"npu:{device_id}").to(dataType)

        if enableBias == True:
            output = torch.ops.mxrec.hstu_dense(
                q_npu, k_npu, v_npu, mask_npu, bias_npu, maskType, max_seq_len, siluScale, "jagged", seq_offset
            )
        else:
            output = torch.ops.mxrec.hstu_dense(
                q_npu, k_npu, v_npu, mask_npu, None, maskType, max_seq_len, siluScale, "jagged", seq_offset
            )
        torch.npu.synchronize()
        return output.cpu().to(dataType).reshape(-1)
    
    def execute(self, batch_size, max_seq_len, head_num, head_dim, enableBias, maskType, siluScale, dataType):
        q, k, v, seq_offset, bias, mask, max_seq_len = jagged_data_gen(batch_size, max_seq_len, head_num, head_dim, dataType, maskType)

        output = self.custom_op_exec(q, k, v, seq_offset, bias, mask, max_seq_len, enableBias, maskType, siluScale, dataType)
        gloden = self.gloden_op_exec(q, k, v, seq_offset, bias, mask, max_seq_len, enableBias, maskType, siluScale, dataType)

        if dataType == torch.bfloat16:
            res = torch.allclose(output, gloden, 1e-2, 1e-2)
        elif dataType == torch.float16:
            res = torch.allclose(output, gloden, 1e-3, 1e-3)
        else:
            res = torch.allclose(output, gloden, 1e-4, 1e-4)
        assert res == True

    @pytest.mark.parametrize("batch_size", [1, 16])
    @pytest.mark.parametrize("head_num", [2, 4])
    @pytest.mark.parametrize("max_seq_len", [15, 1024])
    @pytest.mark.parametrize("head_dim", [16, 128])
    @pytest.mark.parametrize("enableBias", [True, False])
    @pytest.mark.parametrize("maskType", [mask_tril, mask_none, mask_custom])
    @pytest.mark.parametrize("siluScale", [0, 1/1024])
    @pytest.mark.parametrize("dataType", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.skipif(get_chip(), reason="This test case is Skipped for Ascend310P.")
    def test_hstu_dens_forward(self, batch_size, head_num, max_seq_len, head_dim, enableBias, maskType, siluScale, dataType):
        self.execute(batch_size, max_seq_len, head_num, head_dim, enableBias, maskType, siluScale, dataType)

    @pytest.mark.parametrize("head_num", [2])
    @pytest.mark.parametrize("max_seq_len", [2570])
    @pytest.mark.parametrize("head_dim", [256])
    @pytest.mark.parametrize("enableBias", [True, False])
    @pytest.mark.parametrize("maskType", [mask_tril, mask_none, mask_custom])
    @pytest.mark.parametrize("siluScale", [0, 1/1024])
    @pytest.mark.parametrize("dataType", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.skipif(get_chip(), reason="This test case is Skipped for Ascend310P.")
    def test_hstu_dens_forward_128bs(self, head_num, max_seq_len, head_dim, enableBias, maskType, siluScale, dataType):
        self.execute(128, max_seq_len, head_num, head_dim, enableBias, maskType, siluScale, dataType)

    @pytest.mark.parametrize("head_num", [2])
    @pytest.mark.parametrize("max_seq_len", [16])
    @pytest.mark.parametrize("head_dim", [256])
    @pytest.mark.parametrize("enableBias", [True, False])
    @pytest.mark.parametrize("maskType", [mask_tril, mask_none, mask_custom])
    @pytest.mark.parametrize("siluScale", [0, 1/1024])
    @pytest.mark.parametrize("dataType", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.skipif(get_chip(), reason="This test case is Skipped for Ascend310P.")
    def test_hstu_dens_forward_2048bs(self, head_num, max_seq_len, head_dim, enableBias, maskType, siluScale, dataType):
        self.execute(2048, max_seq_len, head_num, head_dim, enableBias, maskType, siluScale, dataType)


class TestHstuNormalDemo:
    def gloden_op_exec(self, q, k, v, bias, mask, maskType, max_seq_len, siluScale, enableBias, dataType):
        B, n, num_heads, linear_dim = q.shape
        siluScale = 1 / max_seq_len if siluScale == 0 else siluScale
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        qk_attn = torch.matmul(q, k)

        qk_attn = qk_attn.to(torch.float32)
        bias = bias.to(torch.float32)
        mask = mask.to(torch.float32)
        if enableBias:
            qk_attn = qk_attn + bias

        qk_attn = F.silu(qk_attn) * siluScale

        if get_chip():
            mask = mask.repeat(1, num_heads, 1, 1)
            qk_attn = qk_attn * mask
        elif maskType != mask_none:
            qk_attn = qk_attn * mask

        v = v.permute(0, 2, 1, 3)

        qk_attn = qk_attn.to(dataType)
        atten_output = torch.matmul(qk_attn, v)
        atten_output = atten_output.permute(0, 2, 1, 3)
        torch.npu.synchronize()
        return atten_output.cpu().to(dataType).reshape(-1)
    
    def custom_op_exec(self, q, k, v, bias, mask, maskType, max_seq_len, siluScale, enableBias, dataType):
        if enableBias == True:
            output = torch.ops.mxrec.hstu_dense(
                q, k, v, mask, bias, maskType, max_seq_len, siluScale, "normal"
            )
        else:
            output = torch.ops.mxrec.hstu_dense(
                q, k, v, mask, None, maskType, max_seq_len, siluScale, "normal"
            )

        torch.npu.synchronize()
        return output.cpu().to(dataType).reshape(-1)
    
    def execute(self, batch_size, max_seq_len, head_num, head_dim, enableBias, maskType, siluScale, dataType):
        q, k, v, bias, mask = generate_tensor(batch_size, max_seq_len, head_num, head_dim, dataType, maskType)

        torch.npu.synchronize()

        output = self.custom_op_exec(q, k, v, bias, mask, maskType, max_seq_len, siluScale, enableBias, dataType)
        gloden = self.gloden_op_exec(q, k, v, bias, mask, maskType, max_seq_len, siluScale, enableBias, dataType)

        torch.npu.synchronize()

        if dataType == torch.bfloat16:
            res = torch.allclose(output, gloden, 1e-2, 1e-2)
        elif dataType == torch.float16:
            res = torch.allclose(output, gloden, 1e-3, 1e-3)
        else:
            res = torch.allclose(output, gloden, 1e-4, 1e-4)
        assert res == True

    max_seq_len = [1, 15, 31, 256, 768, 1023, 4095]
    paramFalse = pytest.param(False, marks=pytest.mark.skipif(get_chip(), reason="This test case is Skipped for Ascend310P."))
    paramFp32 = pytest.param(torch.float32, marks=pytest.mark.skipif(get_chip(), reason="This test case is Skipped for Ascend310P."))
    parambF16 = pytest.param(torch.bfloat16, marks=pytest.mark.skipif(get_chip(), reason="This test case is Skipped for Ascend310P."))
    paramsSeqlen = [pytest.param(i, marks=pytest.mark.skipif(skip_seq_len(i),
                    reason="This test case is Skipped for Ascend310P.")) for i in max_seq_len]
    @pytest.mark.parametrize("batch_size", [1, 16])
    @pytest.mark.parametrize("head_num", [2, 4])
    @pytest.mark.parametrize("max_seq_len", paramsSeqlen)
    @pytest.mark.parametrize("head_dim", [32, 64])
    @pytest.mark.parametrize("enableBias", [True, paramFalse])
    @pytest.mark.parametrize("maskType", [mask_tril, mask_none, mask_custom])
    @pytest.mark.parametrize("siluScale", [1/256])
    @pytest.mark.parametrize("dataType", [torch.float16, paramFp32, parambF16])
    def test_hstu_dens_normal(self, batch_size, head_num, max_seq_len, head_dim, enableBias, maskType, siluScale, dataType):
        self.execute(batch_size, max_seq_len, head_num, head_dim, enableBias, maskType, siluScale, dataType)
