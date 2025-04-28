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

import logging
import numpy as np
import torch_npu
import torch
logging.getLogger().setLevel(logging.INFO)


EMBD_DIM = (129,)
INDEX_DIM = (128, 256, 256)
weight = np.random.randn(EMBD_DIM[0]).astype(np.float32)
index_np = np.random.randint(0, EMBD_DIM[0], INDEX_DIM).astype(np.int64)


class EmbedRank1Select(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, index):
        result = torch_npu.gather_for_rank1(x, index=index)
        ctx.save_for_backward(x, index)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, index = ctx.saved_tensors
        grad_x, grad_index = torch_npu.index_select_for_rank1_backward(grad_output, x, index)
        return grad_x, grad_index


def get_loss(device):
    weight_tensor = torch.nn.Parameter(torch.from_numpy(weight)).to(torch.float64)
    weight_tensor.retain_grad()

    index_tensor = torch.from_numpy(index_np).to(torch.int64)

    result = torch.index_select(weight_tensor, dim=0, index=index_tensor.view(-1))

    loss = torch.mean(result)
    loss.backward()

    grad = weight_tensor.grad.cpu().clone()
    return result.cpu(), grad


def get_loss_op(device):
    weight_tensor = torch.nn.Parameter(torch.from_numpy(weight)).to(device)
    weight_tensor.retain_grad()

    index_tensor = torch.from_numpy(index_np).to(device)

    op = EmbedRank1Select()
    result = op.apply(weight_tensor, index_tensor.view(-1))

    loss = torch.mean(result)
    loss.backward()

    grad = weight_tensor.grad.cpu().clone()
    return result.cpu(), grad


gloden = get_loss(torch.device("cpu"))
npu_result = get_loss_op(torch.device("npu"))
result_forward = torch.abs(gloden[0] - npu_result[0]) < 0.0001
result_grad = torch.abs(gloden[1] - npu_result[1]) < 0.0001
logging.info(result_forward.all().item())
logging.info(result_grad.all().item())