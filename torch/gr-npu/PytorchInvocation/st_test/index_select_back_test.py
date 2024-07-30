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


import torch
import torch_npu
import numpy as np


EMBD_DIM = (129,)
INDEX_DIM = (128, 211, 211)
weight = np.random.randn(EMBD_DIM[0]).astype(np.float32)
index = np.random.randint(0, EMBD_DIM[0], INDEX_DIM).astype(np.int64)


class EmbedRank1Select(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, index):
        result =  torch.index_select(x, dim=0, index=index)
        ctx.save_for_backward(x, index)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, index = ctx.saved_tensors
        gradX, gradIndex = torch_npu.index_select_for_rank1_backward(grad_output, x, index)
        return gradX, gradIndex


def get_loss(device):
    weightTensor = torch.nn.Parameter(torch.from_numpy(weight)).to(device)
    weightTensor.retain_grad()

    indexTensor = torch.from_numpy(index).to(device)

    result = torch.index_select(weightTensor, dim=0, index=indexTensor.view(-1))

    loss = torch.mean(result)
    loss.backward()

    grad = weightTensor.grad.cpu().clone()
    return grad



def get_loss_op(device):
    weightTensor = torch.nn.Parameter(torch.from_numpy(weight)).to(device)
    weightTensor.retain_grad()

    indexTensor = torch.from_numpy(index).to(device)

    op = EmbedRank1Select()
    result = op.apply(weightTensor, indexTensor.view(-1))

    loss = torch.mean(result)
    loss.backward()

    grad = weightTensor.grad.cpu().clone()
    return grad


gloden = get_loss(torch.device("cpu"))
npu_result = get_loss_op(torch.device("npu"))
result = torch.abs(gloden-npu_result)<0.0001
print(result.all().item())