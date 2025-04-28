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
import fbgemm_gpu
import numpy as np
import torch_npu
import torch
logging.getLogger().setLevel(logging.INFO)


DENSE_DIM = (128, 210, 1)
DENSE_DIM_1 = (40, 4, 1)
DENSE_DIMS = [DENSE_DIM, DENSE_DIM_1]


def get_result(device, dense_dim):
    dense_torch = torch.nn.Parameter(torch.from_numpy(denses).to(torch.float32)).to(device)
    dense_torch.retain_grad()

    offsets_torch = torch.from_numpy(offsets).to(torch.int64).to(device)

    jagged_id_offset = torch.ops.fbgemm.asynchronous_complete_cumsum(offsets_torch)

    output_size = jagged_id_offset[-1]

    jagged_embeding = torch.ops.fbgemm.dense_to_jagged(dense_torch, [jagged_id_offset], output_size)[0]

    output_embeddings = torch.ops.fbgemm.jagged_to_padded_dense(jagged_embeding, [jagged_id_offset], 
                                                                max_lengths=[dense_dim[1]], padding_value=0.0)

    loss = torch.mean(output_embeddings)
    loss.backward()

    return jagged_embeding.cpu(), dense_torch.grad.cpu().clone()

for dense_dim in DENSE_DIMS:
    denses = np.random.randn(*dense_dim).astype(np.float32)
    offsets = np.random.randint(0, dense_dim[1], dense_dim[0])
    gloden = get_result(torch.device("cpu"), dense_dim)
    npu_result = get_result(torch.device("npu"), dense_dim)
    result_forward = torch.abs(gloden[0] - npu_result[0]) < 0.0001
    result_grad = torch.abs(gloden[1] - npu_result[1]) < 0.0001
    logging.info(result_forward.all().item())
    logging.info(result_grad.all().item())