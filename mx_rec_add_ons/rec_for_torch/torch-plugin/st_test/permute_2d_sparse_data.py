# !/usr/bin/env python3
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

# input
input_permute = np.arange(26).astype(np.int32)
np.random.shuffle(input_permute)
input_lengths = np.ones((26, 204800), dtype=np.int64)
input_values = np.arange(0, 5324800).astype(np.int64)
input_weights = np.arange(0, 5324800).astype(np.float32)
input_permute_sum = 5324800


def get_result(device):
    input_permute_torch = torch.from_numpy(input_permute).to(torch.int32).to(device)
    input_lengths_torch = torch.from_numpy(input_lengths).to(torch.int64).to(device)
    input_values_torch = torch.from_numpy(input_values).to(torch.int64).to(device)
    input_weights_torch = torch.from_numpy(input_weights).to(torch.float32).to(device)

    (permuted_lengths, permuted_values, permuted_weights) = torch.ops.fbgemm.permute_2D_sparse_data(input_permute_torch,
                                                                                                    input_lengths_torch,
                                                                                                    input_values_torch,
                                                                                                    input_weights_torch,
                                                                                                    input_permute_sum)

    return permuted_lengths.cpu(), permuted_values.cpu()


gloden = get_result(torch.device("cpu"))
npu_result = get_result(torch.device("npu"))
result_forward = torch.abs(gloden[0] - npu_result[0]) < 0.0001
result_grad = torch.abs(gloden[1] - npu_result[1]) < 0.0001
logging.info(result_forward.all().item())
logging.info(result_grad.all().item())
