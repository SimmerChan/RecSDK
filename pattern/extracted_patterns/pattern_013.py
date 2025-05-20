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

import torch
import torch.nn as nn
import torch_npu

from utils.logger import default_logger

device = torch.device("npu")


class PatternModel(nn.Module):
    def __init__(self, input_shape, start_indices, slice_len):
        """
        input_shape: 输入张量形状 (row, col)
        start_indices: 切片起始索引列表
        slice_len: 每个切片长度
        """
        super().__init__()
        self.register_buffer(
            "start_indices", torch.tensor(start_indices, dtype=torch.long)
        )
        self.slice_len = slice_len
        self.output_shape = (len(start_indices), input_shape[0], slice_len)

    def forward(self, x):
        """
        x: 输入张量形状 [batch, row, col]
        输出形状 [batch, num_slices, row, slice_len]
        """
        batch_size = x.size(0)
        outputs = []

        for idx in self.start_indices:
            start_col = idx.item()
            end_col = start_col + self.slice_len

            sliced = x[..., start_col:end_col]

            outputs.append(sliced.unsqueeze(1))

        # 拼接所有切片
        return torch.cat(outputs, dim=1)


def main():
    batch_size = 256
    input_row = 1000
    input_col = 2048
    slice_num = 4
    slice_len = 128
    start_indices = [256, 512, 768, 1024]

    torch.manual_seed(2025)
    input_tensor_cpu = torch.randn(batch_size, input_row, input_col)
    input_tensor_npu = input_tensor_cpu.to(device)

    model = PatternModel(
        input_shape=(input_row, input_col),
        start_indices=start_indices,
        slice_len=slice_len,
    )

    # 执行推理
    with torch.no_grad():
        output_cpu = model(input_tensor_cpu)
        output_npu = model(input_tensor_npu)

    # 结果验证
    default_logger.info("Input shape: %s", input_tensor_npu.shape)
    default_logger.info("Output shape: %s", output_npu.shape)

    if not torch.allclose(output_cpu, output_npu.to("cpu"), rtol=1e-3, atol=1e-3):
        default_logger.error("precision failed!!")
    else:
        default_logger.info("precision OK!")


if __name__ == "__main__":
    main()
