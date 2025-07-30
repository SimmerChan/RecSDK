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

from pattern.util import perform_test


class PatternModel(torch.nn.Module):
    def forward(self, tensors):
        # Step 1: 对每个 tensor 应用 torch.clamp
        clipped_tensors = [torch.clamp(t, min=-1.0, max=1.0) for t in tensors]

        # Step 2: 使用 torch.cat 将所有 tensor 沿 dim=1 拼接
        output_tensor = torch.cat(clipped_tensors, dim=1)  # shape: (128, 201)

        return output_tensor


def main():
    # 创建 201 个形状为 (128, 1) 的输入张量
    input_tensor = [torch.randn(128, 1) for _ in range(201)]
    input_list = [input_tensor]

    perform_test(PatternModel(), input_list)
    
if __name__ == "__main__":
    main()