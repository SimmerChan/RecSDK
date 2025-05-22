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

from utils.logger import default_logger

class PatternModel(torch.nn.Module):
    def __init__(self):
        super(PatternModel, self).__init__()

    def forward(self, input_tensor_div1, input_tensor_div2):
        # Step 1: 输出shape 为 (135780963)
        div_result = torch.div(input_tensor_div1, input_tensor_div2)
        # Step 2: 输出shape 为 (1)
        final_result = torch.sum(torch.square(div_result), dim=0)
        return final_result

def main():
    # 示例输入
    input_tensor_div1 = torch.randn([135780963])
    input_tensor_div2 = torch.randn([135780963])
    model = PatternModel()

    output_tensor = model(input_tensor_div1, input_tensor_div2)

    # 打印输出形状
    default_logger.info("Output shape: %s", output_tensor.shape)  # 应该输出: torch.Size([128, 192, 256])

if __name__ == "__main__":
    main()