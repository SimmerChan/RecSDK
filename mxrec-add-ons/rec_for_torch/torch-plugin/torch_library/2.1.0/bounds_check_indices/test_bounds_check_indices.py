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
#
# # Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# ==============================================================================

import logging
import random
import unittest
from itertools import accumulate

import fbgemm_gpu
import numpy as np
import torch
import torch_npu

torch.ops.load_library("./build/libbounds_check_indices.so")


class TestBoundsCheck(unittest.TestCase):
    def test_bounds_check(self):
        T = random.randint(1, 32)
        B = random.randint(1, 64)
        max_L = random.randint(1, 64)
        logging.info(f"T: {T} B: {B} max_L: {max_L}")

        rows_per_table = torch.tensor(
            np.random.randint(low=1, high=1000, size=(T,))
        ).long()
        Bs = [B] * T
        B_offsets = [0] + list(accumulate(Bs))
        Ls = np.random.randint(low=max_L // 2, high=max_L, size=(B_offsets[-1],))
        indices = [
            np.random.randint(
                low=0,
                high=rows_per_table[t] * 2,
                size=sum(Ls[B_offsets[t] : B_offsets[t + 1]]),
            )
            for t in range(T)
        ]
        indices = torch.tensor(np.concatenate(indices, axis=0)).to(torch.int64)
        indices_copy = indices.clone()
        offsets = torch.tensor([0] + np.cumsum(Ls.flatten()).tolist()).to(torch.int64)
        offsets_copy = offsets.clone()
        warning = torch.tensor([0]).long()
        warning_copy = warning.clone()

        self.assertEqual(indices.numel(), np.sum(Ls).item())
        self.assertEqual(offsets[-1], np.sum(Ls).item())

        logging.info(
            f"tables: {T}, indices: {offsets_copy[-1]}, totalBatch: {offsets_copy.numel() - 1}"
        )

        bounds_check_mode_int = 1
        # npu
        indices, offsets, warning = (
            indices.to("npu"),
            offsets.to("npu"),
            warning.to("npu"),
        )
        torch.ops.fbgemm.bounds_check_indices(
            rows_per_table.to("npu"),
            indices,
            offsets,
            bounds_check_mode_int,
            warning,
            None,
            None,
            -1,
        )
        # cpu
        torch.ops.fbgemm.bounds_check_indices(
            rows_per_table,
            indices_copy,
            offsets_copy,
            bounds_check_mode_int,
            warning_copy,
            None,
            None,
            -1,
        )
        logging.info(torch.equal(offsets.cpu(), offsets_copy))
        logging.info(torch.equal(indices.cpu(), indices_copy))
        logging.info(torch.equal(warning.cpu(), warning_copy))

        logging.info(f"npu: {warning.cpu()}, cpu: {warning_copy}")
        np.savetxt("indices_npu.txt", indices.cpu().numpy(), fmt="%d")
        np.savetxt("indices.txt", indices_copy.numpy(), fmt="%d")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()
