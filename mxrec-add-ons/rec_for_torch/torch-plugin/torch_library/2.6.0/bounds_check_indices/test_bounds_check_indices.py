#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.

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
        indices_cpu = indices.clone()
        offsets = torch.tensor([0] + np.cumsum(Ls.flatten()).tolist()).to(torch.int64)
        offsets_cpu = offsets.clone()
        warning = torch.tensor([0]).long()
        warning_cpu = warning.clone()

        self.assertEqual(indices.numel(), np.sum(Ls).item())
        self.assertEqual(offsets[-1], np.sum(Ls).item())

        logging.info(
            f"tables: {T}, indices: {offsets_cpu[-1]}, totalBatch: {offsets_cpu.numel() - 1}"
        )

        bounds_check_mode_int = 1
        # npu
        indices_npu, offsets_npu, warning_npu = (
            indices.to("npu"),
            offsets.to("npu"),
            warning.to("npu"),
        )
        torch.ops.fbgemm.bounds_check_indices(
            rows_per_table.to("npu"),
            indices_npu,
            offsets_npu,
            bounds_check_mode_int,
            warning_npu,
            None,
            None,
            -1,
        )
        # cpu
        torch.ops.fbgemm.bounds_check_indices(
            rows_per_table,
            indices_cpu,
            offsets_cpu,
            bounds_check_mode_int,
            warning_cpu,
            None,
            None,
            -1,
        )
        logging.info(torch.equal(offsets_npu.cpu(), offsets_cpu))
        logging.info(torch.equal(indices_npu.cpu(), indices_cpu))
        logging.info(torch.equal(warning_npu.cpu(), warning_cpu))

        logging.info(f"npu: {warning_npu.cpu()}, cpu: {warning_cpu}")
        np.savetxt("indices_npu.txt", indices_npu.cpu().numpy(), fmt="%d")
        np.savetxt("indices_cpu.txt", indices_cpu.numpy(), fmt="%d")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()
