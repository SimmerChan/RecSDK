#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import logging
from hybrid_torchrec.modules.ids_process import IdsMapper, unique
import time
import numpy as np

TEST_NUM = 10
logging.basicConfig(level=logging.DEBUG)
IDS_RANGE_TIMES = 10

def test_ids2indices_sequential(input_size=1000000):
    """Test ids2indices with sequential numbers"""

    logging.info("Testing sequential ids mapping")
    input_ids = np.random.choice(input_size*IDS_RANGE_TIMES, (input_size,), replace=False)
    input_ids = torch.from_numpy(input_ids)

    use_time = 0
    mapper = IdsMapper()
    mapper.ids2indices(input_ids)
    for _ in range(TEST_NUM):
        start = time.time()
        indices = mapper.ids2indices(input_ids)
        end = time.time()
        use_time += end - start
    logging.info("Lookup time %f", use_time / TEST_NUM)


def test_unique(input_size=1000000):
    """Test ids2indices with sequential numbers"""

    logging.info("Testing sequential ids mapping")
    input_ids = torch.randint(0, input_size // IDS_RANGE_TIMES, (input_size,))
    start_time = time.time()
    for _ in range(TEST_NUM):
        indices = unique(input_ids)
    end_time = time.time()
    logging.info("Lookup time %f", (end_time - start_time) / TEST_NUM)


if __name__ == "__main__":
    test_ids2indices_sequential()
    test_unique()
