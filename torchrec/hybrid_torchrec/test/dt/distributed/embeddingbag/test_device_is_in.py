#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Union

import pytest
import torch
from hybrid_torchrec.distributed.embeddingbag import device_is_in


@pytest.mark.parametrize("device", [torch.device("cuda:0"), torch.device("cpu"), "npu:0", "cpu"])
@pytest.mark.parametrize("check_device", [["meta", "cpu"]])
def test_device_check_func(device: Union[torch.device|str], check_device: list[str]):
    device_is_in(device, check_device)
