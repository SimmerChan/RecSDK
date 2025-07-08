#!/usr/bin/env python3
# Copyright (c) Huawei Platforms, Inc. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Union

import torch
from hybrid_torchrec.distributed.embeddingbag import _pin_and_move


def test_pin_and_move_cpu():
    device = torch.device("cpu")
    tensor = torch.tensor([1, 2, 3])

    result = _pin_and_move(tensor, device)

    assert result.device.type == "cpu"
    assert torch.equal(result, tensor)
    assert not result.is_pinned()  # CPU上不应被pin
